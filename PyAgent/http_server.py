"""
http_server.py  —  ZENTRAA HTTP/WebSocket Bridge  v2.0
=======================================================
Powered By : Pythonaibrain
Author     : Divyanshu Sinha

New in v2.0 (supports frontend v4.0 WhatsApp-feature-set):
  ─ Typing indicators  (TYPING / TYPING_STOP) — relayed between browser sessions
  ─ Read receipts      (READ_RECEIPT)          — relayed between browser sessions
  ─ Emoji reactions    (REACTION)              — stored + relayed bridge-side
  ─ Message delivery   (DELIVERED)             — sent when target session receives
  ─ Rich USER_LIST     (includes status text, online/offline per user)
  ─ REST /api/upload   (multipart file upload → base64 → relay via WS)
  ─ REST /api/users    (online users + metadata)
  ─ REST /api/history/{conv_id}  (last N messages — bridge-side store)
  ─ WebSocket /ws/{userid}  (unchanged path, enriched protocol)
  ─ Bridge-side message store  (last 500 msgs per conv, reactions, receipts)
  ─ Chunked large-payload support (files split into ≤128 KB chunks)
  ─ Auto port-selection, colorama banner

Architecture
─────────────────────────────────────────────────────────────
  Browser  ←── WS/JSON ──→  BridgeSession  ←── TCP/Encrypted ──→  ZENTRAServer
                                   │
                      BridgeRelay (in-process pub/sub)
                        (typing, reactions, read-receipts)
─────────────────────────────────────────────────────────────

Protocol additions (all JSON, all over the browser WebSocket):

  Browser → Bridge (client-only, NOT forwarded to TCP server):
    { type:"TYPING",        to: str|"ALL" }
    { type:"TYPING_STOP",   to: str|"ALL" }
    { type:"READ_RECEIPT",  msg_id: str, conv_id: str }
    { type:"REACTION",      msg_id: str, conv_id: str, emoji: str, action:"add"|"remove" }
    { type:"DELIVERED",     msg_id: str }
    { type:"CALL_OFFER",    to: str, sdp: str }
    { type:"CALL_ANSWER",   to: str, sdp: str }
    { type:"CALL_ICE",      to: str, candidate: str }
    { type:"CALL_END",      to: str, reason: str }

  Bridge → Browser (injected by bridge, not from TCP server):
    { type:"TYPING",        from: str, to: str }
    { type:"TYPING_STOP",   from: str, to: str }
    { type:"READ_RECEIPT",  msg_id: str, from: str, conv_id: str }
    { type:"REACTION",      msg_id: str, from: str, emoji: str, action:"add"|"remove", reactions: dict }
    { type:"DELIVERED",     msg_id: str, from: str }
    { type:"USER_LIST",     users: list[dict] }   ← enriched (was list[str])
    { type:"CONNECTED",     session_id, welcome, bridge_features: list }
    { type:"CALL_OFFER",    from: str, to: str, sdp: str }
    { type:"CALL_ANSWER",   from: str, to: str, sdp: str }
    { type:"CALL_ICE",      from: str, to: str, candidate: str }
    { type:"CALL_END",      from: str, to: str, reason: str }

  Forwarded to TCP server unchanged:
    MSG, AI_QUERY, PING, PONG, USER_LIST (plain pass-through)

Run
───
  pip install fastapi uvicorn cryptography python-multipart colorama aiofiles
  python http_server.py --http-port 5500
  python http_server.py --http-port 5500 --tcp-host 127.0.0.1 --tcp-port 9999
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import mimetypes
import re
import socket
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import .zentraa_crypto as crypto
import .zentraa_protocol as proto
from .zentraa_protocol import MT

try:
    import colorama
    colorama.init()
    _C = colorama.Fore
    _R = colorama.Style.RESET_ALL
except ImportError:
    class _C:  # type: ignore
        CYAN = GREEN = YELLOW = RED = MAGENTA = ""
    _R = ""

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("zentraa.bridge")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class BridgeConfig:
    http_host:        str   = "0.0.0.0"
    http_port:        int   = 7080
    tcp_host:         str   = "127.0.0.1"
    tcp_port:         int   = 9999
    static_dir:       str   = "static"
    rsa_bits:         int   = 2048
    ping_interval:    float = 20.0
    max_history:      int   = 500      # messages stored per conversation
    max_upload_mb:    int   = 64       # max file upload size
    chunk_size:       int   = 131072   # 128 KB WebSocket chunk size for large files
    typing_timeout:   float = 8.0      # seconds before auto-clearing typing indicator
    tcp_read_timeout: float = 120.0    # seconds before treating a silent TCP connection as dead
                                       # (fixes WinError 121 semaphore timeout on Windows)

cfg = BridgeConfig()

# Thread-pool for blocking crypto + socket calls
_pool = ThreadPoolExecutor(max_workers=64, thread_name_prefix="zbridge")

# ─────────────────────────────────────────────────────────────────────────────
# Bridge-side in-memory store
# ─────────────────────────────────────────────────────────────────────────────

class MessageStore:
    """
    Lightweight in-process store for:
      • per-conversation message history  (last cfg.max_history)
      • per-message reactions             { msg_id → { emoji → set(userids) } }
      • per-message read receipts         { msg_id → set(userids) }
      • user status text                  { userid → str }
    """

    def __init__(self) -> None:
        # conv_id → deque of message dicts
        self._history:   Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=cfg.max_history))
        # msg_id → { emoji → set(userid) }
        self._reactions: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        # msg_id → set(userid)
        self._receipts:  Dict[str, Set[str]] = defaultdict(set)
        # userid → status string
        self._status:    Dict[str, str] = {}

    def add_message(self, conv_id: str, msg: dict) -> None:
        self._history[conv_id].append(msg)

    def get_history(self, conv_id: str, limit: int = 100) -> List[dict]:
        hist = list(self._history[conv_id])
        return hist[-limit:]

    def add_reaction(self, msg_id: str, userid: str, emoji: str) -> Dict[str, List[str]]:
        self._reactions[msg_id][emoji].add(userid)
        return {e: list(u) for e, u in self._reactions[msg_id].items()}

    def remove_reaction(self, msg_id: str, userid: str, emoji: str) -> Dict[str, List[str]]:
        self._reactions[msg_id][emoji].discard(userid)
        if not self._reactions[msg_id][emoji]:
            del self._reactions[msg_id][emoji]
        return {e: list(u) for e, u in self._reactions[msg_id].items()}

    def mark_read(self, msg_id: str, userid: str) -> None:
        self._receipts[msg_id].add(userid)

    def readers(self, msg_id: str) -> List[str]:
        return list(self._receipts[msg_id])

    def set_status(self, userid: str, status: str) -> None:
        self._status[userid] = status

    def get_status(self, userid: str) -> str:
        return self._status.get(userid, "")

    def all_statuses(self) -> Dict[str, str]:
        return dict(self._status)


store = MessageStore()


# ─────────────────────────────────────────────────────────────────────────────
# Bridge Relay  —  pub/sub between browser sessions (no TCP round-trip)
# ─────────────────────────────────────────────────────────────────────────────

class BridgeRelay:
    """
    Delivers bridge-only events (typing, reactions, read receipts, delivery)
    directly between connected BridgeSessions without touching the TCP server.
    """

    async def broadcast(self, sender_id: str, payload: dict, exclude_sender: bool = True) -> None:
        """Send *payload* to all connected sessions except (optionally) *sender_id*."""
        for uid, session in list(BridgeSession._active.items()):
            if exclude_sender and uid == sender_id:
                continue
            await session._ws_send(payload)

    async def send_to(self, target_id: str, payload: dict) -> bool:
        """Send *payload* to a specific session. Returns True if delivered."""
        session = BridgeSession._active.get(target_id)
        if session:
            await session._ws_send(payload)
            return True
        return False

    async def send_typing(self, sender: str, to: str, is_typing: bool) -> None:
        payload = {
            "type": "TYPING" if is_typing else "TYPING_STOP",
            "from": sender,
            "to":   to,
            "ts":   time.time(),
        }
        if to == "ALL" or to == "broadcast":
            await self.broadcast(sender, payload)
        else:
            await self.send_to(to, payload)

    async def send_reaction(
        self, sender: str, msg_id: str, conv_id: str,
        emoji: str, action: str, reactions: dict,
    ) -> None:
        payload = {
            "type":      "REACTION",
            "from":      sender,
            "msg_id":    msg_id,
            "conv_id":   conv_id,
            "emoji":     emoji,
            "action":    action,
            "reactions": reactions,
            "ts":        time.time(),
        }
        # Broadcast to everyone (conv members see the reaction update)
        await self.broadcast(sender, payload, exclude_sender=False)

    async def send_read_receipt(self, sender: str, msg_id: str, conv_id: str) -> None:
        payload = {
            "type":    "READ_RECEIPT",
            "from":    sender,
            "msg_id":  msg_id,
            "conv_id": conv_id,
            "ts":      time.time(),
        }
        await self.broadcast(sender, payload)

    async def send_delivered(self, sender: str, msg_id: str) -> None:
        payload = {
            "type":   "DELIVERED",
            "from":   sender,
            "msg_id": msg_id,
            "ts":     time.time(),
        }
        # Find who sent this message and notify them
        # (We broadcast; the frontend filters by msg_id)
        await self.broadcast(sender, payload)


relay = BridgeRelay()


# ─────────────────────────────────────────────────────────────────────────────
# Async helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _run(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_pool, fn, *args)


async def _recv_frame(sock: socket.socket) -> bytes:
    return await _run(crypto.recv_frame, sock)


async def _send_frame(sock: socket.socket, data: bytes) -> None:
    framed = crypto.frame(data)
    await _run(sock.sendall, framed)


# ─────────────────────────────────────────────────────────────────────────────
# Bridge-only message types (handled locally, NOT forwarded to TCP server)
# ─────────────────────────────────────────────────────────────────────────────

_BRIDGE_ONLY_TYPES = {
    "TYPING",
    "TYPING_STOP",
    "READ_RECEIPT",
    "REACTION",
    "DELIVERED",
    "STATUS_UPDATE",
    # WebRTC call signalling — relayed peer-to-peer via bridge, never sent to TCP server
    "CALL_OFFER",
    "CALL_ANSWER",
    "CALL_ICE",
    "CALL_END",
}

# These pass through to the TCP server unchanged
_TCP_PASSTHROUGH_TYPES = {
    MT.MSG,
    MT.AI_QUERY,
    MT.PING,
    MT.PONG,
    MT.USER_LIST,
    "MSG",
    "AI_QUERY",
    "PING",
    "PONG",
    "USER_LIST",
}


# ─────────────────────────────────────────────────────────────────────────────
# Bridge Session
# ─────────────────────────────────────────────────────────────────────────────

class BridgeSession:
    """One browser WebSocket ↔ one ZENTRAA TCP connection."""

    _active: Dict[str, "BridgeSession"] = {}
    _lock = asyncio.Lock()

    def __init__(self, ws: WebSocket, userid: str) -> None:
        self.ws          = ws
        self.userid      = userid
        self.session_id  = ""
        self.alive       = False
        self.user_status = ""

        # Typing auto-clear tasks  { target → asyncio.Task }
        self._typing_tasks: Dict[str, asyncio.Task] = {}

        # RSA keypair for this session
        self._rsa_priv, self._rsa_pub = crypto.generate_rsa_keypair(cfg.rsa_bits)
        self._rsa_pub_pem: str = crypto.serialize_public_key(self._rsa_pub).decode()

        # Curve25519 ECDH keypair
        self._ecdh_priv, self._ecdh_pub = crypto.generate_x25519_keypair()
        from cryptography.hazmat.primitives import serialization as _ser
        _ecdh_pub_raw = self._ecdh_pub.public_bytes(_ser.Encoding.Raw, _ser.PublicFormat.Raw)
        self._ecdh_pub_hex: str = _ecdh_pub_raw.hex()

        # Server RSA public key (after handshake)
        self._server_rsa_pub = None

        self._sock: Optional[socket.socket] = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalise_to(to_field: Any) -> List[str]:
        """
        Flatten 'to' field to plain list[str] of user IDs.

        The frontend v4.0 sends rich user objects in 'to' because it builds
        MSG packets from the rich USER_LIST the bridge returns, e.g.:
          [{userid:"alice", online:true, status:"..."}]

        All shapes handled:
          []                   → []         (broadcast)
          ["alice"]            → ["alice"]  (plain string)
          [{"userid":"alice"}] → ["alice"]  (rich object)
          "alice"              → ["alice"]  (accidentally a string)
        """
        if not to_field:
            return []
        if isinstance(to_field, str):
            return [to_field] if to_field else []
        result: List[str] = []
        for item in to_field:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                uid = item.get("userid") or item.get("id") or item.get("user") or ""
                if uid:
                    result.append(str(uid))
        return result

    @staticmethod
    def _blocking_connect(host: str, port: int) -> socket.socket:
        """
        Open TCP connection with Windows-compatible keep-alive.

        WinError 121 (semaphore timeout) occurs when the remote end
        disappears silently and Windows waits 2 hours before detecting
        the dead connection. SIO_KEEPALIVE_VALS cuts that to ~30 s.

        NOTE: socket.ioctl() on Windows takes a 3-tuple of ints, NOT bytes.
        The struct.pack() form raises TypeError — fixed here.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        try:
            # Windows: socket.ioctl(SIO_KEEPALIVE_VALS, (onoff, time_ms, interval_ms))
            # Must be a 3-tuple of ints — NOT struct.pack bytes.
            # onoff=1, idle=15 000 ms (15 s), probe=3 000 ms (3 s)
            SIO_KEEPALIVE_VALS = 0x98000004
            sock.ioctl(SIO_KEEPALIVE_VALS, (1, 15000, 3000))
        except (AttributeError, OSError, TypeError):
            # Linux / macOS fallback
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE,  15)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL,  3)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT,    5)
            except (AttributeError, OSError):
                pass
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except (AttributeError, OSError):
            pass
        sock.settimeout(10.0)
        sock.connect((host, port))
        sock.settimeout(None)
        return sock

    async def _ws_send(self, obj: dict) -> None:
        try:
            await self.ws.send_json(_sanitise(obj))
        except Exception as exc:
            log.debug("[%s] ws_send failed: %s", self.userid, exc)
            self.alive = False

    async def _ws_send_raw(self, text: str) -> None:
        """Send a pre-serialised JSON string (avoids double-serialisation for large payloads)."""
        try:
            await self.ws.send_text(text)
        except Exception:
            self.alive = False

    # ── Handshake ─────────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        # TCP connect
        try:
            self._sock = await _run(self._blocking_connect, cfg.tcp_host, cfg.tcp_port)
        except OSError as exc:
            log.error("[%s] TCP connect failed: %s", self.userid, exc)
            await self._ws_send({"type": "ERROR", "reason": f"Cannot reach server: {exc}"})
            return False

        # Send HELLO
        hello = proto.build_hello(self.userid, self._rsa_pub_pem, self._ecdh_pub_hex)
        try:
            await _send_frame(self._sock, proto.encode(hello))
        except OSError as exc:
            log.error("[%s] HELLO send failed: %s", self.userid, exc)
            await self._ws_send({"type": "ERROR", "reason": "Failed to send handshake"})
            return False

        # Receive HELLO_ACK (plain)
        try:
            raw_ack = await asyncio.wait_for(_recv_frame(self._sock), timeout=12.0)
        except asyncio.TimeoutError:
            await self._ws_send({"type": "ERROR", "reason": "Handshake timed out"})
            return False
        except Exception as exc:
            await self._ws_send({"type": "ERROR", "reason": str(exc)})
            return False

        # Decode ACK
        try:
            ack = proto.decode(raw_ack)
        except Exception as exc:
            await self._ws_send({"type": "ERROR", "reason": f"Protocol error: {exc}"})
            return False

        msg_type = ack.get("type", "")
        if msg_type == MT.ERROR:
            await self._ws_send({"type": "ERROR", "reason": ack.get("reason", "Rejected")})
            return False
        if msg_type != MT.HELLO_ACK:
            await self._ws_send({"type": "ERROR", "reason": f"Unexpected reply: {msg_type}"})
            return False

        # Load server public key
        server_pem = ack.get("server_rsa_pub_pem", "")
        if not server_pem:
            await self._ws_send({"type": "ERROR", "reason": "Server key missing in ACK"})
            return False
        try:
            self._server_rsa_pub = crypto.load_public_key(server_pem.encode())
        except Exception as exc:
            await self._ws_send({"type": "ERROR", "reason": f"Bad server key: {exc}"})
            return False

        self.session_id = ack.get("session_id", "")
        motd            = ack.get("motd", "")
        self.alive      = True

        # Notify browser: connected — include bridge feature list so frontend
        # knows which extra capabilities are available
        await self._ws_send({
            "type":            "CONNECTED",
            "session_id":      self.session_id,
            "welcome":         motd,
            "bridge_version":  "2.0.0",
            "bridge_features": [
                "typing_indicators",
                "read_receipts",
                "reactions",
                "delivery_receipts",
                "rich_user_list",
                "file_upload",
                "message_history",
                "status_text",
                "voice_calls",
            ],
        })
        log.info("[%s] %sHandshake OK%s — session=%s", self.userid, _C.GREEN, _R, self.session_id)
        return True

    # ── Encrypted I/O ─────────────────────────────────────────────────────────

    async def _send_to_server(self, msg: dict) -> None:
        if not self.alive or not self._sock:
            return
        try:
            payload   = proto.encode(msg)
            encrypted = await _run(crypto.encrypt, payload, self._server_rsa_pub, self._rsa_priv)
            await _send_frame(self._sock, encrypted)
        except Exception as exc:
            log.warning("[%s] send_to_server error: %s", self.userid, exc)
            self.alive = False

    async def _recv_from_server(self) -> Optional[dict]:
        try:
            # Wrap recv in a timeout so a silently-dead TCP connection
            # (WinError 121 — semaphore timeout) is detected within
            # tcp_read_timeout seconds instead of hanging forever.
            raw = await asyncio.wait_for(
                _recv_frame(self._sock),
                timeout=cfg.tcp_read_timeout,
            )
            payload = await _run(crypto.decrypt, raw, self._rsa_priv, self._server_rsa_pub)
            return proto.decode(payload)
        except asyncio.TimeoutError:
            # Timeout means the TCP connection is dead / server went away
            if self.alive:
                log.warning("[%s] TCP read timeout — connection appears stale", self.userid)
            return None
        except Exception as exc:
            if self.alive:
                log.warning("[%s] recv_from_server error: %s", self.userid, exc)
            return None

    # ── Bridge-only event handlers ────────────────────────────────────────────

    async def _handle_bridge_event(self, data: dict) -> None:
        """Handle events that are processed by the bridge, not forwarded to TCP."""
        evt = data.get("type", "")

        if evt in ("TYPING", "TYPING_STOP"):
            to  = data.get("to", "ALL")
            active = (evt == "TYPING")
            await relay.send_typing(self.userid, to, active)

            # Auto-clear typing after timeout if client forgets to send TYPING_STOP
            key = f"{self.userid}:{to}"
            if active:
                old = self._typing_tasks.pop(key, None)
                if old: old.cancel()
                async def _auto_stop(k=key, sender=self.userid, target=to):
                    await asyncio.sleep(cfg.typing_timeout)
                    await relay.send_typing(sender, target, False)
                    self._typing_tasks.pop(k, None)
                self._typing_tasks[key] = asyncio.create_task(_auto_stop())
            else:
                old = self._typing_tasks.pop(key, None)
                if old: old.cancel()

        elif evt == "READ_RECEIPT":
            msg_id  = data.get("msg_id", "")
            conv_id = data.get("conv_id", "")
            if msg_id:
                store.mark_read(msg_id, self.userid)
                await relay.send_read_receipt(self.userid, msg_id, conv_id)

        elif evt == "REACTION":
            msg_id  = data.get("msg_id", "")
            conv_id = data.get("conv_id", "")
            emoji   = data.get("emoji", "")
            action  = data.get("action", "add")  # "add" | "remove"
            if msg_id and emoji:
                if action == "add":
                    reactions = store.add_reaction(msg_id, self.userid, emoji)
                else:
                    reactions = store.remove_reaction(msg_id, self.userid, emoji)
                await relay.send_reaction(self.userid, msg_id, conv_id, emoji, action, reactions)

        elif evt == "DELIVERED":
            msg_id = data.get("msg_id", "")
            if msg_id:
                await relay.send_delivered(self.userid, msg_id)

        elif evt == "STATUS_UPDATE":
            status = str(data.get("status", ""))[:128]
            store.set_status(self.userid, status)
            self.user_status = status
            # Broadcast updated user list to everyone
            await _broadcast_rich_user_list()

        # ── WebRTC call signalling ─────────────────────────────────────────────
        # CALL_OFFER / CALL_ANSWER / CALL_ICE are point-to-point — relay only to
        # the named target user.
        # CALL_END is sent to the named target or broadcast-to-all-my-calls.

        elif evt == "CALL_OFFER":
            target = data.get("to", "")
            if not target:
                log.warning("[%s] CALL_OFFER missing 'to' field", self.userid)
                return
            payload = {**data, "from": self.userid, "ts": time.time()}
            delivered = await relay.send_to(target, payload)
            if not delivered:
                # Target not connected — inform caller immediately
                await self._ws_send({
                    "type":   "CALL_END",
                    "from":   target,
                    "to":     self.userid,
                    "reason": "user_offline",
                    "ts":     time.time(),
                })
            log.info("[%s] %sCALL_OFFER%s → %s (delivered=%s)",
                     self.userid, _C.CYAN, _R, target, delivered)

        elif evt == "CALL_ANSWER":
            target = data.get("to", "")
            if not target:
                log.warning("[%s] CALL_ANSWER missing 'to' field", self.userid)
                return
            payload = {**data, "from": self.userid, "ts": time.time()}
            await relay.send_to(target, payload)
            log.info("[%s] %sCALL_ANSWER%s → %s", self.userid, _C.CYAN, _R, target)

        elif evt == "CALL_ICE":
            target = data.get("to", "")
            if not target:
                log.warning("[%s] CALL_ICE missing 'to' field", self.userid)
                return
            payload = {**data, "from": self.userid, "ts": time.time()}
            await relay.send_to(target, payload)

        elif evt == "CALL_END":
            target = data.get("to", "")
            payload = {**data, "from": self.userid, "ts": time.time()}
            if target:
                await relay.send_to(target, payload)
            else:
                # No specific target — notify everyone who might be in a call with us
                await relay.broadcast(self.userid, payload)
            log.info("[%s] %sCALL_END%s → %s reason=%s",
                     self.userid, _C.YELLOW, _R, target or "ALL", data.get("reason", ""))

    # ── Intercept & enrich server→browser messages ────────────────────────────

    async def _process_server_msg(self, msg: dict) -> None:
        """
        Intercept messages from the TCP server, enrich where needed,
        then forward to the browser.
        """
        msg_type = msg.get("type", "")

        if msg_type == "USER_LIST":
            # Replace plain user list with rich metadata
            plain_users = msg.get("users", [])
            rich_users  = _build_rich_user_list(plain_users)
            msg = {**msg, "users": rich_users}

        elif msg_type == "MSG":
            # Store message in history and issue a DELIVERED receipt
            msg_id  = msg.get("id") or str(uuid.uuid4())
            msg["id"] = msg_id
            from_   = msg.get("from", "")
            # FIX: normalise 'to' in server→browser direction too
            to_raw  = msg.get("to", [])
            to_     = BridgeSession._normalise_to(to_raw)
            msg["to"] = to_   # ensure browser always gets plain strings
            conv_id = "broadcast" if not to_ else (to_[0] if self.userid == from_ else from_)
            store.add_message(conv_id, msg)
            # Auto-mark delivered for this session
            if from_ != self.userid:
                await relay.send_delivered(self.userid, msg_id)

        elif msg_type == "AI_REPLY":
            msg_id     = msg.get("id") or str(uuid.uuid4())
            msg["id"]  = msg_id
            store.add_message("broadcast", msg)

        await self._ws_send(msg)

    # ── I/O loops ─────────────────────────────────────────────────────────────

    async def _tcp_reader_loop(self) -> None:
        """Forward decrypted + enriched TCP packets → browser WebSocket."""
        while self.alive:
            msg = await self._recv_from_server()
            if msg is None:
                self.alive = False
                break
            await self._process_server_msg(msg)

    async def _ws_reader_loop(self) -> None:
        """
        Read browser messages.
        Bridge-only types are handled locally.
        Everything else is forwarded to the TCP server.
        """
        try:
            while self.alive:
                raw = await self.ws.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")

                if msg_type in _BRIDGE_ONLY_TYPES:
                    # Handle locally — don't touch TCP server
                    await self._handle_bridge_event(data)

                elif msg_type in _TCP_PASSTHROUGH_TYPES or msg_type in (
                    "MSG", "AI_QUERY", "PING", "PONG", "USER_LIST"
                ):
                    # ── FIX: normalise 'to' field ──────────────────────────────
                    # The frontend v4.0 sends rich user objects in 'to' because
                    # it builds MSG packets from the rich USER_LIST the bridge
                    # returns.  Flatten to plain list[str] before forwarding.
                    # Without this fix: TypeError: unhashable type: 'dict'
                    if "to" in data:
                        data = {**data, "to": self._normalise_to(data["to"])}
                    if "targets" in data:
                        data = {**data, "targets": self._normalise_to(data["targets"])}

                    # Strip large file_data (relayed via REST /api/upload)
                    fwd = {k: v for k, v in data.items() if k != "file_data"}

                    # Ensure msg_id exists so receipts can reference it
                    if "id" not in fwd:
                        fwd["id"] = str(uuid.uuid4())

                    await self._send_to_server(fwd)

                    # Store outgoing MSG in history
                    if msg_type == "MSG":
                        to_list = fwd.get("to", [])
                        conv_id = "broadcast" if not to_list else to_list[0]
                        store.add_message(conv_id, {**fwd, "isSelf": True})

                else:
                    log.debug("[%s] Unknown msg type from browser: %s", self.userid, msg_type)

        except WebSocketDisconnect:
            log.info("[%s] %sWebSocket disconnected%s", self.userid, _C.YELLOW, _R)
        except Exception as exc:
            log.warning("[%s] WS reader error: %s", self.userid, exc)
        finally:
            self.alive = False

    async def _pinger_loop(self) -> None:
        """
        Send periodic pings to keep both the browser WebSocket and the
        TCP connection alive.

        Two separate pings:
         • WebSocket ping  — keeps the browser connection open through
                             proxies / load balancers.
         • TCP PING packet — tells the ZENTRAA server this session is
                             still alive so it doesn't disconnect us.
        """
        while self.alive:
            await asyncio.sleep(cfg.ping_interval)
            if not self.alive:
                break
            # 1. Browser WebSocket keep-alive
            await self._ws_send({"type": "PING", "ts": time.time()})
            # 2. TCP server keep-alive (encrypted PING packet)
            await self._send_to_server({"type": "PING", "ts": time.time()})

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        async with BridgeSession._lock:
            BridgeSession._active[self.userid] = self

        # Announce join to all other sessions via bridge relay
        await relay.broadcast(self.userid, {
            "type":   "PRESENCE",
            "userid": self.userid,
            "status": "join",
            "ts":     time.time(),
        })
        # Send current rich user list to the new session
        await _broadcast_rich_user_list(only=self.userid)

        try:
            await asyncio.gather(
                self._tcp_reader_loop(),
                self._ws_reader_loop(),
                self._pinger_loop(),
                return_exceptions=True,
            )
        finally:
            self.alive = False
            # Cancel pending typing auto-clear tasks
            for task in self._typing_tasks.values():
                task.cancel()
            # Close TCP socket
            if self._sock:
                for fn in (self._sock.shutdown, self._sock.close):
                    try:
                        if fn == self._sock.shutdown:
                            fn(socket.SHUT_RDWR)
                        else:
                            fn()
                    except OSError:
                        pass
            # Remove from registry and announce leave
            async with BridgeSession._lock:
                BridgeSession._active.pop(self.userid, None)
            await relay.broadcast(self.userid, {
                "type":   "PRESENCE",
                "userid": self.userid,
                "status": "leave",
                "ts":     time.time(),
            })
            await _broadcast_rich_user_list()
            log.info("[%s] %sSession closed%s", self.userid, _C.YELLOW, _R)


# ─────────────────────────────────────────────────────────────────────────────
# Rich user list helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_rich_user_list(plain_users: Optional[List] = None) -> List[dict]:
    """
    Build a rich user list from what the TCP server returns (plain strings
    or already-rich dicts) merged with bridge-connected users.

    FIX: plain_users could be list[str] OR list[dict] depending on server
    version — extract userid string from either shape.
    """
    online_ids = set(BridgeSession._active.keys())

    # Extract uid strings regardless of input shape
    server_ids: set = set()
    for u in (plain_users or []):
        if isinstance(u, str):
            server_ids.add(u)
        elif isinstance(u, dict):
            uid = u.get("userid") or u.get("id") or u.get("user") or ""
            if uid:
                server_ids.add(str(uid))

    all_ids = server_ids | online_ids
    result = []
    for uid in sorted(all_ids):
        result.append({
            "userid":  uid,
            "online":  uid in online_ids,
            "status":  store.get_status(uid),
        })
    return result


async def _broadcast_rich_user_list(only: Optional[str] = None) -> None:
    """
    Push a fresh rich USER_LIST to all sessions (or just *only*).
    """
    payload = {
        "type":  "USER_LIST",
        "users": _build_rich_user_list(),
        "ts":    time.time(),
    }
    if only:
        await relay.send_to(only, payload)
    else:
        await relay.broadcast("", payload, exclude_sender=False)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _sanitise(obj: Any) -> Any:
    """Recursively make an object JSON-safe."""
    if isinstance(obj, dict):
        return {
            (k.decode() if isinstance(k, bytes) else str(k)): _sanitise(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_sanitise(i) for i in obj]
    if isinstance(obj, bytes):
        try:
            return obj.decode()
        except Exception:
            return obj.hex()
    if hasattr(obj, "value"):      # Enum → str
        return obj.value
    return obj


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _free_port(host: str, start: int, limit: int = 20) -> int:
    for p in range(start, start + limit):
        if _port_free(host, p):
            return p
    raise OSError(f"No free port in {start}–{start + limit - 1}")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="ZENTRAA HTTP Bridge", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = Path(__file__).parent / cfg.static_dir
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# ── Serve frontend ────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    idx = static_path / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"service": "ZENTRAA HTTP Bridge v2.0.0", "status": "ok"})


# ── Health & status ───────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    backend_ok = False
    try:
        _, w = await asyncio.wait_for(
            asyncio.open_connection(cfg.tcp_host, cfg.tcp_port), timeout=1.5
        )
        w.close()
        backend_ok = True
    except Exception:
        pass
    return {
        "bridge":          "ZENTRAA HTTP Bridge v2.0.0",
        "backend_ok":      backend_ok,
        "active_users":    len(BridgeSession._active),
        "users":           _build_rich_user_list(),
        "bridge_features": [
            "typing_indicators", "read_receipts", "reactions",
            "delivery_receipts", "rich_user_list", "file_upload",
            "message_history", "status_text", "voice_calls",
        ],
        "timestamp": time.time(),
    }


# ── Pre-registration check ────────────────────────────────────────────────────

@app.post("/api/register")
async def api_register(body: dict):
    userid = (body.get("userid") or "").strip()
    if not userid:
        raise HTTPException(422, "userid is required")
    if not re.match(r"^[A-Za-z0-9_\-]{3,32}$", userid):
        raise HTTPException(422, "userid must be 3–32 chars, [A-Za-z0-9_-]")
    if userid in BridgeSession._active:
        raise HTTPException(409, f"'{userid}' is already connected")
    return {"available": True, "userid": userid}


# ── Rich user list ────────────────────────────────────────────────────────────

@app.get("/api/users")
async def api_users():
    """Return all known users with online status and status text."""
    return {"users": _build_rich_user_list(), "timestamp": time.time()}


# ── Message history ───────────────────────────────────────────────────────────

@app.get("/api/history/{conv_id}")
async def api_history(conv_id: str, limit: int = 100):
    """
    Return the last *limit* messages for a conversation.
    conv_id: 'broadcast' or a userid for DM threads.
    """
    if limit < 1 or limit > cfg.max_history:
        raise HTTPException(422, f"limit must be 1–{cfg.max_history}")
    msgs = store.get_history(conv_id, limit)
    return {"conv_id": conv_id, "messages": msgs, "count": len(msgs)}


# ── Reactions ─────────────────────────────────────────────────────────────────

@app.post("/api/reaction")
async def api_reaction(body: dict):
    """
    Add or remove a reaction via REST (fallback for clients that can't use WS events).
    Body: { userid, msg_id, conv_id, emoji, action: "add"|"remove" }
    """
    userid  = (body.get("userid") or "").strip()
    msg_id  = (body.get("msg_id") or "").strip()
    conv_id = (body.get("conv_id") or "broadcast").strip()
    emoji   = (body.get("emoji") or "").strip()
    action  = (body.get("action") or "add").strip()

    if not all([userid, msg_id, emoji]):
        raise HTTPException(422, "userid, msg_id and emoji are required")
    if action not in ("add", "remove"):
        raise HTTPException(422, "action must be 'add' or 'remove'")

    if action == "add":
        reactions = store.add_reaction(msg_id, userid, emoji)
    else:
        reactions = store.remove_reaction(msg_id, userid, emoji)

    # Relay to all WS sessions
    await relay.send_reaction(userid, msg_id, conv_id, emoji, action, reactions)
    return {"msg_id": msg_id, "reactions": reactions}


# ── File / media upload ───────────────────────────────────────────────────────

@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    userid: str = "",
    to: str = "",
    conv_id: str = "broadcast",
):
    """
    Upload a file and relay it as a MSG to all (or specific) WebSocket sessions.
    The file is base64-encoded and sent via the bridge relay — it does NOT go
    through the encrypted TCP server (too large for RSA/AES packet sizes).

    Form fields:
      file     — the file being uploaded
      userid   — sender userid
      to       — target userid (empty = broadcast)
      conv_id  — conversation id (for history storage)
    """
    max_bytes = cfg.max_upload_mb * 1024 * 1024
    content   = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(413, f"File exceeds {cfg.max_upload_mb} MB limit")

    mime      = file.content_type or mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream"
    b64data   = "data:" + mime + ";base64," + base64.b64encode(content).decode()
    fname     = file.filename or "file"
    fsize     = len(content)
    msg_id    = str(uuid.uuid4())
    is_image  = mime.startswith("image/")
    is_video  = mime.startswith("video/")
    msg_type  = "image" if is_image else "video" if is_video else "file"
    ts        = time.time()

    payload = {
        "type":      "MSG",
        "id":        msg_id,
        "from":      userid,
        "to":        [to] if to else [],
        "body":      b64data,
        "msg_type":  msg_type,
        "filename":  fname,
        "filesize":  fsize,
        "mime":      mime,
        "ts":        ts,
    }

    # Store in history
    store.add_message(conv_id, payload)

    # Relay to target WS sessions
    if to and to in BridgeSession._active:
        await relay.send_to(to, payload)
        # Also send back to uploader
        if userid and userid in BridgeSession._active:
            await relay.send_to(userid, {**payload, "isSelf": True})
    else:
        # Broadcast to everyone
        await relay.broadcast("", payload, exclude_sender=False)

    return {
        "ok":       True,
        "msg_id":   msg_id,
        "filename": fname,
        "filesize": fsize,
        "mime":     mime,
        "msg_type": msg_type,
    }


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/{userid}")
async def ws_endpoint(ws: WebSocket, userid: str):
    await ws.accept()

    if userid in BridgeSession._active:
        await ws.send_json({"type": "ERROR", "reason": f"'{userid}' is already in use."})
        await ws.close(code=4409)
        return

    session = BridgeSession(ws, userid)
    ok = await session.connect()
    if not ok:
        try:
            await ws.close(code=4500)
        except Exception:
            pass
        return

    await session.run()


# ─────────────────────────────────────────────────────────────────────────────
# Startup banner
# ─────────────────────────────────────────────────────────────────────────────

def _print_banner(http_port: int) -> None:
    w = 72
    lines = [
        ("", ""),
        ("╔" + "═" * (w - 2) + "╗", ""),
        ("║" + f"  {_C.CYAN}ZENTRAA HTTP BRIDGE  v2.0.0{_R}".center(w + len(_C.CYAN) + len(_R) - 2) + "║", ""),
        ("║" + " " * (w - 2) + "║", ""),
        ("║" + f"  HTTP  →  http://localhost:{http_port}".ljust(w - 2) + "║", ""),
        ("║" + f"  TCP   →  {cfg.tcp_host}:{cfg.tcp_port}".ljust(w - 2) + "║", ""),
        ("║" + " " * (w - 2) + "║", ""),
        ("║" + f"  {_C.GREEN}Features{_R}: Typing · Reactions · Read Receipts · File Upload".ljust(w + len(_C.GREEN) + len(_R) - 2) + "║", ""),
        ("║" + f"           Rich User List · Message History · Delivery Receipts · Voice Calls".ljust(w - 2) + "║", ""),
        ("║" + " " * (w - 2) + "║", ""),
        ("║" + f"  {_C.YELLOW}Author{_R}: Divyanshu Sinha  ·  Powered by Pythonaibrain 1.1.9".ljust(w + len(_C.YELLOW) + len(_R) - 2) + "║", ""),
        ("╚" + "═" * (w - 2) + "╝", ""),
        ("", ""),
    ]
    for line, _ in lines:
        print(line)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ZENTRAA HTTP/WebSocket Bridge v2.0")
    parser.add_argument("--host",           default="0.0.0.0",   help="HTTP bind host")
    parser.add_argument("--http-port",      default=None, type=int, help="HTTP port (auto from 7080)")
    parser.add_argument("--tcp-host",       default="127.0.0.1", help="ZENTRAA TCP server host")
    parser.add_argument("--tcp-port",       default=9999, type=int, help="ZENTRAA TCP server port")
    parser.add_argument("--no-auto-port",   action="store_true",  help="Fail if port busy")
    parser.add_argument("--max-upload-mb",  default=64, type=int, help="Max upload size MB")
    parser.add_argument("--history",        default=500, type=int, help="Messages to store per conv")
    parser.add_argument("--ping-interval",  default=20, type=float, help="WS ping interval (seconds)")
    args = parser.parse_args()

    cfg.http_host      = args.host
    cfg.tcp_host       = args.tcp_host
    cfg.tcp_port       = args.tcp_port
    cfg.max_upload_mb  = args.max_upload_mb
    cfg.max_history    = args.history
    cfg.ping_interval  = args.ping_interval

    desired = args.http_port or 7080
    if args.no_auto_port:
        cfg.http_port = desired
    elif _port_free(cfg.http_host, desired):
        cfg.http_port = desired
    else:
        log.warning("Port %d busy — scanning for free port…", desired)
        try:
            cfg.http_port = _free_port(cfg.http_host, desired)
            log.warning("Using port %d instead.", cfg.http_port)
        except OSError as exc:
            log.error("%s", exc)
            raise SystemExit(1)

    _print_banner(cfg.http_port)
    uvicorn.run(app, host=cfg.http_host, port=cfg.http_port, log_level="info")


if __name__ == "__main__":
    main()