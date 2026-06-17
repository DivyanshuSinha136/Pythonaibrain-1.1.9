"""
server.py  —  ZENTRAA Encrypted Chat Server
============================================
  Z one for
  E ncrypted
  N etworked
  T alks &
  R eal-time
  A I  A gent

Powered By        : Pythonaibrain
Author            : Divyanshu Sinha
Version           : 1.0.0
Pythonaibrain Ver : 1.1.9
AI Name           : TIGER AI (Tactical Intelligent Generative Expert Responser/Responder AI)

Run:
    python server.py
    python server.py --config /path/to/ZENTRAA.pbcfg
    python server.py --host 0.0.0.0 --port 9999
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import secrets
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

import PyAgent.zentraa_crypto as crypto
import PyAgent.zentraa_protocol as proto
from PyAgent.zentraa_config import ZENTRAConfig, load_config
from PyAgent.zentraa_protocol import MT, ErrCode

# ─────────────────────────────────────────────────────────────────────────────
# Rich theme
# ─────────────────────────────────────────────────────────────────────────────
ZENTRAA_THEME = Theme({
    "banner":        "bold cyan",
    "info":          "bold green",
    "warn":          "bold yellow",
    "error":         "bold red",
    "highlight":     "bold magenta",
    "dim":           "dim white",
    "user":          "bold bright_cyan",
    "ai":            "bold bright_yellow",
    "enc":           "bold bright_green",
    "server_label":  "bold white on dark_blue",
})

console = Console(theme=ZENTRAA_THEME)


# ─────────────────────────────────────────────────────────────────────────────
# ZENTRAA Banner
# ─────────────────────────────────────────────────────────────────────────────

BANNER = r"""
 ███████╗███████╗███╗   ██╗████████╗██████╗  █████╗  █████╗ 
 ╚══███╔╝██╔════╝████╗  ██║╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗
   ███╔╝ █████╗  ██╔██╗ ██║   ██║   ██████╔╝███████║███████║
  ███╔╝  ██╔══╝  ██║╚██╗██║   ██║   ██╔══██╗██╔══██║██╔══██║
 ███████╗███████╗██║ ╚████║   ██║   ██║  ██║██║  ██║██║  ██║
 ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝
"""

def print_banner(cfg: ZENTRAConfig) -> None:
    if cfg.ui.banner_style == "none":
        return
    console.print(Text(BANNER, style="banner"))
    info = Table.grid(padding=(0, 2))
    info.add_column(style="dim")
    info.add_column(style="highlight")
    rows = [
        ("Zone for Encrypted Networked Talks & Real-time AI Agent", ""),
        ("Powered By",         "Pythonaibrain"),
        ("Author",             "Divyanshu Sinha"),
        ("Version",            "1.0.0"),
        ("Pythonaibrain Ver",  "1.1.9"),
        ("AI Name",            "TIGER AI — Tactical Intelligent Generative Expert Responser"),
    ]
    for k, v in rows:
        info.add_row(k + ("" if not v else " :"), v)
    console.print(Panel(info, border_style="cyan", title="[bold cyan]SERVER", title_align="left"))
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Client session
# ─────────────────────────────────────────────────────────────────────────────

class ClientSession:
    """Represents one authenticated connected client."""

    def __init__(
        self,
        sock: socket.socket,
        addr: tuple,
        server: "ZENTRAServer",
    ) -> None:
        self.sock         = sock
        self.addr         = addr
        self.server       = server
        self.userid: Optional[str] = None
        self.rsa_pub: Optional[crypto.RSAPublicKey] = None
        self.session_id   = secrets.token_hex(16)
        self.alive        = True
        self.msg_count    = 0
        self.last_ping    = time.time()
        self._lock        = threading.Lock()

    # ── I/O ──────────────────────────────────────────────────────────────────

    def send_raw(self, data: bytes) -> None:
        """Send a framed raw bytes packet. Silently drops if session is dead."""
        if not self.alive:
            return
        try:
            with self._lock:
                self.sock.sendall(crypto.frame(data))
        except OSError:
            self.alive = False

    def send_plain(self, msg: dict) -> None:
        """Encode a protocol dict and send unencrypted (handshake only)."""
        self.send_raw(proto.encode(msg))

    def send_encrypted(self, msg: dict) -> None:
        """Encrypt *msg* with client's public key and send."""
        if self.rsa_pub is None:
            return
        try:
            payload = proto.encode(msg)
            packet  = crypto.encrypt(payload, self.rsa_pub, self.server.private_key)
            self.send_raw(packet)
        except Exception as exc:
            log.warning("send_encrypted to %s failed: %s", self.userid, exc)

    def send_error(self, code: ErrCode, reason: str) -> None:
        self.send_plain(proto.build_error(code, reason))

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self.alive = False
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# ZENTRAA Server
# ─────────────────────────────────────────────────────────────────────────────

class ZENTRAServer:
    """
    Thread-per-client encrypted chat server.

    Each client connection spawns two threads:
      • reader — receives and dispatches incoming packets
      • pinger — sends periodic PINGs to detect stale connections
    """

    def __init__(self, cfg: ZENTRAConfig) -> None:
        self.cfg          = cfg
        self._clients: Dict[str, ClientSession] = {}   # userid → session
        self._sessions: Set[ClientSession]       = set()
        self._lock        = threading.RLock()

        # RSA keypair
        kdir  = cfg.server.keys_dir
        kpriv = cfg.server.server_private_key
        kpub  = cfg.server.server_public_key
        if Path(kdir, kpriv).exists():
            self.private_key, self.public_key = crypto.load_keypair(kdir, kpriv, kpub)
            log.info("Loaded existing server RSA keypair from %s", kdir)
        else:
            log.info("Generating new RSA-%d keypair…", cfg.encryption.rsa_key_bits)
            self.private_key, self.public_key = crypto.generate_rsa_keypair(
                cfg.encryption.rsa_key_bits
            )
            crypto.save_keypair(self.private_key, self.public_key, kdir, kpriv, kpub)
            log.info("Keypair saved to %s", kdir)

        self.server_pub_pem = crypto.serialize_public_key(self.public_key).decode()

    # ── Routing ───────────────────────────────────────────────────────────────

    def _get_session(self, userid: str) -> Optional[ClientSession]:
        with self._lock:
            return self._clients.get(userid)

    def _online_users(self) -> List[str]:
        with self._lock:
            return list(self._clients.keys())

    def _register(self, session: ClientSession, userid: str) -> bool:
        with self._lock:
            if userid in self._clients:
                return False
            self._clients[userid] = session
            session.userid = userid
            return True

    def _deregister(self, session: ClientSession) -> None:
        with self._lock:
            uid = session.userid
            if uid and self._clients.get(uid) is session:
                del self._clients[uid]
            self._sessions.discard(session)

    # ── Broadcast ─────────────────────────────────────────────────────────────

    def broadcast(self, msg: dict, exclude: Optional[str] = None) -> None:
        with self._lock:
            targets = list(self._clients.values())
        for s in targets:
            if s.userid != exclude:
                s.send_encrypted(msg)

    def route_msg(self, session: ClientSession, msg: dict) -> None:
        """Route a MSG packet to its target(s)."""
        to: List[str] = msg.get("to", [])
        body          = msg.get("body", "")
        from_id       = session.userid or "unknown"

        if not to:                              # broadcast
            out = proto.build_msg(from_id, [], body)
            self.broadcast(out, exclude=from_id)
            log.info("[BROADCAST] %s → ALL", from_id)
        else:
            out = proto.build_msg(from_id, to, body)
            delivered: List[str] = []
            for uid in to:
                target = self._get_session(uid)
                if target:
                    target.send_encrypted(out)
                    delivered.append(uid)
                else:
                    session.send_encrypted(
                        proto.build_error(ErrCode.TARGET_OFFLINE, f"{uid} is offline")
                    )
            # Echo back to sender so they see their own targeted message
            session.send_encrypted(out)
            log.info("[MSG] %s → %s", from_id, delivered)

    def route_ai_query(self, session: ClientSession, msg: dict) -> None:
        """Forward AI_QUERY to the AI client if online."""
        ai_id = self.cfg.ai.ai_userid
        ai_session = self._get_session(ai_id)
        if not ai_session:
            session.send_encrypted(
                proto.build_error(ErrCode.TARGET_OFFLINE, "TIGER AI is not connected.")
            )
            return
        ai_session.send_encrypted(msg)

    def route_ai_reply(self, _session: ClientSession, msg: dict) -> None:
        """Distribute an AI_REPLY to the original requester + tagged targets."""
        targets: List[str] = msg.get("targets", [])
        for uid in targets:
            s = self._get_session(uid)
            if s:
                s.send_encrypted(msg)

    # ── Client handler ────────────────────────────────────────────────────────

    def _handle_client(self, session: ClientSession) -> None:
        """Main reader loop for one client connection."""
        cfg = self.cfg
        addr_str = f"{session.addr[0]}:{session.addr[1]}"
        log.info("New connection from %s", addr_str)

        try:
            # ── Handshake: receive HELLO ──────────────────────────────────────
            raw = crypto.recv_frame(session.sock)
            hello = proto.decode(raw)
            if hello.get("type") != MT.HELLO:
                session.send_error(ErrCode.UNKNOWN_MSG, "Expected HELLO")
                session.close()
                return

            userid    = hello.get("userid", "").strip()
            pub_pem   = hello.get("rsa_pub_pem", "")

            # Validate user ID
            uid_re = rf"^[{cfg.server.userid_chars}]{{{cfg.server.userid_min_length},{cfg.server.userid_max_length}}}$"
            if not re.match(uid_re, userid):
                session.send_error(ErrCode.BAD_USERID,
                    f"User ID must be {cfg.server.userid_min_length}–{cfg.server.userid_max_length} "
                    f"chars, pattern [{cfg.server.userid_chars}]")
                session.close()
                return

            if not self._register(session, userid):
                session.send_error(ErrCode.USERID_TAKEN, f"'{userid}' is already in use.")
                session.close()
                return

            # Store client public key
            try:
                session.rsa_pub = crypto.load_public_key(pub_pem.encode())
            except Exception as exc:
                session.send_error(ErrCode.INTERNAL, f"Bad RSA public key: {exc}")
                session.close()
                return

            # ── Send HELLO_ACK (plain — client has no session key yet) ─────────
            ack = proto.build_hello_ack(
                self.server_pub_pem,
                session.session_id,
                f"Welcome to ZENTRAA v1.0.0, {userid}! 🔐",
            )
            session.send_plain(ack)

            console.print(f"  [info]✔[/info] [user]{userid}[/user] connected from {addr_str}")

            # Announce presence
            if cfg.server.broadcast_presence:
                self.broadcast(proto.build_presence(userid, "join"), exclude=userid)

            # Start pinger thread
            ping_thread = threading.Thread(
                target=self._pinger, args=(session,), daemon=True
            )
            ping_thread.start()

            # ── Main read loop ────────────────────────────────────────────────
            while session.alive:
                try:
                    raw = crypto.recv_frame(session.sock)
                except ConnectionError:
                    # Covers EOF, WinError 10038, ECONNRESET, and shutdown(SHUT_RD)
                    # signals from the pinger — all normalised by recv_frame.
                    break

                if len(raw) > cfg.server.max_message_size:
                    session.send_encrypted(
                        proto.build_error(ErrCode.MSG_TOO_LARGE, "Message exceeds size limit.")
                    )
                    continue

                # Attempt to decrypt
                try:
                    payload = crypto.decrypt(raw, self.private_key, session.rsa_pub)
                    msg = proto.decode(payload)
                except ValueError as exc:
                    log.warning("Decryption failed from %s: %s", userid, exc)
                    session.send_encrypted(
                        proto.build_error(ErrCode.DECRYPTION_FAIL, str(exc))
                    )
                    continue

                mt = msg.get("type")
                session.msg_count += 1

                if mt == MT.MSG:
                    self.route_msg(session, msg)

                elif mt == MT.AI_QUERY:
                    self.route_ai_query(session, msg)

                elif mt == MT.AI_REPLY:
                    self.route_ai_reply(session, msg)

                elif mt == MT.PING:
                    session.last_ping = time.time()
                    session.send_encrypted(proto.build_pong())

                elif mt == MT.PONG:
                    session.last_ping = time.time()
                    session.send_encrypted(proto.build_ping())

                elif mt == MT.USER_LIST:
                    session.send_encrypted(proto.build_user_list(self._online_users()))

                else:
                    session.send_encrypted(
                        proto.build_error(ErrCode.UNKNOWN_MSG, f"Unknown type: {mt}")
                    )

        except OSError as exc:
            # WinError 10038 and similar socket teardown errors are expected when
            # the pinger signals shutdown via shutdown(SHUT_RD).  Log at DEBUG only.
            if not session.alive:
                log.debug("Socket closed during shutdown for %s: %s", session.addr, exc)
            else:
                log.error("Socket error for %s: %s", session.addr, exc, exc_info=True)
        except Exception as exc:
            log.error("Unhandled error for %s: %s", session.addr, exc, exc_info=True)
        finally:
            uid = session.userid
            self._deregister(session)
            session.close()
            if uid:
                console.print(f"  [warn]✖[/warn] [user]{uid}[/user] disconnected")
                if cfg.server.broadcast_presence:
                    self.broadcast(proto.build_presence(uid, "leave"))

    def _pinger(self, session: ClientSession) -> None:
        interval = self.cfg.server.ping_interval
        timeout  = self.cfg.network.socket_timeout or (interval * 3)
        while session.alive:
            time.sleep(interval)
            if not session.alive:
                break
            now = time.time()
            if now - session.last_ping > timeout:
                log.info("Ping timeout for %s", session.userid)
                # Signal dead — do NOT close the socket here.
                # The reader is blocked on recv(); closing the socket from a
                # different thread races with recv() on Windows and produces
                # WinError 10038 ("not a socket").  Instead just mark alive=False
                # and let the reader's socket timeout fire so it exits cleanly.
                session.alive = False
                # Interrupt the blocking recv() by shutting down the read-half.
                # shutdown() is safe to call from another thread on all platforms.
                try:
                    session.sock.shutdown(socket.SHUT_RD)
                except OSError:
                    pass
                break
            session.send_encrypted(proto.build_ping())

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        host = self.cfg.network.host
        port = self.cfg.network.port
        backlog = self.cfg.network.backlog

        srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv_sock.bind((host, port))
        srv_sock.listen(backlog)

        console.print(
            Panel(
                f"[enc]🔐 End-to-end encryption active[/enc]\n"
                f"[info]Listening on[/info] [highlight]{host}:{port}[/highlight]\n"
                f"[dim]RSA-2048-OAEP  •  AES-256-GCM  •  RSA-PSS  •  Curve25519[/dim]",
                title="[server_label] ZENTRAA SERVER ",
                border_style="cyan",
            )
        )

        try:
            while True:
                try:
                    conn, addr = srv_sock.accept()
                    if self.cfg.network.socket_timeout:
                        conn.settimeout(self.cfg.network.socket_timeout)
                    session = ClientSession(conn, addr, self)
                    with self._lock:
                        self._sessions.add(session)
                    t = threading.Thread(
                        target=self._handle_client, args=(session,), daemon=True
                    )
                    t.start()
                except OSError as exc:
                    log.error("Accept error: %s", exc)
        except KeyboardInterrupt:
            console.print("\n[warn]Shutting down ZENTRAA server…[/warn]")
        finally:
            srv_sock.close()


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(cfg: ZENTRAConfig) -> None:
    level = getattr(logging, cfg.server.log_level.upper(), logging.INFO)
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if cfg.server.log_file:
        handlers.append(logging.FileHandler(cfg.server.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=handlers,
    )


log = logging.getLogger("zentraa.server")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="zentraa-server",
        description="ZENTRAA — Zone for Encrypted Networked Talks & Real-time AI Agent",
    )
    parser.add_argument("--config", "-c", default=None, help="Path to ZENTRAA.pbcfg")
    parser.add_argument("--host",   "-H", default=None, help="Override bind host")
    parser.add_argument("--port",   "-p", default=None, type=int, help="Override port")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.host:
        cfg.network.host = args.host
    if args.port:
        cfg.network.port = args.port

    _setup_logging(cfg)
    print_banner(cfg)

    server = ZENTRAServer(cfg)
    server.run()


if __name__ == "__main__":
    main()
