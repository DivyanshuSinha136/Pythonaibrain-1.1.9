"""
clientAI.py  —  ZENTRAA TIGER AI Client
=========================================
  TIGER AI — Tactical Intelligent Generative Expert Responser/Responder AI

Powered By        : Pythonaibrain
Author            : Divyanshu Sinha
Version           : 1.0.0
Pythonaibrain Ver : 1.1.9

TIGER AI joins the ZENTRAA network as a special client.
It listens for AI_QUERY packets, generates responses using AdvanceBrain
(or Brain fallback), then sends AI_REPLY packets back through the server.

Import pattern (as per spec):
    from pyaitk import Brain, AdvanceBrain, SmartMemory, build_memory, VectorizerMode

Run:
    python clientAI.py
    python clientAI.py --config /path/to/ZENTRAA.pbcfg
    python clientAI.py --smart  # force AdvanceBrain LLM
    python clientAI.py --basic  # force Brain (no LLM)
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

# ── Pythonaibrain imports (from pyaitk as specified) ─────────────────────────
try:
    from pyaitk import Brain, AdvanceBrain, SmartMemory, build_memory, VectorizerMode  # type: ignore
    _PYAITK_AVAILABLE = True
except ImportError:
    # Fallback: attempt direct import from pythonaibrain
    try:
        from pythonaibrain.core import Brain, AdvanceBrain, VectorizerMode           # type: ignore
        from pythonaibrain.core import SmartMemory, build_memory                      # type: ignore
        _PYAITK_AVAILABLE = True
    except ImportError:
        _PYAITK_AVAILABLE = False
        Brain = AdvanceBrain = SmartMemory = build_memory = VectorizerMode = None    # type: ignore

import .zentraa_crypto as crypto
import .zentraa_protocol as proto
from .zentraa_config import ZENTRAConfig, load_config
from .zentraa_protocol import MT, ErrCode

# ─────────────────────────────────────────────────────────────────────────────
# Rich theme & console
# ─────────────────────────────────────────────────────────────────────────────

AI_THEME = Theme({
    "banner":    "bold bright_yellow",
    "info":      "bold green",
    "warn":      "bold yellow",
    "error":     "bold red",
    "highlight": "bold magenta",
    "enc":       "bold bright_green",
    "query":     "bold bright_cyan",
    "reply":     "bold bright_yellow",
    "dim_text":  "dim white",
    "sys":       "bold dim cyan",
})

console = Console(theme=AI_THEME)

AI_BANNER = r"""
  ████████╗██╗ ██████╗ ███████╗██████╗      █████╗ ██╗
  ╚══██╔══╝██║██╔════╝ ██╔════╝██╔══██╗    ██╔══██╗██║
     ██║   ██║██║  ███╗█████╗  ██████╔╝    ███████║██║
     ██║   ██║██║   ██║██╔══╝  ██╔══██╗    ██╔══██║██║
     ██║   ██║╚██████╔╝███████╗██║  ██║    ██║  ██║██║
     ╚═╝   ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝
"""

log = logging.getLogger("zentraa.tiger_ai")


# ─────────────────────────────────────────────────────────────────────────────
# Brain Wrapper — SmartAI abstraction
# ─────────────────────────────────────────────────────────────────────────────

class TigerAIEngine:
    """
    Wraps Pythonaibrain Brain / AdvanceBrain transparently.
    SmartAI=True → AdvanceBrain (LLM-powered).
    SmartAI=False → Brain (intent-matching).
    Falls back gracefully if pythonaibrain is not installed.
    """

    def __init__(self, cfg: ZENTRAConfig) -> None:
        self.cfg        = cfg
        self.ai_cfg     = cfg.ai
        self._brain     = None
        self._adv_brain = None
        self._lock      = threading.Lock()
        self._ready     = False
        self._mode      = "none"

        if not _PYAITK_AVAILABLE:
            console.print("[warn]⚠  pyaitk / pythonaibrain not installed. TIGER AI will use echo fallback.[/warn]")
            return

        vm_str = self.ai_cfg.vectorizer_mode.lower()
        try:
            vm = VectorizerMode(vm_str)
        except (ValueError, TypeError):
            vm = VectorizerMode.BOW  # type: ignore
            vm = VectorizerMode.GENSIM  # type: ignore

        if self.ai_cfg.smart_ai:
            self._init_advance_brain(vm)
        else:
            self._init_brain(vm)

    def _init_advance_brain(self, vm) -> None:
        try:
            self._adv_brain = AdvanceBrain(
                intents_path=self.ai_cfg.intents_path if self.ai_cfg.intents_path else None,
                vectorizer_mode=vm,
            )
            self._adv_brain.load()
            if not self._adv_brain.is_loaded():
                log.warning("AdvanceBrain.load() did not complete — training instead.")
                self._adv_brain.train()
                self._adv_brain.save()
            self._mode  = "advance_brain"
            self._ready = True
            console.print("[info]✔ TIGER AI — AdvanceBrain (LLM) loaded[/info]")
        except Exception as exc:
            log.warning("AdvanceBrain init failed (%s) — falling back to Brain.", exc)
            self._init_brain(vm)

    def _init_brain(self, vm) -> None:
        try:
            memory = build_memory(
                self.ai_cfg.memory_path,
                smart=True,
                fit_interval=self.ai_cfg.memory_fit_interval,
            ) if build_memory else None
            self._brain = Brain(
                intents_path=self.ai_cfg.intents_path if self.ai_cfg.intents_path else None,
                vectorizer_mode=vm,
                memory_path=self.ai_cfg.memory_path,
                smart_memory=True,
                memory_fit_interval=self.ai_cfg.memory_fit_interval,
            )
            # self._brain.train()
            # self._brain.save()
            self._brain.load()
            self._mode  = "brain"
            self._ready = True
            console.print("[info]✔ TIGER AI — Brain (intent-matching) loaded[/info]")
        except Exception as exc:
            log.error("Brain init failed: %s", exc)
            self._ready = False

    def think(self, message: str, user_id: str = "user") -> str:
        """Generate a response for *message* from *user_id*."""
        if not self._ready:
            return f"[TIGER AI] I'm still initialising — please try again shortly."

        with self._lock:
            try:
                if self._mode == "advance_brain" and self._adv_brain:
                    raw = self._adv_brain.process_messages(
                        message,
                        grammar=self.ai_cfg.grammar_correction,
                        advance=True,
                        TTS=self.ai_cfg.tts,
                    )
                elif self._mode == "brain" and self._brain:
                    raw = self._brain.process_messages(
                        message=message,
                        grammar=self.ai_cfg.grammar_correction,
                        TTS=self.ai_cfg.tts,
                    )
                else:
                    return "[TIGER AI] Engine unavailable."

                # Truncate if configured
                max_len = self.ai_cfg.max_response_length
                if max_len and len(raw) > max_len:
                    raw = raw[:max_len].rstrip() + "…"
                return raw

            except Exception as exc:
                log.error("TigerAIEngine.think error: %s", exc)
                return "[TIGER AI] I encountered an error processing your query."

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def ready(self) -> bool:
        return self._ready


# ─────────────────────────────────────────────────────────────────────────────
# TIGER AI Network Client
# ─────────────────────────────────────────────────────────────────────────────

class TigerAIClient:
    """
    Connects to the ZENTRAA server as the TIGER AI user,
    listens for AI_QUERY packets and responds with AI_REPLY.
    """

    def __init__(self, cfg: ZENTRAConfig) -> None:
        self.cfg     = cfg
        self.ai_cfg  = cfg.ai
        self.userid  = self.ai_cfg.ai_userid
        self.alive   = False
        self.sock: Optional[socket.socket] = None
        self.session_id: Optional[str] = None

        # Crypto
        self.private_key, self.public_key = crypto.generate_rsa_keypair(
            cfg.encryption.rsa_key_bits
        )
        self.pub_pem = crypto.serialize_public_key(self.public_key).decode()
        self.server_pub: Optional[crypto.RSAPublicKey] = None

        # AI engine
        self.engine = TigerAIEngine(cfg)
        self._lock  = threading.Lock()

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _send_plain(self, msg: dict) -> None:
        data = proto.encode(msg)
        self.sock.sendall(crypto.frame(data))  # type: ignore[union-attr]

    def _send_encrypted(self, msg: dict) -> None:
        if not self.server_pub:
            return
        payload = proto.encode(msg)
        packet  = crypto.encrypt(payload, self.server_pub, self.private_key)
        try:
            self.sock.sendall(crypto.frame(packet))  # type: ignore[union-attr]
        except OSError:
            self.alive = False

    # ── Handshake ─────────────────────────────────────────────────────────────

    def _handshake(self) -> bool:
        from cryptography.hazmat.primitives import serialization as _ser
        eph_priv, eph_pub = crypto.generate_x25519_keypair()
        eph_pub_hex = eph_pub.public_bytes(
            _ser.Encoding.Raw, _ser.PublicFormat.Raw
        ).hex()

        self._send_plain(proto.build_hello(self.userid, self.pub_pem, eph_pub_hex))

        try:
            raw = crypto.recv_frame(self.sock)
        except ConnectionError:
            return False

        msg = proto.decode(raw)
        if msg.get("type") == MT.ERROR:
            console.print(f"[error]✗ Server rejected: {msg.get('reason')}[/error]")
            return False
        if msg.get("type") != MT.HELLO_ACK:
            console.print(f"[error]✗ Unexpected: {msg.get('type')}[/error]")
            return False

        srv_pub_pem = msg.get("server_rsa_pub_pem", "")
        self.server_pub = crypto.load_public_key(srv_pub_pem.encode())
        self.session_id = msg.get("session_id")
        console.print(f"[enc]🔐 Encrypted session established[/enc]  [dim_text]ID: {(self.session_id or '')[:8]}…[/dim_text]")
        return True

    # ── Query handler ─────────────────────────────────────────────────────────

    def _handle_query(self, msg: dict) -> None:
        from_id  = msg.get("from", "unknown")
        targets  = msg.get("targets", [from_id])
        body     = msg.get("body", "")

        console.print(
            f"  [query]❓ Query from[/query] [highlight]{from_id}[/highlight]"
            + (f" (visible to: {', '.join(t for t in targets if t != from_id)})" if len(targets) > 1 else "")
            + f"  [dim_text]{body[:80]}{'…' if len(body) > 80 else ''}[/dim_text]"
        )

        # Generate response in background thread to not block receiver
        def respond() -> None:
            answer = self.engine.think(body, user_id=from_id)
            prefix = self.ai_cfg.ai_prefix
            full   = f"{prefix} {answer}" if prefix else answer

            # Add typing delay if configured
            delay = self.ai_cfg.typing_delay
            if delay:
                time.sleep(min(len(full) * delay, 3.0))

            reply = proto.build_ai_reply(self.userid, targets, full)
            self._send_encrypted(reply)

            console.print(
                f"  [reply]✔ Reply sent[/reply] → {', '.join(targets)}  "
                f"[dim_text]{full[:60]}{'…' if len(full) > 60 else ''}[/dim_text]"
            )

        t = threading.Thread(target=respond, daemon=True)
        t.start()

    # ── Main receive loop ─────────────────────────────────────────────────────

    def _receiver(self) -> None:
        while self.alive:
            try:
                raw = crypto.recv_frame(self.sock)
            except ConnectionError:
                break

            try:
                payload = crypto.decrypt(raw, self.private_key, self.server_pub)
                msg = proto.decode(payload)
            except ValueError as exc:
                log.warning("Decrypt error: %s", exc)
                continue

            mt = msg.get("type")

            if mt == MT.AI_QUERY:
                self._handle_query(msg)

            elif mt == MT.PING:
                self._send_encrypted(proto.build_pong())

            elif mt == MT.PRESENCE:
                uid    = msg.get("userid", "")
                status = msg.get("status", "")
                console.print(
                    f"  [sys]{'▶' if status == 'join' else '◀'} {uid} {status}ed[/sys]"
                )

            elif mt == MT.ERROR:
                console.print(f"[error]⚠ [{msg.get('code')}] {msg.get('reason')}[/error]")

            elif mt == MT.MSG:
                # TIGER AI can see broadcast messages — optional: react
                pass

        self.alive = False
        console.print("[warn]Disconnected from ZENTRAA.[/warn]")

    # ── Connect ───────────────────────────────────────────────────────────────

    def connect(self, host: str, port: int) -> None:
        max_attempts = self.cfg.network.max_reconnect_attempts
        delay        = self.cfg.network.reconnect_delay
        attempt      = 0

        while True:
            attempt += 1
            console.print(
                f"[info]Connecting as[/info] [highlight]{self.userid}[/highlight] "
                f"to [highlight]{host}:{port}[/highlight]"
                + (f" [dim_text](attempt {attempt})[/dim_text]" if attempt > 1 else "")
                + "…"
            )
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if self.cfg.network.socket_timeout:
                    self.sock.settimeout(self.cfg.network.socket_timeout)
                self.sock.connect((host, port))
                self.sock.settimeout(None)
                break
            except OSError as exc:
                console.print(f"[error]  Connection failed: {exc}[/error]")
                if max_attempts and attempt >= max_attempts:
                    console.print("[error]Max attempts reached.[/error]")
                    sys.exit(1)
                console.print(f"[dim_text]  Retrying in {delay}s…[/dim_text]")
                time.sleep(delay)

        self.alive = True

        if not self._handshake():
            console.print("[error]Handshake failed.[/error]")
            sys.exit(1)

        console.print(
            f"[info]✔ TIGER AI online[/info]  "
            f"[dim_text]Mode: {self.engine.mode} | Smart: {self.ai_cfg.smart_ai}[/dim_text]"
        )

        # Announce arrival
        join_msg = self.ai_cfg.join_message
        if join_msg:
            self._send_encrypted(proto.build_msg(self.userid, [], join_msg))

        recv_thread = threading.Thread(target=self._receiver, daemon=True)
        recv_thread.start()

        console.print("[dim_text]  TIGER AI is listening for @ai queries…  (Ctrl+C to stop)[/dim_text]\n")

        try:
            while self.alive:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            # Announce departure
            leave_msg = self.ai_cfg.leave_message
            if leave_msg and self.alive:
                try:
                    self._send_encrypted(proto.build_msg(self.userid, [], leave_msg))
                    time.sleep(0.5)
                except Exception:
                    pass
            self.alive = False
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
                self.sock.close()
            except OSError:
                pass
            console.print("[info]TIGER AI disconnected.[/info]")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(cfg: ZENTRAConfig) -> None:
    level = getattr(logging, cfg.server.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="zentraa-tiger-ai",
        description="TIGER AI — Tactical Intelligent Generative Expert Responser AI for ZENTRAA",
    )
    parser.add_argument("--config", "-c", default=None, help="Path to ZENTRAA.pbcfg")
    parser.add_argument("--host",   "-H", default=None, help="Server host")
    parser.add_argument("--port",   "-p", default=None, type=int, help="Server port")
    parser.add_argument("--smart",  action="store_true", help="Force AdvanceBrain (LLM)")
    parser.add_argument("--basic",  action="store_true", help="Force Brain (intent-matching)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.smart:
        cfg.ai.smart_ai = True
    if args.basic:
        cfg.ai.smart_ai = False

    host = args.host or cfg.client.default_host
    port = args.port or cfg.client.default_port

    _setup_logging(cfg)

    console.print(Text(AI_BANNER, style="banner"))
    console.print(
        Panel(
            f"[bold bright_yellow]Tactical Intelligent Generative Expert Responser/Responder AI[/bold bright_yellow]\n"
            f"[dim_text]Powered by Pythonaibrain v1.1.9  •  Author: Divyanshu Sinha[/dim_text]\n"
            f"[dim_text]Smart AI: {'✔ AdvanceBrain LLM' if cfg.ai.smart_ai else '✘ Brain (intent-matching)'}[/dim_text]",
            title="[bold bright_yellow] TIGER AI — ZENTRAA AI Client ",
            border_style="bright_yellow",
        )
    )

    ai_client = TigerAIClient(cfg)
    ai_client.connect(host, port)


if __name__ == "__main__":
    main()
