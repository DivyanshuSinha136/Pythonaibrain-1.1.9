"""
client.py  —  ZENTRAA Encrypted Chat Client  [Enhanced TUI v2.1]
=================================================================
  Z one for Encrypted Networked Talks & Real-time AI Agent

Powered By        : Pythonaibrain
Author            : Divyanshu Sinha
Version           : 2.1.0
Pythonaibrain Ver : 1.1.9
AI Name           : TIGER AI (Tactical Intelligent Generative Expert Responser/Responder AI)

Run:
    python client.py
    python client.py --host 127.0.0.1 --port 9999 --userid Alice
    python client.py --config /path/to/ZENTRAA.pbcfg

Commands:
    @<userid> <message>                   Direct message
    @<uid1> @<uid2> ... <message>         Multi-user DM
    @ai <message>                         Ask TIGER AI
    @ai @<uid> <message>                  Ask AI, share reply with <uid>
    /help                                 Show help
    /clear or /cls                        Clear screen
    /ai                                   AI info
    /setting                              Settings
    /users                                List online users
    /me <action>                          Send action/emote message
    /whois <userid>                       Show info about a user
    /ping                                 Ping server manually
    /stats                                Show session statistics
    /notify <on|off>                      Toggle notifications
    /timestamps <on|off>                  Toggle timestamps
    /quit or /exit                        Disconnect

UI FIXES (v2.1):
    - Removed ANSI escape codes from prompt → no more ←[1;96m bleeding
    - Prompt is now plain "Yash >>> " via sys.stdout.write only
    - Status bar prints on ONE line using truncation, no wrap-bleed
    - Status bar refresh thread REMOVED — no more mid-type interruptions
      (status shows at startup, /clear, /stats, /setting)
    - @ai parse fixed: no longer echoes as "You → ALL @@ai hello"
    - Broadcast self-echo de-duped: local print on send, skip server re-echo
"""

from __future__ import annotations

import argparse
import collections
import re
import readline as _rl       # ↑/↓ history + Tab completion
import shutil
import socket
import sys
import threading
import time
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

import .zentraa_crypto as crypto
import .zentraa_protocol as proto
from .zentraa_config import ZENTRAConfig, load_config
from .zentraa_protocol import MT, ErrCode

# ─────────────────────────────────────────────────────────────────────────────
# Theme & console
# ─────────────────────────────────────────────────────────────────────────────

ZENTRAA_THEME = Theme({
    "banner":          "bold cyan",
    "info":            "bold green",
    "warn":            "bold yellow",
    "error":           "bold red",
    "highlight":       "bold magenta",
    "dim_text":        "dim white",
    "user":            "bold bright_cyan",
    "self_user":       "bold bright_white",
    "ai_label":        "bold bright_yellow",
    "ai_body":         "bright_yellow",
    "sys":             "bold dim cyan",
    "enc":             "bold bright_green",
    "ts":              "dim cyan",
    "dm_arrow":        "bold magenta",
    "broadcast":       "bright_white",
    "action_msg":      "italic bright_magenta",
    "presence_join":   "bold green",
    "presence_leave":  "bold red",
    "cmd_hint":        "dim italic cyan",
    "status_ok":       "bold bright_green",
    "status_warn":     "bold bright_yellow",
    "notif":           "bold bright_magenta",
    "divider":         "dim blue",
    "stat_key":        "bold cyan",
    "stat_val":        "bright_white",
    "online_dot":      "bold bright_green",
})

console = Console(theme=ZENTRAA_THEME, highlight=False)

# ─────────────────────────────────────────────────────────────────────────────
# Banners / static text
# ─────────────────────────────────────────────────────────────────────────────

BANNER = r"""
 ███████╗███████╗███╗   ██╗████████╗██████╗  █████╗  █████╗ 
 ╚══███╔╝██╔════╝████╗  ██║╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗
   ███╔╝ █████╗  ██╔██╗ ██║   ██║   ██████╔╝███████║███████║
  ███╔╝  ██╔══╝  ██║╚██╗██║   ██║   ██╔══██╗██╔══██║██╔══██║
 ███████╗███████╗██║ ╚████║   ██║   ██║  ██║██║  ██║██║  ██║
 ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝
"""

HELP_TEXT = """
[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]
[bold white]  ZENTRAA — Help Terminal[/bold white]
[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]

[bold yellow]── Messaging ─────────────────────────────────────[/bold yellow]
  [highlight]<message>[/highlight]               Broadcast to all users
  [highlight]@<userid> <msg>[/highlight]         Direct message a user
  [highlight]@u1 @u2 ... <msg>[/highlight]       Multi-user DM
  [highlight]@ai <msg>[/highlight]               Ask TIGER AI (private)
  [highlight]@ai @<uid> <msg>[/highlight]        AI reply visible to <uid>

[bold yellow]── Commands ──────────────────────────────────────[/bold yellow]
  [info]/help[/info]                   Show this help
  [info]/clear[/info]  [info]/cls[/info]           Clear screen
  [info]/ai[/info]                    TIGER AI information
  [info]/setting[/info]               View current settings
  [info]/users[/info]                 List online users
  [info]/me <action>[/info]           Send an action message
  [info]/whois <userid>[/info]        Info about a user
  [info]/ping[/info]                  Ping the server manually
  [info]/stats[/info]                 Session statistics
  [info]/notify <on|off>[/info]       Toggle bell notifications
  [info]/timestamps <on|off>[/info]   Toggle message timestamps
  [info]/quit[/info]  [info]/exit[/info]           Disconnect

[bold yellow]── Shortcuts ─────────────────────────────────────[/bold yellow]
  [highlight]↑ / ↓[/highlight]                  Command history
  [highlight]Tab[/highlight]                     Autocomplete @userid / /cmd
  [highlight]Ctrl+C / Ctrl+D[/highlight]         Quit

[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]
"""

AI_INFO_TEXT = """
[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]
[bold bright_yellow]  🐯 TIGER AI[/bold bright_yellow]
[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]
  [dim_text]Full Name  :[/dim_text] Tactical Intelligent Generative Expert
             Responser/Responder AI
  [dim_text]Engine     :[/dim_text] Pythonaibrain v1.1.9
  [dim_text]Author     :[/dim_text] Divyanshu Sinha
  [dim_text]Smart Mode :[/dim_text] AdvanceBrain LLM

  [bold yellow]Usage:[/bold yellow]
    @ai <question>           — Ask TIGER AI privately
    @ai @<uid> <question>    — Ask AI, share reply with <uid>

[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]
"""

# ─────────────────────────────────────────────────────────────────────────────
# Session statistics
# ─────────────────────────────────────────────────────────────────────────────

class SessionStats:
    def __init__(self) -> None:
        self.connected_at      = time.time()
        self.messages_sent     = 0
        self.messages_received = 0
        self.dms_sent          = 0
        self.dms_received      = 0
        self.ai_queries        = 0
        self.bytes_sent        = 0
        self.bytes_received    = 0
        self.ping_ms: Optional[float] = None
        self._ping_sent_at: Optional[float] = None

    def uptime(self) -> str:
        s = int(time.time() - self.connected_at)
        h, r = divmod(s, 3600)
        m, s = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def record_ping_sent(self) -> None:
        self._ping_sent_at = time.time()

    def record_pong(self) -> None:
        if self._ping_sent_at:
            self.ping_ms = (time.time() - self._ping_sent_at) * 1000
            self._ping_sent_at = None


# ─────────────────────────────────────────────────────────────────────────────
# Status bar  — single line, no wrap
# ─────────────────────────────────────────────────────────────────────────────

def _print_status(client: "ZENTRAClient") -> None:
    """
    Render a compact one-line status bar.
    We build a plain string first (so we can measure its true visible width),
    then print it with a Rich style.  This avoids the markup-vs-visible-width
    mismatch that caused 'SID:' to spill onto a second line.
    """
    with client._lock:
        oc = len(client.online_users)

    ping_str = f"{client.stats.ping_ms:.0f}ms" if client.stats.ping_ms else "---"
    sid      = (client.session_id or "N/A")[:8]
    notif    = "🔔" if client.notifications else "🔕"
    ts_flag  = "TS:ON" if client.cfg.client.show_timestamps else "TS:OFF"
    h        = client.cfg.client.default_host
    p        = client.cfg.client.default_port

    left  = (f"  🔐 ENC  │  👤 {client.userid}  │  🌐 {h}:{p}"
             f"  │  👥 {oc} online  │  🏓 {ping_str}  │  ⏱ {client.stats.uptime()}")
    right = f"  {notif} {ts_flag}  SID:{sid}  "

    tw  = shutil.get_terminal_size((100, 24)).columns
    gap = tw - len(left) - len(right)
    if gap < 1:
        left = left[: tw - len(right) - 1]
        gap  = 1

    line = left + " " * gap + right

    console.print(Rule(style="dim blue"))
    # no_wrap + overflow=crop guarantees exactly one line
    console.print(line, style="bold white on grey11", no_wrap=True, overflow="crop")
    console.print(Rule(style="dim blue"))


# ─────────────────────────────────────────────────────────────────────────────
# Input prompt  — plain text, NO embedded ANSI codes
# ─────────────────────────────────────────────────────────────────────────────

def _write_prompt(userid: str) -> None:
    """
    Write "Yash >>> " directly to stdout.

    Why not ANSI codes here?
    readline uses the raw byte-length of the prompt string to calculate cursor
    column position.  If you embed ANSI escape sequences without wrapping them
    in \\001...\\002 (RL_PROMPT_START_IGNORE / RL_PROMPT_END_IGNORE), readline
    over-counts the prompt width and produces the ←[1;96m artefacts seen in
    the previous version.  Plain text sidesteps the issue entirely.
    """
    sys.stdout.write(f"{userid} >>> ")
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Message parsing
# ─────────────────────────────────────────────────────────────────────────────

_AT_USER = re.compile(r"^@([\w\-]+)")


def parse_input(line: str, ai_userid: str) -> Tuple[str, List[str], str]:
    """
    Returns (kind, targets, body).
    kind: "broadcast" | "dm" | "ai"
    """
    line = line.strip()
    if not line:
        return "broadcast", [], ""

    tokens  = line.split()
    targets: List[str] = []
    i       = 0
    is_ai   = False

    while i < len(tokens):
        m = _AT_USER.match(tokens[i])
        if not m:
            break
        uid = m.group(1)
        if uid.lower() == ai_userid.lower() or uid.lower() == "ai":
            is_ai = True
        else:
            targets.append(uid)
        i += 1

    body = " ".join(tokens[i:])

    if is_ai:
        return "ai", targets, body
    if targets:
        return "dm", targets, body
    return "broadcast", [], body


# ─────────────────────────────────────────────────────────────────────────────
# Bell
# ─────────────────────────────────────────────────────────────────────────────

def _bell() -> None:
    sys.stdout.write("\a")
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# ZENTRAA Client
# ─────────────────────────────────────────────────────────────────────────────

class ZENTRAClient:

    def __init__(self, cfg: ZENTRAConfig, userid: str) -> None:
        self.cfg       = cfg
        self.userid    = userid
        self.sock: Optional[socket.socket] = None
        self.alive     = False
        self.session_id: Optional[str] = None

        # Crypto
        self.private_key, self.public_key = crypto.generate_rsa_keypair(
            cfg.encryption.rsa_key_bits
        )
        self.pub_pem    = crypto.serialize_public_key(self.public_key).decode()
        self.server_pub: Optional[crypto.RSAPublicKey] = None

        # State
        self.online_users: List[str]     = []
        self._lock                       = threading.Lock()
        self._recv_thread: Optional[threading.Thread] = None

        # Enhanced state
        self.stats         = SessionStats()
        self.notifications = True
        self._ai_thinking  = False
        self._ai_lock      = threading.Lock()
        self._manual_ping  = False
        self._last_event: Dict[str, str] = {}   # dedup presence spam

        # readline: Tab completion
        _rl.set_completer(self._tab_completer)
        _rl.parse_and_bind("tab: complete")
        _rl.set_completer_delims(" \t\n")

    # ── Tab completion ────────────────────────────────────────────────────────

    def _tab_completer(self, text: str, state: int) -> Optional[str]:
        buf    = _rl.get_line_buffer()
        tokens = buf.lstrip().split()
        last   = tokens[-1] if tokens else ""

        # @-mention completion
        if last.startswith("@"):
            prefix = last[1:].lower()
            with self._lock:
                hits = ["@" + u for u in self.online_users if u.lower().startswith(prefix)]
            if "ai".startswith(prefix):
                hits.append("@ai")
            if state < len(hits):
                return hits[state] + " "
            return None

        # /command completion
        if not tokens or (len(tokens) == 1 and buf.startswith("/")):
            cmds = [
                "/help", "/clear", "/cls", "/ai", "/setting", "/users",
                "/me ", "/whois ", "/ping", "/stats",
                "/notify ", "/timestamps ", "/quit", "/exit",
            ]
            prefix = (tokens[0] if tokens else "/").lower()
            hits   = [c for c in cmds if c.startswith(prefix)]
            if state < len(hits):
                return hits[state]
            return None

        return None

    # ── Timestamp string ──────────────────────────────────────────────────────

    def _ts_str(self, ts: float = 0) -> str:
        if not self.cfg.client.show_timestamps:
            return ""
        src = ts if ts else time.time()
        return datetime.fromtimestamp(src).strftime(self.cfg.client.timestamp_format) + "  "

    # ── Message display ───────────────────────────────────────────────────────

    def _print_msg(
        self,
        from_id: str,
        to: List[str],
        body: str,
        ts: float = 0,
        is_ai: bool = False,
        is_action: bool = False,
    ) -> None:
        ts_str = self._ts_str(ts)

        if is_ai:
            # Distinct bordered panel for AI replies
            console.print(
                Panel(
                    f"[ai_body]{body}[/ai_body]",
                    title="[ai_label]🐯 TIGER AI[/ai_label]",
                    border_style="bright_yellow",
                    padding=(0, 1),
                    expand=False,
                )
            )
            return

        if is_action:
            console.print(f"[ts]{ts_str}[/ts][action_msg]  ✦  {from_id} {body}[/action_msg]")
            return

        if from_id == self.userid:
            dest = ", ".join(to) if to else "ALL"
            console.print(
                f"[ts]{ts_str}[/ts]"
                f"[self_user]{from_id}[/self_user] "
                f"[dm_arrow]>>>[/dm_arrow] "
                f"[highlight]{dest}[/highlight]   "
                f"[broadcast]{body}[/broadcast]"
            )
        elif to:
            # DM received
            console.print(
                f"[ts]{ts_str}[/ts]"
                f"[user]{from_id}[/user] "
                f"[dm_arrow]>>>[/dm_arrow] "
                f"[highlight]You[/highlight]   "
                f"[broadcast]{body}[/broadcast]"
            )
            if self.notifications:
                _bell()
        else:
            # Broadcast from others
            console.print(
                f"[ts]{ts_str}[/ts]"
                f"[user]{from_id}[/user]   "
                f"[broadcast]{body}[/broadcast]"
            )

    def _sys(self, msg: str, style: str = "sys") -> None:
        console.print(f"[{style}]  ℹ  {msg}[/{style}]")

    # ── Encryption I/O ────────────────────────────────────────────────────────

    def _send_encrypted(self, msg: dict) -> None:
        if not self.server_pub:
            return
        payload = proto.encode(msg)
        packet  = crypto.encrypt(payload, self.server_pub, self.private_key)
        self.stats.bytes_sent += len(packet)
        try:
            self.sock.sendall(crypto.frame(packet))      # type: ignore[union-attr]
        except OSError:
            self.alive = False

    def _send_plain(self, msg: dict) -> None:
        data = proto.encode(msg)
        try:
            self.sock.sendall(crypto.frame(data))        # type: ignore[union-attr]
        except OSError:
            self.alive = False

    # ── Handshake ─────────────────────────────────────────────────────────────

    def _handshake(self) -> bool:
        eph_priv, eph_pub = crypto.generate_x25519_keypair()
        from cryptography.hazmat.primitives import serialization as _ser
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
            console.print(f"[error]  ✗  Server rejected: {msg.get('reason')}[/error]")
            return False
        if msg.get("type") != MT.HELLO_ACK:
            console.print(f"[error]  ✗  Unexpected response: {msg.get('type')}[/error]")
            return False

        srv_pub_pem     = msg.get("server_rsa_pub_pem", "")
        self.server_pub = crypto.load_public_key(srv_pub_pem.encode())
        self.session_id = msg.get("session_id")
        motd            = msg.get("motd", "")
        motd_line       = f"\n[sys]{motd}[/sys]" if motd else ""

        console.print(
            Panel(
                f"[enc]🔐  End-to-end encrypted session established[/enc]\n"
                f"[dim_text]Session ID : {self.session_id}\n"
                f"Cipher     : RSA-2048-OAEP + AES-256-GCM + RSA-PSS + Curve25519"
                f"[/dim_text]{motd_line}",
                title="[enc] Secure Channel [/enc]",
                border_style="bright_green",
            )
        )
        return True

    # ── Receive loop ──────────────────────────────────────────────────────────

    def _receiver(self) -> None:
        while self.alive:
            try:
                raw = crypto.recv_frame(self.sock)
            except ConnectionError:
                break

            self.stats.bytes_received += len(raw)

            try:
                payload = crypto.decrypt(raw, self.private_key, self.server_pub)
                msg     = proto.decode(payload)
            except ValueError as exc:
                console.print(f"[error]  ⚠  Decrypt error: {exc}[/error]")
                continue

            mt = msg.get("type")
            ts = msg.get("ts", 0.0)

            if mt == MT.MSG:
                from_id   = msg.get("from", "")
                to        = msg.get("to", [])
                is_action = msg.get("action", False)

                # Skip server re-echo of our own broadcasts (we already
                # printed locally on send).  DMs and actions still show.
                if from_id == self.userid and not to and not is_action:
                    continue

                self.stats.messages_received += 1
                if to:
                    self.stats.dms_received += 1
                self._print_msg(from_id, to, msg.get("body", ""), ts,
                                is_action=is_action)

            elif mt == MT.AI_REPLY:
                with self._ai_lock:
                    self._ai_thinking = False
                self._print_msg(
                    msg.get("from", ""), msg.get("targets", []),
                    msg.get("body", ""), ts, is_ai=True,
                )

            elif mt == MT.PRESENCE:
                uid    = msg.get("userid", "")
                status = msg.get("status", "")
                if self._last_event.get(uid) == status:
                    continue
                self._last_event[uid] = status

                if status == "join":
                    with self._lock:
                        if uid not in self.online_users:
                            self.online_users.append(uid)
                    console.print(f"[presence_join]  ●  {uid} joined the zone[/presence_join]")
                    if self.notifications and uid != self.userid:
                        _bell()
                elif status == "leave":
                    with self._lock:
                        self.online_users = [u for u in self.online_users if u != uid]
                    console.print(f"[presence_leave]  ○  {uid} left the zone[/presence_leave]")

            elif mt == MT.USER_LIST:
                with self._lock:
                    self.online_users = msg.get("users", [])
                self._cmd_users()

            elif mt == MT.PING:
                self._send_encrypted(proto.build_pong())

            elif mt == MT.PONG:
                self.stats.record_pong()
                if self._manual_ping:
                    self._manual_ping = False
                    console.print(
                        f"[info]  🏓  Pong — {self.stats.ping_ms:.1f} ms[/info]"
                    )

            elif mt == MT.ERROR:
                console.print(
                    Panel(
                        f"[error]{msg.get('reason', 'Unknown error')}[/error]",
                        title=f"[error]  Server Error  [{msg.get('code', '')}]  [/error]",
                        border_style="red",
                        expand=False,
                    )
                )

            elif mt == MT.SYS:
                self._sys(msg.get("body", ""))

        self.alive = False
        console.print("\n[warn]  ⚡  Connection closed.[/warn]")

    # ── Keepalive pinger ──────────────────────────────────────────────────────

    def _pinger(self) -> None:
        interval = self.cfg.server.ping_interval
        while self.alive:
            time.sleep(interval)
            if not self.alive:
                break
            self.stats.record_ping_sent()
            self._send_encrypted(proto.build_ping())

    # ── Command helpers ───────────────────────────────────────────────────────

    def _cmd_users(self) -> None:
        with self._lock:
            users = list(self.online_users)
        ai_uid = self.cfg.ai.ai_userid

        t = Table(
            title="Online Users",
            box=box.ROUNDED,
            border_style="cyan",
            header_style="bold cyan",
        )
        t.add_column("#",       style="dim", width=4)
        t.add_column("User ID", style="user")
        t.add_column("Status",  style="info")
        t.add_column("Note",    style="ai_label")

        for i, u in enumerate(users, 1):
            note = ("🐯 TIGER AI" if u == ai_uid
                    else "👤 You" if u == self.userid
                    else "")
            t.add_row(str(i), u, "[presence_join]● online[/presence_join]", note)

        console.print(t)
        console.print(
            f"[dim_text]  {len(users)} user(s)  ·  @userid to DM  ·  @ai to query TIGER AI[/dim_text]"
        )

    def _cmd_settings(self) -> None:
        cfg = self.cfg
        t = Table(
            title="ZENTRAA Settings",
            box=box.ROUNDED,
            border_style="cyan",
            header_style="bold cyan",
        )
        t.add_column("Setting", style="stat_key", min_width=22)
        t.add_column("Value",   style="stat_val")
        rows = [
            ("Host",          f"{cfg.client.default_host}:{cfg.client.default_port}"),
            ("User ID",        self.userid),
            ("Session ID",    (self.session_id or "N/A")[:32]),
            ("Encryption",     "RSA-2048-OAEP + AES-256-GCM + RSA-PSS + Curve25519"),
            ("Timestamps",     "ON" if cfg.client.show_timestamps else "OFF"),
            ("Notifications",  "ON" if self.notifications else "OFF"),
            ("Theme",          cfg.client.theme),
            ("Color Scheme",   cfg.ui.color_scheme),
            ("AI User ID",     cfg.ai.ai_userid),
            ("Smart AI",       str(cfg.ai.smart_ai)),
            ("Uptime",         self.stats.uptime()),
            ("Ping",           f"{self.stats.ping_ms:.1f} ms" if self.stats.ping_ms else "N/A"),
        ]
        for k, v in rows:
            t.add_row(k, v)
        console.print(t)

    def _cmd_stats(self) -> None:
        s = self.stats
        t = Table(
            title="Session Statistics",
            box=box.ROUNDED,
            border_style="cyan",
            header_style="bold cyan",
        )
        t.add_column("Metric", style="stat_key", min_width=22)
        t.add_column("Value",  style="stat_val")
        rows = [
            ("Uptime",            s.uptime()),
            ("Ping (last)",       f"{s.ping_ms:.1f} ms" if s.ping_ms else "N/A"),
            ("Messages Sent",     str(s.messages_sent)),
            ("Messages Received", str(s.messages_received)),
            ("DMs Sent",          str(s.dms_sent)),
            ("DMs Received",      str(s.dms_received)),
            ("AI Queries",        str(s.ai_queries)),
            ("Bytes Sent",        f"{s.bytes_sent:,}"),
            ("Bytes Received",    f"{s.bytes_received:,}"),
        ]
        for k, v in rows:
            t.add_row(k, v)
        console.print(t)

    def _cmd_whois(self, uid: str) -> None:
        with self._lock:
            users = list(self.online_users)
        if uid not in users:
            console.print(f"[warn]  User '{uid}' is not currently online.[/warn]")
            return
        ai_uid = self.cfg.ai.ai_userid
        t = Table(
            title=f"Whois: {uid}",
            box=box.ROUNDED,
            border_style="cyan",
            header_style="bold cyan",
        )
        t.add_column("Field", style="stat_key")
        t.add_column("Info",  style="stat_val")
        t.add_row("User ID", uid)
        t.add_row("Status",  "[presence_join]● online[/presence_join]")
        t.add_row("Type",    ("🐯 AI Agent" if uid == ai_uid
                              else "👤 You" if uid == self.userid
                              else "👤 Human"))
        if uid == ai_uid:
            t.add_row("Engine", "Pythonaibrain v1.1.9 — AdvanceBrain LLM")
        console.print(t)

    # ── Input loop ────────────────────────────────────────────────────────────

    def _input_loop(self) -> None:
        ai_uid   = self.cfg.ai.ai_userid
        stop_evt = threading.Event()

        # Print status once at startup — not in a background thread
        _print_status(self)
        console.print(
            "[cmd_hint]  Type a message · @userid msg · @ai msg"
            " · /help · Tab autocomplete[/cmd_hint]\n"
        )

        def _read_lines() -> None:
            while self.alive:
                try:
                    _write_prompt(self.userid)      # plain "Yash >>> "
                    line = sys.stdin.readline()
                    if not line:                    # EOF / Ctrl-D
                        break
                    line = line.rstrip("\n").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not line:
                    continue

                _rl.add_history(line)

                # ── Slash commands ─────────────────────────────────────────────
                if line.startswith("/"):
                    parts = line.split(maxsplit=2)
                    cmd   = parts[0].lower()

                    if cmd in ("/quit", "/exit"):
                        break

                    elif cmd in ("/clear", "/cls"):
                        console.clear()
                        console.print(Text(BANNER, style="banner"))
                        _print_status(self)

                    elif cmd == "/help":
                        console.print(HELP_TEXT)

                    elif cmd == "/ai":
                        console.print(AI_INFO_TEXT)

                    elif cmd == "/setting":
                        self._cmd_settings()
                        _print_status(self)

                    elif cmd == "/users":
                        self._send_encrypted(proto.build_user_list([]))

                    elif cmd == "/stats":
                        self._cmd_stats()

                    elif cmd == "/ping":
                        self._manual_ping = True
                        self.stats.record_ping_sent()
                        self._send_encrypted(proto.build_ping())
                        console.print("[info]  🏓  PING sent — waiting for pong…[/info]")

                    elif cmd == "/me":
                        if len(parts) < 2:
                            console.print("[warn]  Usage: /me <action text>[/warn]")
                        else:
                            action_body = " ".join(parts[1:])
                            m = proto.build_msg(self.userid, [], action_body)
                            m["action"] = True
                            self._send_encrypted(m)
                            self.stats.messages_sent += 1
                            console.print(
                                f"[action_msg]  ✦  {self.userid} {action_body}[/action_msg]"
                            )

                    elif cmd == "/whois":
                        if len(parts) < 2:
                            console.print("[warn]  Usage: /whois <userid>[/warn]")
                        else:
                            self._cmd_whois(parts[1])

                    elif cmd == "/notify":
                        if len(parts) < 2 or parts[1].lower() not in ("on", "off"):
                            state = "ON" if self.notifications else "OFF"
                            console.print(
                                f"[warn]  Notifications: {state}. Use /notify on|off[/warn]"
                            )
                        else:
                            self.notifications = parts[1].lower() == "on"
                            icon = "🔔" if self.notifications else "🔕"
                            console.print(
                                f"[info]  {icon}  Notifications "
                                f"{'ON' if self.notifications else 'OFF'}[/info]"
                            )

                    elif cmd == "/timestamps":
                        if len(parts) < 2 or parts[1].lower() not in ("on", "off"):
                            state = "ON" if self.cfg.client.show_timestamps else "OFF"
                            console.print(
                                f"[warn]  Timestamps: {state}. Use /timestamps on|off[/warn]"
                            )
                        else:
                            self.cfg.client.show_timestamps = parts[1].lower() == "on"
                            console.print(
                                f"[info]  Timestamps "
                                f"{'ON' if self.cfg.client.show_timestamps else 'OFF'}[/info]"
                            )

                    else:
                        console.print(f"[warn]  Unknown command: {cmd}  (/help)[/warn]")
                    continue

                # ── Message dispatch ───────────────────────────────────────────
                if not self.alive:
                    break

                kind, targets, body = parse_input(line, ai_uid)

                if not body:
                    console.print("[warn]  Empty message body.[/warn]")
                    continue

                if kind == "broadcast":
                    self._send_encrypted(proto.build_msg(self.userid, [], body))
                    self.stats.messages_sent += 1
                    # Echo locally; _receiver skips server re-echo for self-broadcasts
                    self._print_msg(self.userid, [], body)

                elif kind == "dm":
                    self._send_encrypted(proto.build_msg(self.userid, targets, body))
                    self.stats.messages_sent += 1
                    self.stats.dms_sent      += 1
                    self._print_msg(self.userid, targets, body)

                elif kind == "ai":
                    # ── FIX: do NOT call _print_msg here — the raw line
                    # contained "@ai …" which caused "You → ALL @@ai …".
                    # We only print the "Asking…" notice instead.
                    self.stats.ai_queries += 1
                    all_targets = [self.userid] + targets
                    self._send_encrypted(
                        proto.build_ai_query(self.userid, all_targets, body)
                    )
                    with self._ai_lock:
                        self._ai_thinking = True
                    preview = body[:55] + ("…" if len(body) > 55 else "")
                    console.print(
                        f"[ai_label]  🐯  Asking TIGER AI…[/ai_label]  "
                        f"[dim_text]{preview}[/dim_text]"
                    )

            stop_evt.set()
            self.alive = False

        reader = threading.Thread(target=_read_lines, daemon=True)
        reader.start()

        # Wake stop_evt if the connection drops
        def _watch() -> None:
            while self.alive:
                time.sleep(0.5)
            stop_evt.set()

        threading.Thread(target=_watch, daemon=True).start()

        # ── NOTE: No background status-bar refresh thread ──────────────────
        # The previous version had a thread that printed a status bar every
        # 30 s.  This raced with sys.stdin.readline() and produced output in
        # the middle of the user's typed line.  Status is now static and only
        # refreshed explicitly (/clear, /setting, startup).

        stop_evt.wait()
        self.alive = False

    # ── Connect ───────────────────────────────────────────────────────────────

    def connect(self, host: str, port: int) -> None:
        max_attempts = self.cfg.network.max_reconnect_attempts
        delay        = self.cfg.network.reconnect_delay
        attempt      = 0

        while True:
            attempt += 1
            console.print(
                f"[info]  Connecting to[/info] [highlight]{host}:{port}[/highlight]"
                + (f" [dim_text](attempt {attempt})[/dim_text]" if attempt > 1 else "")
                + " …"
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
                    console.print("[error]  Max reconnect attempts reached.[/error]")
                    sys.exit(1)
                console.print(f"[dim_text]  Retrying in {delay}s…[/dim_text]")
                time.sleep(delay)

        self.alive = True

        if not self._handshake():
            console.print("[error]  Handshake failed. Exiting.[/error]")
            sys.exit(1)

        self._send_encrypted(proto.build_user_list([]))

        self._recv_thread = threading.Thread(target=self._receiver, daemon=True)
        self._recv_thread.start()

        threading.Thread(target=self._pinger, daemon=True).start()

        self._input_loop()

        # Clean shutdown
        self.alive = False
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
            self.sock.close()
        except OSError:
            pass

        console.print()
        self._cmd_stats()
        console.print("[info]  Disconnected from ZENTRAA. Stay safe. 🔐[/info]")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="zentraa-client",
        description="ZENTRAA — Zone for Encrypted Networked Talks & Real-time AI Agent",
    )
    parser.add_argument("--config", "-c", default=None, help="Path to ZENTRAA.pbcfg")
    parser.add_argument("--host",   "-H", default=None, help="Server host")
    parser.add_argument("--port",   "-p", default=None, type=int, help="Server port")
    parser.add_argument("--userid", "-u", default=None, help="Your user ID")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    host = args.host or cfg.client.default_host
    port = args.port or cfg.client.default_port

    console.clear()
    console.print(Text(BANNER, style="banner"))
    console.print(
        Panel(
            Group(
                Text(
                    "RSA-2048-OAEP  •  AES-256-GCM  •  RSA-PSS  •  Curve25519",
                    style="dim_text", justify="center",
                ),
                Text(
                    "Powered by Pythonaibrain v1.1.9  •  Author: Divyanshu Sinha",
                    style="dim_text", justify="center",
                ),
                Text(
                    "↑/↓ history  •  Tab autocomplete  •  /help for all commands",
                    style="cmd_hint", justify="center",
                ),
            ),
            title="[bold cyan] ZENTRAA CLIENT v2.1.0 [/bold cyan]",
            border_style="cyan",
            padding=(1, 4),
        )
    )

    userid = args.userid
    if not userid:
        userid_re = re.compile(
            rf"^[{cfg.server.userid_chars}]{{{cfg.server.userid_min_length},{cfg.server.userid_max_length}}}$"
        )
        while True:
            userid = Prompt.ask("[highlight]  Enter your User ID[/highlight]").strip()
            if userid_re.match(userid):
                break
            console.print(
                f"[warn]  User ID must be {cfg.server.userid_min_length}–"
                f"{cfg.server.userid_max_length} chars [{cfg.server.userid_chars}][/warn]"
            )

    ZENTRAClient(cfg, userid).connect(host, port)


if __name__ == "__main__":
    main()
