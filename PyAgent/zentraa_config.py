"""
zentraa_config.py  —  ZENTRAA Configuration System
====================================================
Powered By        : Pythonaibrain
Author            : Divyanshu Sinha
Version           : 1.0.0
Pythonaibrain Ver : 1.1.9
AI Name           : TIGER AI (Tactical Intelligent Generative Expert Responser/Responder AI)

Loads and exposes all settings from ZENTRAA.pbcfg.
Import as:  from zentraa_config import cfg
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Default config file search order ─────────────────────────────────────────
_DEFAULT_SEARCH = [
    Path("ZENTRAA.pbcfg"),
    Path.home() / ".zentraa" / "ZENTRAA.pbcfg",
    Path("/etc/zentraa/ZENTRAA.pbcfg"),
]

_CONFIG_TEMPLATE = Path(__file__).parent / "ZENTRAA.pbcfg"


# ─────────────────────────────────────────────────────────────────────────────
# Typed section dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NetworkConfig:
    host: str = "0.0.0.0"
    port: int = 9999
    max_connections: int = 100
    backlog: int = 10
    socket_timeout: int = 30
    reconnect_delay: int = 3
    max_reconnect_attempts: int = 5


@dataclass
class EncryptionConfig:
    rsa_key_bits: int = 2048
    aes_key_bytes: int = 32
    aes_nonce_bytes: int = 12
    aes_tag_bytes: int = 16
    curve25519_enabled: bool = True
    rsa_pss_salt_length: int = -1
    rsa_pss_hash: str = "sha256"
    rsa_oaep_hash: str = "sha256"
    session_key_rotation: int = 500


@dataclass
class ServerConfig:
    keys_dir: str = ".zentraa_keys"
    server_private_key: str = "server_private.pem"
    server_public_key: str = "server_public.pem"
    log_level: str = "INFO"
    log_file: str = "zentraa_server.log"
    broadcast_presence: bool = True
    max_message_size: int = 1_048_576
    ping_interval: int = 15
    userid_min_length: int = 3
    userid_max_length: int = 24
    userid_chars: str = "a-zA-Z0-9_-"


@dataclass
class ClientConfig:
    default_host: str = "127.0.0.1"
    default_port: int = 9999
    show_timestamps: bool = True
    timestamp_format: str = "%H:%M:%S"
    theme: str = "dark"
    notifications: bool = True
    history_size: int = 500
    auto_scroll: bool = True


@dataclass
class AIConfig:
    ai_userid: str = "TIGER_AI"
    ai_display_name: str = "TIGER AI"
    smart_ai: bool = True
    intents_path: str = "intents.json"
    memory_path: str = "tiger_memory.json"
    memory_fit_interval: int = 100
    ai_prefix: str = "[TIGER AI]"
    max_response_length: int = 1024
    typing_delay: float = 0.02
    grammar_correction: bool = True
    tts: bool = False
    vectorizer_mode: str = "bow"
    join_message: str = "TIGER AI has entered the zone."
    leave_message: str = "TIGER AI has left the zone."


@dataclass
class LoggingConfig:
    level: str = "WARNING"
    format: str = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"


@dataclass
class UIConfig:
    banner_style: str = "full"
    border_style: str = "rounded"
    color_scheme: str = "cyber"
    show_user_list: bool = True
    show_encryption_status: bool = True
    spinner_style: str = "dots"


@dataclass
class ZENTRAConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    encryption: EncryptionConfig = field(default_factory=EncryptionConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    _path: Optional[Path] = field(default=None, repr=False)

    def reload(self) -> None:
        """Re-read config from disk and update all sections in place."""
        if self._path and self._path.exists():
            _load_into(self, self._path)

    @property
    def config_path(self) -> Optional[Path]:
        return self._path


# ─────────────────────────────────────────────────────────────────────────────
# Internal parser helpers
# ─────────────────────────────────────────────────────────────────────────────

def _b(parser: configparser.ConfigParser, sec: str, key: str, default: bool) -> bool:
    try:
        return parser.getboolean(sec, key)
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
        return default


def _i(parser: configparser.ConfigParser, sec: str, key: str, default: int) -> int:
    try:
        return parser.getint(sec, key)
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
        return default


def _f(parser: configparser.ConfigParser, sec: str, key: str, default: float) -> float:
    try:
        return parser.getfloat(sec, key)
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
        return default


def _s(parser: configparser.ConfigParser, sec: str, key: str, default: str) -> str:
    try:
        return parser.get(sec, key)
    except (configparser.NoSectionError, configparser.NoOptionError):
        return default


def _load_into(cfg: ZENTRAConfig, path: Path) -> None:
    """Parse *path* and populate *cfg* sections."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(str(path), encoding="utf-8")

    n = cfg.network
    n.host                    = _s(parser, "network", "host", n.host)
    n.port                    = _i(parser, "network", "port", n.port)
    n.max_connections         = _i(parser, "network", "max_connections", n.max_connections)
    n.backlog                 = _i(parser, "network", "backlog", n.backlog)
    n.socket_timeout          = _i(parser, "network", "socket_timeout", n.socket_timeout)
    n.reconnect_delay         = _i(parser, "network", "reconnect_delay", n.reconnect_delay)
    n.max_reconnect_attempts  = _i(parser, "network", "max_reconnect_attempts", n.max_reconnect_attempts)

    e = cfg.encryption
    e.rsa_key_bits            = _i(parser, "encryption", "rsa_key_bits", e.rsa_key_bits)
    e.aes_key_bytes           = _i(parser, "encryption", "aes_key_bytes", e.aes_key_bytes)
    e.aes_nonce_bytes         = _i(parser, "encryption", "aes_nonce_bytes", e.aes_nonce_bytes)
    e.aes_tag_bytes           = _i(parser, "encryption", "aes_tag_bytes", e.aes_tag_bytes)
    e.curve25519_enabled      = _b(parser, "encryption", "curve25519_enabled", e.curve25519_enabled)
    e.rsa_pss_salt_length     = _i(parser, "encryption", "rsa_pss_salt_length", e.rsa_pss_salt_length)
    e.rsa_pss_hash            = _s(parser, "encryption", "rsa_pss_hash", e.rsa_pss_hash)
    e.rsa_oaep_hash           = _s(parser, "encryption", "rsa_oaep_hash", e.rsa_oaep_hash)
    e.session_key_rotation    = _i(parser, "encryption", "session_key_rotation", e.session_key_rotation)

    sv = cfg.server
    sv.keys_dir               = _s(parser, "server", "keys_dir", sv.keys_dir)
    sv.server_private_key     = _s(parser, "server", "server_private_key", sv.server_private_key)
    sv.server_public_key      = _s(parser, "server", "server_public_key", sv.server_public_key)
    sv.log_level              = _s(parser, "server", "log_level", sv.log_level)
    sv.log_file               = _s(parser, "server", "log_file", sv.log_file)
    sv.broadcast_presence     = _b(parser, "server", "broadcast_presence", sv.broadcast_presence)
    sv.max_message_size       = _i(parser, "server", "max_message_size", sv.max_message_size)
    sv.ping_interval          = _i(parser, "server", "ping_interval", sv.ping_interval)
    sv.userid_min_length      = _i(parser, "server", "userid_min_length", sv.userid_min_length)
    sv.userid_max_length      = _i(parser, "server", "userid_max_length", sv.userid_max_length)
    sv.userid_chars           = _s(parser, "server", "userid_chars", sv.userid_chars)

    cl = cfg.client
    cl.default_host           = _s(parser, "client", "default_host", cl.default_host)
    cl.default_port           = _i(parser, "client", "default_port", cl.default_port)
    cl.show_timestamps        = _b(parser, "client", "show_timestamps", cl.show_timestamps)
    cl.timestamp_format       = _s(parser, "client", "timestamp_format", cl.timestamp_format)
    cl.theme                  = _s(parser, "client", "theme", cl.theme)
    cl.notifications          = _b(parser, "client", "notifications", cl.notifications)
    cl.history_size           = _i(parser, "client", "history_size", cl.history_size)
    cl.auto_scroll            = _b(parser, "client", "auto_scroll", cl.auto_scroll)

    ai = cfg.ai
    ai.ai_userid              = _s(parser, "ai", "ai_userid", ai.ai_userid)
    ai.ai_display_name        = _s(parser, "ai", "ai_display_name", ai.ai_display_name)
    ai.smart_ai               = _b(parser, "ai", "smart_ai", ai.smart_ai)
    ai.intents_path           = _s(parser, "ai", "intents_path", ai.intents_path)
    ai.memory_path            = _s(parser, "ai", "memory_path", ai.memory_path)
    ai.memory_fit_interval    = _i(parser, "ai", "memory_fit_interval", ai.memory_fit_interval)
    ai.ai_prefix              = _s(parser, "ai", "ai_prefix", ai.ai_prefix)
    ai.max_response_length    = _i(parser, "ai", "max_response_length", ai.max_response_length)
    ai.typing_delay           = _f(parser, "ai", "typing_delay", ai.typing_delay)
    ai.grammar_correction     = _b(parser, "ai", "grammar_correction", ai.grammar_correction)
    ai.tts                    = _b(parser, "ai", "tts", ai.tts)
    ai.vectorizer_mode        = _s(parser, "ai", "vectorizer_mode", ai.vectorizer_mode)
    ai.join_message           = _s(parser, "ai", "join_message", ai.join_message)
    ai.leave_message          = _s(parser, "ai", "leave_message", ai.leave_message)

    lg = cfg.logging
    lg.level                  = _s(parser, "logging", "level", lg.level)
    lg.format                 = _s(parser, "logging", "format", lg.format)

    ui = cfg.ui
    ui.banner_style           = _s(parser, "ui", "banner_style", ui.banner_style)
    ui.border_style           = _s(parser, "ui", "border_style", ui.border_style)
    ui.color_scheme           = _s(parser, "ui", "color_scheme", ui.color_scheme)
    ui.show_user_list         = _b(parser, "ui", "show_user_list", ui.show_user_list)
    ui.show_encryption_status = _b(parser, "ui", "show_encryption_status", ui.show_encryption_status)
    ui.spinner_style          = _s(parser, "ui", "spinner_style", ui.spinner_style)


# ─────────────────────────────────────────────────────────────────────────────
# Public loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: Optional[str | Path] = None) -> ZENTRAConfig:
    """
    Load ZENTRAA.pbcfg and return a typed ZENTRAConfig.

    Search order (when *path* is None):
      1. ./ZENTRAA.pbcfg
      2. ~/.zentraa/ZENTRAA.pbcfg
      3. /etc/zentraa/ZENTRAA.pbcfg
      4. Bundled defaults (no file required)
    """
    config = ZENTRAConfig()

    if path is not None:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Config file not found: {resolved}")
        _load_into(config, resolved)
        object.__setattr__(config, "_path", resolved)
        return config

    for candidate in _DEFAULT_SEARCH:
        if candidate.exists():
            _load_into(config, candidate)
            object.__setattr__(config, "_path", candidate)
            return config

    # No file found — use defaults silently
    return config


# ── Module-level singleton ────────────────────────────────────────────────────
cfg: ZENTRAConfig = load_config()


def reload() -> ZENTRAConfig:
    """Reload the config from disk and return the updated singleton."""
    global cfg
    cfg.reload()
    return cfg