"""
zentraa_protocol.py  —  ZENTRAA Wire Protocol
==============================================
Powered By        : Pythonaibrain
Author            : Divyanshu Sinha
Version           : 1.0.0

Defines all JSON message types exchanged between server and clients.
Every message is serialised as UTF-8 JSON, then encrypted and framed.

Handshake flow:
  Client → Server :  HELLO      { userid, rsa_pub_pem, ecdh_pub_bytes }
  Server → Client :  HELLO_ACK  { server_rsa_pub_pem, session_id }  [encrypted]
  Server → All    :  PRESENCE   { userid, status: join|leave }        [encrypted]
  Client → Server :  MSG        { from, to: list|"ALL", body, ts }    [encrypted]
  Server → Client :  MSG        { from, to, body, ts }                [encrypted]
  Client → Server :  PING       {}
  Server → Client :  PONG       {}
  Server → Client :  ERROR      { code, reason }
  Client → Server :  AI_QUERY   { from, targets: list|[], body, ts }  [encrypted]
  Server → AI     :  AI_QUERY   { from, targets, body, ts }           [encrypted]
  AI     → Server :  AI_REPLY   { from, targets, body, ts }           [encrypted]
  Server → Client :  AI_REPLY   { from, targets, body, ts }           [encrypted]
  Client → Server :  USER_LIST  {}
  Server → Client :  USER_LIST  { users: list[str] }
"""

from __future__ import annotations

import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Message type constants
# ─────────────────────────────────────────────────────────────────────────────

class MT(str, Enum):
    HELLO      = "HELLO"
    HELLO_ACK  = "HELLO_ACK"
    PRESENCE   = "PRESENCE"
    MSG        = "MSG"
    PING       = "PING"
    PONG       = "PONG"
    ERROR      = "ERROR"
    AI_QUERY   = "AI_QUERY"
    AI_REPLY   = "AI_REPLY"
    USER_LIST  = "USER_LIST"
    SYS        = "SYS"


# ─────────────────────────────────────────────────────────────────────────────
# Error codes
# ─────────────────────────────────────────────────────────────────────────────

class ErrCode(str, Enum):
    BAD_USERID      = "BAD_USERID"
    USERID_TAKEN    = "USERID_TAKEN"
    DECRYPTION_FAIL = "DECRYPTION_FAIL"
    TARGET_OFFLINE  = "TARGET_OFFLINE"
    MSG_TOO_LARGE   = "MSG_TOO_LARGE"
    RATE_LIMIT      = "RATE_LIMIT"
    INTERNAL        = "INTERNAL"
    UNKNOWN_MSG     = "UNKNOWN_MSG"


# ─────────────────────────────────────────────────────────────────────────────
# Builder helpers — return plain dicts ready for json.dumps()
# ─────────────────────────────────────────────────────────────────────────────

def _base(mt: MT) -> Dict[str, Any]:
    return {"type": mt.value, "ts": time.time()}


def build_hello(userid: str, rsa_pub_pem: str, ecdh_pub_hex: str) -> Dict[str, Any]:
    d = _base(MT.HELLO)
    d["userid"]      = userid
    d["rsa_pub_pem"] = rsa_pub_pem
    d["ecdh_pub_hex"] = ecdh_pub_hex
    return d


def build_hello_ack(
    server_rsa_pub_pem: str,
    session_id: str,
    motd: str = "",
) -> Dict[str, Any]:
    d = _base(MT.HELLO_ACK)
    d["server_rsa_pub_pem"] = server_rsa_pub_pem
    d["session_id"]         = session_id
    d["motd"]               = motd
    return d


def build_presence(userid: str, status: str) -> Dict[str, Any]:
    d = _base(MT.PRESENCE)
    d["userid"] = userid
    d["status"] = status          # "join" | "leave"
    return d


def build_msg(
    from_id: str,
    to: List[str],
    body: str,
) -> Dict[str, Any]:
    d = _base(MT.MSG)
    d["from"] = from_id
    d["to"]   = to               # empty list → broadcast
    d["body"] = body
    return d


def build_ping() -> Dict[str, Any]:
    return _base(MT.PING)


def build_pong() -> Dict[str, Any]:
    return _base(MT.PONG)


def build_error(code: ErrCode, reason: str) -> Dict[str, Any]:
    d = _base(MT.ERROR)
    d["code"]   = code.value
    d["reason"] = reason
    return d


def build_ai_query(
    from_id: str,
    targets: List[str],
    body: str,
) -> Dict[str, Any]:
    d = _base(MT.AI_QUERY)
    d["from"]    = from_id
    d["targets"] = targets       # users who should also see the reply
    d["body"]    = body
    return d


def build_ai_reply(
    from_id: str,
    targets: List[str],
    body: str,
) -> Dict[str, Any]:
    d = _base(MT.AI_REPLY)
    d["from"]    = from_id
    d["targets"] = targets
    d["body"]    = body
    return d


def build_user_list(users: List[str]) -> Dict[str, Any]:
    d = _base(MT.USER_LIST)
    d["users"] = users
    return d


def build_sys(body: str) -> Dict[str, Any]:
    d = _base(MT.SYS)
    d["body"] = body
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────────────────

def encode(msg: Dict[str, Any]) -> bytes:
    return json.dumps(msg, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def decode(data: bytes) -> Dict[str, Any]:
    return json.loads(data.decode("utf-8"))