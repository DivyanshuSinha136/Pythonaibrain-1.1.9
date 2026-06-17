"""
zentraa_crypto.py  —  ZENTRAA Cryptographic Engine
====================================================
Powered By        : Pythonaibrain
Author            : Divyanshu Sinha
Version           : 1.0.0
Pythonaibrain Ver : 1.1.9

Implements full end-to-end encryption stack:
  • RSA-2048-OAEP    — asymmetric key wrapping
  • AES-256-GCM      — authenticated symmetric encryption
  • RSA-PSS          — digital signatures (authenticity + integrity)
  • Curve25519 ECDH  — ephemeral Diffie-Hellman key exchange

Wire protocol for each encrypted packet (all lengths little-endian uint32):
  [4]  total_length
  [4]  ecdh_pub_len       → sender's ephemeral Curve25519 public key
  [N]  ecdh_pub
  [4]  wrapped_key_len    → RSA-OAEP wrapped AES session key
  [N]  wrapped_key
  [4]  nonce_len          → AES-GCM nonce (12 bytes)
  [N]  nonce
  [4]  ciphertext_len     → AES-GCM encrypted payload
  [N]  ciphertext
  [4]  tag_len            → AES-GCM authentication tag (16 bytes)
  [N]  tag
  [4]  signature_len      → RSA-PSS signature over (nonce + ciphertext + tag)
  [N]  signature
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives.asymmetric.rsa import (
    RSAPrivateKey, RSAPublicKey, generate_private_key,
)
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey, X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric import padding as rsa_padding
from cryptography.hazmat.backends import default_backend

# ─────────────────────────────────────────────────────────────────────────────
# Constants (overridable from zentraa_config)
# ─────────────────────────────────────────────────────────────────────────────
RSA_KEY_BITS:    int = 2048
AES_KEY_BYTES:   int = 32      # AES-256
AES_NONCE_BYTES: int = 12      # GCM nonce
AES_TAG_BYTES:   int = 16      # GCM authentication tag


def _oaep() -> asym_padding.OAEP:
    return asym_padding.OAEP(
        mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None,
    )


def _pss() -> rsa_padding.PSS:
    return rsa_padding.PSS(
        mgf=rsa_padding.MGF1(hashes.SHA256()),
        salt_length=rsa_padding.PSS.MAX_LENGTH,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RSA key management
# ─────────────────────────────────────────────────────────────────────────────

def generate_rsa_keypair(bits: int = RSA_KEY_BITS) -> Tuple[RSAPrivateKey, RSAPublicKey]:
    """Generate a fresh RSA keypair."""
    private_key = generate_private_key(
        public_exponent=65537,
        key_size=bits,
        backend=default_backend(),
    )
    return private_key, private_key.public_key()


def serialize_public_key(pub: RSAPublicKey) -> bytes:
    return pub.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def serialize_private_key(priv: RSAPrivateKey, password: Optional[bytes] = None) -> bytes:
    enc = (
        serialization.BestAvailableEncryption(password)
        if password
        else serialization.NoEncryption()
    )
    return priv.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, enc)


def load_public_key(pem: bytes) -> RSAPublicKey:
    return serialization.load_pem_public_key(pem, backend=default_backend())  # type: ignore[return-value]


def load_private_key(pem: bytes, password: Optional[bytes] = None) -> RSAPrivateKey:
    return serialization.load_pem_private_key(pem, password=password, backend=default_backend())  # type: ignore[return-value]


def save_keypair(
    private_key: RSAPrivateKey,
    public_key: RSAPublicKey,
    keys_dir: str,
    private_name: str = "server_private.pem",
    public_name: str = "server_public.pem",
    password: Optional[bytes] = None,
) -> None:
    d = Path(keys_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / private_name).write_bytes(serialize_private_key(private_key, password))
    (d / public_name).write_bytes(serialize_public_key(public_key))
    # Restrict private key permissions on POSIX
    try:
        (d / private_name).chmod(0o600)
    except AttributeError:
        pass


def load_keypair(
    keys_dir: str,
    private_name: str = "server_private.pem",
    public_name: str = "server_public.pem",
    password: Optional[bytes] = None,
) -> Tuple[RSAPrivateKey, RSAPublicKey]:
    d = Path(keys_dir)
    priv = load_private_key((d / private_name).read_bytes(), password)
    pub  = load_public_key((d / public_name).read_bytes())
    return priv, pub


# ─────────────────────────────────────────────────────────────────────────────
# Curve25519 ECDH ephemeral key exchange
# ─────────────────────────────────────────────────────────────────────────────

def generate_x25519_keypair() -> Tuple[X25519PrivateKey, X25519PublicKey]:
    priv = X25519PrivateKey.generate()
    return priv, priv.public_key()


def derive_shared_secret(
    local_private: X25519PrivateKey,
    peer_public_bytes: bytes,
    salt: Optional[bytes] = None,
) -> bytes:
    """ECDH + HKDF-SHA256 → 32-byte AES session key."""
    peer_pub = X25519PublicKey.from_public_bytes(peer_public_bytes)
    raw_secret = local_private.exchange(peer_pub)
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=AES_KEY_BYTES,
        salt=salt,
        info=b"ZENTRAA-session-key-v1",
        backend=default_backend(),
    )
    return hkdf.derive(raw_secret)


# ─────────────────────────────────────────────────────────────────────────────
# Wire packet helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pack_field(data: bytes) -> bytes:
    """Prefix *data* with a little-endian uint32 length."""
    return struct.pack("<I", len(data)) + data


def _unpack_field(buf: bytes, offset: int) -> Tuple[bytes, int]:
    """Read a length-prefixed field from *buf* at *offset*."""
    length = struct.unpack_from("<I", buf, offset)[0]
    offset += 4
    return buf[offset: offset + length], offset + length


# ─────────────────────────────────────────────────────────────────────────────
# Encrypted packet dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EncryptedPacket:
    ecdh_pub:    bytes   # sender's ephemeral Curve25519 public key (32 bytes)
    wrapped_key: bytes   # RSA-OAEP wrapped AES key
    nonce:       bytes   # AES-GCM nonce
    ciphertext:  bytes   # encrypted payload
    tag:         bytes   # AES-GCM tag
    signature:   bytes   # RSA-PSS over (nonce + ciphertext + tag)

    def to_bytes(self) -> bytes:
        body = (
            _pack_field(self.ecdh_pub)
            + _pack_field(self.wrapped_key)
            + _pack_field(self.nonce)
            + _pack_field(self.ciphertext)
            + _pack_field(self.tag)
            + _pack_field(self.signature)
        )
        return _pack_field(body)  # outer length-prefix

    @classmethod
    def from_bytes(cls, raw: bytes) -> "EncryptedPacket":
        body, _ = _unpack_field(raw, 0)
        offset = 0
        ecdh_pub,    offset = _unpack_field(body, offset)
        wrapped_key, offset = _unpack_field(body, offset)
        nonce,       offset = _unpack_field(body, offset)
        ciphertext,  offset = _unpack_field(body, offset)
        tag,         offset = _unpack_field(body, offset)
        signature,   offset = _unpack_field(body, offset)
        return cls(ecdh_pub, wrapped_key, nonce, ciphertext, tag, signature)


# ─────────────────────────────────────────────────────────────────────────────
# Core encrypt / decrypt
# ─────────────────────────────────────────────────────────────────────────────

def encrypt(
    plaintext: bytes,
    recipient_rsa_pub: RSAPublicKey,
    sender_rsa_priv: RSAPrivateKey,
) -> bytes:
    """
    Encrypt *plaintext* for *recipient_rsa_pub*, signed by *sender_rsa_priv*.

    Returns wire-format bytes (EncryptedPacket.to_bytes()).
    """
    # 1. Ephemeral Curve25519 keypair
    eph_priv, eph_pub = generate_x25519_keypair()
    eph_pub_bytes = eph_pub.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )

    # 2. AES-256 session key — random (ECDH-derived keys are sent wrapped)
    aes_key   = os.urandom(AES_KEY_BYTES)
    nonce     = os.urandom(AES_NONCE_BYTES)

    # 3. AES-256-GCM encrypt
    aesgcm    = AESGCM(aes_key)
    ct_with_tag = aesgcm.encrypt(nonce, plaintext, None)  # tag appended by library
    ciphertext  = ct_with_tag[:-AES_TAG_BYTES]
    tag         = ct_with_tag[-AES_TAG_BYTES:]

    # 4. RSA-OAEP wrap the AES key
    wrapped_key = recipient_rsa_pub.encrypt(aes_key, _oaep())

    # 5. RSA-PSS sign (nonce ‖ ciphertext ‖ tag)
    signed_payload = nonce + ciphertext + tag
    signature = sender_rsa_priv.sign(signed_payload, _pss(), hashes.SHA256())

    packet = EncryptedPacket(
        ecdh_pub=eph_pub_bytes,
        wrapped_key=wrapped_key,
        nonce=nonce,
        ciphertext=ciphertext,
        tag=tag,
        signature=signature,
    )
    return packet.to_bytes()


def decrypt(
    packet_bytes: bytes,
    recipient_rsa_priv: RSAPrivateKey,
    sender_rsa_pub: RSAPublicKey,
) -> bytes:
    """
    Decrypt *packet_bytes* using *recipient_rsa_priv*, verifying *sender_rsa_pub*.

    Raises ValueError on signature/authentication failure.
    """
    packet = EncryptedPacket.from_bytes(packet_bytes)

    # 1. Verify RSA-PSS signature first (fail fast)
    signed_payload = packet.nonce + packet.ciphertext + packet.tag
    try:
        sender_rsa_pub.verify(packet.signature, signed_payload, _pss(), hashes.SHA256())
    except Exception as exc:
        raise ValueError(f"Signature verification failed: {exc}") from exc

    # 2. Unwrap AES key via RSA-OAEP
    try:
        aes_key = recipient_rsa_priv.decrypt(packet.wrapped_key, _oaep())
    except Exception as exc:
        raise ValueError(f"Key unwrap failed: {exc}") from exc

    # 3. AES-256-GCM decrypt + authenticate
    aesgcm = AESGCM(aes_key)
    ct_with_tag = packet.ciphertext + packet.tag
    try:
        plaintext = aesgcm.decrypt(packet.nonce, ct_with_tag, None)
    except Exception as exc:
        raise ValueError(f"AES-GCM decryption failed: {exc}") from exc

    return plaintext


# ─────────────────────────────────────────────────────────────────────────────
# Simple symmetric-only path (server→client broadcasts using ECDH shared key)
# ─────────────────────────────────────────────────────────────────────────────

def sym_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """AES-256-GCM encrypt with a pre-shared *key*. Returns nonce+ct+tag."""
    nonce = os.urandom(AES_NONCE_BYTES)
    ct_with_tag = AESGCM(key).encrypt(nonce, plaintext, None)
    return nonce + ct_with_tag


def sym_decrypt(data: bytes, key: bytes) -> bytes:
    """Inverse of sym_encrypt."""
    nonce = data[:AES_NONCE_BYTES]
    ct_with_tag = data[AES_NONCE_BYTES:]
    return AESGCM(key).decrypt(nonce, ct_with_tag, None)


# ─────────────────────────────────────────────────────────────────────────────
# Framing helpers for socket I/O
# ─────────────────────────────────────────────────────────────────────────────

def frame(data: bytes) -> bytes:
    """Prefix raw bytes with 4-byte big-endian length for socket send."""
    return struct.pack(">I", len(data)) + data


def recv_frame(sock) -> bytes:
    """
    Block-read exactly one length-prefixed frame from *sock*.
    Returns the payload bytes (without the length prefix).
    Raises ConnectionError on EOF or any socket/OS error (including
    WinError 10038 on Windows when the socket is shut down mid-recv).
    """
    try:
        raw_len = _recvall(sock, 4)
        if not raw_len:
            raise ConnectionError("Connection closed")
        length = struct.unpack(">I", raw_len)[0]
        data = _recvall(sock, length)
        if not data:
            raise ConnectionError("Connection closed mid-frame")
        return data
    except ConnectionError:
        raise
    except OSError as exc:
        # Normalise platform socket errors (e.g. WinError 10038, EBADF, ECONNRESET)
        # into ConnectionError so all callers can use a single except clause.
        raise ConnectionError(f"Socket error: {exc}") from exc


def _recvall(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return b""
        buf += chunk
    return buf