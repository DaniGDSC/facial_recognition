"""Cryptographic utilities for biometric data protection.

Provides:
- AES-256-GCM encryption for biometric images at rest
- HMAC-SHA256 integrity verification for stored embedding vectors
"""

import hashlib
import hmac
import logging
import os
import struct

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

# --- Key management ---

_ENCRYPTION_KEY_ENV = "BIOMETRIC_ENCRYPTION_KEY"
_HMAC_KEY_ENV = "EMBEDDING_HMAC_KEY"


def _get_key(env_var: str, key_len: int = 32) -> bytes:
    """Load a key from an environment variable (hex-encoded) or raise."""
    raw = os.environ.get(env_var)
    if not raw:
        raise ValueError(
            f"{env_var} environment variable is required. "
            f"Generate one with: python -c \"import os; print(os.urandom({key_len}).hex())\""
        )
    key = bytes.fromhex(raw)
    if len(key) != key_len:
        raise ValueError(f"{env_var} must be exactly {key_len} bytes ({key_len * 2} hex chars)")
    return key


def get_encryption_key() -> bytes:
    """Return the 256-bit AES encryption key."""
    return _get_key(_ENCRYPTION_KEY_ENV, 32)


def get_hmac_key() -> bytes:
    """Return the 256-bit HMAC signing key."""
    return _get_key(_HMAC_KEY_ENV, 32)


# --- Image encryption (AES-256-GCM) ---

_NONCE_LEN = 12  # 96-bit nonce for AES-GCM


def encrypt_image(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt image bytes using AES-256-GCM.

    Returns: nonce (12 bytes) || ciphertext+tag
    """
    nonce = os.urandom(_NONCE_LEN)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return nonce + ciphertext


def decrypt_image(data: bytes, key: bytes) -> bytes:
    """Decrypt AES-256-GCM encrypted image bytes.

    Expects: nonce (12 bytes) || ciphertext+tag
    """
    if len(data) < _NONCE_LEN + 16:  # nonce + minimum tag
        raise ValueError("Encrypted data too short")
    nonce = data[:_NONCE_LEN]
    ciphertext = data[_NONCE_LEN:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)


# --- Embedding HMAC integrity ---

def compute_embedding_hmac(embedding_vector: list[float], key: bytes) -> str:
    """Compute HMAC-SHA256 over an embedding vector.

    The vector is serialized as packed IEEE 754 doubles for deterministic hashing.
    Returns hex-encoded HMAC digest.
    """
    packed = struct.pack(f">{len(embedding_vector)}d", *embedding_vector)
    return hmac.new(key, packed, hashlib.sha256).hexdigest()


def verify_embedding_hmac(embedding_vector: list[float], expected_hmac: str, key: bytes) -> bool:
    """Verify HMAC-SHA256 of an embedding vector (constant-time comparison)."""
    computed = compute_embedding_hmac(embedding_vector, key)
    return hmac.compare_digest(computed, expected_hmac)


def get_optional_key(env_var: str) -> bytes | None:
    """Load an optional hex-encoded key from env, return None if not set."""
    raw = os.environ.get(env_var)
    if not raw:
        return None
    return bytes.fromhex(raw)
