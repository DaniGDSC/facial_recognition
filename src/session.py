"""Session management with JWT-based token access post-authentication.

Provides:
- Token generation after successful facial authentication
- Token verification with expiry and revocation
- Session tracking with configurable timeouts
"""

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Session defaults
_DEFAULT_SESSION_TTL = 900  # 15 minutes
_DEFAULT_MAX_SESSIONS = 100

_SESSION_SECRET_ENV = "SESSION_SECRET_KEY"


def _get_session_secret() -> bytes:
    """Load session signing secret from environment."""
    raw = os.environ.get(_SESSION_SECRET_ENV)
    if not raw:
        env = os.environ.get("FLASK_ENV", os.environ.get("APP_ENV", ""))
        if env.lower() == "production":
            raise ValueError(
                f"{_SESSION_SECRET_ENV} is required in production. "
                "Generate with: python -c \"import os; print(os.urandom(32).hex())\""
            )
        logger.warning(
            "SESSION_SECRET_KEY not set — generating ephemeral key (sessions will not survive restart)"
        )
        return os.urandom(32)
    return bytes.fromhex(raw)


def _compute_signature(payload_b64: str, secret: bytes) -> str:
    """HMAC-SHA256 signature over a base64 payload."""
    return hmac.new(secret, payload_b64.encode(), hashlib.sha256).hexdigest()


class SessionManager:
    """Manages authenticated sessions with token-based access."""

    def __init__(
        self,
        session_ttl: int = _DEFAULT_SESSION_TTL,
        max_sessions: int = _DEFAULT_MAX_SESSIONS,
    ):
        self.session_ttl = session_ttl
        self.max_sessions = max_sessions
        self._secret = _get_session_secret()
        # session_id -> session data
        self._sessions: dict[str, dict] = {}
        # set of revoked token hashes
        self._revoked: set[str] = set()

    def create_session(
        self,
        user_id: str,
        user_code: str,
        full_name: str,
        security_level: str = "STANDARD",
    ) -> str:
        """Create a new session and return a signed token."""
        self._evict_expired()

        if len(self._sessions) >= self.max_sessions:
            self._evict_oldest()

        now = int(time.time())
        session_id = hashlib.sha256(os.urandom(32)).hexdigest()[:24]

        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'user_code': user_code,
            'full_name': full_name,
            'security_level': security_level,
            'created_at': now,
            'expires_at': now + self.session_ttl,
            'last_activity': now,
        }
        self._sessions[session_id] = session_data

        token = self._encode_token(session_id, now)
        logger.info(
            "Session created: session_id=%s, user=%s, expires_in=%ds",
            session_id, user_code, self.session_ttl,
        )
        return token

    def verify_token(self, token: str) -> Optional[dict]:
        """Verify a token and return session data, or None if invalid/expired."""
        # Check revocation
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        if token_hash in self._revoked:
            logger.warning("Token has been revoked")
            return None

        # Decode
        session_id = self._decode_token(token)
        if session_id is None:
            return None

        session = self._sessions.get(session_id)
        if session is None:
            logger.warning("Session not found: %s", session_id)
            return None

        now = int(time.time())
        if now > session['expires_at']:
            logger.info("Session expired: %s", session_id)
            del self._sessions[session_id]
            return None

        # Sliding window: refresh expiry on activity
        session['last_activity'] = now
        session['expires_at'] = now + self.session_ttl

        return {
            'user_id': session['user_id'],
            'user_code': session['user_code'],
            'full_name': session['full_name'],
            'security_level': session['security_level'],
            'session_id': session_id,
            'expires_in': session['expires_at'] - now,
        }

    def revoke_token(self, token: str) -> bool:
        """Revoke a token (logout)."""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        self._revoked.add(token_hash)

        session_id = self._decode_token(token)
        if session_id and session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Session revoked: %s", session_id)
            return True
        return False

    def revoke_all_for_user(self, user_id: str) -> int:
        """Revoke all sessions for a user. Returns count revoked."""
        to_remove = [
            sid for sid, s in self._sessions.items() if s['user_id'] == user_id
        ]
        for sid in to_remove:
            del self._sessions[sid]
        logger.info("Revoked %d sessions for user %s", len(to_remove), user_id)
        return len(to_remove)

    @property
    def active_session_count(self) -> int:
        self._evict_expired()
        return len(self._sessions)

    def _encode_token(self, session_id: str, issued_at: int) -> str:
        """Encode session_id into a signed token (header.payload.signature)."""
        import base64

        header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "SES"}).encode()).decode().rstrip("=")
        payload_data = {"sid": session_id, "iat": issued_at}
        payload = base64.urlsafe_b64encode(json.dumps(payload_data).encode()).decode().rstrip("=")
        signature = _compute_signature(f"{header}.{payload}", self._secret)

        return f"{header}.{payload}.{signature}"

    def _decode_token(self, token: str) -> Optional[str]:
        """Decode and verify a signed token. Returns session_id or None."""
        import base64

        parts = token.split(".")
        if len(parts) != 3:
            logger.warning("Malformed token (expected 3 parts)")
            return None

        header_b64, payload_b64, signature = parts

        # Verify signature
        expected_sig = _compute_signature(f"{header_b64}.{payload_b64}", self._secret)
        if not hmac.compare_digest(signature, expected_sig):
            logger.warning("Invalid token signature")
            return None

        try:
            # Add padding
            padding = 4 - len(payload_b64) % 4
            payload_json = base64.urlsafe_b64decode(payload_b64 + "=" * padding)
            payload_data = json.loads(payload_json)
            return payload_data.get("sid")
        except Exception:
            logger.warning("Failed to decode token payload")
            return None

    def _evict_expired(self):
        now = int(time.time())
        expired = [sid for sid, s in self._sessions.items() if now > s['expires_at']]
        for sid in expired:
            del self._sessions[sid]

    def _evict_oldest(self):
        if not self._sessions:
            return
        oldest = min(self._sessions, key=lambda s: self._sessions[s]['created_at'])
        del self._sessions[oldest]
