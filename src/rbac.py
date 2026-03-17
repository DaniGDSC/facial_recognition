"""Role-Based Access Control for facial recognition operations.

Roles:
- OPERATOR: Can enroll users and manage the system
- USER: Can authenticate / be recognized
- ADMIN: Full access (operator + user + system config)

Operator authentication is required before enrollment operations.
"""

import hashlib
import logging
import os
import sqlite3

logger = logging.getLogger(__name__)

# Role definitions
ROLE_USER = "USER"
ROLE_OPERATOR = "OPERATOR"
ROLE_ADMIN = "ADMIN"

# Permission -> required roles
PERMISSIONS = {
    "enroll": {ROLE_OPERATOR, ROLE_ADMIN},
    "recognize": {ROLE_USER, ROLE_OPERATOR, ROLE_ADMIN},
    "authenticate": {ROLE_USER, ROLE_OPERATOR, ROLE_ADMIN},
    "view_statistics": {ROLE_OPERATOR, ROLE_ADMIN},
    "manage_users": {ROLE_ADMIN},
}


def _hash_pin(pin: str, salt: bytes) -> str:
    """Hash a PIN with PBKDF2-HMAC-SHA256."""
    dk = hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, iterations=100_000)
    return dk.hex()


class RBACManager:
    """Manages operator accounts and permission checks."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database", "metadata.db")
            db_path = os.environ.get("DATABASE_FILE", default)
        self.db_path = db_path
        self._ensure_table()
        self._current_role: str | None = None
        self._current_operator: str | None = None

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self):
        """Create operator_accounts table if it doesn't exist."""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS operator_accounts (
                    operator_id TEXT PRIMARY KEY,
                    pin_hash TEXT NOT NULL,
                    pin_salt TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'OPERATOR',
                    created_at INTEGER NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def register_operator(self, operator_id: str, pin: str, role: str = ROLE_OPERATOR) -> bool:
        """Register a new operator account."""
        if role not in (ROLE_OPERATOR, ROLE_ADMIN):
            logger.error("Invalid role for operator registration: %s", role)
            return False

        salt = os.urandom(16)
        pin_hash = _hash_pin(pin, salt)

        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO operator_accounts (operator_id, pin_hash, pin_salt, role, created_at) VALUES (?, ?, ?, ?, ?)",
                (operator_id, pin_hash, salt.hex(), role, int(__import__("time").time())),
            )
            conn.commit()
            logger.info("Registered operator: %s (role=%s)", operator_id, role)
            return True
        except sqlite3.IntegrityError:
            logger.warning("Operator %s already exists", operator_id)
            return False
        finally:
            conn.close()

    def authenticate_operator(self, operator_id: str, pin: str) -> bool:
        """Authenticate an operator by ID and PIN."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT pin_hash, pin_salt, role FROM operator_accounts WHERE operator_id = ?",
                (operator_id,),
            ).fetchone()

            if not row:
                logger.warning("Operator authentication failed: %s not found", operator_id)
                return False

            salt = bytes.fromhex(row["pin_salt"])
            expected_hash = row["pin_hash"]
            computed_hash = _hash_pin(pin, salt)

            if computed_hash == expected_hash:
                self._current_role = row["role"]
                self._current_operator = operator_id
                logger.info("Operator %s authenticated (role=%s)", operator_id, self._current_role)
                return True
            else:
                logger.warning("Operator authentication failed: wrong PIN for %s", operator_id)
                return False
        finally:
            conn.close()

    def check_permission(self, action: str) -> bool:
        """Check if the current authenticated role has permission for an action."""
        if self._current_role is None:
            logger.warning("Permission check failed: no operator authenticated")
            return False

        required_roles = PERMISSIONS.get(action)
        if required_roles is None:
            logger.warning("Unknown action: %s", action)
            return False

        allowed = self._current_role in required_roles
        if not allowed:
            logger.warning(
                "Permission denied: %s requires %s, current role is %s",
                action, required_roles, self._current_role,
            )
        return allowed

    def require_permission(self, action: str) -> None:
        """Raise PermissionError if the current role lacks the given permission."""
        if not self.check_permission(action):
            raise PermissionError(
                f"Action '{action}' requires one of {PERMISSIONS.get(action, set())}. "
                f"Current role: {self._current_role or 'unauthenticated'}"
            )

    @property
    def current_role(self) -> str | None:
        return self._current_role

    @property
    def current_operator(self) -> str | None:
        return self._current_operator

    def logout(self):
        """Clear the current operator session."""
        logger.info("Operator %s logged out", self._current_operator)
        self._current_role = None
        self._current_operator = None

    def has_operators(self) -> bool:
        """Check whether any operator accounts exist (for first-run setup)."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM operator_accounts").fetchone()
            return row["cnt"] > 0
        finally:
            conn.close()
