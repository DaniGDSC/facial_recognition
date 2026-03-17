"""Database migration system for SQLite metadata and PostgreSQL vector databases.

Replaces destructive setup scripts with versioned, forward-only migrations.
Each migration has a unique version number and runs exactly once.
"""

import logging
import sqlite3
import psycopg2
from psycopg2 import sql as pgsql

from config import DATABASE_FILE
from database_config import (
    PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD,
    COLLECTION_NAME, VECTOR_DIMENSION,
)

logger = logging.getLogger(__name__)


# ---------- Migration definitions ----------

SQLITE_MIGRATIONS = [
    {
        'version': 1,
        'description': 'Create user_profiles table',
        'sql': """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                user_code TEXT UNIQUE NOT NULL,
                first_name TEXT,
                last_name TEXT,
                account_status TEXT NOT NULL,
                enrollment_date INTEGER,
                biometric_consent_date INTEGER
            );
        """,
    },
    {
        'version': 2,
        'description': 'Create biometric_audit_log table',
        'sql': """
            CREATE TABLE IF NOT EXISTS biometric_audit_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                timestamp INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                similarity_score REAL,
                spoof_detected INTEGER,
                device_ip TEXT,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
            );
        """,
    },
    {
        'version': 3,
        'description': 'Create operator_accounts table',
        'sql': """
            CREATE TABLE IF NOT EXISTS operator_accounts (
                operator_id TEXT PRIMARY KEY,
                pin_hash TEXT NOT NULL,
                pin_salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'OPERATOR',
                created_at INTEGER NOT NULL
            );
        """,
    },
    {
        'version': 4,
        'description': 'Create sessions table',
        'sql': """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
            );
        """,
    },
]


PG_MIGRATIONS = [
    {
        'version': 1,
        'description': 'Enable pgvector extension',
        'sql': "CREATE EXTENSION IF NOT EXISTS vector;",
    },
    {
        'version': 2,
        'description': 'Create face_templates table',
        'sql': f"""
            CREATE TABLE IF NOT EXISTS {COLLECTION_NAME} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id VARCHAR(100) UNIQUE NOT NULL,
                enrollment_date BIGINT NOT NULL,
                face_vector vector({VECTOR_DIMENSION})
            );
        """,
    },
    {
        'version': 3,
        'description': 'Create IVFFlat index on face_vector',
        'sql': f"""
            CREATE INDEX IF NOT EXISTS face_vector_ivfflat_idx
            ON {COLLECTION_NAME} USING ivfflat (face_vector) WITH (lists = 100);
        """,
    },
    {
        'version': 4,
        'description': 'Add hmac_tag column for embedding integrity',
        'sql': f"""
            ALTER TABLE {COLLECTION_NAME}
            ADD COLUMN IF NOT EXISTS hmac_tag TEXT;
        """,
    },
]


# ---------- Migration runner ----------

class SQLiteMigrator:
    """Runs versioned migrations on the SQLite metadata database."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DATABASE_FILE
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_version_table()

    def _ensure_version_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at INTEGER NOT NULL
            )
        """)
        self.conn.commit()

    def current_version(self) -> int:
        row = self.conn.execute("SELECT MAX(version) as v FROM schema_version").fetchone()
        return row['v'] or 0

    def migrate(self) -> int:
        """Apply all pending migrations. Returns count applied."""
        current = self.current_version()
        applied = 0

        for migration in SQLITE_MIGRATIONS:
            if migration['version'] <= current:
                continue

            logger.info(
                "SQLite migration v%d: %s", migration['version'], migration['description']
            )
            try:
                self.conn.execute(migration['sql'])
                self.conn.execute(
                    "INSERT INTO schema_version (version, description, applied_at) VALUES (?, ?, ?)",
                    (migration['version'], migration['description'], int(__import__('time').time())),
                )
                self.conn.commit()
                applied += 1
            except sqlite3.Error as e:
                logger.error("SQLite migration v%d failed: %s", migration['version'], e)
                self.conn.rollback()
                raise

        if applied:
            logger.info("SQLite: applied %d migration(s), now at v%d", applied, self.current_version())
        else:
            logger.info("SQLite: already at latest v%d", current)

        return applied

    def close(self):
        self.conn.close()


class PostgreSQLMigrator:
    """Runs versioned migrations on the PostgreSQL vector database."""

    def __init__(self):
        self.conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, database=PG_DATABASE,
            user=PG_USER, password=PG_PASSWORD,
        )
        self.conn.autocommit = True
        self._ensure_version_table()

    def _ensure_version_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at BIGINT NOT NULL
                )
            """)

    def current_version(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT MAX(version) FROM schema_version")
            row = cur.fetchone()
            return row[0] or 0

    def migrate(self) -> int:
        """Apply all pending migrations. Returns count applied."""
        current = self.current_version()
        applied = 0

        for migration in PG_MIGRATIONS:
            if migration['version'] <= current:
                continue

            logger.info(
                "PostgreSQL migration v%d: %s", migration['version'], migration['description']
            )
            try:
                with self.conn.cursor() as cur:
                    cur.execute(migration['sql'])
                    cur.execute(
                        "INSERT INTO schema_version (version, description, applied_at) VALUES (%s, %s, %s)",
                        (migration['version'], migration['description'], int(__import__('time').time())),
                    )
                applied += 1
            except psycopg2.Error as e:
                logger.error("PostgreSQL migration v%d failed: %s", migration['version'], e)
                raise

        if applied:
            logger.info("PostgreSQL: applied %d migration(s), now at v%d", applied, self.current_version())
        else:
            logger.info("PostgreSQL: already at latest v%d", current)

        return applied

    def close(self):
        self.conn.close()


def run_all_migrations():
    """Run all pending migrations on both databases."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=== Running database migrations ===")

    # SQLite
    sqlite_migrator = SQLiteMigrator()
    try:
        sqlite_migrator.migrate()
    finally:
        sqlite_migrator.close()

    # PostgreSQL
    pg_migrator = PostgreSQLMigrator()
    try:
        pg_migrator.migrate()
    finally:
        pg_migrator.close()

    logger.info("=== All migrations complete ===")


if __name__ == "__main__":
    run_all_migrations()
