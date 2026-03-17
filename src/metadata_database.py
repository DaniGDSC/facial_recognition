import logging
import sqlite3
import os
from config import DATABASE_FILE

logger = logging.getLogger(__name__)


class MetadataDBSetup:

    def __init__(self):
        self.conn = sqlite3.connect(DATABASE_FILE)
        self.cursor = self.conn.cursor()
        logger.info("Connected to SQLite database: %s", DATABASE_FILE)
        self.conn.row_factory = sqlite3.Row

    def create_user_profiles_table(self):
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    user_code TEXT UNIQUE NOT NULL,
                    first_name TEXT,
                    last_name TEXT,
                    account_status TEXT NOT NULL,
                    enrollment_date INTEGER,
                    biometric_consent_date INTEGER
                );
            """)
            logger.info("Table 'user_profiles' ready")
        except sqlite3.Error as e:
            logger.error("Error creating user_profiles table: %s", e)

    def create_audit_log_table(self):
        """Create biometric audit log table."""
        try:
            self.cursor.execute("""
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
            """)
            logger.info("Table 'biometric_audit_log' ready")
        except sqlite3.Error as e:
            logger.error("Error creating biometric_audit_log table: %s", e)

    def close(self):
        """Close database connection and save changes."""
        self.conn.commit()
        self.conn.close()
        logger.info("Closed connection to %s", DATABASE_FILE)


def main():
    """Run metadata database setup."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    logger.info("--- METADATA DATABASE SETUP (SQLITE) ---")
    setup = None
    try:
        setup = MetadataDBSetup()
        setup.create_user_profiles_table()
        setup.create_audit_log_table()
    except RuntimeError:
        logger.error("SETUP FAILED")
    finally:
        if setup:
            setup.close()


if __name__ == "__main__":
    main()
