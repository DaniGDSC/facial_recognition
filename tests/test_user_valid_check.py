import sys
import os
import sqlite3
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Patch env vars before importing src modules
os.environ.setdefault("PG_PASSWORD", "test_password")

try:
    import config
    _has_config = True
except ImportError:
    _has_config = False


@unittest.skipIf(not _has_config, "torch not installed (required by config)")
class TestUserCodeCheckSystem(unittest.TestCase):

    def setUp(self):
        """Create a temporary SQLite database with user_profiles table."""
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE user_profiles (
                user_id TEXT PRIMARY KEY,
                user_code TEXT UNIQUE NOT NULL,
                first_name TEXT,
                last_name TEXT,
                account_status TEXT NOT NULL,
                enrollment_date INTEGER,
                biometric_consent_date INTEGER
            )
        """)
        conn.execute("""
            INSERT INTO user_profiles
            (user_id, user_code, first_name, last_name, account_status, enrollment_date, biometric_consent_date)
            VALUES ('uuid-1', '1234567890', 'John', 'Doe', 'ACTIVE', 1000000, 1000000)
        """)
        conn.commit()
        conn.close()

        # Monkey-patch DATABASE_FILE so UserCodeCheckSystem uses our temp DB
        self._original_db = config.DATABASE_FILE
        config.DATABASE_FILE = self.db_path

        from user_valid_check import UserCodeCheckSystem
        self.system = UserCodeCheckSystem()

    def tearDown(self):
        self.system.close()
        config.DATABASE_FILE = self._original_db
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_existing_code_returns_failure(self):
        result = self.system.check_user_code_uniqueness("1234567890")
        self.assertIn("THẤT BẠI", result)
        self.assertIn("already exists", result)

    def test_new_code_returns_success(self):
        result = self.system.check_user_code_uniqueness("9999999999")
        self.assertIn("THÀNH CÔNG", result)
        self.assertIn("is available", result)

    def test_get_user_info_found(self):
        info = self.system.get_user_info("1234567890")
        self.assertTrue(info['found'])
        self.assertEqual(info['first_name'], 'John')
        self.assertEqual(info['last_name'], 'Doe')
        self.assertEqual(info['full_name'], 'John Doe')
        self.assertEqual(info['account_status'], 'ACTIVE')

    def test_get_user_info_not_found(self):
        info = self.system.get_user_info("0000000000")
        self.assertFalse(info['found'])

    def test_uniqueness_check_sql_injection_safe(self):
        """Ensure parameterized queries prevent SQL injection."""
        malicious = "'; DROP TABLE user_profiles; --"
        result = self.system.check_user_code_uniqueness(malicious)
        self.assertIn("THÀNH CÔNG", result)
        # Table should still exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM user_profiles")
        count = cursor.fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)


@unittest.skipIf(not _has_config, "torch not installed (required by config)")
class TestInputValidation(unittest.TestCase):
    """Test the input validation functions from enrollment_client."""

    def test_valid_user_codes(self):
        from enrollment_client import validate_user_code
        self.assertTrue(validate_user_code("1234567890"))
        self.assertTrue(validate_user_code("0000000000"))

    def test_invalid_user_codes(self):
        from enrollment_client import validate_user_code
        self.assertFalse(validate_user_code("12345"))
        self.assertFalse(validate_user_code("12345678901"))
        self.assertFalse(validate_user_code("abcdefghij"))
        self.assertFalse(validate_user_code(""))
        self.assertFalse(validate_user_code("123-456-78"))

    def test_valid_names(self):
        from enrollment_client import validate_name
        self.assertTrue(validate_name("John"))
        self.assertTrue(validate_name("Mary Jane"))
        self.assertTrue(validate_name("A"))

    def test_invalid_names(self):
        from enrollment_client import validate_name
        self.assertFalse(validate_name(""))
        self.assertFalse(validate_name("   "))
        self.assertFalse(validate_name("John123"))
        self.assertFalse(validate_name("O'Brien"))


if __name__ == "__main__":
    unittest.main()
