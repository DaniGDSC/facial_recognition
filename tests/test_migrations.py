import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ.setdefault("PG_PASSWORD", "test_password")

try:
    from migrations import SQLiteMigrator, SQLITE_MIGRATIONS
except ImportError:
    SQLiteMigrator = None
    SQLITE_MIGRATIONS = []


@unittest.skipIf(SQLiteMigrator is None, "psycopg2 not installed")
class TestSQLiteMigrator(unittest.TestCase):

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.migrator = SQLiteMigrator(db_path=self.db_path)

    def tearDown(self):
        self.migrator.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_starts_at_version_zero(self):
        self.assertEqual(self.migrator.current_version(), 0)

    def test_migrate_applies_all(self):
        count = self.migrator.migrate()
        self.assertEqual(count, len(SQLITE_MIGRATIONS))
        self.assertEqual(self.migrator.current_version(), len(SQLITE_MIGRATIONS))

    def test_migrate_idempotent(self):
        self.migrator.migrate()
        count = self.migrator.migrate()
        self.assertEqual(count, 0)

    def test_tables_exist_after_migration(self):
        self.migrator.migrate()
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        self.assertIn('user_profiles', tables)
        self.assertIn('biometric_audit_log', tables)
        self.assertIn('operator_accounts', tables)
        self.assertIn('sessions', tables)
        self.assertIn('schema_version', tables)

    def test_version_table_tracks_all_migrations(self):
        self.migrator.migrate()
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM schema_version")
        count = cursor.fetchone()[0]
        conn.close()
        self.assertEqual(count, len(SQLITE_MIGRATIONS))


@unittest.skipIf(SQLiteMigrator is None, "psycopg2 not installed")
class TestMigrationDefinitions(unittest.TestCase):

    def test_sqlite_versions_are_sequential(self):
        versions = [m['version'] for m in SQLITE_MIGRATIONS]
        self.assertEqual(versions, list(range(1, len(SQLITE_MIGRATIONS) + 1)))

    def test_all_migrations_have_required_fields(self):
        for m in SQLITE_MIGRATIONS:
            self.assertIn('version', m)
            self.assertIn('description', m)
            self.assertIn('sql', m)
            self.assertIsInstance(m['version'], int)
            self.assertIsInstance(m['description'], str)
            self.assertIsInstance(m['sql'], str)


if __name__ == "__main__":
    unittest.main()
