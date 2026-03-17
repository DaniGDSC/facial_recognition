import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ.setdefault("PG_PASSWORD", "test_password")

from session import SessionManager


class TestSessionManager(unittest.TestCase):

    def setUp(self):
        os.environ["SESSION_SECRET_KEY"] = os.urandom(32).hex()
        self.mgr = SessionManager(session_ttl=5, max_sessions=10)

    def test_create_and_verify(self):
        token = self.mgr.create_session("u1", "1234567890", "John Doe")
        session = self.mgr.verify_token(token)
        self.assertIsNotNone(session)
        self.assertEqual(session['user_id'], "u1")
        self.assertEqual(session['user_code'], "1234567890")
        self.assertEqual(session['full_name'], "John Doe")

    def test_invalid_token_returns_none(self):
        self.assertIsNone(self.mgr.verify_token("invalid.token.here"))

    def test_tampered_token_returns_none(self):
        token = self.mgr.create_session("u1", "1234567890", "John")
        tampered = token[:-1] + ("a" if token[-1] != "a" else "b")
        self.assertIsNone(self.mgr.verify_token(tampered))

    def test_revoke_token(self):
        token = self.mgr.create_session("u1", "1234567890", "John")
        self.assertTrue(self.mgr.revoke_token(token))
        self.assertIsNone(self.mgr.verify_token(token))

    def test_expired_session_returns_none(self):
        mgr = SessionManager(session_ttl=1, max_sessions=10)
        token = mgr.create_session("u1", "1234567890", "John")
        time.sleep(1.5)
        self.assertIsNone(mgr.verify_token(token))

    def test_sliding_window_extends_expiry(self):
        mgr = SessionManager(session_ttl=2, max_sessions=10)
        token = mgr.create_session("u1", "1234567890", "John")
        time.sleep(1)
        session = mgr.verify_token(token)  # Should refresh
        self.assertIsNotNone(session)
        time.sleep(1)
        session2 = mgr.verify_token(token)  # Still valid due to refresh
        self.assertIsNotNone(session2)

    def test_active_session_count(self):
        self.assertEqual(self.mgr.active_session_count, 0)
        self.mgr.create_session("u1", "1", "A")
        self.mgr.create_session("u2", "2", "B")
        self.assertEqual(self.mgr.active_session_count, 2)

    def test_revoke_all_for_user(self):
        self.mgr.create_session("u1", "1", "A")
        self.mgr.create_session("u1", "1", "A")
        self.mgr.create_session("u2", "2", "B")
        count = self.mgr.revoke_all_for_user("u1")
        self.assertEqual(count, 2)
        self.assertEqual(self.mgr.active_session_count, 1)

    def test_max_sessions_evicts_oldest(self):
        mgr = SessionManager(session_ttl=60, max_sessions=3)
        mgr.create_session("u1", "1", "A")
        mgr.create_session("u2", "2", "B")
        mgr.create_session("u3", "3", "C")
        mgr.create_session("u4", "4", "D")  # Should evict oldest
        self.assertEqual(mgr.active_session_count, 3)

    def test_security_level_preserved(self):
        token = self.mgr.create_session("u1", "1", "A", security_level="HIGH")
        session = self.mgr.verify_token(token)
        self.assertEqual(session['security_level'], "HIGH")


if __name__ == "__main__":
    unittest.main()
