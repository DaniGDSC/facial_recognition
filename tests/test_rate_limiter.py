import time
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Patch database_config before any src module tries to import it
os.environ.setdefault("PG_PASSWORD", "test_password")

from rate_limiter import RateLimiter


class TestRateLimiter(unittest.TestCase):

    def test_allows_within_limit(self):
        limiter = RateLimiter(max_attempts=3, cooldown_seconds=10, window_seconds=3600)
        self.assertTrue(limiter.is_allowed("user1"))
        self.assertEqual(limiter.remaining_attempts("user1"), 3)

    def test_blocks_after_limit_reached(self):
        limiter = RateLimiter(max_attempts=2, cooldown_seconds=10, window_seconds=3600)

        limiter.record_attempt("user1")
        self.assertTrue(limiter.is_allowed("user1"))
        self.assertEqual(limiter.remaining_attempts("user1"), 1)

        limiter.record_attempt("user1")
        self.assertFalse(limiter.is_allowed("user1"))
        self.assertEqual(limiter.remaining_attempts("user1"), 0)

    def test_cooldown_expiry(self):
        limiter = RateLimiter(max_attempts=1, cooldown_seconds=1, window_seconds=3600)

        limiter.record_attempt("user1")
        self.assertFalse(limiter.is_allowed("user1"))

        time.sleep(1.1)
        # Cooldown expired, but attempts still in window
        self.assertFalse(limiter.is_allowed("user1"))

    def test_window_expiry(self):
        limiter = RateLimiter(max_attempts=1, cooldown_seconds=0, window_seconds=1)

        limiter.record_attempt("user1")
        self.assertFalse(limiter.is_allowed("user1"))

        time.sleep(1.1)
        # Window expired, attempt should be pruned
        self.assertTrue(limiter.is_allowed("user1"))

    def test_independent_identifiers(self):
        limiter = RateLimiter(max_attempts=1, cooldown_seconds=10, window_seconds=3600)

        limiter.record_attempt("user1")
        self.assertFalse(limiter.is_allowed("user1"))
        self.assertTrue(limiter.is_allowed("user2"))

    def test_remaining_attempts_decreases(self):
        limiter = RateLimiter(max_attempts=3, cooldown_seconds=10, window_seconds=3600)

        self.assertEqual(limiter.remaining_attempts("user1"), 3)
        limiter.record_attempt("user1")
        self.assertEqual(limiter.remaining_attempts("user1"), 2)
        limiter.record_attempt("user1")
        self.assertEqual(limiter.remaining_attempts("user1"), 1)
        limiter.record_attempt("user1")
        self.assertEqual(limiter.remaining_attempts("user1"), 0)

    def test_cooldown_remaining_returns_positive_when_active(self):
        limiter = RateLimiter(max_attempts=1, cooldown_seconds=60, window_seconds=3600)

        self.assertEqual(limiter.cooldown_remaining("user1"), 0)
        limiter.record_attempt("user1")
        self.assertGreater(limiter.cooldown_remaining("user1"), 0)

    def test_cooldown_remaining_zero_when_no_cooldown(self):
        limiter = RateLimiter(max_attempts=5, cooldown_seconds=60, window_seconds=3600)
        limiter.record_attempt("user1")
        self.assertEqual(limiter.cooldown_remaining("user1"), 0)


if __name__ == "__main__":
    unittest.main()
