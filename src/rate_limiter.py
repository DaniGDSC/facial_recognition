import time
import logging
from collections import defaultdict
from recognition_constants import MAX_AUTH_ATTEMPTS_PER_HOUR, COOLDOWN_PERIOD_SECONDS, SECONDS_PER_HOUR

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for authentication attempts.

    Tracks attempts per identifier (e.g. device IP or session) and enforces:
    - Maximum attempts per hour
    - Cooldown period between attempts after limit is reached
    """

    def __init__(
        self,
        max_attempts: int = MAX_AUTH_ATTEMPTS_PER_HOUR,
        cooldown_seconds: int = COOLDOWN_PERIOD_SECONDS,
        window_seconds: int = SECONDS_PER_HOUR,
    ):
        self.max_attempts = max_attempts
        self.cooldown_seconds = cooldown_seconds
        self.window_seconds = window_seconds
        # identifier -> list of attempt timestamps
        self._attempts: dict[str, list[float]] = defaultdict(list)
        # identifier -> cooldown expiry timestamp
        self._cooldowns: dict[str, float] = {}

    def _prune_old_attempts(self, identifier: str) -> None:
        """Remove attempts outside the current window."""
        cutoff = time.time() - self.window_seconds
        self._attempts[identifier] = [
            t for t in self._attempts[identifier] if t > cutoff
        ]

    def is_allowed(self, identifier: str) -> bool:
        """Check whether the identifier is allowed to attempt authentication."""
        now = time.time()

        # Check active cooldown
        cooldown_expiry = self._cooldowns.get(identifier, 0)
        if now < cooldown_expiry:
            remaining = int(cooldown_expiry - now)
            logger.warning(
                "Rate limit cooldown active for %s (%ds remaining)", identifier, remaining
            )
            return False

        self._prune_old_attempts(identifier)
        return len(self._attempts[identifier]) < self.max_attempts

    def record_attempt(self, identifier: str) -> None:
        """Record an authentication attempt. Activates cooldown if limit reached."""
        now = time.time()
        self._attempts[identifier].append(now)
        self._prune_old_attempts(identifier)

        if len(self._attempts[identifier]) >= self.max_attempts:
            self._cooldowns[identifier] = now + self.cooldown_seconds
            logger.warning(
                "Rate limit reached for %s — cooldown %ds activated",
                identifier,
                self.cooldown_seconds,
            )

    def remaining_attempts(self, identifier: str) -> int:
        """Return how many attempts remain in the current window."""
        self._prune_old_attempts(identifier)
        return max(0, self.max_attempts - len(self._attempts[identifier]))

    def cooldown_remaining(self, identifier: str) -> int:
        """Return seconds until cooldown expires (0 if not in cooldown)."""
        remaining = self._cooldowns.get(identifier, 0) - time.time()
        return max(0, int(remaining))
