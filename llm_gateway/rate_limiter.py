"""
Token-bucket rate limiter for the LLM Gateway.

Each provider gets its own bucket, configured via ``RateLimitConfig`` in the
gateway config.  The limiter enforces a requests-per-minute ceiling with a
configurable burst multiplier.  When the bucket is empty, ``acquire`` blocks
(sleeps) until capacity is available or retries are exhausted.

Thread-safety is provided by ``threading.Lock`` per bucket.

Usage::

    limiter = RateLimiter(config)
    limiter.acquire("claude")  # blocks if needed
"""

from __future__ import annotations

import logging
import threading
import time

from llm_gateway.config import GatewayConfig, RateLimitConfig

logger = logging.getLogger(__name__)


class _TokenBucket:
    """A single token-bucket rate limiter.

    Tokens refill at a constant rate.  ``consume`` tries to remove one token;
    if the bucket is empty it returns the wait time required.

    Attributes:
        capacity:   Maximum tokens in the bucket (burst size).
        refill_rate: Tokens added per second.
        tokens:     Current token count.
    """

    def __init__(self, capacity: float, refill_rate: float) -> None:
        """Initialise the bucket.

        Args:
            capacity:    Maximum burst size.
            refill_rate: Tokens per second.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self._last_refill = now

    def consume(self, n: float = 1.0) -> float:
        """Try to consume *n* tokens.

        Args:
            n: Number of tokens to consume.

        Returns:
            ``0.0`` if tokens were consumed successfully, otherwise the number
            of seconds to wait before retrying.
        """
        with self._lock:
            self._refill()
            if self.tokens >= n:
                self.tokens -= n
                return 0.0
            deficit = n - self.tokens
            return deficit / self.refill_rate


class RateLimiter:
    """Per-provider token-bucket rate limiter.

    Maintains one ``_TokenBucket`` per provider.  ``acquire`` blocks until the
    caller is allowed to proceed, using exponential back-off up to a
    configured ceiling.

    Attributes:
        config:  Gateway configuration.
        buckets: Mapping of provider name → ``_TokenBucket``.
    """

    def __init__(self, config: GatewayConfig) -> None:
        """Initialise rate limiters for all configured providers.

        Args:
            config: Gateway configuration with ``rate_limits`` section.
        """
        self.config = config
        self.buckets: dict[str, _TokenBucket] = {}

        for provider, rl_cfg in config.rate_limits.items():
            capacity = rl_cfg.requests_per_minute * rl_cfg.burst_multiplier
            refill_rate = rl_cfg.requests_per_minute / 60.0
            self.buckets[provider] = _TokenBucket(capacity, refill_rate)

    def acquire(
        self,
        provider: str,
        max_wait_seconds: float | None = None,
    ) -> float:
        """Block until one request slot is available for *provider*.

        Uses exponential back-off with jitter when the bucket is exhausted.

        Args:
            provider:          Provider name (e.g. ``"claude"``).
            max_wait_seconds:  Override for maximum total wait.  Defaults to
                the provider's ``retry_max_seconds``.

        Returns:
            Total seconds spent waiting (``0.0`` if no wait was needed).

        Raises:
            RateLimitExceeded: If *max_wait_seconds* is exceeded.
        """
        bucket = self.buckets.get(provider)
        if bucket is None:
            # Unknown provider → no limit enforced.
            return 0.0

        rl_cfg = self.config.rate_limits.get(
            provider, RateLimitConfig()
        )
        ceiling = max_wait_seconds or rl_cfg.retry_max_seconds
        base = rl_cfg.retry_base_seconds

        total_waited = 0.0
        attempt = 0

        while True:
            wait = bucket.consume()
            if wait == 0.0:
                if total_waited > 0:
                    logger.info(
                        "Rate-limit wait complete for %s: %.2fs",
                        provider,
                        total_waited,
                    )
                return total_waited

            # Exponential back-off with ceiling.
            sleep_time = min(base * (2 ** attempt), ceiling - total_waited)
            if sleep_time <= 0:
                raise RateLimitExceeded(
                    f"Rate limit for {provider!r}: waited {total_waited:.1f}s "
                    f"(ceiling {ceiling:.1f}s)"
                )

            logger.debug(
                "Rate-limited on %s — sleeping %.2fs (attempt %d)",
                provider,
                sleep_time,
                attempt + 1,
            )
            time.sleep(sleep_time)
            total_waited += sleep_time
            attempt += 1

    def remaining(self, provider: str) -> float:
        """Return the approximate remaining token count for *provider*.

        Args:
            provider: Provider name.

        Returns:
            Estimated remaining tokens (may be fractional).
        """
        bucket = self.buckets.get(provider)
        if bucket is None:
            return float("inf")
        with bucket._lock:
            bucket._refill()
            return bucket.tokens

    def reset(self, provider: str | None = None) -> None:
        """Reset bucket(s) to full capacity.

        Args:
            provider: If given, reset only that provider's bucket.
                Otherwise reset all.
        """
        targets = (
            [provider] if provider else list(self.buckets.keys())
        )
        for p in targets:
            bucket = self.buckets.get(p)
            if bucket:
                with bucket._lock:
                    bucket.tokens = bucket.capacity
                    bucket._last_refill = time.monotonic()


class RateLimitExceeded(Exception):
    """Raised when a rate-limit wait exceeds the configured ceiling."""
