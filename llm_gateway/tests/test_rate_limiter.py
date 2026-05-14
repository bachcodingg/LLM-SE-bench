"""
Tests for the token-bucket RateLimiter.
"""

from __future__ import annotations

import time

import pytest

from llm_gateway.config import GatewayConfig, RateLimitConfig
from llm_gateway.rate_limiter import RateLimiter, RateLimitExceeded, _TokenBucket


class TestTokenBucket:
    """Unit tests for the internal _TokenBucket."""

    def test_consume_success(self) -> None:
        """Consuming from a full bucket returns 0 wait."""
        bucket = _TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.consume(1.0) == 0.0

    def test_consume_empty(self) -> None:
        """Consuming from an empty bucket returns positive wait."""
        bucket = _TokenBucket(capacity=1, refill_rate=1.0)
        bucket.consume(1.0)  # drain it
        wait = bucket.consume(1.0)
        assert wait > 0.0

    def test_refill_over_time(self) -> None:
        """Bucket refills after sleeping."""
        bucket = _TokenBucket(capacity=2, refill_rate=100.0)  # fast refill
        bucket.consume(2.0)  # drain
        time.sleep(0.05)
        assert bucket.consume(1.0) == 0.0


class TestRateLimiter:
    """Tests for the per-provider RateLimiter."""

    def test_acquire_no_wait(self, gateway_config: GatewayConfig) -> None:
        """First acquire on a fresh bucket returns immediately."""
        limiter = RateLimiter(gateway_config)
        waited = limiter.acquire("claude")
        assert waited == 0.0

    def test_acquire_unknown_provider(self, gateway_config: GatewayConfig) -> None:
        """Unknown provider is unconstrained."""
        limiter = RateLimiter(gateway_config)
        waited = limiter.acquire("unknown-provider")
        assert waited == 0.0

    def test_remaining(self, gateway_config: GatewayConfig) -> None:
        """remaining returns current token count."""
        limiter = RateLimiter(gateway_config)
        r = limiter.remaining("claude")
        assert r > 0

    def test_remaining_unknown(self, gateway_config: GatewayConfig) -> None:
        """remaining returns inf for unknown provider."""
        limiter = RateLimiter(gateway_config)
        assert limiter.remaining("nope") == float("inf")

    def test_reset_single(self, gateway_config: GatewayConfig) -> None:
        """reset restores a provider's bucket to full."""
        limiter = RateLimiter(gateway_config)
        limiter.acquire("claude")
        limiter.reset("claude")
        # After reset, remaining should be at capacity.
        r = limiter.remaining("claude")
        rl_cfg = gateway_config.rate_limits["claude"]
        expected_cap = rl_cfg.requests_per_minute * rl_cfg.burst_multiplier
        assert abs(r - expected_cap) < 1.0

    def test_reset_all(self, gateway_config: GatewayConfig) -> None:
        """reset() without args resets all buckets."""
        limiter = RateLimiter(gateway_config)
        limiter.acquire("claude")
        limiter.acquire("gpt4")
        limiter.reset()
        assert limiter.remaining("claude") > 0
        assert limiter.remaining("gpt4") > 0

    def test_rate_limit_exceeded(self) -> None:
        """RateLimitExceeded is raised when ceiling is hit."""
        cfg = GatewayConfig(
            rate_limits={
                "test": RateLimitConfig(
                    requests_per_minute=1,
                    burst_multiplier=1.0,
                    retry_base_seconds=0.01,
                    retry_max_seconds=0.05,
                ),
            }
        )
        limiter = RateLimiter(cfg)
        limiter.acquire("test")  # consume the single token
        with pytest.raises(RateLimitExceeded):
            limiter.acquire("test", max_wait_seconds=0.05)
