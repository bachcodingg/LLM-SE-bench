"""
Tests for the SQLite response cache (``ResponseCache``).

Uses a temporary database for every test to ensure isolation.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from contracts import LLMResponse
from llm_gateway.cache import ResponseCache

# ── helpers ────────────────────────────────────────────────────────────

def _resp(response_id: str = "resp-c-001", model_id: str = "gpt-4o") -> LLMResponse:
    """Build a minimal LLMResponse."""
    return LLMResponse(
        response_id=response_id,
        prompt_id="pmt-001",
        model_id=model_id,
        raw_text="hello world",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=100.0,
        created_at=datetime(2026, 1, 1),
        metadata={"cache_hit": False},
    )


# ── basic put / get ───────────────────────────────────────────────────

class TestCachePutGet:
    """Tests for basic cache store and retrieve."""

    def test_put_then_get(self, tmp_path: Path) -> None:
        """Stored response is retrievable."""
        cache = ResponseCache(db_path=tmp_path / "c.db")
        r = _resp()
        cache.put("hash-a", "gpt-4o", r)
        got = cache.get("hash-a", "gpt-4o")
        assert got is not None
        assert got.response_id == r.response_id
        assert got.raw_text == r.raw_text

    def test_get_miss(self, tmp_path: Path) -> None:
        """Cache miss returns None."""
        cache = ResponseCache(db_path=tmp_path / "c.db")
        assert cache.get("nonexistent", "gpt-4o") is None

    def test_different_model_miss(self, tmp_path: Path) -> None:
        """Same hash but different model is a miss."""
        cache = ResponseCache(db_path=tmp_path / "c.db")
        cache.put("hash-b", "gpt-4o", _resp())
        assert cache.get("hash-b", "claude-3-5-sonnet") is None

    def test_overwrite(self, tmp_path: Path) -> None:
        """Second put with same key overwrites the first."""
        cache = ResponseCache(db_path=tmp_path / "c.db")
        cache.put("hash-c", "gpt-4o", _resp(response_id="old"))
        cache.put("hash-c", "gpt-4o", _resp(response_id="new"))
        got = cache.get("hash-c", "gpt-4o")
        assert got is not None
        assert got.response_id == "new"


# ── TTL / expiry ──────────────────────────────────────────────────────

class TestCacheExpiry:
    """Tests for TTL-based expiration."""

    def test_expired_entry_is_miss(self, tmp_path: Path) -> None:
        """Entry with 0-second TTL expires immediately."""
        cache = ResponseCache(db_path=tmp_path / "c.db", ttl_seconds=0)
        cache.put("hash-d", "gpt-4o", _resp())
        # Tiny sleep to pass the 0-second TTL.
        time.sleep(0.05)
        assert cache.get("hash-d", "gpt-4o") is None

    def test_custom_ttl(self, tmp_path: Path) -> None:
        """Per-entry TTL override is respected."""
        cache = ResponseCache(db_path=tmp_path / "c.db", ttl_seconds=3600)
        cache.put("hash-e", "gpt-4o", _resp(), ttl_seconds=0)
        time.sleep(0.05)
        assert cache.get("hash-e", "gpt-4o") is None

    def test_purge_expired(self, tmp_path: Path) -> None:
        """purge_expired removes only expired entries."""
        cache = ResponseCache(db_path=tmp_path / "c.db")
        cache.put("hash-f", "gpt-4o", _resp(), ttl_seconds=0)
        cache.put("hash-g", "gpt-4o", _resp(), ttl_seconds=3600)
        time.sleep(0.05)
        removed = cache.purge_expired()
        assert removed == 1
        assert cache.get("hash-g", "gpt-4o") is not None


# ── invalidate / clear ────────────────────────────────────────────────

class TestCacheInvalidate:
    """Tests for selective and bulk deletion."""

    def test_invalidate_existing(self, tmp_path: Path) -> None:
        """Invalidating an existing key returns True and removes it."""
        cache = ResponseCache(db_path=tmp_path / "c.db")
        cache.put("hash-h", "gpt-4o", _resp())
        assert cache.invalidate("hash-h", "gpt-4o") is True
        assert cache.get("hash-h", "gpt-4o") is None

    def test_invalidate_missing(self, tmp_path: Path) -> None:
        """Invalidating a nonexistent key returns False."""
        cache = ResponseCache(db_path=tmp_path / "c.db")
        assert cache.invalidate("nope", "gpt-4o") is False

    def test_clear(self, tmp_path: Path) -> None:
        """Clear removes all entries."""
        cache = ResponseCache(db_path=tmp_path / "c.db")
        cache.put("hash-i", "gpt-4o", _resp())
        cache.put("hash-j", "gpt-4o", _resp())
        removed = cache.clear()
        assert removed == 2
        assert cache.get("hash-i", "gpt-4o") is None


# ── stats ──────────────────────────────────────────────────────────────

class TestCacheStats:
    """Tests for cache statistics."""

    def test_stats_empty(self, tmp_path: Path) -> None:
        """Empty cache reports all zeros."""
        cache = ResponseCache(db_path=tmp_path / "c.db")
        s = cache.stats()
        assert s["total_entries"] == 0
        assert s["total_hits"] == 0

    def test_stats_after_hits(self, tmp_path: Path) -> None:
        """Hit counter increments on successful gets."""
        cache = ResponseCache(db_path=tmp_path / "c.db")
        cache.put("hash-k", "gpt-4o", _resp())
        cache.get("hash-k", "gpt-4o")
        cache.get("hash-k", "gpt-4o")
        s = cache.stats()
        assert s["total_entries"] == 1
        assert s["total_hits"] == 2
