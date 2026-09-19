"""
SQLite-backed response cache for the LLM Gateway.

Stores ``LLMResponse`` objects keyed by ``(prompt_hash, model_id)`` so that
identical prompts sent to the same model are never re-executed.  The prompt
hash is a SHA-256 digest computed by ``PromptRenderer.hash_prompt``.

Features:
* Content-addressed (deterministic key from prompt text).
* Transparent serialisation via Pydantic ``model_dump_json`` / ``model_validate_json``.
* TTL-based expiry (default 30 days, configurable).
* Thread-safe (SQLite WAL mode + per-call connection reuse).

Usage::

    cache = ResponseCache("llm_cache.db")
    cache.put(prompt_hash, model_id, response)
    hit = cache.get(prompt_hash, model_id)  # LLMResponse | None
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

from contracts import LLMResponse

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS response_cache (
    prompt_hash  TEXT    NOT NULL,
    model_id     TEXT    NOT NULL,
    response_json TEXT   NOT NULL,
    created_at   REAL    NOT NULL,
    expires_at   REAL    NOT NULL,
    hit_count    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (prompt_hash, model_id)
);

CREATE INDEX IF NOT EXISTS idx_cache_expires
    ON response_cache (expires_at);
"""


class ResponseCache:
    """SQLite content-addressed cache for ``LLMResponse`` objects.

    Each entry is keyed by ``(prompt_hash, model_id)`` and carries an
    expiration timestamp.  Expired entries are lazily purged on reads.

    Attributes:
        db_path:     Path to the SQLite database file.
        ttl_seconds: Time-to-live for cache entries in seconds.
    """

    def __init__(
        self,
        db_path: str | Path = "llm_cache.db",
        ttl_seconds: int = 30 * 24 * 3600,
    ) -> None:
        """Open (or create) the cache database.

        Args:
            db_path:     Filesystem path to the SQLite file.
            ttl_seconds: Default TTL for new entries (default 30 days).
        """
        self.db_path = str(db_path)
        self.ttl_seconds = ttl_seconds
        self._init_db()

    def _init_db(self) -> None:
        """Create the schema if it does not exist."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    def _connect(self) -> sqlite3.Connection:
        """Return a new connection with WAL mode enabled.

        Returns:
            An open ``sqlite3.Connection``.
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ── public API ─────────────────────────────────────────────────────

    def get(self, prompt_hash: str, model_id: str) -> Optional[LLMResponse]:
        """Retrieve a cached response, or ``None`` on cache miss / expiry.

        Increments the hit counter on a successful retrieval.

        Args:
            prompt_hash: SHA-256 hex digest of the prompt content.
            model_id:    Model identifier string.

        Returns:
            The cached ``LLMResponse``, or ``None``.
        """
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT response_json FROM response_cache "
                "WHERE prompt_hash = ? AND model_id = ? AND expires_at > ?",
                (prompt_hash, model_id, now),
            ).fetchone()

            if row is None:
                return None

            conn.execute(
                "UPDATE response_cache SET hit_count = hit_count + 1 "
                "WHERE prompt_hash = ? AND model_id = ?",
                (prompt_hash, model_id),
            )

        try:
            return LLMResponse.model_validate_json(row[0])
        except Exception:
            logger.warning(
                "Corrupt cache entry for hash=%s model=%s — treating as miss",
                prompt_hash[:12],
                model_id,
            )
            return None

    def put(
        self,
        prompt_hash: str,
        model_id: str,
        response: LLMResponse,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store a response in the cache.

        Overwrites any existing entry for the same key pair.

        Args:
            prompt_hash: SHA-256 hex digest of the prompt content.
            model_id:    Model identifier string.
            response:    The ``LLMResponse`` to cache.
            ttl_seconds: Override the default TTL for this entry.
        """
        ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        now = time.time()
        response_json = response.model_dump_json()

        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO response_cache "
                "(prompt_hash, model_id, response_json, created_at, expires_at, hit_count) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (prompt_hash, model_id, response_json, now, now + ttl),
            )

        logger.debug("Cached response for hash=%s model=%s", prompt_hash[:12], model_id)

    def invalidate(self, prompt_hash: str, model_id: str) -> bool:
        """Remove a specific cache entry.

        Args:
            prompt_hash: SHA-256 hex digest of the prompt content.
            model_id:    Model identifier string.

        Returns:
            ``True`` if an entry was deleted, ``False`` if nothing matched.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM response_cache WHERE prompt_hash = ? AND model_id = ?",
                (prompt_hash, model_id),
            )
        return cursor.rowcount > 0

    def purge_expired(self) -> int:
        """Delete all expired entries.

        Returns:
            Number of entries removed.
        """
        now = time.time()
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM response_cache WHERE expires_at <= ?", (now,)
            )
        removed = cursor.rowcount
        if removed:
            logger.info("Purged %d expired cache entries", removed)
        return removed

    def clear(self) -> int:
        """Delete **all** cache entries.

        Returns:
            Number of entries removed.
        """
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM response_cache")
        return cursor.rowcount

    def stats(self) -> dict[str, int]:
        """Return basic cache statistics.

        Returns:
            Dict with ``total_entries``, ``total_hits``, and ``expired``
            counts.
        """
        now = time.time()
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM response_cache"
            ).fetchone()[0]
            hits = conn.execute(
                "SELECT COALESCE(SUM(hit_count), 0) FROM response_cache"
            ).fetchone()[0]
            expired = conn.execute(
                "SELECT COUNT(*) FROM response_cache WHERE expires_at <= ?",
                (now,),
            ).fetchone()[0]
        return {"total_entries": total, "total_hits": hits, "expired": expired}
