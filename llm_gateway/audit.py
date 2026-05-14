"""
Append-only JSONL audit logger for the LLM Gateway.

Records every API interaction (and cache hit) as a single JSON line in an
append-only file.  Each entry captures: timestamp, event type, model id,
prompt hash, response id, latency, token counts, cost, and cache-hit status.

This log supports reproducibility analysis and post-hoc cost / latency
auditing across benchmark runs.

Usage::

    audit = AuditLogger("audit_log.jsonl")
    audit.log_event(
        event_type="api_call",
        prompt=prompt,
        response=response,
        prompt_hash=hash_str,
        cost_usd=0.0042,
    )
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contracts import LLMResponse, Prompt

logger = logging.getLogger(__name__)


class AuditLogger:
    """Append-only JSONL audit trail for LLM interactions.

    Each ``log_event`` call writes a single JSON line.  The file is opened in
    append mode and flushed after every write.  A ``threading.Lock`` ensures
    thread-safe writes.

    Attributes:
        log_path: Path to the JSONL file.
    """

    def __init__(self, log_path: str | Path = "audit_log.jsonl") -> None:
        """Open the audit log file (creating it if needed).

        Args:
            log_path: Filesystem path for the JSONL log.
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log_event(
        self,
        *,
        event_type: str,
        prompt: Prompt | None = None,
        response: LLMResponse | None = None,
        prompt_hash: str = "",
        cost_usd: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write one audit entry.

        Args:
            event_type:  Label such as ``"api_call"`` or ``"cache_hit"``.
            prompt:      The ``Prompt`` object (optional).
            response:    The ``LLMResponse`` object (optional).
            prompt_hash: SHA-256 prompt hash for traceability.
            cost_usd:    Computed cost for this call.
            extra:       Arbitrary additional key-value pairs.

        Returns:
            The dict that was serialised to JSON (useful for testing).
        """
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
        }

        if prompt is not None:
            entry["prompt_id"] = prompt.prompt_id
            entry["problem_id"] = prompt.problem_id
            entry["model_id"] = prompt.model_id
            entry["temperature"] = prompt.temperature
            entry["max_tokens"] = prompt.max_tokens

        if response is not None:
            entry["response_id"] = response.response_id
            entry["model_id"] = response.model_id
            entry["finish_reason"] = response.finish_reason
            entry["prompt_tokens"] = response.prompt_tokens
            entry["completion_tokens"] = response.completion_tokens
            entry["total_tokens"] = response.total_tokens
            entry["latency_ms"] = response.latency_ms
            entry["cache_hit"] = response.metadata.get("cache_hit", False)

        if prompt_hash:
            entry["prompt_hash"] = prompt_hash

        if cost_usd > 0:
            entry["cost_usd"] = cost_usd

        if extra:
            entry.update(extra)

        self._write(entry)
        return entry

    def log_error(
        self,
        *,
        error_type: str,
        message: str,
        prompt: Prompt | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write an error audit entry.

        Args:
            error_type: Exception class name or error category.
            message:    Human-readable error description.
            prompt:     The ``Prompt`` that triggered the error (optional).
            extra:      Additional context.

        Returns:
            The dict that was serialised to JSON.
        """
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "error",
            "error_type": error_type,
            "error_message": message,
        }

        if prompt is not None:
            entry["prompt_id"] = prompt.prompt_id
            entry["model_id"] = prompt.model_id

        if extra:
            entry.update(extra)

        self._write(entry)
        return entry

    def _write(self, entry: dict[str, Any]) -> None:
        """Serialise *entry* and append to the log file.

        Args:
            entry: The audit record to persist.
        """
        line = json.dumps(entry, default=str) + "\n"
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line)

    # ── read-back helpers ──────────────────────────────────────────────

    def read_all(self) -> list[dict[str, Any]]:
        """Read and parse all log entries.

        Returns:
            List of dicts, one per log line.
        """
        if not self.log_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with open(self.log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def count(self) -> int:
        """Return the number of log entries.

        Returns:
            Entry count.
        """
        if not self.log_path.exists():
            return 0
        with open(self.log_path, "r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())

    def clear(self) -> None:
        """Truncate the log file (use with caution — data loss)."""
        with self._lock:
            with open(self.log_path, "w", encoding="utf-8") as fh:
                fh.truncate(0)
