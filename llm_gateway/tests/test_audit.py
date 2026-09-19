"""
Tests for the JSONL AuditLogger.
"""

from __future__ import annotations

import json
from pathlib import Path

from contracts import LLMResponse, Prompt
from llm_gateway.audit import AuditLogger


def _prompt() -> Prompt:
    """Build a test Prompt."""
    return Prompt(
        prompt_id="pmt-aud-001",
        problem_id="prob-001",
        model_id="gpt-4o",
        user_message="test",
    )


def _response() -> LLMResponse:
    """Build a test LLMResponse."""
    return LLMResponse(
        response_id="resp-aud-001",
        prompt_id="pmt-aud-001",
        model_id="gpt-4o",
        raw_text="output",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=250.0,
        metadata={"cache_hit": False},
    )


class TestAuditLogger:
    """Tests for append-only audit logging."""

    def test_log_event_creates_file(self, tmp_path: Path) -> None:
        """First log_event creates the JSONL file."""
        log_path = tmp_path / "audit.jsonl"
        audit = AuditLogger(log_path=log_path)
        audit.log_event(event_type="test")
        assert log_path.exists()

    def test_log_event_content(self, tmp_path: Path) -> None:
        """Logged event contains expected fields."""
        audit = AuditLogger(log_path=tmp_path / "a.jsonl")
        entry = audit.log_event(
            event_type="api_call",
            prompt=_prompt(),
            response=_response(),
            prompt_hash="abc123",
            cost_usd=0.005,
        )
        assert entry["event_type"] == "api_call"
        assert entry["prompt_id"] == "pmt-aud-001"
        assert entry["response_id"] == "resp-aud-001"
        assert entry["prompt_hash"] == "abc123"
        assert entry["cost_usd"] == 0.005
        assert entry["prompt_tokens"] == 10
        assert entry["total_tokens"] == 15

    def test_log_error(self, tmp_path: Path) -> None:
        """log_error writes an error-type entry."""
        audit = AuditLogger(log_path=tmp_path / "a.jsonl")
        entry = audit.log_error(
            error_type="APIError",
            message="Rate limited",
            prompt=_prompt(),
        )
        assert entry["event_type"] == "error"
        assert entry["error_type"] == "APIError"
        assert entry["prompt_id"] == "pmt-aud-001"

    def test_read_all(self, tmp_path: Path) -> None:
        """read_all returns all logged entries."""
        audit = AuditLogger(log_path=tmp_path / "a.jsonl")
        audit.log_event(event_type="first")
        audit.log_event(event_type="second")
        entries = audit.read_all()
        assert len(entries) == 2
        assert entries[0]["event_type"] == "first"
        assert entries[1]["event_type"] == "second"

    def test_count(self, tmp_path: Path) -> None:
        """count returns the number of entries."""
        audit = AuditLogger(log_path=tmp_path / "a.jsonl")
        assert audit.count() == 0
        audit.log_event(event_type="x")
        audit.log_event(event_type="y")
        assert audit.count() == 2

    def test_clear(self, tmp_path: Path) -> None:
        """clear empties the log file."""
        audit = AuditLogger(log_path=tmp_path / "a.jsonl")
        audit.log_event(event_type="x")
        audit.clear()
        assert audit.count() == 0

    def test_extra_fields(self, tmp_path: Path) -> None:
        """Extra kwargs are merged into the entry."""
        audit = AuditLogger(log_path=tmp_path / "a.jsonl")
        entry = audit.log_event(event_type="test", extra={"custom_key": 42})
        assert entry["custom_key"] == 42

    def test_read_all_empty_file(self, tmp_path: Path) -> None:
        """read_all on a nonexistent file returns empty list."""
        audit = AuditLogger(log_path=tmp_path / "nonexistent.jsonl")
        assert audit.read_all() == []

    def test_log_event_without_prompt_or_response(self, tmp_path: Path) -> None:
        """log_event works with neither prompt nor response."""
        audit = AuditLogger(log_path=tmp_path / "a.jsonl")
        entry = audit.log_event(event_type="bare_event")
        assert "prompt_id" not in entry
        assert "response_id" not in entry

    def test_jsonl_format(self, tmp_path: Path) -> None:
        """Each line is valid JSON."""
        log_path = tmp_path / "a.jsonl"
        audit = AuditLogger(log_path=log_path)
        audit.log_event(event_type="a")
        audit.log_event(event_type="b")

        with open(log_path) as fh:
            lines = fh.readlines()
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "timestamp" in parsed
