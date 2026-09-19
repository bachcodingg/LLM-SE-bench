"""
Tests for mcp_servers.truncation.

The contract that matters: a truncated payload must say so, must keep the
first error, and must never silently look like a complete log.
"""

from __future__ import annotations

from mcp_servers.truncation import (
    DEFAULT_MAX_LINES,
    Truncation,
    truncate_log,
)


class TestShortInput:
    def test_empty(self):
        text, info = truncate_log("")
        assert text == ""
        assert info.truncated is False
        assert info.original_lines == 0

    def test_under_budget_is_untouched(self):
        original = "line one\nline two\nline three"
        text, info = truncate_log(original)
        assert text == original
        assert info.truncated is False
        assert info.strategy == "none"
        assert info.dropped_lines == 0

    def test_exactly_at_budget_is_untouched(self):
        original = "\n".join(f"line {i}" for i in range(DEFAULT_MAX_LINES))
        text, info = truncate_log(original)
        assert text == original
        assert info.truncated is False


class TestLineBudget:
    def test_over_budget_is_truncated_and_says_so(self):
        original = "\n".join(f"line {i}" for i in range(1000))
        text, info = truncate_log(original, max_lines=100)
        assert info.truncated is True
        assert info.strategy == "head+errors+tail"
        assert info.original_lines == 1000
        assert info.dropped_lines > 0
        assert info.kept_lines + info.dropped_lines == 1000
        assert len(text.splitlines()) < 1000

    def test_keeps_both_ends(self):
        original = "\n".join(f"line {i}" for i in range(1000))
        text, _ = truncate_log(original, max_lines=100, head_lines=10)
        assert "line 0" in text
        assert "line 999" in text
        assert "line 500" not in text

    def test_marks_the_gap(self):
        original = "\n".join(f"line {i}" for i in range(1000))
        text, _ = truncate_log(original, max_lines=100)
        assert "omitted by llm-se-bench" in text


class TestErrorPreservation:
    def test_buried_error_survives(self):
        """The first error is the informative one and is usually not at either end."""
        lines = [f"[INFO] compiling module {i}" for i in range(500)]
        lines[250] = "Main.java:12: error: cannot find symbol"
        text, info = truncate_log("\n".join(lines), max_lines=60, head_lines=10)
        assert "cannot find symbol" in text
        assert info.error_lines_preserved >= 1

    def test_several_error_markers_survive(self):
        lines = [f"noise {i}" for i in range(400)]
        lines[100] = "A.java:1: error: first"
        lines[200] = "[ERROR] second"
        lines[300] = "Caused by: java.lang.NullPointerException"
        text, info = truncate_log("\n".join(lines), max_lines=60, head_lines=5)
        assert "first" in text
        assert "second" in text
        assert "NullPointerException" in text
        assert info.error_lines_preserved == 3

    def test_no_markers_means_none_preserved(self):
        text, info = truncate_log("\n".join(f"quiet {i}" for i in range(400)), max_lines=50)
        assert info.error_lines_preserved == 0
        assert info.truncated is True
        assert text


class TestCharBudget:
    def test_long_single_line_is_capped(self):
        text, info = truncate_log("x" * 50_000, max_chars=1_000)
        assert len(text) <= 1_000
        assert info.truncated is True
        assert "char-cap" in info.strategy
        assert info.original_chars == 50_000

    def test_char_cap_composes_with_line_cap(self):
        original = "\n".join("y" * 500 for _ in range(1000))
        _, info = truncate_log(original, max_lines=100, max_chars=2_000)
        assert info.truncated is True
        assert "head+errors+tail" in info.strategy
        assert "char-cap" in info.strategy


class TestTruncationModel:
    def test_default_is_the_untruncated_case(self):
        info = Truncation()
        assert info.truncated is False
        assert info.strategy == "none"
        assert info.dropped_lines == 0

    def test_serialises_for_the_wire(self):
        _, info = truncate_log("\n".join(str(i) for i in range(500)), max_lines=50)
        payload = info.model_dump()
        assert payload["truncated"] is True
        assert payload["original_lines"] == 500
