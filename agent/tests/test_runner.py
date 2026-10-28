"""
Tests for agent.runner's episode setup.

The writable-tests mode exists to measure tampering, so the one thing it
must get right is that only the enforcement changes: the instruction not to
edit the tests stays, in both the task prompt and the system prompt.
"""

from __future__ import annotations

from pathlib import Path

from agent.loop import SYSTEM_PROMPT
from agent.runner import (
    _ENFORCED_TESTS_LINE,
    _UNENFORCED_TESTS_LINE,
    build_episode,
)

DATA_DIR = str(Path(__file__).resolve().parents[2] / "data")
TASK = "D4J_Lang_1"


def _test_files(workspace) -> list[str]:
    return [path for path in workspace.snapshot() if path.endswith("Test.java")]


class TestBuildEpisode:
    def test_tests_are_read_only_by_default(self):
        workspace, prompt, _, _ = build_episode(TASK, DATA_DIR)
        assert _test_files(workspace)
        assert all(workspace.is_read_only(path) for path in _test_files(workspace))
        assert "read-only" in prompt

    def test_writable_tests_lifts_only_the_enforcement(self):
        workspace, prompt, _, _ = build_episode(TASK, DATA_DIR, writable_tests=True)
        assert not any(workspace.is_read_only(path) for path in _test_files(workspace))
        assert "Do not modify the test" in prompt

    def test_the_system_prompt_line_being_replaced_exists(self):
        """A silent no-op replace would leave 'read-only' in a writable run."""
        assert _ENFORCED_TESTS_LINE in SYSTEM_PROMPT
        unenforced = SYSTEM_PROMPT.replace(_ENFORCED_TESTS_LINE, _UNENFORCED_TESTS_LINE)
        assert "Fix the code, not the test" in unenforced
        assert "apply_patch will refuse" not in unenforced
