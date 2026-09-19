"""
Tests for agent.workspace.

The workspace is the agent's only writable surface, so the properties that
matter are: it cannot be escaped, read-only files cannot be written, and
every mutation is recorded accurately enough for the no-progress and churn
metrics to mean something.
"""

from __future__ import annotations

import pytest

from agent.workspace import Workspace, WorkspaceError


class TestPathHandling:
    @pytest.mark.parametrize("path", ["../secrets", "a/../../b", "../../etc/passwd"])
    def test_upward_traversal_is_rejected(self, path):
        workspace = Workspace()
        with pytest.raises(WorkspaceError, match=r"\.\."):
            workspace.write(path, "x")

    def test_absolute_paths_are_rejected(self):
        with pytest.raises(WorkspaceError, match="absolute"):
            Workspace().write("/etc/passwd", "x")

    def test_empty_path_is_rejected(self):
        with pytest.raises(WorkspaceError, match="Empty"):
            Workspace().write("   ", "x")

    def test_leading_dot_slash_is_normalised(self):
        workspace = Workspace(files={"./A.java": "class A {}"})
        assert workspace.read("A.java") == "class A {}"
        assert "A.java" in workspace

    def test_backslashes_are_normalised(self):
        workspace = Workspace(files={"src/A.java": "x"})
        assert workspace.read("src\\A.java") == "x"

    def test_contains_does_not_raise_on_a_bad_path(self):
        assert ("../x" in Workspace()) is False


class TestReading:
    def test_reads_a_file(self, workspace):
        assert "Calculator" in workspace.read("Calculator.java")

    def test_missing_file_error_lists_what_exists(self, workspace):
        with pytest.raises(WorkspaceError, match="CalculatorTest.java"):
            workspace.read("Nope.java")

    def test_lists_paths_sorted(self, workspace):
        assert workspace.list_paths() == ["Calculator.java", "CalculatorTest.java"]

    def test_glob_filters(self, workspace):
        assert workspace.list_paths("*Test.java") == ["CalculatorTest.java"]

    def test_glob_matches_the_basename_too(self):
        workspace = Workspace(files={"src/main/A.java": "x", "README.md": "y"})
        assert workspace.list_paths("*.java") == ["src/main/A.java"]

    def test_snapshot_is_a_copy(self, workspace):
        snapshot = workspace.snapshot()
        snapshot["Calculator.java"] = "tampered"
        assert workspace.read("Calculator.java") != "tampered"


class TestReadOnly:
    def test_write_to_a_read_only_path_is_refused(self, workspace):
        with pytest.raises(WorkspaceError, match="read-only"):
            workspace.write("CalculatorTest.java", "anything")

    def test_delete_of_a_read_only_path_is_refused(self, workspace):
        with pytest.raises(WorkspaceError, match="read-only"):
            workspace.delete("CalculatorTest.java")

    def test_the_refusal_explains_why(self, workspace):
        """The agent should learn to fix the code, not fight the harness."""
        with pytest.raises(WorkspaceError, match="scored"):
            workspace.write("CalculatorTest.java", "x")

    def test_read_only_files_are_still_readable(self, workspace):
        assert workspace.read("CalculatorTest.java")
        assert workspace.is_read_only("CalculatorTest.java")


class TestWriting:
    def test_creating_records_a_create(self, workspace):
        edit = workspace.write("New.java", "class New {}", step_index=3)
        assert edit.operation == "create"
        assert edit.step_index == 3
        assert workspace.read("New.java") == "class New {}"

    def test_updating_records_an_update(self, workspace):
        edit = workspace.write("Calculator.java", "class Calculator {}")
        assert edit.operation == "update"
        assert edit.before_hash and edit.before_hash != edit.after_hash

    def test_identical_content_is_a_no_op(self, workspace):
        """Rewriting the same bytes costs a step but is not progress."""
        original = workspace.read("Calculator.java")
        edit = workspace.write("Calculator.java", original)
        assert edit.operation == "no-op"
        assert edit.reverted is True
        assert workspace.effective_edits() == 0

    def test_oversized_content_is_refused(self):
        workspace = Workspace(max_file_chars=10)
        with pytest.raises(WorkspaceError, match="limit"):
            workspace.write("A.java", "x" * 11)

    def test_line_counts_are_recorded(self, workspace):
        edit = workspace.write("Calculator.java", "a\nb\nc\nd\ne\nf\n")
        assert edit.lines_added > 0

    def test_delete_removes_and_records(self, workspace):
        workspace.delete("Calculator.java")
        assert "Calculator.java" not in workspace
        assert workspace.edits[-1].operation == "delete"

    def test_deleting_a_missing_file_is_an_error(self, workspace):
        with pytest.raises(WorkspaceError, match="does not exist"):
            workspace.delete("Nope.java")


class TestHistory:
    def test_files_touched_is_in_first_touch_order(self, workspace):
        workspace.write("B.java", "x")
        workspace.write("A.java", "y")
        workspace.write("B.java", "z")
        assert workspace.files_touched() == ["B.java", "A.java"]

    def test_effective_edits_ignores_no_ops(self, workspace):
        original = workspace.read("Calculator.java")
        workspace.write("Calculator.java", "changed")
        workspace.write("Calculator.java", "changed")  # no-op
        workspace.write("Calculator.java", original)
        assert workspace.effective_edits() == 2

    def test_diff_summary_classifies_each_change(self, workspace):
        workspace.write("Calculator.java", "modified")
        workspace.write("New.java", "created")
        workspace.delete("Calculator.java")
        summary = workspace.diff_summary()
        assert summary["Calculator.java"] == "deleted"
        assert summary["New.java"] == "created"

    def test_a_file_restored_to_its_original_is_not_in_the_diff(self, workspace):
        original = workspace.read("Calculator.java")
        workspace.write("Calculator.java", "temporary")
        workspace.write("Calculator.java", original)
        assert "Calculator.java" not in workspace.diff_summary()

    def test_churn_counts_work_that_left_no_trace(self, workspace):
        """Writing and reverting is real work with no diff to show for it."""
        original = workspace.read("Calculator.java")
        workspace.write("Calculator.java", original + "\n// a\n// b\n// c\n")
        workspace.write("Calculator.java", original)
        assert workspace.churn() > 0

    def test_a_straight_edit_has_little_churn(self, workspace):
        original = workspace.read("Calculator.java")
        workspace.write("Calculator.java", original.replace("a - b", "a + b"))
        assert workspace.churn() == 0


class TestMaterialise:
    def test_writes_every_file_to_disk(self, workspace, tmp_path):
        root = workspace.materialise(tmp_path / "out")
        assert (root / "Calculator.java").read_text(encoding="utf-8")
        assert (root / "CalculatorTest.java").exists()

    def test_creates_nested_directories(self, tmp_path):
        workspace = Workspace(files={"src/main/java/A.java": "class A {}"})
        root = workspace.materialise(tmp_path / "out")
        assert (root / "src/main/java/A.java").exists()

    def test_len_and_contains(self, workspace):
        assert len(workspace) == 2
        assert "Calculator.java" in workspace
