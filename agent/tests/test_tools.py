"""
Tests for agent.tools.

The contract being protected: a tool never raises for anything the agent
did. Every mistake — a missing file, a bad regex, a wrong argument name, a
tool that does not exist — comes back as an error result the agent can read
and recover from, because recovering is the behaviour under measurement.
"""

from __future__ import annotations

import pytest

from agent.tools import TOOL_SCHEMAS, AgentToolset, ToolContext, ToolOutcome
from agent.workspace import Workspace


@pytest.fixture
def toolset(workspace, test_runner) -> AgentToolset:
    return AgentToolset(ToolContext(workspace=workspace, test_runner=test_runner))


class TestSchemas:
    def test_the_five_tools_are_declared(self):
        assert {schema.name for schema in TOOL_SCHEMAS} == {
            "read_file", "list_dir", "grep", "apply_patch", "run_tests"
        }

    def test_schemas_match_the_dispatcher(self, toolset):
        assert sorted(schema.name for schema in TOOL_SCHEMAS) == toolset.names

    def test_every_schema_is_a_json_schema_object(self):
        for schema in TOOL_SCHEMAS:
            assert schema.parameters["type"] == "object"
            assert "properties" in schema.parameters

    def test_descriptions_are_written_for_a_model(self):
        for schema in TOOL_SCHEMAS:
            assert len(schema.description) > 150, schema.name

    def test_apply_patch_warns_that_it_is_a_whole_file_write(self):
        """The most expensive misunderstanding available to the agent."""
        schema = next(s for s in TOOL_SCHEMAS if s.name == "apply_patch")
        assert "complete" in schema.description
        assert "not a diff" in schema.description

    def test_run_tests_warns_that_zero_of_zero_is_not_a_pass(self):
        schema = next(s for s in TOOL_SCHEMAS if s.name == "run_tests")
        assert "0/0" in schema.description


class TestDispatch:
    def test_unknown_tool_is_an_error_result_not_an_exception(self, toolset):
        outcome = toolset.dispatch("teleport", {})
        assert outcome.is_error
        assert "read_file" in outcome.content  # tells it what does exist

    def test_wrong_argument_name_is_an_error_result(self, toolset):
        outcome = toolset.dispatch("read_file", {"filename": "Calculator.java"})
        assert outcome.is_error
        assert "Invalid arguments" in outcome.content

    def test_missing_required_argument_is_an_error_result(self, toolset):
        outcome = toolset.dispatch("read_file", {})
        assert outcome.is_error

    def test_workspace_errors_surface_as_readable_text(self, toolset):
        outcome = toolset.dispatch("apply_patch", {
            "path": "CalculatorTest.java", "content": "x"
        })
        assert outcome.is_error
        assert "read-only" in outcome.content

    def test_an_internal_failure_is_reported_not_propagated(self, workspace):
        def explode(_workspace):
            raise RuntimeError("sandbox on fire")

        toolset = AgentToolset(ToolContext(workspace=workspace, test_runner=explode))
        outcome = toolset.dispatch("run_tests", {})
        assert outcome.is_error
        assert "sandbox on fire" in outcome.content


class TestReadFile:
    def test_returns_numbered_lines(self, toolset):
        outcome = toolset.dispatch("read_file", {"path": "Calculator.java"})
        assert not outcome.is_error
        assert "    1  public class Calculator" in outcome.content

    def test_header_reports_the_range(self, toolset):
        outcome = toolset.dispatch("read_file", {"path": "Calculator.java"})
        assert "lines 1-" in outcome.content

    def test_marks_read_only_files(self, toolset):
        outcome = toolset.dispatch("read_file", {"path": "CalculatorTest.java"})
        assert "[read-only]" in outcome.content

    def test_line_range_is_respected(self, toolset):
        outcome = toolset.dispatch(
            "read_file", {"path": "Calculator.java", "start_line": 2, "end_line": 3}
        )
        assert "    2  " in outcome.content
        assert "    1  " not in outcome.content

    def test_start_past_the_end_is_an_error(self, toolset):
        outcome = toolset.dispatch(
            "read_file", {"path": "Calculator.java", "start_line": 500}
        )
        assert outcome.is_error
        assert "past the end" in outcome.content

    def test_missing_file_lists_what_exists(self, toolset):
        outcome = toolset.dispatch("read_file", {"path": "Nope.java"})
        assert outcome.is_error
        assert "CalculatorTest.java" in outcome.content

    def test_a_long_file_is_truncated_and_says_how_to_page(self, test_runner):
        workspace = Workspace(files={"Big.java": "\n".join(f"// {i}" for i in range(2000))})
        toolset = AgentToolset(ToolContext(workspace=workspace, test_runner=test_runner))
        outcome = toolset.dispatch("read_file", {"path": "Big.java"})
        assert outcome.truncated
        assert "lines omitted" in outcome.content
        assert "start_line" in outcome.content


class TestListDir:
    def test_lists_every_file_with_line_counts(self, toolset):
        outcome = toolset.dispatch("list_dir", {})
        assert "Calculator.java" in outcome.content
        assert "lines)" in outcome.content

    def test_marks_read_only_files(self, toolset):
        outcome = toolset.dispatch("list_dir", {})
        assert "[read-only]" in outcome.content

    def test_glob_filters(self, toolset):
        outcome = toolset.dispatch("list_dir", {"pattern": "*Test.java"})
        assert "CalculatorTest.java" in outcome.content
        assert "1 file(s)" in outcome.content

    def test_no_match_is_not_an_error(self, toolset):
        outcome = toolset.dispatch("list_dir", {"pattern": "*.py"})
        assert not outcome.is_error
        assert "No files" in outcome.content


class TestGrep:
    def test_finds_matches_with_path_and_line(self, toolset):
        outcome = toolset.dispatch("grep", {"pattern": "public static"})
        assert "Calculator.java:2:" in outcome.content

    def test_no_match_is_a_normal_result(self, toolset):
        outcome = toolset.dispatch("grep", {"pattern": "zzzz"})
        assert not outcome.is_error
        assert "No match" in outcome.content

    def test_invalid_regex_is_an_error_result_not_a_crash(self, toolset):
        outcome = toolset.dispatch("grep", {"pattern": "[unclosed"})
        assert outcome.is_error
        assert "Invalid regular expression" in outcome.content

    def test_ignore_case(self, toolset):
        assert "No match" in toolset.dispatch("grep", {"pattern": "CALCULATOR"}).content
        outcome = toolset.dispatch("grep", {"pattern": "CALCULATOR", "ignore_case": True})
        assert not outcome.is_error
        assert "match(es)" in outcome.content

    def test_path_glob_limits_the_search(self, toolset):
        outcome = toolset.dispatch(
            "grep", {"pattern": "class", "path_glob": "*Test.java"}
        )
        assert "CalculatorTest.java" in outcome.content
        assert "Calculator.java:" not in outcome.content


class TestApplyPatch:
    def test_updates_a_file_and_reports_the_change(self, toolset, workspace):
        outcome = toolset.dispatch("apply_patch", {
            "path": "Calculator.java", "content": "class Calculator {}"
        })
        assert not outcome.is_error
        assert "Updated" in outcome.content
        assert workspace.read("Calculator.java") == "class Calculator {}"

    def test_creates_a_new_file(self, toolset, workspace):
        outcome = toolset.dispatch("apply_patch", {"path": "New.java", "content": "x"})
        assert "Created" in outcome.content
        assert "New.java" in workspace

    def test_identical_content_is_reported_as_a_no_op(self, toolset, workspace):
        original = workspace.read("Calculator.java")
        outcome = toolset.dispatch("apply_patch", {
            "path": "Calculator.java", "content": original
        })
        assert "nothing changed" in outcome.content
        assert workspace.effective_edits() == 0

    def test_refuses_a_read_only_path(self, toolset):
        outcome = toolset.dispatch("apply_patch", {
            "path": "CalculatorTest.java", "content": "@Test public void nothing() {}"
        })
        assert outcome.is_error
        assert "read-only" in outcome.content

    def test_records_the_step_index(self, workspace, test_runner):
        context = ToolContext(workspace=workspace, test_runner=test_runner, step_index=7)
        AgentToolset(context).dispatch("apply_patch", {"path": "A.java", "content": "x"})
        assert workspace.edits[-1].step_index == 7

    def test_suggests_running_the_tests(self, toolset):
        outcome = toolset.dispatch("apply_patch", {"path": "A.java", "content": "x"})
        assert "run_tests" in outcome.content


class TestRunTests:
    def test_delegates_to_the_injected_runner(self, toolset, test_runner):
        toolset.dispatch("run_tests", {})
        assert test_runner.calls == 1

    def test_reports_failure_on_the_buggy_source(self, toolset):
        outcome = toolset.dispatch("run_tests", {})
        assert outcome.is_error
        assert outcome.tests_passed == 0
        assert outcome.tests_total == 2

    def test_reports_success_once_the_source_is_fixed(self, toolset, workspace):
        workspace.write("Calculator.java", "class C { int f() { return a + b; } }")
        outcome = toolset.dispatch("run_tests", {})
        assert not outcome.is_error
        assert outcome.tests_passed == outcome.tests_total

    def test_carries_counts_so_the_loop_need_not_reparse(self, toolset):
        outcome: ToolOutcome = toolset.dispatch("run_tests", {})
        assert outcome.tests_total is not None
        assert outcome.compiled is True
