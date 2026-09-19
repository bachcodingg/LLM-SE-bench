"""
Tests for agent.stub.

The stub backs ``--dry-run`` and the CI smoke job, so two properties matter
more than anything it does: it never claims to solve a task, and it starts
each episode from scratch even though one instance serves the whole run.
"""

from __future__ import annotations

import pytest

from agent.loop import AgentConfig, AgentLoop
from agent.stub import StubAgentClient
from agent.termination import StopCondition, TerminationPolicy
from agent.tools import ToolOutcome
from agent.workspace import Workspace


@pytest.fixture
def stub_policy() -> TerminationPolicy:
    """A policy wide enough for the stub's plan to finish.

    The stub only explores, and exploring is correctly *not* progress, so
    the default no-progress window stops it before it reaches run_tests.
    That is the loop behaving properly; these tests are about the stub, so
    they give it room.
    """
    return TerminationPolicy(
        max_cost_eur=1.0, max_steps=30, no_progress_steps=25, wall_clock_seconds=60.0
    )


def _unexecuted_runner(_workspace: Workspace) -> ToolOutcome:
    """A test runner standing in for 'Docker is not available'."""
    return ToolOutcome(
        content="EXECUTION SKIPPED — structural check only.",
        tests_passed=2, tests_total=2, compiled=True, executed=False,
    )


def _run(client, workspace, policy):
    return AgentLoop(
        client=client,
        workspace=workspace,
        test_runner=_unexecuted_runner,
        config=AgentConfig(model_id="stub", policy=policy, task_id="T"),
        cost_of=lambda turn: 0.0,
    ).run("Fix it.")


class TestStubBehaviour:
    def test_it_explores_then_stops(self, workspace, stub_policy):
        trajectory = _run(StubAgentClient(), workspace, stub_policy)
        tools = [step.tool_name for step in trajectory.steps if step.tool_name]
        assert tools[0] == "list_dir"
        assert "read_file" in tools
        assert tools[-1] == "run_tests"

    def test_it_never_claims_a_solve(self, workspace, stub_policy):
        """A stub that solved tasks would make a green dry run look real."""
        trajectory = _run(StubAgentClient(), workspace, stub_policy)
        assert trajectory.solved is False
        assert trajectory.stop_condition == StopCondition.ABANDONED.value

    def test_exploring_alone_trips_the_no_progress_rule(self, workspace, policy):
        """With the default window the stub is stopped, correctly, for idling."""
        trajectory = _run(StubAgentClient(), workspace, policy)
        assert trajectory.solved is False
        assert trajectory.stop_condition == StopCondition.NO_PROGRESS.value

    def test_it_never_edits_anything(self, workspace, stub_policy):
        trajectory = _run(StubAgentClient(), workspace, stub_policy)
        assert trajectory.files_changed == {}
        assert trajectory.steps_to_first_edit is None

    def test_it_spends_no_tokens(self, workspace, stub_policy):
        """A dry run must not consume the token ceiling or look like spend."""
        trajectory = _run(StubAgentClient(), workspace, stub_policy)
        assert trajectory.total_tokens == 0
        assert trajectory.total_cost_eur == 0.0

    def test_it_reads_the_files_list_dir_reported(self, workspace, stub_policy):
        trajectory = _run(StubAgentClient(), workspace, stub_policy)
        read = [
            step.tool_args["path"]
            for step in trajectory.steps
            if step.tool_name == "read_file"
        ]
        assert set(read) <= {"Calculator.java", "CalculatorTest.java"}
        assert read

    def test_max_reads_bounds_the_exploration(self, stub_policy):
        workspace = Workspace(files={f"F{i}.java": "class F {}" for i in range(20)})
        trajectory = _run(StubAgentClient(max_reads=2), workspace, stub_policy)
        reads = [s for s in trajectory.steps if s.tool_name == "read_file"]
        assert len(reads) == 2


class TestEpisodeIsolation:
    def test_one_instance_restarts_its_plan_for_each_episode(
        self, workspace, stub_policy
    ):
        """One stub serves a whole run; episode two must not resume episode one."""
        client = StubAgentClient()
        first = _run(client, workspace, stub_policy)

        fresh = Workspace(
            files={"Calculator.java": "x", "CalculatorTest.java": "y"},
            read_only={"CalculatorTest.java"},
        )
        second = _run(client, fresh, stub_policy)

        assert second.num_steps == first.num_steps
        assert [s.tool_name for s in second.steps] == [s.tool_name for s in first.steps]

    def test_the_call_counter_survives_the_reset(self, workspace, stub_policy):
        client = StubAgentClient()
        _run(client, workspace, stub_policy)
        calls_after_one = client.calls
        _run(client, workspace, stub_policy)
        assert client.calls > calls_after_one


class TestPathParsing:
    def test_extracts_paths_from_a_listing(self):
        listing = (
            "2 file(s):\n"
            "  Calculator.java  (5 lines)\n"
            "  CalculatorTest.java  (12 lines)  [read-only]\n"
        )
        assert StubAgentClient._paths_in(listing) == [
            "Calculator.java", "CalculatorTest.java"
        ]

    def test_ignores_the_header_line(self):
        assert "2 file" not in StubAgentClient._paths_in("2 file(s):\n  A.java  (1 lines)\n")

    def test_empty_listing_yields_nothing(self):
        assert StubAgentClient._paths_in("No files in the workspace.") == []
