"""
Tests for agent.scaffolds (M1).

The reason these exist: a comparison across scaffolds is only valid if every
scaffold returns the same shape, scored the same way, on the same tasks.
Most of these tests pin exactly that — the interface, not the strategy.
"""

from __future__ import annotations

import pytest

from agent.loop import AgentConfig
from agent.scaffolds import (
    SCAFFOLDS,
    ExternalHarnessScaffold,
    PlanThenExecuteScaffold,
    ReActScaffold,
    SingleShotScaffold,
    get_scaffold,
)
from agent.termination import StopCondition
from agent.tests.conftest import FIXED_SOURCE, ScriptedClient, final_turn, tool_turn
from agent.trajectory import Trajectory
from llm_gateway.conversation import AssistantTurn, TokenUsage


def _config(policy, **kwargs) -> AgentConfig:
    defaults = {"model_id": "test-model", "policy": policy, "task_id": "T1"}
    defaults.update(kwargs)
    return AgentConfig(**defaults)


class TestRegistry:
    def test_known_scaffolds(self):
        assert set(SCAFFOLDS) == {"react", "plan_then_execute", "single_shot"}

    @pytest.mark.parametrize("name", ["react", "plan_then_execute", "single_shot"])
    def test_get_scaffold(self, name):
        assert get_scaffold(name).name == name

    def test_unknown_scaffold_lists_the_known_ones(self):
        with pytest.raises(KeyError, match="react"):
            get_scaffold("telepathy")

    def test_external_is_excluded_because_it_needs_a_command(self):
        with pytest.raises(KeyError):
            get_scaffold("external")


class TestReAct:
    def test_solves_and_tags_the_trajectory(self, workspace, test_runner, policy):
        client = ScriptedClient(turns=[
            tool_turn("apply_patch", {"path": "Calculator.java", "content": FIXED_SOURCE}),
            tool_turn("run_tests", {}),
        ])
        trajectory = ReActScaffold().run(
            client, workspace, test_runner, _config(policy), "Fix add()."
        )
        assert trajectory.solved is True
        assert trajectory.scaffold == "react"


class TestSingleShot:
    def test_applies_one_completion_and_scores_it(self, workspace, test_runner, policy):
        client = ScriptedClient(turns=[
            AssistantTurn(
                text=f"```java\n{FIXED_SOURCE}```",
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=500, output_tokens=200),
            )
        ])
        trajectory = SingleShotScaffold().run(
            client, workspace, test_runner, _config(policy), "Fix add()."
        )
        assert trajectory.solved is True
        assert trajectory.scaffold == "single_shot"
        assert trajectory.num_steps == 1

    def test_gets_exactly_one_chance(self, workspace, test_runner, policy):
        """No second attempt is the whole point of the baseline."""
        client = ScriptedClient(turns=[
            AssistantTurn(
                text="```java\npublic class Calculator { }```",
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=100, output_tokens=50),
            ),
            AssistantTurn(text=f"```java\n{FIXED_SOURCE}```", stop_reason="end_turn"),
        ])
        trajectory = SingleShotScaffold().run(
            client, workspace, test_runner, _config(policy), "Fix add()."
        )
        assert trajectory.solved is False
        assert len(client.calls) == 1

    def test_is_offered_no_tools(self, workspace, test_runner, policy):
        client = ScriptedClient(turns=[final_turn("no code")])
        SingleShotScaffold().run(
            client, workspace, test_runner, _config(policy), "Fix it."
        )
        assert client.calls[0]["tools"] == []

    def test_a_provider_failure_is_recorded(self, workspace, test_runner, policy):
        client = ScriptedClient(turns=[], raise_on_call=RuntimeError("503"))
        trajectory = SingleShotScaffold().run(
            client, workspace, test_runner, _config(policy), "Fix it."
        )
        assert trajectory.stop_condition == StopCondition.ERROR.value
        assert "503" in trajectory.error

    def test_a_baseline_is_still_measured(self, workspace, test_runner, policy):
        client = ScriptedClient(turns=[final_turn("nothing")])
        trajectory = SingleShotScaffold().run(
            client, workspace, test_runner, _config(policy), "Fix it."
        )
        assert trajectory.baseline_passed == 0
        assert trajectory.baseline_total == 2

    def test_unexecuted_tests_are_never_a_solve(self, workspace, policy):
        def never_executes(_workspace):
            from agent.tools import ToolOutcome

            return ToolOutcome(
                content="EXECUTION SKIPPED", tests_passed=2, tests_total=2,
                compiled=True, executed=False,
            )

        client = ScriptedClient(turns=[
            AssistantTurn(text=f"```java\n{FIXED_SOURCE}```", stop_reason="end_turn")
        ])
        trajectory = SingleShotScaffold().run(
            client, workspace, never_executes, _config(policy), "Fix it."
        )
        assert trajectory.solved is False
        assert trajectory.tests_executed is False


class TestPlanThenExecute:
    def test_plans_before_acting(self, workspace, test_runner, policy):
        client = ScriptedClient(turns=[
            AssistantTurn(text="1. Read Calculator.java\n2. Fix the operator",
                          stop_reason="end_turn",
                          usage=TokenUsage(input_tokens=200, output_tokens=60)),
            tool_turn("apply_patch", {"path": "Calculator.java", "content": FIXED_SOURCE}),
            tool_turn("run_tests", {}),
        ])
        trajectory = PlanThenExecuteScaffold().run(
            client, workspace, test_runner, _config(policy), "Fix add()."
        )
        assert trajectory.solved is True
        assert trajectory.scaffold == "plan_then_execute"
        assert trajectory.steps[0].step_index == 0
        assert "Fix the operator" in trajectory.steps[0].thought_text

    def test_the_planning_turn_gets_no_tools(self, workspace, test_runner, policy):
        """A planning turn that could act is not a planning turn."""
        client = ScriptedClient(turns=[
            AssistantTurn(text="A plan.", stop_reason="end_turn"),
            final_turn(),
        ])
        PlanThenExecuteScaffold().run(
            client, workspace, test_runner, _config(policy), "Fix it."
        )
        assert client.calls[0]["tools"] == []
        assert client.calls[1]["tools"]

    def test_the_plan_reaches_the_execution_prompt(self, workspace, test_runner, policy):
        client = ScriptedClient(turns=[
            AssistantTurn(text="PLAN-MARKER-123", stop_reason="end_turn"),
            final_turn(),
        ])
        PlanThenExecuteScaffold().run(
            client, workspace, test_runner, _config(policy), "Fix it."
        )
        assert len(client.calls) == 2

    def test_planning_cost_counts_against_the_budget(self, workspace, test_runner, policy):
        """Otherwise the ceiling is exceeded by exactly one call, every time."""
        client = ScriptedClient(turns=[
            AssistantTurn(text="A plan.", stop_reason="end_turn",
                          usage=TokenUsage(input_tokens=1000, output_tokens=500)),
            final_turn(),
        ])
        trajectory = PlanThenExecuteScaffold().run(
            client, workspace, test_runner, _config(policy), "Fix it."
        )
        assert trajectory.total_tokens >= 1500

    def test_a_failed_planning_call_does_not_lose_the_episode(
        self, workspace, test_runner, policy
    ):
        class FailingPlanner(ScriptedClient):
            def send_conversation(self, conversation, tools, model_id, max_tokens, temperature):
                if not tools:  # the planning call
                    raise RuntimeError("planner unavailable")
                return super().send_conversation(
                    conversation, tools, model_id, max_tokens, temperature
                )

        client = FailingPlanner(turns=[
            tool_turn("apply_patch", {"path": "Calculator.java", "content": FIXED_SOURCE}),
            tool_turn("run_tests", {}),
        ])
        trajectory = PlanThenExecuteScaffold().run(
            client, workspace, test_runner, _config(policy), "Fix add()."
        )
        assert trajectory.solved is True


class TestExternalHarness:
    def test_a_missing_binary_is_recorded_not_raised(self, workspace, test_runner, policy):
        scaffold = ExternalHarnessScaffold(command=["definitely-not-a-real-binary"])
        trajectory = scaffold.run(
            None, workspace, test_runner, _config(policy), "Fix it."
        )
        assert trajectory.stop_condition == StopCondition.ERROR.value
        assert "not found" in trajectory.error

    def test_records_no_steps_rather_than_inventing_them(
        self, workspace, test_runner, policy
    ):
        """An external harness does not expose its trajectory; faking one is worse."""
        scaffold = ExternalHarnessScaffold(command=["definitely-not-a-real-binary"])
        trajectory = scaffold.run(
            None, workspace, test_runner, _config(policy), "Fix it."
        )
        assert trajectory.steps == []
        assert trajectory.scaffold == "external"

    def test_cost_defaults_to_visibly_zero(self, workspace, test_runner, policy):
        """Visibly wrong beats plausibly wrong when the number is unknowable."""
        scaffold = ExternalHarnessScaffold(command=["true"])
        assert scaffold.cost_eur == 0.0


class TestInterfaceParity:
    """Every scaffold must return the same shape, or comparison is invalid."""

    @pytest.mark.parametrize("scaffold_name", ["react", "plan_then_execute", "single_shot"])
    def test_returns_a_trajectory_with_the_outcome_fields(
        self, scaffold_name, workspace, test_runner, policy
    ):
        client = ScriptedClient(turns=[
            AssistantTurn(text=f"```java\n{FIXED_SOURCE}```", stop_reason="end_turn"),
            tool_turn("apply_patch", {"path": "Calculator.java", "content": FIXED_SOURCE}),
            tool_turn("run_tests", {}),
            final_turn(),
        ])
        trajectory = get_scaffold(scaffold_name).run(
            client, workspace, test_runner, _config(policy), "Fix add()."
        )
        assert isinstance(trajectory, Trajectory)
        assert trajectory.scaffold == scaffold_name
        assert trajectory.stop_condition
        assert isinstance(trajectory.solved, bool)
        assert trajectory.baseline_total == 2
        assert trajectory.final_workspace
