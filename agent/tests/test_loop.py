"""
End-to-end tests for agent.loop.

The model is a scripted stub, so the whole loop runs offline. The loop
cannot tell a scripted turn from a real one — both arrive as the same
neutral ``AssistantTurn`` — which is exactly what makes this worth testing
this way.
"""

from __future__ import annotations

import pytest

from agent.loop import AgentConfig, AgentLoop
from agent.termination import StopCondition, TerminationPolicy
from agent.tests.conftest import (
    FIXED_SOURCE,
    ScriptedClient,
    ScriptedTestRunner,
    final_turn,
    tool_turn,
)
from agent.trajectory import Trajectory, TrajectoryStore, replay_scoring
from agent.workspace import Workspace
from llm_gateway.conversation import AssistantTurn, TokenUsage, ToolCall


def _config(policy, **kwargs) -> AgentConfig:
    defaults = {
        "model_id": "test-model",
        "policy": policy,
        "task_id": "TASK_1",
        "dataset": "test-dataset",
    }
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def _loop(client, workspace, runner, policy, **kwargs) -> AgentLoop:
    return AgentLoop(
        client=client,
        workspace=workspace,
        test_runner=runner,
        config=_config(policy, **kwargs),
        cost_of=lambda turn: 0.0,
    )


class TestSolvingAnEpisode:
    def test_a_correct_fix_ends_in_success(self, workspace, test_runner, policy):
        client = ScriptedClient(turns=[
            tool_turn("list_dir", {}, text="Let me look around."),
            tool_turn("read_file", {"path": "Calculator.java"}),
            tool_turn("apply_patch", {"path": "Calculator.java", "content": FIXED_SOURCE},
                      text="The operator is wrong."),
            tool_turn("run_tests", {}),
        ])
        trajectory = _loop(client, workspace, test_runner, policy).run("Fix add().")

        assert trajectory.solved is True
        assert trajectory.stop_condition == StopCondition.SUCCESS.value
        assert trajectory.final_passed == 2
        assert workspace.read("Calculator.java") == FIXED_SOURCE

    def test_the_baseline_is_measured_before_the_agent_acts(
        self, workspace, test_runner, policy
    ):
        """Without it, 'fixed the target but broke two others' scores as success."""
        client = ScriptedClient(turns=[final_turn()])
        trajectory = _loop(client, workspace, test_runner, policy).run("Fix it.")
        assert trajectory.baseline_passed == 0
        assert trajectory.baseline_total == 2

    def test_files_changed_is_recorded(self, workspace, test_runner, policy):
        client = ScriptedClient(turns=[
            tool_turn("apply_patch", {"path": "Calculator.java", "content": FIXED_SOURCE}),
            tool_turn("run_tests", {}),
        ])
        trajectory = _loop(client, workspace, test_runner, policy).run("Fix it.")
        assert trajectory.files_changed == {"Calculator.java": "modified"}


class TestTerminationInTheLoop:
    def test_max_steps_stops_the_episode(self, workspace, test_runner):
        policy = TerminationPolicy(max_cost_eur=1.0, max_steps=3, no_progress_steps=99)
        client = ScriptedClient(turns=[
            tool_turn("read_file", {"path": "Calculator.java"}),
            tool_turn("read_file", {"path": "CalculatorTest.java"}),
            tool_turn("list_dir", {}),
            tool_turn("grep", {"pattern": "add"}),
        ])
        trajectory = _loop(client, workspace, test_runner, policy).run("Fix it.")
        assert trajectory.stop_condition == StopCondition.MAX_STEPS.value
        assert trajectory.num_steps == 3
        assert trajectory.solved is False

    def test_no_progress_stops_a_loop_going_in_circles(self, workspace, test_runner):
        """Re-reading the same file forever is a named failure, not a wait."""
        policy = TerminationPolicy(max_cost_eur=1.0, max_steps=50, no_progress_steps=3)
        client = ScriptedClient(turns=[
            tool_turn("read_file", {"path": "Calculator.java"}) for _ in range(10)
        ])
        trajectory = _loop(client, workspace, test_runner, policy).run("Fix it.")
        assert trajectory.stop_condition == StopCondition.NO_PROGRESS.value
        assert trajectory.num_steps == 3

    def test_no_op_writes_do_not_count_as_progress(self, workspace, test_runner):
        """Rewriting identical bytes costs a step; it must not reset the counter."""
        original = workspace.read("Calculator.java")
        policy = TerminationPolicy(max_cost_eur=1.0, max_steps=50, no_progress_steps=3)
        client = ScriptedClient(turns=[
            tool_turn("apply_patch", {"path": "Calculator.java", "content": original})
            for _ in range(10)
        ])
        trajectory = _loop(client, workspace, test_runner, policy).run("Fix it.")
        assert trajectory.stop_condition == StopCondition.NO_PROGRESS.value

    def test_max_tokens_stops_the_episode(self, workspace, test_runner):
        policy = TerminationPolicy(max_cost_eur=1.0, max_steps=50, max_tokens=200,
                                   no_progress_steps=99)
        client = ScriptedClient(turns=[
            tool_turn("list_dir", {}, input=500, output=500) for _ in range(10)
        ])
        trajectory = _loop(client, workspace, test_runner, policy).run("Fix it.")
        assert trajectory.stop_condition == StopCondition.MAX_TOKENS.value

    def test_the_cost_ceiling_stops_the_episode_before_overspending(
        self, workspace, test_runner
    ):
        policy = TerminationPolicy(max_cost_eur=0.10, max_steps=50, no_progress_steps=99)
        client = ScriptedClient(turns=[tool_turn("list_dir", {}) for _ in range(20)])
        loop = AgentLoop(
            client=client,
            workspace=workspace,
            test_runner=test_runner,
            config=_config(policy, worst_case_call_eur=0.04),
            cost_of=lambda turn: 0.04,
        )
        trajectory = loop.run("Fix it.")
        assert trajectory.stop_condition == StopCondition.MAX_COST.value
        assert trajectory.total_cost_eur <= policy.max_cost_eur

    def test_abandonment_when_the_model_stops_without_solving(
        self, workspace, test_runner, policy
    ):
        client = ScriptedClient(turns=[final_turn("I give up.")])
        trajectory = _loop(client, workspace, test_runner, policy).run("Fix it.")
        assert trajectory.stop_condition == StopCondition.ABANDONED.value
        assert trajectory.solved is False

    def test_a_provider_failure_is_an_error_not_a_crash(
        self, workspace, test_runner, policy
    ):
        """An exception escaping here would lose the steps that led to it."""
        client = ScriptedClient(turns=[], raise_on_call=RuntimeError("503 upstream"))
        trajectory = _loop(client, workspace, test_runner, policy).run("Fix it.")
        assert trajectory.stop_condition == StopCondition.ERROR.value
        assert "503 upstream" in trajectory.error
        assert trajectory.solved is False


class TestMalformedCalls:
    def _client_with_a_broken_call(self):
        return ScriptedClient(turns=[
            AssistantTurn(
                text="calling",
                tool_calls=[ToolCall(
                    call_id="c1", name="read_file",
                    malformed=True, raw_arguments='{"path": "A.java',
                )],
                stop_reason="tool_use",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            ),
            tool_turn("apply_patch", {"path": "Calculator.java", "content": FIXED_SOURCE}),
            tool_turn("run_tests", {}),
        ])

    def test_reprompt_costs_one_step_not_the_episode(
        self, workspace, test_runner, policy
    ):
        trajectory = _loop(
            self._client_with_a_broken_call(), workspace, test_runner, policy,
            malformed_call_policy="reprompt",
        ).run("Fix it.")
        assert trajectory.solved is True
        assert trajectory.steps[0].malformed_call is True
        assert trajectory.steps[0].tool_error is True

    def test_the_malformed_rate_is_measurable(self, workspace, test_runner, policy):
        trajectory = _loop(
            self._client_with_a_broken_call(), workspace, test_runner, policy
        ).run("Fix it.")
        assert trajectory.malformed_call_rate == pytest.approx(1 / 3)

    def test_abort_policy_ends_the_episode(self, workspace, test_runner, policy):
        trajectory = _loop(
            self._client_with_a_broken_call(), workspace, test_runner, policy,
            malformed_call_policy="abort",
        ).run("Fix it.")
        assert trajectory.stop_condition == StopCondition.ERROR.value
        assert "unparseable" in trajectory.error

    def test_fail_step_policy_discards_the_step(self, workspace, test_runner, policy):
        trajectory = _loop(
            self._client_with_a_broken_call(), workspace, test_runner, policy,
            malformed_call_policy="fail_step",
        ).run("Fix it.")
        assert "discarded" in trajectory.steps[0].result_preview


class TestTrajectoryRecording:
    @pytest.fixture
    def solved(self, workspace, test_runner, policy) -> Trajectory:
        client = ScriptedClient(turns=[
            tool_turn("list_dir", {}, text="Looking around.", input=100, output=30, cached=900),
            tool_turn("read_file", {"path": "Calculator.java"}),
            tool_turn("apply_patch", {"path": "Calculator.java", "content": FIXED_SOURCE}),
            tool_turn("run_tests", {}),
        ])
        return _loop(client, workspace, test_runner, policy).run("Fix add().")

    def test_one_row_per_tool_call(self, solved):
        assert solved.num_steps == 4
        assert [step.tool_name for step in solved.steps] == [
            "list_dir", "read_file", "apply_patch", "run_tests"
        ]

    def test_thought_text_is_captured(self, solved):
        assert solved.steps[0].thought_text == "Looking around."

    def test_results_are_hashed_not_stored_whole(self, solved):
        assert all(len(step.tool_result_hash) == 64 for step in solved.steps)
        assert all(len(step.result_preview) <= 400 for step in solved.steps)

    def test_cache_tokens_are_recorded_separately(self, solved):
        assert solved.steps[0].cache_read_tokens == 900
        assert solved.steps[0].tokens_in == 100
        assert solved.cache_hit_rate > 0

    def test_files_touched_is_attributed_to_the_right_step(self, solved):
        edit_step = next(s for s in solved.steps if s.tool_name == "apply_patch")
        assert edit_step.files_touched == ["Calculator.java"]
        read_step = next(s for s in solved.steps if s.tool_name == "read_file")
        assert read_step.files_touched == []

    def test_test_state_is_recorded_after_the_run(self, solved):
        run_step = next(s for s in solved.steps if s.tool_name == "run_tests")
        assert run_step.tests_passing_after == 2
        assert run_step.tests_total_after == 2

    def test_steps_to_first_edit(self, solved):
        assert solved.steps_to_first_edit == 3

    def test_tool_call_distribution(self, solved):
        assert solved.tool_call_distribution == {
            "apply_patch": 1, "list_dir": 1, "read_file": 1, "run_tests": 1
        }

    def test_the_summary_row_is_flat(self, solved):
        summary = solved.summary()
        assert summary["solved"] is True
        assert summary["task_id"] == "TASK_1"
        assert all(not isinstance(value, (dict, list)) for value in summary.values())

    def test_repeated_calls_are_counted(self, workspace, test_runner):
        policy = TerminationPolicy(max_cost_eur=1.0, max_steps=4, no_progress_steps=99)
        client = ScriptedClient(turns=[
            tool_turn("read_file", {"path": "Calculator.java"}) for _ in range(4)
        ])
        trajectory = _loop(client, workspace, test_runner, policy).run("Fix it.")
        assert trajectory.repeated_calls == 3

    def test_no_edit_leaves_steps_to_first_edit_none(
        self, workspace, test_runner, policy
    ):
        client = ScriptedClient(turns=[tool_turn("list_dir", {}), final_turn()])
        trajectory = _loop(client, workspace, test_runner, policy).run("Fix it.")
        assert trajectory.steps_to_first_edit is None


class TestMultipleToolCallsInOneTurn:
    def test_each_call_becomes_its_own_step(self, workspace, test_runner, policy):
        client = ScriptedClient(turns=[
            AssistantTurn(
                text="Reading both files.",
                tool_calls=[
                    ToolCall(call_id="a", name="read_file", arguments={"path": "Calculator.java"}),
                    ToolCall(call_id="b", name="read_file", arguments={"path": "CalculatorTest.java"}),
                ],
                stop_reason="tool_use",
                usage=TokenUsage(input_tokens=100, output_tokens=50),
            ),
            final_turn(),
        ])
        trajectory = _loop(client, workspace, test_runner, policy).run("Fix it.")
        assert trajectory.num_steps == 3  # two tool steps plus the final text turn

    def test_the_call_cost_is_attributed_once(self, workspace, test_runner, policy):
        """A turn requesting three tools made one API call, not three."""
        client = ScriptedClient(turns=[
            AssistantTurn(
                tool_calls=[
                    ToolCall(call_id=str(i), name="list_dir", arguments={})
                    for i in range(3)
                ],
                stop_reason="tool_use",
                usage=TokenUsage(input_tokens=300, output_tokens=60),
            ),
            final_turn(),
        ])
        loop = AgentLoop(
            client=client, workspace=workspace, test_runner=test_runner,
            config=_config(policy), cost_of=lambda turn: 0.01,
        )
        trajectory = loop.run("Fix it.")
        tool_steps = [s for s in trajectory.steps if s.tool_name == "list_dir"]
        assert [step.cost_eur for step in tool_steps] == [0.01, 0.0, 0.0]
        assert [step.tokens_in for step in tool_steps] == [300, 0, 0]


class TestConversationConstruction:
    def test_tools_are_offered_on_every_call(self, workspace, test_runner, policy):
        client = ScriptedClient(turns=[tool_turn("list_dir", {}), final_turn()])
        _loop(client, workspace, test_runner, policy).run("Fix it.")
        for call in client.calls:
            assert set(call["tools"]) == {
                "read_file", "list_dir", "grep", "apply_patch", "run_tests"
            }

    def test_the_conversation_grows_with_each_exchange(
        self, workspace, test_runner, policy
    ):
        client = ScriptedClient(turns=[
            tool_turn("list_dir", {}), tool_turn("list_dir", {}), final_turn()
        ])
        _loop(client, workspace, test_runner, policy).run("Fix it.")
        lengths = [call["messages"] for call in client.calls]
        assert lengths == sorted(lengths)
        assert lengths[0] == 1  # just the task
        assert lengths[-1] > lengths[0]


class TestPersistenceAndReplay:
    def test_round_trips_through_the_store(
        self, workspace, test_runner, policy, tmp_path
    ):
        client = ScriptedClient(turns=[
            tool_turn("apply_patch", {"path": "Calculator.java", "content": FIXED_SOURCE}),
            tool_turn("run_tests", {}),
        ])
        trajectory = _loop(client, workspace, test_runner, policy).run("Fix it.")

        store = TrajectoryStore(tmp_path)
        store.save(trajectory)
        loaded = store.load(trajectory.episode_id)

        assert loaded is not None
        assert loaded.episode_id == trajectory.episode_id
        assert loaded.solved == trajectory.solved
        assert loaded.num_steps == trajectory.num_steps

    def test_the_index_gets_a_summary_row(
        self, workspace, test_runner, policy, tmp_path
    ):
        client = ScriptedClient(turns=[final_turn()])
        store = TrajectoryStore(tmp_path)
        store.save(_loop(client, workspace, test_runner, policy).run("Fix it."))
        assert (tmp_path / "index.jsonl").read_text(encoding="utf-8").strip()

    def test_unknown_episode_is_none(self, tmp_path):
        assert TrajectoryStore(tmp_path).load("ep-nope") is None

    @pytest.mark.parametrize("episode_id", ["../escape", "a/b", ".."])
    def test_traversal_in_an_episode_id_is_refused(self, tmp_path, episode_id):
        store = TrajectoryStore(tmp_path)
        assert store.load(episode_id) is None
        with pytest.raises(ValueError):
            store.save(Trajectory(episode_id=episode_id))

    def test_replay_rescores_without_calling_any_api(
        self, workspace, test_runner, policy
    ):
        """The point: changing a scoring rule costs a CPU second, not a run."""
        client = ScriptedClient(turns=[
            tool_turn("apply_patch", {"path": "Calculator.java", "content": FIXED_SOURCE}),
            tool_turn("run_tests", {}),
        ])
        trajectory = _loop(client, workspace, test_runner, policy).run("Fix it.")
        calls_before = len(client.calls)

        replayed = replay_scoring(
            trajectory,
            score=lambda files: {"has_fix": "a + b" in files["Calculator.java"]},
        )
        assert replayed["score"]["has_fix"] is True
        assert len(client.calls) == calls_before  # nothing was called

    def test_replay_refuses_a_trajectory_with_no_workspace(self):
        with pytest.raises(ValueError, match="cannot be replayed"):
            replay_scoring(Trajectory(episode_id="ep-old"), score=lambda files: files)

    def test_the_final_workspace_is_stored(self, workspace, test_runner, policy):
        client = ScriptedClient(turns=[
            tool_turn("apply_patch", {"path": "Calculator.java", "content": FIXED_SOURCE}),
            tool_turn("run_tests", {}),
        ])
        trajectory = _loop(client, workspace, test_runner, policy).run("Fix it.")
        assert trajectory.final_workspace["Calculator.java"] == FIXED_SOURCE


class TestUnexecutedTestsAreNeverASolve:
    def test_an_unexecuted_pass_does_not_count(self, workspace, policy):
        """A structural check reports every @Test as passing. It is not a solve."""
        def never_executes(_workspace):
            from agent.tools import ToolOutcome

            return ToolOutcome(
                content="EXECUTION SKIPPED — structural check only.",
                tests_passed=2, tests_total=2, compiled=True, executed=False,
            )

        client = ScriptedClient(turns=[
            tool_turn("run_tests", {}),
            final_turn("Looks done."),
        ])
        trajectory = _loop(client, workspace, never_executes, policy).run("Fix it.")
        assert trajectory.solved is False
        assert trajectory.tests_executed is False
        assert trajectory.stop_condition == StopCondition.ABANDONED.value


class TestRegressionIsNotSuccess:
    def test_breaking_other_tests_does_not_count_as_solved(self, policy):
        """Fixing the target by breaking something else is not a solution."""
        workspace = Workspace(
            files={"A.java": "good", "ATest.java": "test"},
            read_only={"ATest.java"},
        )
        runner = ScriptedTestRunner(passing_marker="good", total=5, source_path="A.java")
        # Baseline: "good" is present, so 5/5 pass. The agent then breaks it.
        client = ScriptedClient(turns=[
            tool_turn("apply_patch", {"path": "A.java", "content": "broken"}),
            tool_turn("run_tests", {}),
            final_turn("Done!"),
        ])
        trajectory = _loop(client, workspace, runner, policy).run("Improve it.")
        assert trajectory.solved is False
        assert trajectory.regressed is True
        assert trajectory.final_passed == 0
