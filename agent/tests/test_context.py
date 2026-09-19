"""
Tests for agent.context and agent.parallel (M1).

Compaction's own failure mode is what most of these pin: an agent that
forgets what it tried and repeats it. The summary is written to prevent
that, so the tests check the summary *says* what was tried.
"""

from __future__ import annotations

import pytest

from agent.context import (
    CONTEXT_WINDOWS,
    CompactionPolicy,
    ContextManager,
    estimate_tokens,
)
from agent.parallel import ParallelRunner, RunBudget
from agent.trajectory import Trajectory
from llm_gateway.conversation import Conversation, Message, ToolCall, ToolResult


def _conversation(exchanges: int, result_size: int = 200) -> Conversation:
    conversation = Conversation(system="You are an agent.", messages=[Message.user("Fix it.")])
    for index in range(exchanges):
        conversation.append(Message(
            role="assistant",
            text=f"Step {index}.",
            tool_calls=[ToolCall(
                call_id=f"c{index}", name="read_file",
                arguments={"path": f"File{index}.java"},
            )],
        ))
        conversation.append(Message(
            role="user",
            tool_results=[ToolResult(
                call_id=f"c{index}", name="read_file", content="x" * result_size,
            )],
        ))
    return conversation


class TestTokenEstimation:
    def test_empty(self):
        assert estimate_tokens("") == 1

    def test_scales_with_length(self):
        assert estimate_tokens("x" * 1000) > estimate_tokens("x" * 100)

    def test_biased_high(self):
        """Overestimating makes compaction fire early, which is the safe side."""
        text = "public class Foo { }" * 100
        assert estimate_tokens(text) > len(text) / 4


class TestCompactionPolicy:
    def test_rejects_an_implausible_trigger(self):
        with pytest.raises(ValueError):
            CompactionPolicy(trigger_fraction=0.99)

    def test_requires_keeping_at_least_two_messages(self):
        """A tool call whose result was summarised away is unusable."""
        with pytest.raises(ValueError, match="tool call whose result is gone"):
            CompactionPolicy(keep_recent_exchanges=1)


class TestContextManager:
    def test_window_is_looked_up_per_model(self):
        manager = ContextManager("claude-sonnet-4-6")
        assert manager.window == CONTEXT_WINDOWS["claude-sonnet-4-6"]

    def test_unknown_model_gets_a_small_window(self):
        """Compacting unnecessarily costs a summary; not compacting costs the episode."""
        manager = ContextManager("some-new-model")
        assert manager.window < CONTEXT_WINDOWS["claude-sonnet-4-6"]

    def test_reserve_is_held_back_for_the_response(self):
        manager = ContextManager("gpt-4o", reserve_tokens=8000)
        assert manager.usable_tokens == manager.window - 8000

    def test_a_short_conversation_needs_no_compaction(self):
        manager = ContextManager("claude-sonnet-4-6")
        assert manager.needs_compaction(_conversation(2)) is False

    def test_a_long_conversation_triggers_compaction(self):
        manager = ContextManager("gpt-4o", reserve_tokens=1000)
        manager.window = 5000
        assert manager.needs_compaction(_conversation(40, result_size=2000)) is True

    def test_compaction_shrinks_the_conversation(self):
        manager = ContextManager("gpt-4o")
        conversation = _conversation(20, result_size=1000)
        before = manager.measure(conversation)

        manager.compact(conversation, step_index=20)

        assert manager.measure(conversation) < before
        assert manager.events
        assert manager.events[0].tokens_after < manager.events[0].tokens_before

    def test_the_task_survives_compaction(self):
        """An agent that loses the task starts solving a summary of it."""
        manager = ContextManager("gpt-4o")
        conversation = _conversation(20)
        manager.compact(conversation)
        assert conversation.messages[0].text == "Fix it."

    def test_recent_exchanges_survive_verbatim(self):
        manager = ContextManager("gpt-4o", policy=CompactionPolicy(keep_recent_exchanges=4))
        conversation = _conversation(20)
        manager.compact(conversation)
        assert conversation.messages[-1].tool_results
        assert conversation.messages[-2].tool_calls

    def test_the_summary_lists_what_was_already_tried(self):
        """Compaction's failure mode is an agent repeating itself."""
        manager = ContextManager("gpt-4o")
        conversation = _conversation(20)
        manager.compact(conversation)

        summary = next(
            message.text for message in conversation.messages
            if "removed to stay inside" in message.text
        )
        assert "Read:" in summary
        assert "File0.java" in summary
        assert "Do not repeat work listed above" in summary

    def test_summary_records_edits_and_test_runs(self):
        manager = ContextManager("gpt-4o")
        conversation = _conversation(2)
        conversation.append(Message(
            role="assistant",
            tool_calls=[ToolCall(call_id="e", name="apply_patch",
                                 arguments={"path": "Fix.java"})],
        ))
        conversation.append(Message(role="user", tool_results=[
            ToolResult(call_id="e", name="run_tests", content="2/3 tests passed.")
        ]))
        for _ in range(10):
            conversation.append(Message.user("filler"))

        manager.compact(conversation)
        summary = next(
            m.text for m in conversation.messages if "removed to stay inside" in m.text
        )
        assert "Fix.java" in summary
        assert "2/3 tests passed" in summary

    def test_compacting_a_short_conversation_is_a_no_op(self):
        manager = ContextManager("gpt-4o")
        conversation = _conversation(1)
        before = list(conversation.messages)
        manager.compact(conversation)
        assert conversation.messages == before
        assert manager.events == []

    def test_stats_report_what_was_saved(self):
        manager = ContextManager("gpt-4o")
        manager.compact(_conversation(20, result_size=1000))
        stats = manager.stats()
        assert stats["compactions"] == 1
        assert stats["tokens_saved"] > 0


class TestCacheBreakpoint:
    def test_a_short_prefix_is_not_worth_caching(self):
        """Below about a thousand tokens the cache write costs more than it saves."""
        manager = ContextManager("claude-sonnet-4-6")
        assert manager.cache_breakpoint(_conversation(1, result_size=10)) == -1

    def test_a_long_stable_prefix_gets_a_breakpoint(self):
        manager = ContextManager("claude-sonnet-4-6")
        index = manager.cache_breakpoint(_conversation(20, result_size=2000))
        assert index > 0

    def test_the_breakpoint_precedes_the_growing_tail(self):
        manager = ContextManager(
            "claude-sonnet-4-6", policy=CompactionPolicy(keep_recent_exchanges=6)
        )
        conversation = _conversation(20, result_size=2000)
        index = manager.cache_breakpoint(conversation)
        assert index < len(conversation.messages) - 6


class TestRunBudget:
    def test_rejects_a_non_positive_total(self):
        with pytest.raises(ValueError):
            RunBudget(total_eur=0.0)

    def test_reserve_and_settle(self):
        budget = RunBudget(total_eur=1.0)
        assert budget.reserve(0.3) is True
        assert budget.available_eur == pytest.approx(0.7)
        budget.settle(reserved_eur=0.3, actual_eur=0.1)
        assert budget.spent_eur == pytest.approx(0.1)
        assert budget.available_eur == pytest.approx(0.9)

    def test_reservation_prevents_double_spending(self):
        """Two workers each seeing the same headroom must not both start."""
        budget = RunBudget(total_eur=1.0)
        assert budget.reserve(0.6) is True
        assert budget.reserve(0.6) is False

    def test_unused_reservation_returns_to_the_pool(self):
        budget = RunBudget(total_eur=1.0)
        budget.reserve(0.5)
        budget.settle(reserved_eur=0.5, actual_eur=0.05)
        assert budget.reserve(0.9) is True

    def test_snapshot(self):
        budget = RunBudget(total_eur=2.0)
        budget.reserve(0.5)
        snapshot = budget.snapshot()
        assert snapshot["total_eur"] == 2.0
        assert snapshot["reserved_eur"] == 0.5


class TestParallelRunner:
    def _trajectory(self, task_id: str, cost: float = 0.01) -> Trajectory:
        return Trajectory(
            episode_id=f"ep-{task_id}", task_id=task_id,
            solved=task_id.endswith("1"), total_cost_eur=cost,
        )

    def test_runs_every_task(self):
        runner = ParallelRunner(workers=3)
        result = runner.run(["T1", "T2", "T3"], lambda task: self._trajectory(task))
        assert len(result.trajectories) == 3
        assert [t.task_id for t in result.trajectories] == ["T1", "T2", "T3"]

    def test_rejects_zero_workers(self):
        with pytest.raises(ValueError):
            ParallelRunner(workers=0)

    def test_a_failing_episode_does_not_stop_the_others(self):
        def episode(task_id: str) -> Trajectory:
            if task_id == "T2":
                raise RuntimeError("container died")
            return self._trajectory(task_id)

        result = ParallelRunner(workers=2).run(["T1", "T2", "T3"], episode)
        assert len(result.trajectories) == 2
        assert "T2" in result.failed

    def test_the_run_budget_stops_dispatch(self):
        """Without a shared budget, N workers spend N times the ceiling."""
        budget = RunBudget(total_eur=0.10)
        runner = ParallelRunner(workers=2, budget=budget, per_episode_reserve_eur=0.04)
        result = runner.run(
            [f"T{i}" for i in range(10)], lambda task: self._trajectory(task, cost=0.04)
        )
        assert len(result.trajectories) < 10
        assert result.skipped
        assert budget.spent_eur <= budget.total_eur

    def test_trajectories_are_saved_as_they_finish(self, tmp_path):
        """A run that dies at episode 40 of 50 should not lose 39 results."""
        from agent.trajectory import TrajectoryStore

        store = TrajectoryStore(tmp_path)
        ParallelRunner(workers=2, store=store).run(
            ["T1", "T2"], lambda task: self._trajectory(task)
        )
        assert sorted(store.episode_ids()) == ["ep-T1", "ep-T2"]

    def test_summary(self):
        result = ParallelRunner(workers=2).run(
            ["T1", "T2"], lambda task: self._trajectory(task)
        )
        summary = result.summary()
        assert summary["attempted"] == 2
        assert summary["solved"] == 1
        assert summary["resolve_rate"] == 0.5
