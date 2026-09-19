"""
Tests for the trajectory dashboard page (M9).

Only the pure functions: row building, failure colouring, loading. The Dash
layout and callbacks need a browser to mean anything, and a test that
asserts a component tree exists proves nothing about whether the page is
usable.

What these do protect is the part that can be quietly wrong — a running
cost that does not accumulate, a divergence marker on the wrong row, an
environment failure coloured like a model failure.
"""

from __future__ import annotations

import pytest

from agent.trajectory import Trajectory, TrajectoryStep
from framework.dashboard.pages.trajectory import (
    FAILURE_COLOURS,
    build_diff_rows,
    build_replay_rows,
    failure_distribution,
    load_trajectories,
)


def _step(index: int, tool: str = "read_file", **kwargs) -> TrajectoryStep:
    defaults = {"step_index": index, "tool_name": tool}
    defaults.update(kwargs)
    return TrajectoryStep(**defaults)


def _trajectory(**kwargs) -> Trajectory:
    defaults = {"episode_id": "ep-1", "task_id": "T1", "model_id": "m", "scaffold": "react"}
    defaults.update(kwargs)
    return Trajectory(**defaults)


class TestReplayRows:
    def test_one_row_per_step(self):
        trajectory = _trajectory(steps=[_step(1), _step(2, "run_tests")])
        assert len(build_replay_rows(trajectory)) == 2

    def test_cost_accumulates(self):
        """The running total is what makes a replay useful for cost work."""
        trajectory = _trajectory(steps=[
            _step(1, cost_eur=0.01), _step(2, cost_eur=0.02), _step(3, cost_eur=0.03),
        ])
        rows = build_replay_rows(trajectory)
        assert [row["cumulative_cost_eur"] for row in rows] == [0.01, 0.03, 0.06]

    def test_tokens_accumulate(self):
        trajectory = _trajectory(steps=[
            _step(1, tokens_in=100, tokens_out=50),
            _step(2, tokens_in=200, tokens_out=50),
        ])
        rows = build_replay_rows(trajectory)
        assert rows[-1]["cumulative_tokens"] == 400

    def test_errors_are_flagged_for_the_conditional_style(self):
        trajectory = _trajectory(steps=[_step(1, tool_error=True), _step(2)])
        rows = build_replay_rows(trajectory)
        assert rows[0]["error"] == "yes"
        assert rows[1]["error"] == ""

    def test_test_state_is_rendered_as_a_fraction(self):
        trajectory = _trajectory(steps=[
            _step(1, "run_tests", tests_passing_after=2, tests_total_after=3)
        ])
        assert build_replay_rows(trajectory)[0]["tests"] == "2/3"

    def test_a_step_with_no_test_run_shows_nothing(self):
        assert build_replay_rows(_trajectory(steps=[_step(1)]))[0]["tests"] == ""

    def test_a_step_with_no_tool_is_labelled(self):
        trajectory = _trajectory(steps=[_step(1, tool="")])
        assert build_replay_rows(trajectory)[0]["tool"] == "(no tool call)"

    def test_long_values_are_clipped(self):
        trajectory = _trajectory(steps=[
            _step(1, result_preview="x" * 5000, thought_text="y" * 5000)
        ])
        row = build_replay_rows(trajectory)[0]
        assert len(row["result"]) <= 200
        assert len(row["thought"]) <= 200

    def test_empty_trajectory(self):
        assert build_replay_rows(_trajectory()) == []


class TestDiffRows:
    def test_identical_episodes_never_diverge(self):
        steps = [_step(1, "list_dir"), _step(2, "read_file", tool_args={"path": "A.java"})]
        rows, divergence = build_diff_rows(
            _trajectory(steps=list(steps)), _trajectory(steps=list(steps))
        )
        assert divergence is None
        assert all(row["diverged"] == "" for row in rows)

    def test_the_divergence_row_is_marked(self):
        left = _trajectory(steps=[_step(1, "list_dir"), _step(2, "read_file")])
        right = _trajectory(steps=[_step(1, "list_dir"), _step(2, "grep")])
        rows, divergence = build_diff_rows(left, right)
        assert divergence == 2
        assert rows[0]["diverged"] == ""
        assert rows[1]["diverged"] == "<<<"

    def test_unequal_lengths_align(self):
        left = _trajectory(steps=[_step(1), _step(2), _step(3)])
        right = _trajectory(steps=[_step(1)])
        rows, _ = build_diff_rows(left, right)
        assert len(rows) == 3
        assert rows[2]["right"] == "—"

    def test_test_state_is_shown_for_both_sides(self):
        left = _trajectory(steps=[_step(1, "run_tests", tests_passing_after=3)])
        right = _trajectory(steps=[_step(1, "run_tests", tests_passing_after=1)])
        rows, _ = build_diff_rows(left, right)
        assert rows[0]["left_tests"] == 3
        assert rows[0]["right_tests"] == 1

    def test_absent_test_state_is_blank_not_zero(self):
        rows, _ = build_diff_rows(
            _trajectory(steps=[_step(1)]), _trajectory(steps=[_step(1)])
        )
        assert rows[0]["left_tests"] == ""


class TestFailureDistribution:
    def test_counts_by_class(self):
        trajectories = [
            _trajectory(episode_id="a", solved=True),
            _trajectory(episode_id="b", stop_condition="no_progress"),
            _trajectory(episode_id="c", stop_condition="no_progress"),
        ]
        summary = failure_distribution(trajectories)
        assert summary["distribution"]["solved"] == 1
        assert summary["distribution"]["loop"] == 2

    def test_episodes_are_drillable_from_a_class(self):
        trajectories = [
            _trajectory(episode_id="a", solved=True),
            _trajectory(episode_id="b", solved=True),
        ]
        summary = failure_distribution(trajectories)
        assert summary["episodes_by_class"]["solved"] == ["a", "b"]

    def test_every_failure_class_has_a_colour(self):
        from agent.analysis import FailureClass

        assert {c.value for c in FailureClass} <= set(FAILURE_COLOURS)

    def test_environment_failures_are_grey(self):
        """Colouring the harness's fault like a model failure misleads."""
        assert FAILURE_COLOURS["environment_failure"] == "#a0aec0"
        assert FAILURE_COLOURS["environment_failure"] != FAILURE_COLOURS["semantic_failure"]


class TestLoading:
    def test_a_missing_directory_returns_nothing(self, tmp_path):
        assert load_trajectories(tmp_path / "does-not-exist") == []

    def test_loads_and_orders_newest_first(self, tmp_path):
        from agent.trajectory import TrajectoryStore

        store = TrajectoryStore(tmp_path)
        store.save(_trajectory(episode_id="old", started_at="2026-01-01T00:00:00"))
        store.save(_trajectory(episode_id="new", started_at="2026-06-01T00:00:00"))

        loaded = load_trajectories(tmp_path)
        assert [t.episode_id for t in loaded] == ["new", "old"]


class TestLayoutImportsLazily:
    def test_the_module_imports_without_dash(self):
        """The analysis underneath must not require a web framework."""
        import framework.dashboard.pages.trajectory as page

        assert callable(page.build_replay_rows)

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("dash") is None,
        reason="dash is not installed",
    )
    def test_layout_renders_with_no_episodes(self, tmp_path):
        from framework.dashboard.pages.trajectory import layout

        rendered = layout(tmp_path / "empty")
        assert rendered is not None
