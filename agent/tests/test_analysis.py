"""
Tests for agent.analysis (M4).

The taxonomy's value is that it is reproducible. Every test here pins a rule
to an observable fact about the trajectory, because the moment a class is
assigned by judgement it stops being a measurement.
"""

from __future__ import annotations

import pytest

from agent.analysis import (
    FailureClass,
    aggregate_failures,
    classify_failure,
    compute_localisation,
    compute_process_metrics,
    diff_trajectories,
)
from agent.trajectory import Trajectory, TrajectoryStep


def _step(index: int, tool: str = "read_file", **kwargs) -> TrajectoryStep:
    defaults = {
        "step_index": index,
        "tool_name": tool,
        "tool_args": kwargs.pop("args", {}),
    }
    defaults.update(kwargs)
    return TrajectoryStep(**defaults)


def _trajectory(**kwargs) -> Trajectory:
    defaults = {
        "episode_id": "ep-test",
        "task_id": "T1",
        "model_id": "m",
        "tests_executed": True,
    }
    defaults.update(kwargs)
    return Trajectory(**defaults)


class TestFailureTaxonomy:
    def test_solved(self):
        verdict = classify_failure(_trajectory(solved=True))
        assert verdict.failure_class is FailureClass.SOLVED
        assert verdict.confidence == "rule"

    def test_tampering_pre_empts_a_solve(self):
        """A solve that cheated is not a solve, and every later rule mis-reads it."""
        verdict = classify_failure(_trajectory(solved=True), tampered=True)
        assert verdict.failure_class is FailureClass.TAMPER

    def test_regression(self):
        verdict = classify_failure(_trajectory(
            regressed=True, baseline_passed=5, final_passed=2,
        ))
        assert verdict.failure_class is FailureClass.REGRESSION
        assert "5" in verdict.evidence

    def test_unexecuted_tests_are_an_environment_failure(self):
        verdict = classify_failure(_trajectory(tests_executed=False))
        assert verdict.failure_class is FailureClass.ENVIRONMENT

    def test_provider_error_is_an_environment_failure(self):
        verdict = classify_failure(_trajectory(stop_condition="error", error="503"))
        assert verdict.failure_class is FailureClass.ENVIRONMENT

    def test_environment_marker_in_output(self):
        verdict = classify_failure(_trajectory(
            stop_condition="abandoned",
            steps=[_step(1, "run_tests", result_preview="EXECUTION SKIPPED — no docker")],
        ))
        assert verdict.failure_class is FailureClass.ENVIRONMENT

    def test_localisation_failure(self):
        trajectory = _trajectory(
            stop_condition="max_steps",
            steps=[_step(1, "read_file", args={"path": "Wrong.java"})],
        )
        verdict = classify_failure(trajectory, golden_files=["Right.java"])
        assert verdict.failure_class is FailureClass.LOCALISATION
        assert "Right.java" in verdict.evidence

    def test_opening_the_right_file_avoids_localisation_failure(self):
        trajectory = _trajectory(
            stop_condition="max_steps",
            steps=[_step(1, "read_file", args={"path": "Right.java"})],
        )
        verdict = classify_failure(trajectory, golden_files=["Right.java"])
        assert verdict.failure_class is not FailureClass.LOCALISATION

    def test_without_golden_files_the_localisation_rule_is_skipped(self):
        """Unanswerable is not the same as false, so the rule does not fire."""
        trajectory = _trajectory(stop_condition="max_steps", steps=[_step(1)])
        verdict = classify_failure(trajectory)
        assert verdict.failure_class is FailureClass.BUDGET

    def test_loop(self):
        verdict = classify_failure(_trajectory(stop_condition="no_progress"))
        assert verdict.failure_class is FailureClass.LOOP

    def test_syntactic_failure_beats_semantic(self):
        """A patch that does not compile has not been given the chance to be wrong."""
        verdict = classify_failure(_trajectory(
            stop_condition="max_steps",
            steps=[_step(1, "run_tests", result_preview="COMPILE FAILED — error: cannot find symbol")],
        ))
        assert verdict.failure_class is FailureClass.SYNTACTIC

    @pytest.mark.parametrize("condition", ["max_steps", "max_tokens", "max_cost_eur", "wall_clock"])
    def test_budget_conditions(self, condition):
        verdict = classify_failure(_trajectory(stop_condition=condition))
        assert verdict.failure_class is FailureClass.BUDGET

    def test_abandonment_without_an_edit(self):
        verdict = classify_failure(_trajectory(
            stop_condition="abandoned", steps=[_step(1, "read_file")],
        ))
        assert verdict.failure_class is FailureClass.ABANDONMENT

    def test_abandonment_after_an_edit_is_semantic(self):
        """It attempted and failed, which is a different thing from giving up."""
        verdict = classify_failure(_trajectory(
            stop_condition="abandoned",
            final_passed=1, final_total=3,
            steps=[_step(1, "apply_patch", files_touched=["A.java"])],
        ))
        assert verdict.failure_class is FailureClass.SEMANTIC

    def test_unrecognised_stop_condition_is_ambiguous_not_guessed(self):
        verdict = classify_failure(_trajectory(stop_condition="something_new"))
        assert verdict.failure_class is FailureClass.UNKNOWN
        assert verdict.confidence == "ambiguous"

    def test_every_class_has_a_description(self):
        for failure_class in FailureClass:
            assert failure_class.description


class TestAggregation:
    def test_distribution_and_rule_coverage(self):
        verdicts = [
            classify_failure(_trajectory(solved=True)),
            classify_failure(_trajectory(stop_condition="max_steps")),
            classify_failure(_trajectory(stop_condition="mystery")),
        ]
        summary = aggregate_failures(verdicts)
        assert summary["episodes"] == 3
        assert summary["distribution"]["solved"] == 1
        assert summary["ambiguous"] == 1
        assert summary["rule_coverage"] == pytest.approx(2 / 3, abs=1e-4)

    def test_empty(self):
        assert aggregate_failures([])["episodes"] == 0


class TestLocalisation:
    def test_recall_and_first_touch_step(self):
        trajectory = _trajectory(steps=[
            _step(1, "read_file", args={"path": "Other.java"}),
            _step(2, "read_file", args={"path": "Target.java"}),
            _step(3, "apply_patch", args={"path": "Target.java"}, files_touched=["Target.java"]),
        ])
        accuracy = compute_localisation(trajectory, ["Target.java"])
        assert accuracy.found is True
        assert accuracy.recall == 1.0
        assert accuracy.step_first_golden_opened == 2
        assert accuracy.step_first_golden_edited == 3

    def test_precision_falls_when_wandering(self):
        trajectory = _trajectory(steps=[
            _step(i, "read_file", args={"path": f"F{i}.java"}) for i in range(1, 5)
        ] + [_step(5, "read_file", args={"path": "Target.java"})])
        accuracy = compute_localisation(trajectory, ["Target.java"])
        assert accuracy.recall == 1.0
        assert accuracy.precision == pytest.approx(0.2)

    def test_never_found(self):
        trajectory = _trajectory(steps=[_step(1, "read_file", args={"path": "Wrong.java"})])
        accuracy = compute_localisation(trajectory, ["Target.java"])
        assert accuracy.found is False
        assert accuracy.recall == 0.0
        assert accuracy.step_first_golden_opened is None

    def test_edit_precision_catches_collateral_edits(self):
        trajectory = _trajectory(steps=[
            _step(1, "apply_patch", files_touched=["Target.java"]),
            _step(2, "apply_patch", files_touched=["Unrelated.java"]),
        ])
        accuracy = compute_localisation(trajectory, ["Target.java"])
        assert accuracy.edit_precision == pytest.approx(0.5)


class TestProcessMetrics:
    def test_counts_by_tool(self):
        trajectory = _trajectory(steps=[
            _step(1, "read_file"), _step(2, "read_file"),
            _step(3, "apply_patch", files_touched=["A.java"]),
            _step(4, "run_tests", tests_passing_after=2, tests_total_after=2),
        ])
        metrics = compute_process_metrics(trajectory)
        assert metrics.read_steps == 2
        assert metrics.edit_steps == 1
        assert metrics.test_runs == 1
        assert metrics.steps_to_first_edit == 3

    def test_backtracking_is_measured(self):
        """Oscillating is different from idling, and the idle counter misses it."""
        trajectory = _trajectory(steps=[
            _step(1, "run_tests", tests_passing_after=2),
            _step(2, "run_tests", tests_passing_after=1),
            _step(3, "run_tests", tests_passing_after=3),
        ])
        metrics = compute_process_metrics(trajectory)
        assert metrics.backtracking_rate == pytest.approx(0.5)

    def test_dead_end_depth(self):
        trajectory = _trajectory(steps=[
            _step(1, "read_file"), _step(2, "read_file"), _step(3, "read_file"),
            _step(4, "apply_patch", files_touched=["A.java"]),
            _step(5, "read_file"),
        ])
        assert compute_process_metrics(trajectory).dead_end_depth == 3

    def test_cost_to_solution_only_when_solved(self):
        unsolved = compute_process_metrics(_trajectory(steps=[_step(1)], total_cost_eur=0.5))
        assert unsolved.cost_to_solution_eur is None
        solved = compute_process_metrics(
            _trajectory(steps=[_step(1)], solved=True, total_cost_eur=0.5)
        )
        assert solved.cost_to_solution_eur == pytest.approx(0.5)

    def test_empty_trajectory(self):
        metrics = compute_process_metrics(_trajectory())
        assert metrics.steps == 0
        assert metrics.to_dict()["steps"] == 0


class TestTrajectoryDiff:
    def test_identical_trajectories_do_not_diverge(self):
        steps = [_step(1, "list_dir"), _step(2, "read_file", args={"path": "A.java"})]
        left = _trajectory(model_id="left", steps=list(steps))
        right = _trajectory(model_id="right", steps=list(steps))
        diff = diff_trajectories(left, right)
        assert diff.divergence_step is None
        assert all(row["same"] for row in diff.aligned)

    def test_divergence_is_located(self):
        left = _trajectory(model_id="left", steps=[
            _step(1, "list_dir"), _step(2, "read_file", args={"path": "A.java"}),
        ])
        right = _trajectory(model_id="right", steps=[
            _step(1, "list_dir"), _step(2, "grep", args={"pattern": "x"}),
        ])
        diff = diff_trajectories(left, right)
        assert diff.divergence_step == 2

    def test_different_lengths_align(self):
        left = _trajectory(steps=[_step(1, "list_dir"), _step(2, "read_file")])
        right = _trajectory(steps=[_step(1, "list_dir")])
        diff = diff_trajectories(left, right)
        assert len(diff.aligned) == 2
        assert diff.aligned[1]["right"] == "—"

    def test_renders_side_by_side(self):
        left = _trajectory(model_id="claude", steps=[_step(1, "list_dir")])
        right = _trajectory(model_id="gpt-4o", steps=[_step(1, "grep")])
        text = diff_trajectories(left, right).render()
        assert "claude" in text and "gpt-4o" in text
        assert "diverged at step 1" in text

    def test_serialises(self):
        left = _trajectory(steps=[_step(1, "list_dir")])
        payload = diff_trajectories(left, left).to_dict()
        assert payload["task_id"] == "T1"
        assert len(payload["steps"]) == 1
