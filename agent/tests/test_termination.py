"""
Tests for agent.termination.

All six conditions, plus the two properties that are easy to get wrong:

* success requires *no regression*, not just a passing target;
* the cost ceiling is enforced before a call, not after.
"""

from __future__ import annotations

import time

import pytest

from agent.termination import (
    BudgetExceeded,
    BudgetGovernor,
    EpisodeState,
    StopCondition,
    TerminationPolicy,
    check_termination,
)


def _state(**kwargs) -> EpisodeState:
    return EpisodeState(**kwargs)


class TestPolicyValidation:
    def test_cost_ceiling_is_required_to_be_positive(self):
        """There is no safe default for someone else's account."""
        with pytest.raises(ValueError, match="no safe default"):
            TerminationPolicy(max_cost_eur=0.0)

    def test_negative_cost_ceiling_is_rejected(self):
        with pytest.raises(ValueError):
            TerminationPolicy(max_cost_eur=-1.0)

    def test_zero_steps_is_rejected(self):
        with pytest.raises(ValueError, match="max_steps"):
            TerminationPolicy(max_cost_eur=1.0, max_steps=0)

    def test_zero_no_progress_window_is_rejected(self):
        with pytest.raises(ValueError, match="no_progress_steps"):
            TerminationPolicy(max_cost_eur=1.0, no_progress_steps=0)


class TestSuccess:
    def test_all_tests_passing_is_success(self, policy):
        state = _state(tests_passed=3, tests_total=3, baseline_passed=0, baseline_total=3)
        assert check_termination(state, policy) is StopCondition.SUCCESS

    def test_zero_of_zero_is_not_success(self, policy):
        """A compile failure gives 0/0. That must never read as 'all passed'."""
        state = _state(tests_passed=0, tests_total=0)
        assert check_termination(state, policy) is None

    def test_partial_pass_is_not_success(self, policy):
        state = _state(tests_passed=2, tests_total=3)
        assert check_termination(state, policy) is None

    def test_never_running_tests_is_not_success(self, policy):
        assert check_termination(_state(), policy) is None

    def test_unexecuted_tests_can_never_be_a_success(self, policy):
        """The structural fallback credits every @Test as passing.

        Without this guard a dry run — or any machine without Docker —
        produces a 100% resolve rate that looks exactly like a real one.
        """
        state = _state(
            tests_passed=3, tests_total=3,
            baseline_passed=0, baseline_total=3,
            tests_executed=False,
        )
        assert check_termination(state, policy) is not StopCondition.SUCCESS
        assert state.tests_all_pass is False

    def test_the_same_numbers_do_succeed_when_executed(self, policy):
        state = _state(
            tests_passed=3, tests_total=3,
            baseline_passed=0, baseline_total=3,
            tests_executed=True,
        )
        assert check_termination(state, policy) is StopCondition.SUCCESS


class TestRegression:
    def test_a_regression_is_detected(self):
        state = _state(tests_passed=1, tests_total=5, baseline_passed=3, baseline_total=5)
        assert state.regressed is True

    def test_no_baseline_means_no_regression_claim(self):
        """Without a baseline nothing can be said, so nothing is claimed."""
        state = _state(tests_passed=1, tests_total=5)
        assert state.regressed is False

    def test_improvement_is_not_a_regression(self):
        state = _state(tests_passed=5, tests_total=5, baseline_passed=3, baseline_total=5)
        assert state.regressed is False


class TestCeilings:
    def test_max_steps(self, policy):
        assert check_termination(_state(steps=policy.max_steps), policy) is StopCondition.MAX_STEPS

    def test_max_tokens(self, policy):
        state = _state(total_tokens=policy.max_tokens)
        assert check_termination(state, policy) is StopCondition.MAX_TOKENS

    def test_max_cost(self, policy):
        state = _state(cost_eur=policy.max_cost_eur)
        assert check_termination(state, policy) is StopCondition.MAX_COST

    def test_wall_clock(self, policy):
        state = _state(started_at=time.monotonic() - policy.wall_clock_seconds - 1)
        assert check_termination(state, policy) is StopCondition.WALL_CLOCK

    def test_success_beats_a_ceiling_reached_on_the_same_step(self, policy):
        """Solving it on the final permitted step is a success, not a timeout."""
        state = _state(
            steps=policy.max_steps,
            tests_passed=3, tests_total=3,
            baseline_passed=0, baseline_total=3,
        )
        assert check_termination(state, policy) is StopCondition.SUCCESS

    def test_cost_is_checked_before_the_other_ceilings(self, policy):
        state = _state(steps=policy.max_steps, cost_eur=policy.max_cost_eur)
        assert check_termination(state, policy) is StopCondition.MAX_COST


class TestNoProgress:
    def test_fires_at_the_configured_window(self, policy):
        state = _state(steps_without_progress=policy.no_progress_steps)
        assert check_termination(state, policy) is StopCondition.NO_PROGRESS

    def test_does_not_fire_one_step_early(self, policy):
        state = _state(steps_without_progress=policy.no_progress_steps - 1)
        assert check_termination(state, policy) is None

    def test_progress_resets_the_counter(self):
        state = _state(steps_without_progress=4)
        state.record_progress(True)
        assert state.steps_without_progress == 0

    def test_no_progress_increments_the_counter(self):
        state = _state()
        state.record_progress(False)
        state.record_progress(False)
        assert state.steps_without_progress == 2


class TestOtherConditions:
    def test_stopping_without_solving_is_abandonment(self, policy):
        state = _state(model_stopped_calling_tools=True)
        assert check_termination(state, policy) is StopCondition.ABANDONED

    def test_an_error_pre_empts_everything(self, policy):
        state = _state(error="provider exploded", tests_passed=3, tests_total=3)
        assert check_termination(state, policy) is StopCondition.ERROR

    def test_error_is_not_counted_as_a_budget_stop(self):
        assert StopCondition.ERROR.is_budget is False
        assert StopCondition.ERROR.is_success is False

    def test_the_four_ceilings_classify_as_budget(self):
        budget = {c for c in StopCondition if c.is_budget}
        assert budget == {
            StopCondition.MAX_STEPS,
            StopCondition.MAX_TOKENS,
            StopCondition.MAX_COST,
            StopCondition.WALL_CLOCK,
        }


class TestBudgetGovernor:
    def test_rejects_a_non_positive_ceiling(self):
        with pytest.raises(ValueError):
            BudgetGovernor(ceiling_eur=0.0)

    def test_allows_a_call_within_budget(self):
        BudgetGovernor(ceiling_eur=1.0, worst_case_call_eur=0.01).check()

    def test_refuses_once_the_ceiling_is_reached(self):
        governor = BudgetGovernor(ceiling_eur=1.0)
        governor.record(1.0)
        with pytest.raises(BudgetExceeded, match="exhausted"):
            governor.check()

    def test_refuses_before_a_call_that_could_cross(self):
        """The whole point: the check happens before the money is spent."""
        governor = BudgetGovernor(ceiling_eur=1.0, worst_case_call_eur=0.5)
        governor.record(0.8)
        with pytest.raises(BudgetExceeded, match="Refusing"):
            governor.check()
        assert governor.spent_eur == 0.8  # nothing further was spent

    def test_the_refusal_states_the_numbers(self):
        governor = BudgetGovernor(ceiling_eur=1.0, worst_case_call_eur=0.5)
        governor.record(0.8)
        with pytest.raises(BudgetExceeded) as caught:
            governor.check()
        message = str(caught.value)
        assert "0.8" in message and "0.5" in message and "1.0" in message

    def test_without_a_projection_it_can_only_stop_afterwards(self):
        """Documented consequence: no projection means up to one call of overshoot."""
        governor = BudgetGovernor(ceiling_eur=1.0, worst_case_call_eur=0.0)
        governor.record(0.99)
        governor.check()  # allowed, even though the next call may overshoot

    def test_tracks_spend_and_call_count(self):
        governor = BudgetGovernor(ceiling_eur=10.0)
        governor.record(1.0)
        governor.record(2.0)
        assert governor.spent_eur == pytest.approx(3.0)
        assert governor.calls == 2
        assert governor.remaining_eur == pytest.approx(7.0)

    def test_negative_costs_are_ignored(self):
        governor = BudgetGovernor(ceiling_eur=10.0)
        governor.record(-5.0)
        assert governor.spent_eur == 0.0

    def test_remaining_never_goes_negative(self):
        governor = BudgetGovernor(ceiling_eur=1.0)
        governor.record(5.0)
        assert governor.remaining_eur == 0.0
