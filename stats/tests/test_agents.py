"""
Tests for stats.agents (M8).

Two of these matter more than the rest: pass@k must refuse to extrapolate,
and the bootstrap must resample clusters rather than observations. Both have
an incorrect version that runs fine and gives a nicer-looking answer, which
is what makes them worth pinning.
"""

from __future__ import annotations

import pytest

from stats.agents import (
    AlphaSpending,
    SequentialTest,
    clustered_bootstrap_ci,
    decompose_variance,
    pass_at_k,
    pass_at_k_from_episodes,
    required_episodes,
)


class TestPassAtK:
    def test_all_pass(self):
        assert pass_at_k(n=5, c=5, k=1) == 1.0
        assert pass_at_k(n=5, c=5, k=3) == 1.0

    def test_none_pass(self):
        assert pass_at_k(n=5, c=0, k=1) == 0.0
        assert pass_at_k(n=5, c=0, k=5) == 0.0

    def test_pass_at_1_is_the_success_fraction(self):
        assert pass_at_k(n=4, c=1, k=1) == pytest.approx(0.25)
        assert pass_at_k(n=10, c=3, k=1) == pytest.approx(0.3)

    def test_increases_with_k(self):
        values = [pass_at_k(n=10, c=2, k=k) for k in (1, 2, 5, 8)]
        assert values == sorted(values)

    def test_known_value(self):
        # 1 success in 4: pass@2 = 1 - C(3,2)/C(4,2) = 1 - 3/6 = 0.5
        assert pass_at_k(n=4, c=1, k=2) == pytest.approx(0.5)

    def test_refuses_to_extrapolate(self):
        """pass@5 from 3 samples is extrapolation dressed as a measurement."""
        with pytest.raises(ValueError, match="at least 5 samples"):
            pass_at_k(n=3, c=1, k=5)

    def test_rejects_more_successes_than_samples(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            pass_at_k(n=3, c=4, k=1)

    @pytest.mark.parametrize("n,c,k", [(10, 3, 5), (8, 2, 3), (6, 4, 2), (20, 1, 7)])
    def test_matches_the_exact_combinatorial_formula(self, n, c, k):
        """1 - C(n-c, k) / C(n, k), computed the slow, obvious way."""
        from math import comb

        expected = 1 - comb(n - c, k) / comb(n, k)
        assert pass_at_k(n, c, k) == pytest.approx(expected)


class TestPassAtKFromEpisodes:
    def test_aggregates_across_tasks(self):
        outcomes = {
            "T1": [True, True, True],
            "T2": [False, False, False],
            "T3": [True, False, False],
        }
        result = pass_at_k_from_episodes(outcomes, k_values=(1,))
        # 1.0 + 0.0 + 1/3, averaged
        assert result["estimates"]["pass@1"]["value"] == pytest.approx(0.4444, abs=1e-3)

    def test_excludes_tasks_with_too_few_samples(self):
        outcomes = {"T1": [True] * 5, "T2": [True, False]}
        result = pass_at_k_from_episodes(outcomes, k_values=(5,))
        estimate = result["estimates"]["pass@5"]
        assert estimate["tasks_included"] == 1
        assert estimate["tasks_excluded"] == 1

    def test_reports_none_when_no_task_qualifies(self):
        result = pass_at_k_from_episodes({"T1": [True]}, k_values=(5,))
        assert result["estimates"]["pass@5"]["value"] is None
        assert "no task has 5" in result["estimates"]["pass@5"]["reason"]


class TestVarianceDecomposition:
    def test_identical_outcomes_have_no_variance(self):
        rows = [{"score": 1.0, "task": f"T{i}", "model": "m"} for i in range(5)]
        decomposition = decompose_variance(rows)
        assert decomposition.total == 0.0
        assert decomposition.shares == {}

    def test_task_variance_is_attributed(self):
        rows = [
            {"score": s, "task": task, "model": model}
            for task, base in (("easy", 1.0), ("hard", 0.0))
            for model in ("a", "b")
            for s in (base,)
        ]
        decomposition = decompose_variance(rows, factors=("task", "model"))
        assert decomposition.components["task"] > decomposition.components.get("model", 0)
        assert decomposition.dominant == "task"

    def test_interpretation_warns_when_task_variance_dominates(self):
        rows = [
            {"score": 1.0 if task == "easy" else 0.0, "task": task, "model": model}
            for task in ("easy", "hard")
            for model in ("a", "b", "c")
        ]
        payload = decompose_variance(rows, factors=("task", "model")).to_dict()
        assert "Task variance dominates" in payload["interpretation"]

    def test_too_few_observations(self):
        assert decompose_variance([{"score": 1.0}]).total == 0.0


class TestRequiredEpisodes:
    def test_more_variance_needs_more_tasks(self):
        low = required_episodes(observed_variance=0.01)["tasks_required"]
        high = required_episodes(observed_variance=0.1)["tasks_required"]
        assert high > low

    def test_smaller_effects_need_more_tasks(self):
        big = required_episodes(observed_variance=0.05, effect=0.2)["tasks_required"]
        small = required_episodes(observed_variance=0.05, effect=0.02)["tasks_required"]
        assert small > big

    def test_repeats_help_less_when_correlated(self):
        """At ICC 1.0 extra repeats add nothing; pretending otherwise inflates n."""
        independent = required_episodes(
            observed_variance=0.05, repeats=3, intraclass_correlation=0.0
        )
        correlated = required_episodes(
            observed_variance=0.05, repeats=3, intraclass_correlation=1.0
        )
        assert correlated["tasks_required"] > independent["tasks_required"]
        assert correlated["effective_sample_per_task"] == pytest.approx(1.0)

    def test_zero_variance(self):
        assert required_episodes(observed_variance=0.0)["tasks_required"] == 0

    def test_rejects_non_positive_effect(self):
        with pytest.raises(ValueError):
            required_episodes(observed_variance=0.1, effect=0.0)


class TestClusteredBootstrap:
    def test_produces_an_interval_around_the_observed_value(self):
        clusters = {f"T{i}": [float(i % 2)] * 3 for i in range(20)}
        result = clustered_bootstrap_ci(clusters, n_resamples=500, seed=1)
        assert result["ci_lower"] <= result["observed"] <= result["ci_upper"]

    def test_reports_clusters_and_observations_separately(self):
        clusters = {f"T{i}": [1.0, 0.0, 1.0] for i in range(10)}
        result = clustered_bootstrap_ci(clusters, n_resamples=200, seed=1)
        assert result["n_clusters"] == 10
        assert result["n_observations"] == 30

    def test_is_wider_than_an_uncluster_aware_interval(self):
        """Resampling observations treats correlated repeats as independent."""
        clusters = {f"T{i}": [float(i % 2)] * 5 for i in range(20)}
        clustered = clustered_bootstrap_ci(clusters, n_resamples=1000, seed=7)

        flattened = {
            f"obs{index}": [value]
            for index, value in enumerate(
                v for values in clusters.values() for v in values
            )
        }
        naive = clustered_bootstrap_ci(flattened, n_resamples=1000, seed=7)

        clustered_width = clustered["ci_upper"] - clustered["ci_lower"]
        naive_width = naive["ci_upper"] - naive["ci_lower"]
        assert clustered_width > naive_width

    def test_is_reproducible_from_a_seed(self):
        clusters = {f"T{i}": [float(i % 3)] for i in range(15)}
        first = clustered_bootstrap_ci(clusters, n_resamples=300, seed=42)
        second = clustered_bootstrap_ci(clusters, n_resamples=300, seed=42)
        assert first["ci_lower"] == second["ci_lower"]

    def test_supports_median(self):
        clusters = {f"T{i}": [float(i)] for i in range(11)}
        result = clustered_bootstrap_ci(clusters, statistic="median", n_resamples=200)
        assert result["observed"] == pytest.approx(5.0)

    def test_rejects_an_unknown_statistic(self):
        with pytest.raises(ValueError, match="unknown statistic"):
            clustered_bootstrap_ci({"T": [1.0]}, statistic="mode")

    def test_empty_input(self):
        assert "error" in clustered_bootstrap_ci({})


class TestAlphaSpending:
    def test_early_looks_are_strict(self):
        """O'Brien-Fleming spends almost nothing early, which is the point."""
        spending = AlphaSpending(alpha=0.05, total_looks=5)
        assert spending.threshold_at(1) < spending.threshold_at(5)
        assert spending.threshold_at(1) < 0.01

    def test_cumulative_alpha_reaches_the_budget(self):
        spending = AlphaSpending(alpha=0.05, total_looks=4)
        assert spending.spent_by(4) == pytest.approx(0.05)

    def test_cumulative_alpha_is_monotonic(self):
        spending = AlphaSpending(total_looks=5)
        values = [spending.spent_by(look) for look in range(1, 6)]
        assert values == sorted(values)

    def test_rejects_a_bad_alpha(self):
        with pytest.raises(ValueError):
            AlphaSpending(alpha=1.5)


class TestSequentialTest:
    def test_stops_when_clearly_significant(self):
        test = SequentialTest(alpha=0.05, total_looks=5)
        test.look(p_value=0.5, n_so_far=10)
        record = test.look(p_value=0.0001, n_so_far=20)
        assert record["stop"] is True
        assert test.summary()["stopped_early"] is True

    def test_reports_the_budget_saved(self):
        test = SequentialTest(alpha=0.05, total_looks=5)
        test.look(p_value=0.00001, n_so_far=10)
        assert test.summary()["budget_saved"] == pytest.approx(0.8)

    def test_a_borderline_p_value_does_not_stop_early(self):
        """0.04 is significant at a fixed alpha and not at look 1 of 5."""
        test = SequentialTest(alpha=0.05, total_looks=5)
        assert test.look(p_value=0.04, n_so_far=10)["stop"] is False

    def test_refuses_an_undeclared_extra_look(self):
        """Adding a look after seeing the data is what this prevents."""
        test = SequentialTest(alpha=0.05, total_looks=2)
        test.look(p_value=0.5, n_so_far=10)
        test.look(p_value=0.5, n_so_far=20)
        with pytest.raises(ValueError, match="exceeds the 2 declared"):
            test.look(p_value=0.01, n_so_far=30)

    def test_summary_without_stopping(self):
        test = SequentialTest(total_looks=3)
        test.look(p_value=0.5, n_so_far=10)
        summary = test.summary()
        assert summary["stopped_early"] is False
        assert summary["budget_saved"] == 0.0
        assert len(summary["schedule"]) == 3
