"""Tests for stats/effect_size.py."""

import numpy as np
import pytest

from stats.effect_size import (
    EffectSizeCalculator,
    cliffs_delta,
    cohens_d,
    rank_biserial,
    vargha_delaney_a,
)


class TestCliffsDelta:
    def test_identical_samples(self):
        x = np.array([1, 2, 3, 4, 5])
        d, label = cliffs_delta(x, x)
        assert d == pytest.approx(0.0, abs=1e-6)
        assert label == "negligible"

    def test_completely_separated(self):
        x = np.array([10, 11, 12, 13])
        y = np.array([1, 2, 3, 4])
        d, label = cliffs_delta(x, y)
        assert d == pytest.approx(1.0, abs=1e-6)
        assert label == "large"

    def test_reversed(self):
        x = np.array([1, 2, 3])
        y = np.array([10, 11, 12])
        d, _ = cliffs_delta(x, y)
        assert d < 0

    def test_empty_arrays(self):
        d, label = cliffs_delta(np.array([]), np.array([1, 2]))
        assert d == 0.0

    def test_nan_handling(self):
        d, _ = cliffs_delta(np.array([1, 2, np.nan, 4]), np.array([0, 1, 2, 3]))
        assert isinstance(d, float)


class TestCohensD:
    def test_identical(self):
        x = np.array([5, 5, 5, 5, 5])
        d, label = cohens_d(x, x)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_large_effect(self):
        rng = np.random.default_rng(42)
        x = rng.normal(10, 1, size=50)
        y = rng.normal(5, 1, size=50)
        d, label = cohens_d(x, y)
        assert abs(d) > 2
        assert label == "large"

    def test_small_sample(self):
        d, label = cohens_d(np.array([1.0]), np.array([2.0]))
        assert d == 0.0  # n < 2


class TestRankBiserial:
    def test_perfect_dominance(self):
        x = np.array([10, 11, 12, 13, 14])
        y = np.array([1, 2, 3, 4, 5])
        r = rank_biserial(x, y)
        assert r == pytest.approx(1.0, abs=1e-6)

    def test_no_difference(self):
        x = np.array([1, 2, 3, 4, 5])
        r = rank_biserial(x, x)
        assert r == pytest.approx(0.0, abs=1e-6)


class TestVarghaDelaneyA:
    def test_no_effect(self):
        x = np.array([1, 2, 3, 4, 5])
        A, label = vargha_delaney_a(x, x)
        assert A == pytest.approx(0.5, abs=0.01)
        assert label == "negligible"

    def test_complete_dominance(self):
        A, label = vargha_delaney_a(np.array([10, 11]), np.array([1, 2]))
        assert A == pytest.approx(1.0, abs=1e-6)
        assert label == "large"


class TestEffectSizeCalculator:
    def test_pairwise_cliffs(self, results_df):
        calc = EffectSizeCalculator(method="cliffs_delta", bootstrap_ci=False)
        results = calc.compute_pairwise(results_df, "weighted_score")
        assert len(results) == 3  # C(3,2)
        for r in results:
            assert -1 <= r.value <= 1
            assert r.measure == "cliffs_delta"

    def test_pairwise_cohens(self, results_df):
        calc = EffectSizeCalculator(method="cohens_d", bootstrap_ci=False)
        results = calc.compute_pairwise(results_df, "weighted_score")
        assert len(results) == 3
        for r in results:
            assert r.measure == "cohens_d"

    def test_bootstrap_ci(self, results_df):
        calc = EffectSizeCalculator(bootstrap_ci=True, n_bootstrap=200)
        results = calc.compute_pairwise(results_df, "weighted_score")
        for r in results:
            assert r.ci_lower <= r.value <= r.ci_upper or abs(r.ci_lower - r.ci_upper) < 0.01

    def test_to_dict(self, results_df):
        calc = EffectSizeCalculator(bootstrap_ci=False)
        results = calc.compute_pairwise(results_df, "weighted_score")
        d = results[0].to_dict()
        assert "measure" in d
        assert "value" in d
