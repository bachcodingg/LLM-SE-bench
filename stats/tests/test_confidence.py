"""Tests for stats/confidence.py."""

import numpy as np
import pytest

from stats.confidence import BootstrapCI


class TestBootstrapCI:
    def test_mean_ci(self):
        rng = np.random.default_rng(42)
        data = rng.normal(50, 5, size=100)
        bci = BootstrapCI(n_bootstrap=2000, seed=42)
        result = bci.ci_mean(data)
        assert result.ci_lower < 50 < result.ci_upper
        assert result.n_samples == 100
        assert result.statistic_name == "mean"

    def test_median_ci(self):
        data = np.arange(1, 101, dtype=float)
        bci = BootstrapCI(n_bootstrap=1000, seed=42)
        result = bci.ci_median(data)
        assert result.ci_lower < result.point_estimate < result.ci_upper

    def test_proportion_ci(self):
        data = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1], dtype=float)
        bci = BootstrapCI(n_bootstrap=1000, seed=42)
        result = bci.ci_proportion(data)
        assert 0 <= result.ci_lower <= result.ci_upper <= 1

    def test_empty_data(self):
        bci = BootstrapCI()
        result = bci.ci_mean(np.array([]))
        assert result.n_samples == 0
        assert result.point_estimate == 0.0

    def test_single_value(self):
        bci = BootstrapCI()
        result = bci.ci_mean(np.array([5.0]))
        assert result.point_estimate == 5.0
        assert result.ci_lower == 5.0

    def test_bca_method(self):
        rng = np.random.default_rng(42)
        data = rng.normal(10, 2, size=50)
        bci = BootstrapCI(n_bootstrap=1000, method="bca", seed=42)
        result = bci.compute(data, np.mean, "mean")
        assert result.method == "bca"
        assert result.ci_lower < result.ci_upper

    def test_ci_difference(self):
        rng = np.random.default_rng(42)
        x = rng.normal(10, 2, size=50)
        y = rng.normal(8, 2, size=50)
        bci = BootstrapCI(n_bootstrap=1000, seed=42)
        result = bci.ci_difference(x, y)
        assert result.point_estimate > 0  # x mean > y mean
        assert result.ci_lower < result.ci_upper

    def test_to_dict(self):
        bci = BootstrapCI(n_bootstrap=500)
        result = bci.ci_mean(np.array([1, 2, 3, 4, 5]))
        d = result.to_dict()
        assert "point_estimate" in d
        assert "ci_lower" in d
        assert "ci_upper" in d

    def test_custom_statistic(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        bci = BootstrapCI(n_bootstrap=500, seed=42)
        result = bci.compute(data, lambda x: np.percentile(x, 75), "q75")
        assert result.statistic_name == "q75"
        assert result.point_estimate == pytest.approx(7.75, abs=0.5)
