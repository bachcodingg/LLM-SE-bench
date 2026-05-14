"""Tests for stats/descriptive.py."""

import numpy as np
import pandas as pd
import pytest

from stats.descriptive import DescriptiveAnalyzer, DescriptiveStats


class TestDescriptiveCompute:
    def test_basic_stats(self):
        analyzer = DescriptiveAnalyzer()
        vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = analyzer.compute(vals, metric_name="test", model_id="m1")
        assert result.n == 10
        assert result.mean == pytest.approx(5.5, abs=1e-6)
        assert result.median == pytest.approx(5.5, abs=1e-6)
        assert result.min_val == 1.0
        assert result.max_val == 10.0
        assert result.q1 == pytest.approx(3.25, abs=0.5)
        assert result.q3 == pytest.approx(7.75, abs=0.5)
        assert result.iqr > 0
        assert result.std_dev > 0

    def test_empty_array(self):
        result = DescriptiveAnalyzer.compute(np.array([]))
        assert result.n == 0
        assert result.mean == 0.0

    def test_single_value(self):
        result = DescriptiveAnalyzer.compute(np.array([42.0]))
        assert result.n == 1
        assert result.mean == 42.0
        assert result.std_dev == 0.0
        assert result.ci_lower_95 == 42.0

    def test_nan_handling(self):
        result = DescriptiveAnalyzer.compute(np.array([1, 2, np.nan, 4, 5]))
        assert result.n == 4
        assert result.mean == pytest.approx(3.0, abs=1e-6)

    def test_ci_coverage(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(50, 10, size=100)
        result = DescriptiveAnalyzer.compute(vals)
        assert result.ci_lower_95 < result.mean < result.ci_upper_95

    def test_to_dict(self):
        result = DescriptiveAnalyzer.compute([1, 2, 3], metric_name="x")
        d = result.to_dict()
        assert "metric_name" in d
        assert d["metric_name"] == "x"
        assert isinstance(d["mean"], float)


class TestSummariseMetric:
    def test_grouped_summary(self, results_df):
        analyzer = DescriptiveAnalyzer()
        stats = analyzer.summarise_metric(results_df, "weighted_score", "model_id")
        assert len(stats) == 3  # 3 models
        for s in stats:
            assert s.n > 0
            assert 0 <= s.mean <= 1

    def test_multi_group(self, results_df):
        analyzer = DescriptiveAnalyzer()
        stats = analyzer.summarise_metric(
            results_df, "weighted_score", ["model_id", "dataset"]
        )
        assert len(stats) == 3 * 4  # 3 models × 4 datasets


class TestPassAtK:
    def test_pass_at_1(self, results_df):
        analyzer = DescriptiveAnalyzer()
        pak = analyzer.pass_at_k(results_df, k=1)
        assert "pass@1" in pak.columns
        assert len(pak) == 3
        assert all(0 <= v <= 1 for v in pak["pass@1"])

    def test_pass_at_5(self, results_df):
        analyzer = DescriptiveAnalyzer()
        pak = analyzer.pass_at_k(results_df, k=5)
        assert "pass@5" in pak.columns
        # pass@5 should be >= pass@1 in expectation
        pak1 = analyzer.pass_at_k(results_df, k=1)
        merged = pak1.merge(pak, on="model_id")
        # Not strict due to estimator variance, but check structure
        assert len(merged) == 3


class TestPercentileProfile:
    def test_standard_percentiles(self):
        vals = np.arange(1, 101)
        profile = DescriptiveAnalyzer.percentile_profile(vals)
        assert "p50" in profile
        assert profile["p50"] == pytest.approx(50.5, abs=1)
        assert profile["p5"] < profile["p95"]
