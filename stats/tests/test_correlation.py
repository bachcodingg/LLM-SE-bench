"""Tests for stats/correlation.py."""

import numpy as np
import pandas as pd
import pytest

from stats.correlation import CorrelationAnalyzer


class TestCorrelate:
    def test_perfect_positive(self):
        ca = CorrelationAnalyzer()
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = ca.correlate(x, x, "x", "x")
        assert result.coefficient == pytest.approx(1.0, abs=1e-6)
        assert result.significant

    def test_perfect_negative(self):
        ca = CorrelationAnalyzer()
        x = np.arange(10, dtype=float)
        y = -x
        result = ca.correlate(x, y, "x", "y")
        assert result.coefficient == pytest.approx(-1.0, abs=1e-6)

    def test_uncorrelated(self):
        rng = np.random.default_rng(42)
        ca = CorrelationAnalyzer()
        x = rng.random(200)
        y = rng.random(200)
        result = ca.correlate(x, y)
        assert abs(result.coefficient) < 0.2

    def test_insufficient_samples(self):
        ca = CorrelationAnalyzer(min_n=10)
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        result = ca.correlate(x, y)
        assert result.coefficient == 0.0

    def test_kendall(self):
        ca = CorrelationAnalyzer(method="kendall")
        x = np.arange(10, dtype=float)
        result = ca.correlate(x, x * 2 + 1, method="kendall")
        assert result.coefficient == pytest.approx(1.0, abs=1e-6)


class TestCorrelationMatrix:
    def test_shape(self, merged_df):
        ca = CorrelationAnalyzer()
        cols = ["cyclomatic_complexity", "maintainability_index"]
        available = [c for c in cols if c in merged_df.columns]
        if len(available) >= 2:
            corr, pval = ca.correlation_matrix(merged_df, available)
            assert corr.shape == (len(available), len(available))
            assert pval.shape == corr.shape
            # Diagonal should be 1
            for i in range(len(available)):
                assert corr.iloc[i, i] == pytest.approx(1.0)


class TestMetricsVsPerformance:
    def test_returns_results(self, merged_df):
        ca = CorrelationAnalyzer()
        metric_cols = [c for c in ["cyclomatic_complexity", "maintainability_index"]
                       if c in merged_df.columns]
        if metric_cols and "weighted_score" in merged_df.columns:
            results = ca.metrics_vs_performance(merged_df, metric_cols)
            assert len(results) > 0
            for r in results:
                assert r.method in ("spearman", "point_biserial")


class TestPerModelCorrelations:
    def test_per_model(self, merged_df):
        ca = CorrelationAnalyzer()
        metric_cols = [c for c in ["cyclomatic_complexity"] if c in merged_df.columns]
        if metric_cols and "weighted_score" in merged_df.columns:
            out = ca.per_model_correlations(merged_df, metric_cols)
            assert len(out) == 3  # 3 models
