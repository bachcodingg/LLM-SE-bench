"""Tests for stats/hypothesis.py."""

import numpy as np
import pandas as pd
import pytest

from stats.hypothesis import HypothesisEngine, HypothesisTestResult


class TestWilcoxon:
    def test_identical_samples(self):
        engine = HypothesisEngine()
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        result = engine.wilcoxon_signed_rank(x, x, metric_name="test")
        assert result.p_value == 1.0
        assert not result.significant

    def test_different_samples(self):
        engine = HypothesisEngine()
        rng = np.random.default_rng(42)
        x = rng.normal(10, 2, size=30)
        y = rng.normal(5, 2, size=30)
        result = engine.wilcoxon_signed_rank(x, y)
        assert result.p_value < 0.05
        assert result.significant

    def test_insufficient_samples(self):
        engine = HypothesisEngine(min_samples=10)
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        result = engine.wilcoxon_signed_rank(x, y)
        assert "Insufficient" in result.notes

    def test_to_dict(self):
        engine = HypothesisEngine()
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        result = engine.wilcoxon_signed_rank(x, x + 1, metric_name="m1",
                                              group_a="A", group_b="B")
        d = result.to_dict()
        assert d["test_name"] == "wilcoxon_signed_rank"
        assert d["group_a"] == "A"


class TestPairwiseWilcoxon:
    def test_all_pairs(self, results_df):
        engine = HypothesisEngine()
        results = engine.pairwise_wilcoxon(results_df, "weighted_score")
        assert len(results) == 3  # C(3,2)
        for r in results:
            assert r.correction_method in ("holm", "bonferroni")

    def test_correction_applied(self, results_df):
        engine = HypothesisEngine(correction="bonferroni")
        results = engine.pairwise_wilcoxon(results_df, "weighted_score")
        for r in results:
            assert r.correction_method == "bonferroni"


class TestFriedman:
    def test_three_models(self, results_df):
        engine = HypothesisEngine()
        result = engine.friedman_test(results_df, "weighted_score")
        assert result.test_name == "friedman"
        assert result.n_samples > 0
        assert 0 <= result.p_value <= 1
        assert result.effect_size >= 0  # Kendall's W

    def test_insufficient_groups(self):
        engine = HypothesisEngine()
        df = pd.DataFrame({
            "problem_id": ["p1", "p2", "p1", "p2"],
            "model_id": ["A", "A", "B", "B"],
            "score": [0.5, 0.6, 0.7, 0.8],
        })
        result = engine.friedman_test(df, "score")
        assert "Need ≥3" in result.notes


class TestNemenyi:
    def test_posthoc(self, results_df):
        engine = HypothesisEngine()
        results = engine.nemenyi_posthoc(results_df, "weighted_score")
        assert len(results) == 3  # C(3,2) pairs
        for r in results:
            assert r.test_name == "nemenyi"
            assert "CD=" in r.notes


class TestCochransQ:
    def test_binary_data(self, results_df):
        engine = HypothesisEngine()
        result = engine.cochrans_q(results_df)
        assert result.test_name == "cochrans_q"
        assert 0 <= result.p_value <= 1

    def test_constant_data(self):
        engine = HypothesisEngine(min_samples=1)
        df = pd.DataFrame({
            "problem_id": ["p1", "p1", "p1"],
            "model_id": ["A", "B", "C"],
            "is_pass": [True, True, True],
        })
        result = engine.cochrans_q(df)
        # All pass → no difference expected
        assert result.p_value >= 0


class TestRunAll:
    def test_full_bundle(self, results_df):
        engine = HypothesisEngine()
        bundle = engine.run_all(results_df, "weighted_score")
        assert "friedman" in bundle
        assert "nemenyi" in bundle
        assert "wilcoxon" in bundle
        assert isinstance(bundle["friedman"], HypothesisTestResult)
