"""Tests for stats/consistency.py."""

import numpy as np
import pandas as pd
import pytest

from stats.consistency import ConsistencyAnalyzer


class TestConsistencyAnalyse:
    def test_profiles_per_model(self, results_df):
        ca = ConsistencyAnalyzer()
        profiles = ca.analyse(results_df, group_by="model_id")
        assert len(profiles) == 3
        for p in profiles:
            assert p.n_problems > 0
            assert 0 <= p.agreement_rate <= 1
            assert 0 <= p.flip_rate <= 1
            assert p.agreement_rate + p.flip_rate <= 1.01  # rounding

    def test_profiles_per_model_dataset(self, results_df):
        ca = ConsistencyAnalyzer()
        profiles = ca.analyse(results_df, group_by=["model_id", "dataset"])
        assert len(profiles) == 3 * 4  # 3 models × 4 datasets

    def test_to_dict(self, results_df):
        ca = ConsistencyAnalyzer()
        profiles = ca.analyse(results_df)
        d = profiles[0].to_dict()
        assert "model_id" in d
        assert "agreement_rate" in d


class TestFlipAnalysis:
    def test_find_flips(self, results_df):
        ca = ConsistencyAnalyzer()
        flips = ca.find_flips(results_df)
        # With 3 runs and realistic data, some flips are expected
        assert isinstance(flips, list)
        if flips:
            f = flips[0]
            assert f.n_pass > 0
            assert f.n_fail > 0
            assert f.n_total == f.n_pass + f.n_fail

    def test_flip_to_dict(self, results_df):
        ca = ConsistencyAnalyzer()
        flips = ca.find_flips(results_df)
        if flips:
            d = flips[0].to_dict()
            assert "problem_id" in d
            assert "n_pass" in d


class TestICC:
    def test_perfect_agreement(self):
        df = pd.DataFrame({
            "problem_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "evaluation_id": ["e1", "e2", "e1", "e2", "e1", "e2"],
            "weighted_score": [0.8, 0.8, 0.5, 0.5, 0.9, 0.9],
        })
        icc = ConsistencyAnalyzer.compute_icc(df)
        assert icc == pytest.approx(1.0, abs=0.01)

    def test_no_agreement(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "problem_id": [f"p{i}" for i in range(20)] * 2,
            "evaluation_id": ["e1"] * 20 + ["e2"] * 20,
            "weighted_score": rng.random(40).tolist(),
        })
        icc = ConsistencyAnalyzer.compute_icc(df)
        assert icc < 0.5  # should be low for random data
