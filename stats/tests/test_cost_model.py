"""Tests for stats/cost_model.py."""

import pytest

from stats.cost_model import CostModel


class TestCostProfiles:
    def test_compute_profiles(self, cost_df, results_df, quality_df):
        cm = CostModel()
        profiles = cm.compute_profiles(cost_df, results_df, quality_df)
        assert len(profiles) == 3  # 3 models
        for p in profiles:
            assert p.total_cost_usd > 0
            assert p.mean_cost_per_task > 0
            assert p.n_tasks > 0
            assert p.total_prompt_tokens > 0

    def test_cost_per_correct(self, cost_df, results_df):
        cm = CostModel()
        profiles = cm.compute_profiles(cost_df, results_df)
        for p in profiles:
            if p.cost_per_correct > 0:
                assert p.cost_per_correct >= p.mean_cost_per_task

    def test_to_dict(self, cost_df, results_df):
        cm = CostModel()
        profiles = cm.compute_profiles(cost_df, results_df)
        d = profiles[0].to_dict()
        assert "model_id" in d
        assert "total_cost_usd" in d


class TestCostByDataset:
    def test_grouping(self, cost_df, results_df):
        cm = CostModel()
        result = cm.cost_by_dataset(cost_df, results_df)
        if not result.empty:
            assert "dataset" in result.columns
            assert "model_id" in result.columns


class TestScalingProjections:
    def test_default_sizes(self, cost_df):
        cm = CostModel()
        projections = cm.project_scaling(cost_df)
        assert len(projections) > 0
        sizes_seen = {p.n_files for p in projections}
        assert 1000 in sizes_seen
        assert 10000 in sizes_seen

    def test_custom_sizes(self, cost_df):
        cm = CostModel(scaling_sizes=[50, 200])
        projections = cm.project_scaling(cost_df)
        sizes = {p.n_files for p in projections}
        assert 50 in sizes
        assert 200 in sizes

    def test_linear_scaling(self, cost_df):
        cm = CostModel(scaling_sizes=[100, 1000])
        projections = cm.project_scaling(cost_df)
        # Group by model, check 1000 is ~10x of 100
        by_model = {}
        for p in projections:
            by_model.setdefault(p.model_id, {})[p.n_files] = p.projected_cost_usd
        for model, costs in by_model.items():
            if 100 in costs and 1000 in costs:
                assert costs[1000] == pytest.approx(costs[100] * 10, rel=0.01)


class TestTokenAnalysis:
    def test_structure(self, cost_df):
        result = CostModel.token_analysis(cost_df)
        assert "mean_prompt" in result.columns
        assert "mean_completion" in result.columns
        assert "input_output_ratio" in result.columns
        assert len(result) == 3


class TestCostEffectiveness:
    def test_pairwise_ratios(self, cost_df, results_df):
        cm = CostModel()
        profiles = cm.compute_profiles(cost_df, results_df)
        ratios = cm.compare_cost_effectiveness(profiles)
        assert len(ratios) >= 3  # at least C(3,2) for mean_cost
        for r in ratios:
            assert r.ratio > 0
