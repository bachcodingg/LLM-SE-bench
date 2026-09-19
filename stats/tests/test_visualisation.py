"""Tests for stats/visualisation.py."""

import pandas as pd
import pytest

from stats.visualisation import VisualisationEngine


@pytest.fixture()
def viz(tmp_path):
    return VisualisationEngine(output_dir=tmp_path / "figures", fmt="pdf")


class TestBarCharts:
    def test_pass_at_1_bar(self, viz, results_df):
        path = viz.pass_at_1_bar(results_df)
        assert path.exists()
        assert path.suffix == ".pdf"

    def test_pass_at_k_bar(self, viz, results_df):
        from stats.descriptive import DescriptiveAnalyzer
        da = DescriptiveAnalyzer()
        pak = da.pass_at_k(results_df, k=1)
        path = viz.pass_at_k_bar(pak)
        assert path.exists()

    def test_cost_bar(self, viz):
        profiles = [
            {"model_id": "A", "total_cost_usd": 10.5},
            {"model_id": "B", "total_cost_usd": 15.2},
        ]
        path = viz.cost_bar(profiles)
        assert path.exists()

    def test_cost_per_correct_bar(self, viz):
        profiles = [
            {"model_id": "A", "cost_per_correct": 0.05},
            {"model_id": "B", "cost_per_correct": 0.08},
        ]
        path = viz.cost_per_correct_bar(profiles)
        assert path.exists()


class TestBoxPlots:
    def test_score_boxplot(self, viz, results_df):
        path = viz.score_boxplot(results_df)
        assert path.exists()

    def test_quality_boxplot(self, viz, quality_df):
        paths = viz.quality_boxplot(quality_df)
        assert len(paths) > 0
        for p in paths:
            assert p.exists()

    def test_boxplot_by_dataset(self, viz, results_df):
        path = viz.boxplot_by_dataset(results_df)
        assert path.exists()


class TestHeatmaps:
    def test_dataset_model(self, viz, results_df):
        path = viz.heatmap_dataset_model(results_df)
        assert path.exists()

    def test_difficulty(self, viz, results_df):
        path = viz.heatmap_difficulty(results_df)
        assert path.exists()

    def test_consistency_heatmap(self, viz):
        profiles = [
            {"model_id": "A", "agreement_rate": 0.9, "mean_cv": 0.1,
             "icc": 0.85, "flip_rate": 0.05},
            {"model_id": "B", "agreement_rate": 0.8, "mean_cv": 0.2,
             "icc": 0.7, "flip_rate": 0.1},
        ]
        path = viz.consistency_heatmap(profiles)
        assert path.exists()

    def test_correlation_heatmap(self, viz):
        corr = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]],
            index=["A", "B"], columns=["A", "B"],
        )
        path = viz.correlation_heatmap(corr)
        assert path.exists()


class TestViolinPlots:
    def test_violin_score(self, viz, results_df):
        path = viz.violin_score(results_df)
        assert path.exists()


class TestRadarChart:
    def test_radar(self, viz):
        profiles = {
            "A": {"Accuracy": 0.8, "Speed": 0.6, "Cost": 0.9},
            "B": {"Accuracy": 0.7, "Speed": 0.8, "Cost": 0.5},
        }
        path = viz.radar_chart(profiles)
        assert path.exists()


class TestCriticalDifference:
    def test_cd_diagram(self, viz):
        ranks = {"A": 1.2, "B": 2.0, "C": 2.8}
        path = viz.critical_difference_diagram(ranks, cd=1.0, n_subjects=50)
        assert path.exists()


class TestForestPlots:
    def test_effect_size_forest(self, viz):
        results = [
            {"group_a": "A", "group_b": "B", "value": 0.3,
             "ci_lower": 0.1, "ci_upper": 0.5},
            {"group_a": "A", "group_b": "C", "value": -0.2,
             "ci_lower": -0.4, "ci_upper": 0.0},
        ]
        path = viz.effect_size_forest(results)
        assert path.exists()

    def test_ci_forest(self, viz):
        results = [
            {"statistic": "mean_A", "point_estimate": 0.7,
             "ci_lower": 0.65, "ci_upper": 0.75},
        ]
        path = viz.ci_forest(results)
        assert path.exists()


class TestScalingLine:
    def test_scaling(self, viz):
        projections = [
            {"model_id": "A", "n_files": 100, "projected_cost_usd": 5,
             "confidence_lower": 4, "confidence_upper": 6},
            {"model_id": "A", "n_files": 1000, "projected_cost_usd": 50,
             "confidence_lower": 40, "confidence_upper": 60},
        ]
        path = viz.scaling_line(projections)
        assert path.exists()


class TestStackedError:
    def test_stacked(self, viz, results_df):
        path = viz.stacked_error(results_df)
        assert path.exists()


class TestCumulativePass:
    def test_cumulative(self, viz, results_df):
        path = viz.cumulative_pass(results_df)
        assert path.exists()


class TestTokenDistribution:
    def test_tokens(self, viz):
        token_df = pd.DataFrame({
            "model_id": ["A", "B"],
            "mean_prompt": [1200, 1500],
            "mean_completion": [800, 600],
        })
        path = viz.token_distribution(token_df)
        assert path.exists()


class TestGenerateAll:
    def test_generates_multiple(self, viz, results_df, quality_df):
        paths = viz.generate_all(
            results_df=results_df,
            quality_df=quality_df,
        )
        assert len(paths) >= 8  # at least the core figures
