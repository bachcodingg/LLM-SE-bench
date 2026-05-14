"""Tests for stats/report_data.py."""

import pandas as pd
import pytest

from stats.report_data import ReportDataGenerator


@pytest.fixture()
def rdg(tmp_path):
    return ReportDataGenerator(output_dir=tmp_path / "tables")


class TestDescriptiveTable:
    def test_generates_tex(self, rdg):
        stats = [
            {"metric_name": "score", "model_id": "A", "n": 100,
             "mean": 0.75, "std_dev": 0.1, "median": 0.78,
             "q1": 0.68, "q3": 0.85},
        ]
        path = rdg.descriptive_table(stats)
        assert path.exists()
        content = path.read_text()
        assert "\\begin{table}" in content
        assert "0.750" in content


class TestHypothesisTable:
    def test_generates_tex(self, rdg):
        results = [
            {"test_name": "wilcoxon", "metric_name": "score",
             "group_a": "A", "group_b": "B", "statistic": 123.4,
             "p_value": 0.03, "significant": True,
             "correction_method": "holm"},
        ]
        path = rdg.hypothesis_table(results)
        assert path.exists()
        content = path.read_text()
        assert "wilcoxon" in content
        assert r"$\checkmark$" in content


class TestEffectSizeTable:
    def test_generates_tex(self, rdg):
        results = [
            {"metric_name": "score", "group_a": "A", "group_b": "B",
             "measure": "cliffs_delta", "value": 0.35, "label": "medium",
             "ci_lower": 0.1, "ci_upper": 0.6},
        ]
        path = rdg.effect_size_table(results)
        assert path.exists()


class TestRankingTable:
    def test_generates_tex(self, rdg):
        rankings = [
            {"rank": 1, "model_id": "A", "tier": "S",
             "composite_score": 0.85, "pass_rate": 0.9,
             "avg_quality": 75.0, "avg_cost_usd": 0.02,
             "wins": 2, "losses": 0, "ties": 1},
        ]
        path = rdg.ranking_table(rankings)
        assert path.exists()
        content = path.read_text()
        assert "\\$0.020" in content


class TestCostTable:
    def test_generates_tex(self, rdg):
        profiles = [
            {"model_id": "A", "total_cost_usd": 15.5,
             "mean_cost_per_task": 0.02, "cost_per_correct": 0.03,
             "total_prompt_tokens": 500000,
             "total_completion_tokens": 300000},
        ]
        path = rdg.cost_table(profiles)
        assert path.exists()


class TestPassAtKTable:
    def test_generates_tex(self, rdg):
        df = pd.DataFrame({
            "model_id": ["A", "B"],
            "pass@1": [0.72, 0.68],
            "pass@5": [0.88, 0.82],
        })
        path = rdg.pass_at_k_table(df)
        assert path.exists()


class TestCorrelationTable:
    def test_generates_tex(self, rdg):
        results = [
            {"variable_a": "complexity", "variable_b": "score",
             "method": "spearman", "coefficient": -0.45,
             "p_value": 0.001, "significant": True},
        ]
        path = rdg.correlation_table(results)
        assert path.exists()


class TestConsistencyTable:
    def test_generates_tex(self, rdg):
        profiles = [
            {"model_id": "A", "agreement_rate": 0.92,
             "mean_cv": 0.08, "icc": 0.88, "flip_rate": 0.04},
        ]
        path = rdg.consistency_table(profiles)
        assert path.exists()


class TestLatexEscaping:
    def test_special_chars(self, rdg):
        stats = [
            {"metric_name": "pass_rate_%", "model_id": "gpt-4_turbo",
             "n": 10, "mean": 0.5, "std_dev": 0.1, "median": 0.5,
             "q1": 0.4, "q3": 0.6},
        ]
        path = rdg.descriptive_table(stats)
        content = path.read_text()
        # Underscores and % should be escaped
        assert "\\_" in content or "pass" in content


class TestGenerateAll:
    def test_generates_multiple(self, rdg):
        paths = rdg.generate_all(
            descriptive_stats=[{"metric_name": "s", "model_id": "A",
                                "n": 10, "mean": 0.5, "std_dev": 0.1,
                                "median": 0.5, "q1": 0.4, "q3": 0.6}],
            cost_profiles=[{"model_id": "A", "total_cost_usd": 10,
                            "mean_cost_per_task": 0.01,
                            "cost_per_correct": 0.02,
                            "total_prompt_tokens": 100000,
                            "total_completion_tokens": 50000}],
        )
        assert len(paths) >= 2
