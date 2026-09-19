"""Tests for stats/_loader.py."""


import pandas as pd
import pytest

from stats._loader import (
    load_cost_data,
    load_evaluation_results,
    load_quality_metrics,
    merge_results_quality,
)


class TestLoadEvaluationResults:
    def test_loads_all(self, results_dir):
        df = load_evaluation_results(results_dir)
        assert not df.empty
        assert "dataset" in df.columns
        assert "model_id" in df.columns
        assert "problem_id" in df.columns
        assert df["model_id"].nunique() == 3
        assert df["dataset"].nunique() == 4

    def test_derived_columns(self, results_dir):
        df = load_evaluation_results(results_dir)
        assert "pass_rate" in df.columns
        assert "is_pass" in df.columns

    def test_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_evaluation_results(tmp_path / "nonexistent")

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty_results"
        d.mkdir()
        df = load_evaluation_results(d)
        assert df.empty


class TestLoadQualityMetrics:
    def test_loads_all(self, quality_dir):
        df = load_quality_metrics(quality_dir)
        assert not df.empty
        assert "dataset" in df.columns
        assert "cyclomatic_complexity" in df.columns

    def test_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_quality_metrics(tmp_path / "nonexistent")


class TestLoadCostData:
    def test_loads_from_sqlite(self, cost_db):
        df = load_cost_data(cost_db)
        assert not df.empty
        assert "model_id" in df.columns
        assert "total_cost_usd" in df.columns

    def test_missing_db(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_cost_data(tmp_path / "missing.db")


class TestMergeResultsQuality:
    def test_merge_on_response_id(self, results_df, quality_df):
        merged = merge_results_quality(results_df, quality_df)
        assert len(merged) >= len(results_df)
        assert "cyclomatic_complexity" in merged.columns or \
               any("cyclomatic" in c for c in merged.columns)

    def test_empty_quality(self, results_df):
        merged = merge_results_quality(results_df, pd.DataFrame())
        assert len(merged) == len(results_df)

    def test_empty_results(self, quality_df):
        merged = merge_results_quality(pd.DataFrame(), quality_df)
        assert merged.empty
