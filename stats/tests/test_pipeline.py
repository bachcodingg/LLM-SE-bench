"""Integration tests for stats/_pipeline.py — full analysis pipeline."""

import json

from stats._pipeline import run_full_analysis


class TestRunFullAnalysis:
    """End-to-end integration tests for the analysis pipeline."""

    def test_full_pipeline(self, results_dir, quality_dir, cost_db, tmp_path):
        output_dir = tmp_path / "analysis"
        summary = run_full_analysis(
            results_dir=results_dir,
            quality_dir=quality_dir,
            db_path=cost_db,
            output_dir=output_dir,
            n_bootstrap=500,  # fast for tests
        )

        # Summary JSON written
        summary_path = output_dir / "statistical_summary.json"
        assert summary_path.exists()
        loaded = json.loads(summary_path.read_text())

        # All top-level keys present
        expected_keys = {
            "meta", "data_summary", "descriptive", "pass_at_k",
            "hypothesis_tests", "effect_sizes", "confidence_intervals",
            "cost_profiles", "scaling_projections", "consistency",
            "flips", "correlations", "rankings",
        }
        assert expected_keys.issubset(set(loaded.keys()))

    def test_data_summary(self, results_dir, quality_dir, cost_db, tmp_path):
        output_dir = tmp_path / "analysis"
        summary = run_full_analysis(
            results_dir=results_dir,
            quality_dir=quality_dir,
            db_path=cost_db,
            output_dir=output_dir,
            n_bootstrap=200,
        )
        ds = summary["data_summary"]
        assert ds["n_results"] > 0
        assert len(ds["models"]) == 3
        assert len(ds["datasets"]) == 4

    def test_figures_generated(self, results_dir, quality_dir, cost_db, tmp_path):
        output_dir = tmp_path / "analysis"
        run_full_analysis(
            results_dir=results_dir,
            quality_dir=quality_dir,
            db_path=cost_db,
            output_dir=output_dir,
            n_bootstrap=200,
        )
        fig_dir = output_dir / "figures"
        assert fig_dir.exists()
        pdfs = list(fig_dir.glob("*.pdf"))
        assert len(pdfs) >= 8, f"Expected ≥8 figures, got {len(pdfs)}: {[p.name for p in pdfs]}"

    def test_latex_tables_generated(self, results_dir, quality_dir, cost_db, tmp_path):
        output_dir = tmp_path / "analysis"
        run_full_analysis(
            results_dir=results_dir,
            quality_dir=quality_dir,
            db_path=cost_db,
            output_dir=output_dir,
            n_bootstrap=200,
        )
        tab_dir = output_dir / "tables"
        assert tab_dir.exists()
        texs = list(tab_dir.glob("*.tex"))
        assert len(texs) >= 4, f"Expected ≥4 tables, got {len(texs)}"

    def test_rankings_ordered(self, results_dir, quality_dir, cost_db, tmp_path):
        output_dir = tmp_path / "analysis"
        summary = run_full_analysis(
            results_dir=results_dir,
            quality_dir=quality_dir,
            db_path=cost_db,
            output_dir=output_dir,
            n_bootstrap=200,
        )
        rankings = summary["rankings"]
        assert len(rankings) == 3
        # Ranks should be 1, 2, 3
        ranks = [r["rank"] for r in rankings]
        assert sorted(ranks) == [1, 2, 3]
        # Composite scores should be descending
        scores = [r["composite_score"] for r in rankings]
        assert scores == sorted(scores, reverse=True)

    def test_without_cost_db(self, results_dir, quality_dir, tmp_path):
        """Pipeline should work even without cost data."""
        output_dir = tmp_path / "analysis"
        summary = run_full_analysis(
            results_dir=results_dir,
            quality_dir=quality_dir,
            db_path=None,
            output_dir=output_dir,
            n_bootstrap=200,
        )
        assert summary["cost_profiles"] == []
        assert summary["data_summary"]["n_results"] > 0

    def test_hypothesis_tests_present(self, results_dir, quality_dir, tmp_path):
        output_dir = tmp_path / "analysis"
        summary = run_full_analysis(
            results_dir=results_dir,
            quality_dir=quality_dir,
            output_dir=output_dir,
            n_bootstrap=200,
        )
        tests = summary["hypothesis_tests"]
        assert len(tests) > 0
        test_names = {t["test_name"] for t in tests}
        assert "friedman" in test_names
        assert "wilcoxon_signed_rank" in test_names
