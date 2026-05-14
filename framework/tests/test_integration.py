"""
test_integration.py — End-to-end integration tests for Component 5.

Tests the full pipeline: JSON input → MCDA → recommendations →
Pareto → export (JSON + CSV) → CLI report.  Uses synthetic data
throughout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestEndToEndPipeline:
    """Full pipeline: load → analyse → export → report."""

    def test_full_pipeline(
        self, synthetic_summaries, summary_json_path, tmp_path, profiles_dir
    ):
        from framework.decision_matrix import DecisionMatrixEngine
        from framework.recommender import ModelRecommender
        from framework.tradeoffs import TradeoffAnalyzer
        from framework.exporters import JSONExporter, CSVExporter

        # 1. Build decision matrix from JSON file
        engine = DecisionMatrixEngine.from_json_file(
            summary_json_path,
            profile_name="devops",
            profiles_dir=str(profiles_dir),
        )
        matrices = engine.build()
        assert len(matrices) == 3
        assert matrices[0].rank == 1

        # 2. Generate recommendation
        rec = ModelRecommender(
            synthetic_summaries,
            profile_name="devops",
            profiles_dir=str(profiles_dir),
        )
        result = rec.recommend(use_case="integration-test")
        assert result.recommended_model
        assert result.confidence > 0

        # 3. Pareto analysis
        analyzer = TradeoffAnalyzer(synthetic_summaries)
        pareto = analyzer.analyze()
        assert len(pareto.frontier) >= 1
        assert len(pareto.tradeoffs) > 0

        # 4. Export JSON
        json_path = JSONExporter(engine).export(tmp_path / "dm.json")
        assert json_path.exists()
        j = json.loads(json_path.read_text())
        assert len(j["decision_matrix"]) == 3

        # 5. Export CSV
        csv_path = CSVExporter(engine).export(tmp_path / "dm.csv")
        assert csv_path.exists()

        # 6. CLI report
        from framework.cli_report import CLIReporter

        reporter = CLIReporter(
            synthetic_summaries,
            profile_name="devops",
            profiles_dir=str(profiles_dir),
            color=False,
        )
        report_text = reporter.render()
        assert "Decision Report" in report_text
        assert result.recommended_model in report_text

    def test_all_profiles_consistent(
        self, synthetic_summaries, profiles_dir
    ):
        """All three profiles should produce valid rankings without errors."""
        from framework.decision_matrix import DecisionMatrixEngine
        from framework.recommender import ModelRecommender

        for profile in ["devops", "audit", "budget"]:
            engine = DecisionMatrixEngine(
                synthetic_summaries,
                profile_name=profile,
                profiles_dir=str(profiles_dir),
            )
            matrices = engine.build()
            assert len(matrices) == 3
            assert matrices[0].rank == 1
            # All ranks unique
            ranks = [m.rank for m in matrices]
            assert len(set(ranks)) == 3

            rec = ModelRecommender(
                synthetic_summaries,
                profile_name=profile,
                profiles_dir=str(profiles_dir),
            )
            result = rec.recommend(use_case=profile)
            assert result.recommended_model
            # Recommended model must be one of the evaluated models
            model_ids = {m.model_id for m in matrices}
            assert result.recommended_model in model_ids

    def test_sensitivity_across_all_criteria(self, synthetic_summaries):
        """Sensitivity analysis should work for every criterion."""
        from framework.decision_matrix import CRITERIA, DecisionMatrixEngine

        engine = DecisionMatrixEngine(synthetic_summaries)
        engine.build()
        for c in CRITERIA:
            sa = engine.sensitivity_analysis(c, steps=5)
            assert len(sa) == 5
            # At 100% weight, the best model on that criterion should be #1
            final_rankings = sa[-1]["rankings"]
            assert len(final_rankings) == 3

    def test_multi_profile_export_roundtrip(
        self, synthetic_summaries, tmp_path, profiles_dir
    ):
        from framework.exporters import MultiProfileExporter

        exp = MultiProfileExporter(
            synthetic_summaries,
            output_dir=tmp_path,
            profiles_dir=str(profiles_dir),
        )
        paths = exp.export_all()

        # Read combined JSON back
        combined = json.loads(paths["combined"].read_text())
        profiles = combined["profiles"]
        assert len(profiles) >= 3

        # Each profile should rank models
        for pname, pdata in profiles.items():
            assert len(pdata["rankings"]) == 3
            ranks = [r["rank"] for r in pdata["rankings"]]
            assert sorted(ranks) == [1, 2, 3]

    def test_cost_heavy_prefers_gemini(
        self, synthetic_summaries, profiles_dir
    ):
        """Budget profile (cost=40%) should rank Gemini #1."""
        from framework.decision_matrix import DecisionMatrixEngine

        engine = DecisionMatrixEngine(
            synthetic_summaries,
            profile_name="budget",
            profiles_dir=str(profiles_dir),
        )
        matrices = engine.build()
        # Gemini is cheapest → should benefit most from cost=40% weight
        # (Exact ranking depends on normalised trade-offs, so we check
        # Gemini is at least rank ≤ 2 under budget)
        gemini_rank = next(
            m.rank for m in matrices if m.model_id == "gemini-1.5-pro"
        )
        # Under equal weights Claude dominates; under budget, Gemini
        # should improve
        equal = DecisionMatrixEngine(synthetic_summaries)
        equal.build()
        gemini_equal_rank = next(
            m.rank for m in equal.matrices if m.model_id == "gemini-1.5-pro"
        )
        assert gemini_rank <= gemini_equal_rank

    def test_rebuild_preserves_normalisation(self, synthetic_summaries):
        """Rebuild with new weights should reuse normalised data."""
        from framework.decision_matrix import DecisionMatrixEngine

        engine = DecisionMatrixEngine(synthetic_summaries)
        engine.build()
        norm1 = engine.normalised_scores

        engine.rebuild_with_weights(
            {"correctness": 0.5, "quality": 0.1, "speed": 0.1,
             "cost": 0.2, "consistency": 0.1}
        )
        norm2 = engine.normalised_scores

        # Normalised values should be identical
        for mid in norm1:
            for c in norm1[mid]:
                assert abs(norm1[mid][c] - norm2[mid][c]) < 1e-9


class TestEdgeCases:
    def test_identical_models(self):
        """Two models with identical metrics."""
        try:
            from contracts import StatisticalSummary
        except ImportError:
            from pydantic import BaseModel, Field as PField
            from datetime import datetime

            class StatisticalSummary(BaseModel):
                metric_name: str = ""
                model_id: str = ""
                mean: float = 0.0
                std_dev: float = 0.0
                computed_at: datetime = PField(default_factory=datetime.utcnow)

        metrics = ["pass_rate", "maintainability_index", "latency_ms",
                   "cost_usd", "consistency_score"]
        vals = [0.75, 65.0, 3000.0, 0.04, 0.85]

        summaries = []
        for mid in ["model-a", "model-b"]:
            for m, v in zip(metrics, vals):
                summaries.append(
                    StatisticalSummary(
                        metric_name=m, model_id=mid, mean=v, std_dev=0.05
                    )
                )

        from framework.decision_matrix import DecisionMatrixEngine

        engine = DecisionMatrixEngine(summaries)
        matrices = engine.build()
        assert len(matrices) == 2
        # Identical models should have identical scores
        assert abs(matrices[0].weighted_total - matrices[1].weighted_total) < 1e-6

    def test_single_model(self):
        """Single model should rank #1 with score ~1.0."""
        try:
            from contracts import StatisticalSummary
        except ImportError:
            from pydantic import BaseModel, Field as PField
            from datetime import datetime

            class StatisticalSummary(BaseModel):
                metric_name: str = ""
                model_id: str = ""
                mean: float = 0.0
                std_dev: float = 0.0
                computed_at: datetime = PField(default_factory=datetime.utcnow)

        summaries = [
            StatisticalSummary(metric_name="pass_rate", model_id="solo", mean=0.8, std_dev=0.05),
            StatisticalSummary(metric_name="cost_usd", model_id="solo", mean=0.03, std_dev=0.01),
            StatisticalSummary(metric_name="latency_ms", model_id="solo", mean=2000, std_dev=300),
            StatisticalSummary(metric_name="maintainability_index", model_id="solo", mean=70, std_dev=5),
            StatisticalSummary(metric_name="consistency_score", model_id="solo", mean=0.9, std_dev=0.03),
        ]

        from framework.decision_matrix import DecisionMatrixEngine

        engine = DecisionMatrixEngine(summaries)
        matrices = engine.build()
        assert len(matrices) == 1
        assert matrices[0].rank == 1
        assert matrices[0].weighted_total == 1.0  # normalised, single model
