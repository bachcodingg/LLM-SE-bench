"""
test_recommender.py — Tests for the constraint-based model recommender.

Covers:
- ConstraintChecker (feasibility, violations)
- ModelRecommender (recommend, all-profiles, confidence, rationale)
- Edge cases (empty data, no feasible models)
"""

from __future__ import annotations

from framework.recommender import (
    ConstraintChecker,
    ConstraintViolation,
    ModelRecommender,
    RationaleGenerator,
)

# =====================================================================
# ConstraintChecker
# =====================================================================


class TestConstraintChecker:
    def test_no_constraints(self, synthetic_summaries):
        checker = ConstraintChecker(synthetic_summaries, {})
        feasible = checker.feasible_models()
        assert len(feasible) == 3

    def test_max_cost_filters(self, synthetic_summaries):
        # GPT-4 cost mean = 0.065 → should be excluded at 0.05
        checker = ConstraintChecker(
            synthetic_summaries, {"max_cost_usd": 0.05}
        )
        feasible = checker.feasible_models()
        assert "gpt-4-turbo" not in feasible
        assert "claude-3.5-sonnet" in feasible
        assert "gemini-1.5-pro" in feasible

    def test_min_pass_rate_filters(self, synthetic_summaries):
        # Gemini pass_rate mean = 0.68 → excluded at 0.70
        checker = ConstraintChecker(
            synthetic_summaries, {"min_pass_rate": 0.70}
        )
        feasible = checker.feasible_models()
        assert "gemini-1.5-pro" not in feasible

    def test_max_latency_filters(self, synthetic_summaries):
        # GPT-4 latency = 5200 → excluded at 4000
        checker = ConstraintChecker(
            synthetic_summaries, {"max_latency_ms": 4000}
        )
        feasible = checker.feasible_models()
        assert "gpt-4-turbo" not in feasible
        assert "gemini-1.5-pro" in feasible

    def test_combined_constraints(self, synthetic_summaries):
        checker = ConstraintChecker(
            synthetic_summaries,
            {"max_cost_usd": 0.05, "min_pass_rate": 0.70},
        )
        feasible = checker.feasible_models()
        # Only Claude survives: Gemini fails pass_rate, GPT-4 fails cost
        assert feasible == ["claude-3.5-sonnet"]

    def test_check_returns_violations(self, synthetic_summaries):
        checker = ConstraintChecker(
            synthetic_summaries, {"max_cost_usd": 0.05}
        )
        vs = checker.check("gpt-4-turbo")
        assert len(vs) == 1
        assert vs[0].constraint == "max_cost_usd"
        assert vs[0].actual > 0.05

    def test_infeasible_models(self, synthetic_summaries):
        checker = ConstraintChecker(
            synthetic_summaries, {"max_cost_usd": 0.05}
        )
        infeasible = checker.infeasible_models()
        assert "gpt-4-turbo" in infeasible
        assert "claude-3.5-sonnet" not in infeasible

    def test_all_pass(self, synthetic_summaries):
        # Very relaxed constraints
        checker = ConstraintChecker(
            synthetic_summaries,
            {"max_cost_usd": 1.0, "min_pass_rate": 0.0, "max_latency_ms": 100000},
        )
        assert len(checker.feasible_models()) == 3

    def test_violation_str(self):
        v = ConstraintViolation(
            model_id="m1", constraint="max_cost_usd",
            limit=0.05, actual=0.10, metric_name="cost_usd",
        )
        s = str(v)
        assert "m1" in s
        assert "max_cost_usd" in s


# =====================================================================
# ModelRecommender
# =====================================================================


class TestModelRecommender:
    def test_recommend_basic(self, synthetic_summaries):
        rec = ModelRecommender(synthetic_summaries)
        result = rec.recommend()
        assert result.recommended_model in {
            "claude-3.5-sonnet", "gpt-4-turbo", "gemini-1.5-pro"
        }
        assert 0.0 <= result.confidence <= 1.0
        assert result.rationale

    def test_recommend_with_profile(self, synthetic_summaries, profiles_dir):
        rec = ModelRecommender(
            synthetic_summaries,
            profile_name="devops",
            profiles_dir=str(profiles_dir),
        )
        result = rec.recommend(use_case="devops")
        assert result.use_case == "devops"
        assert result.recommendation_id.startswith("rec-")

    def test_recommend_with_constraints(self, synthetic_summaries):
        rec = ModelRecommender(
            synthetic_summaries,
            constraints={"max_cost_usd": 0.05},
        )
        result = rec.recommend()
        # GPT-4 should be excluded
        assert result.recommended_model != "gpt-4-turbo"
        assert "max_cost_usd=0.05" in result.constraints_applied

    def test_no_feasible_falls_back(self, synthetic_summaries):
        rec = ModelRecommender(
            synthetic_summaries,
            constraints={"max_cost_usd": 0.001},  # nothing passes
        )
        result = rec.recommend()
        assert result.warnings  # should have a warning
        assert result.recommended_model  # still gives a recommendation

    def test_runner_up(self, synthetic_summaries):
        rec = ModelRecommender(synthetic_summaries)
        result = rec.recommend()
        assert result.runner_up_model

    def test_confidence_single_model(self):
        """With only one model, confidence should be 1.0."""
        try:
            from contracts import StatisticalSummary
        except ImportError:
            from datetime import datetime

            from pydantic import BaseModel
            from pydantic import Field as PField

            class StatisticalSummary(BaseModel):
                metric_name: str = ""
                model_id: str = ""
                mean: float = 0.0
                std_dev: float = 0.0
                computed_at: datetime = PField(default_factory=datetime.utcnow)

        single = [
            StatisticalSummary(metric_name="pass_rate", model_id="only-model", mean=0.9, std_dev=0.02),
            StatisticalSummary(metric_name="cost_usd", model_id="only-model", mean=0.03, std_dev=0.01),
            StatisticalSummary(metric_name="latency_ms", model_id="only-model", mean=3000, std_dev=500),
            StatisticalSummary(metric_name="maintainability_index", model_id="only-model", mean=70, std_dev=5),
            StatisticalSummary(metric_name="consistency_score", model_id="only-model", mean=0.88, std_dev=0.04),
        ]
        rec = ModelRecommender(single)
        result = rec.recommend()
        assert result.confidence == 1.0

    def test_recommend_empty(self):
        rec = ModelRecommender([])
        result = rec.recommend()
        assert result.recommended_model == "none"
        assert result.confidence == 0.0

    def test_recommend_all_profiles(self, synthetic_summaries, profiles_dir):
        rec = ModelRecommender(
            synthetic_summaries,
            profiles_dir=str(profiles_dir),
        )
        recs = rec.recommend_all_profiles()
        assert len(recs) >= 3
        use_cases = {r.use_case for r in recs}
        assert "devops" in use_cases
        assert "audit" in use_cases
        assert "budget" in use_cases

    def test_to_dict(self, synthetic_summaries):
        rec = ModelRecommender(synthetic_summaries)
        result = rec.recommend()
        d = rec.to_dict(result)
        assert "recommendation_id" in d
        assert "recommended_model" in d
        assert "rationale" in d


# =====================================================================
# RationaleGenerator
# =====================================================================


class TestRationaleGenerator:
    def test_generates_text(self, synthetic_summaries):
        from framework.decision_matrix import DecisionMatrixEngine

        engine = DecisionMatrixEngine(synthetic_summaries)
        matrices = engine.build()
        top = matrices[0]
        runner = matrices[1] if len(matrices) > 1 else None

        rationale = RationaleGenerator.generate(
            top=top,
            runner_up=runner,
            use_case="test",
            constraints_applied=["max_cost=0.05"],
            infeasible={},
            confidence=0.85,
        )
        assert "test" in rationale
        assert top.model_id in rationale
        assert "85%" in rationale
