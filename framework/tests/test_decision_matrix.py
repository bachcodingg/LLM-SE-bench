"""
test_decision_matrix.py — Tests for the MCDA decision matrix engine.

Covers:
- ProfileLoader (loading, validation, listing, errors)
- MetricAggregator (aggregation, consistency derivation)
- Normaliser (benefit/cost criteria, edge cases)
- DecisionMatrixEngine (build, ranking, rebuild, sensitivity)
- JSON file round-trip
"""

from __future__ import annotations

import pytest

from framework.decision_matrix import (
    CRITERIA,
    DecisionMatrixEngine,
    MetricAggregator,
    ModelMetrics,
    Normaliser,
    ProfileLoader,
    ProfileSpec,
)

# =====================================================================
# ProfileLoader
# =====================================================================


class TestProfileLoader:
    def test_list_profiles(self, profiles_dir):
        loader = ProfileLoader(profiles_dir)
        names = loader.list_profiles()
        assert "devops" in names
        assert "audit" in names
        assert "budget" in names

    def test_load_devops(self, profiles_dir):
        loader = ProfileLoader(profiles_dir)
        spec = loader.get("devops")
        assert spec.name == "DevOps"
        assert spec.weights["speed"] == 0.35
        assert spec.weights["cost"] == 0.25
        assert abs(sum(spec.weights.values()) - 1.0) < 0.01

    def test_load_audit(self, profiles_dir):
        loader = ProfileLoader(profiles_dir)
        spec = loader.get("audit")
        assert spec.weights["quality"] == 0.30
        assert spec.weights["correctness"] == 0.30

    def test_load_budget(self, profiles_dir):
        loader = ProfileLoader(profiles_dir)
        spec = loader.get("budget")
        assert spec.weights["cost"] == 0.40

    def test_case_insensitive(self, profiles_dir):
        loader = ProfileLoader(profiles_dir)
        spec = loader.get("DevOps")
        assert spec.name == "DevOps"

    def test_not_found(self, profiles_dir):
        loader = ProfileLoader(profiles_dir)
        with pytest.raises(FileNotFoundError):
            loader.get("nonexistent")

    def test_get_weights(self, profiles_dir):
        loader = ProfileLoader(profiles_dir)
        w = loader.get_weights("devops")
        assert isinstance(w, dict)
        assert set(w.keys()) == set(CRITERIA)

    def test_get_constraints(self, profiles_dir):
        loader = ProfileLoader(profiles_dir)
        c = loader.get_constraints("devops")
        assert "max_latency_ms" in c

    def test_caching(self, profiles_dir):
        loader = ProfileLoader(profiles_dir)
        a = loader.get("devops")
        b = loader.get("devops")
        assert a is b

    def test_validation_bad_weights(self, tmp_path):
        import yaml

        bad = {
            "name": "Bad",
            "description": "bad",
            "weights": {"correctness": 0.50, "quality": 0.60,
                        "speed": 0.1, "cost": 0.1, "consistency": 0.1},
        }
        (tmp_path / "bad.yaml").write_text(yaml.dump(bad))
        loader = ProfileLoader(tmp_path)
        with pytest.raises(ValueError, match="sum"):
            loader.get("bad")

    def test_validation_negative_weight(self, tmp_path):
        import yaml

        bad = {
            "name": "Neg",
            "description": "neg",
            "weights": {"correctness": -0.10, "quality": 0.30,
                        "speed": 0.30, "cost": 0.30, "consistency": 0.20},
        }
        (tmp_path / "neg.yaml").write_text(yaml.dump(bad))
        loader = ProfileLoader(tmp_path)
        with pytest.raises(ValueError, match="negative"):
            loader.get("neg")

    def test_validation_unknown_criterion(self, tmp_path):
        import yaml

        bad = {
            "name": "Unk",
            "description": "unk",
            "weights": {"correctness": 0.20, "quality": 0.20,
                        "speed": 0.20, "cost": 0.20, "magic": 0.20},
        }
        (tmp_path / "unk.yaml").write_text(yaml.dump(bad))
        loader = ProfileLoader(tmp_path)
        with pytest.raises(ValueError, match="unknown"):
            loader.get("unk")


# =====================================================================
# MetricAggregator
# =====================================================================


class TestMetricAggregator:
    def test_aggregates_all_models(self, synthetic_summaries):
        agg = MetricAggregator(synthetic_summaries)
        result = agg.aggregate()
        model_ids = {m.model_id for m in result}
        assert model_ids == {"claude-3.5-sonnet", "gpt-4-turbo", "gemini-1.5-pro"}

    def test_maps_metrics_to_criteria(self, synthetic_summaries):
        agg = MetricAggregator(synthetic_summaries)
        result = agg.aggregate()
        for mm in result:
            assert "correctness" in mm.values
            assert "quality" in mm.values
            assert "speed" in mm.values
            assert "cost" in mm.values

    def test_consistency_present(self, synthetic_summaries):
        agg = MetricAggregator(synthetic_summaries)
        result = agg.aggregate()
        for mm in result:
            assert "consistency" in mm.values

    def test_empty_summaries(self):
        agg = MetricAggregator([])
        assert agg.aggregate() == []


# =====================================================================
# Normaliser
# =====================================================================


class TestNormaliser:
    def test_benefit_criterion(self):
        metrics = [
            ModelMetrics("a", {"correctness": 0.90}),
            ModelMetrics("b", {"correctness": 0.60}),
            ModelMetrics("c", {"correctness": 0.75}),
        ]
        norm = Normaliser.normalise(metrics)
        assert norm["a"]["correctness"] == 1.0
        assert norm["b"]["correctness"] == 0.0
        assert 0.0 < norm["c"]["correctness"] < 1.0

    def test_cost_criterion(self):
        metrics = [
            ModelMetrics("a", {"cost": 10.0}),
            ModelMetrics("b", {"cost": 50.0}),
            ModelMetrics("c", {"cost": 30.0}),
        ]
        norm = Normaliser.normalise(metrics)
        # Lowest cost should get highest normalised score
        assert norm["a"]["cost"] == 1.0
        assert norm["b"]["cost"] == 0.0

    def test_all_equal(self):
        metrics = [
            ModelMetrics("a", {"speed": 100.0}),
            ModelMetrics("b", {"speed": 100.0}),
        ]
        norm = Normaliser.normalise(metrics)
        assert norm["a"]["speed"] == 1.0
        assert norm["b"]["speed"] == 1.0

    def test_empty(self):
        assert Normaliser.normalise([]) == {}

    def test_single_model(self):
        metrics = [ModelMetrics("a", {"correctness": 0.7, "cost": 0.05})]
        norm = Normaliser.normalise(metrics)
        assert norm["a"]["correctness"] == 1.0
        assert norm["a"]["cost"] == 1.0

    def test_values_in_range(self, synthetic_summaries):
        agg = MetricAggregator(synthetic_summaries)
        mm = agg.aggregate()
        norm = Normaliser.normalise(mm)
        for mid, scores in norm.items():
            for c, v in scores.items():
                assert 0.0 <= v <= 1.0, f"{mid}.{c} = {v}"


# =====================================================================
# DecisionMatrixEngine
# =====================================================================


class TestDecisionMatrixEngine:
    def test_build_returns_matrices(self, synthetic_summaries):
        engine = DecisionMatrixEngine(synthetic_summaries)
        matrices = engine.build()
        assert len(matrices) == 3

    def test_ranking_order(self, synthetic_summaries):
        engine = DecisionMatrixEngine(synthetic_summaries)
        matrices = engine.build()
        scores = [m.weighted_total for m in matrices]
        assert scores == sorted(scores, reverse=True)
        assert matrices[0].rank == 1
        assert matrices[1].rank == 2
        assert matrices[2].rank == 3

    def test_equal_weights(self, synthetic_summaries):
        engine = DecisionMatrixEngine(synthetic_summaries)
        engine.build()
        w = engine.weights
        assert all(abs(v - 0.20) < 0.001 for v in w.values())

    def test_devops_profile(self, synthetic_summaries, profiles_dir):
        engine = DecisionMatrixEngine(
            synthetic_summaries,
            profile_name="devops",
            profiles_dir=str(profiles_dir),
        )
        matrices = engine.build()
        assert engine.profile_name == "devops"
        assert engine.weights["speed"] == 0.35

    def test_custom_weights(self, synthetic_summaries):
        custom = {"correctness": 0.5, "quality": 0.1, "speed": 0.1,
                  "cost": 0.2, "consistency": 0.1}
        engine = DecisionMatrixEngine(synthetic_summaries, weights=custom)
        engine.build()
        assert engine.weights == custom

    def test_rebuild_with_weights(self, synthetic_summaries):
        engine = DecisionMatrixEngine(synthetic_summaries)
        engine.build()
        original = [m.model_id for m in engine.matrices]

        new_w = {"correctness": 0.0, "quality": 0.0, "speed": 0.0,
                 "cost": 1.0, "consistency": 0.0}
        engine.rebuild_with_weights(new_w)
        # With 100% cost weight, cheapest model should be #1
        assert engine.matrices[0].model_id == "gemini-1.5-pro"

    def test_weighted_total_consistent(self, synthetic_summaries):
        engine = DecisionMatrixEngine(synthetic_summaries)
        engine.build()
        for dm in engine.matrices:
            expected = sum(
                s.weight * s.normalised_value for s in dm.scores
            )
            assert abs(dm.weighted_total - expected) < 1e-6

    def test_all_criteria_present(self, synthetic_summaries):
        engine = DecisionMatrixEngine(synthetic_summaries)
        engine.build()
        for dm in engine.matrices:
            criteria_in_scores = {s.criterion for s in dm.scores}
            assert criteria_in_scores == set(CRITERIA)

    def test_from_json_file(self, summary_json_path, profiles_dir):
        engine = DecisionMatrixEngine.from_json_file(
            summary_json_path,
            profile_name="audit",
            profiles_dir=str(profiles_dir),
        )
        matrices = engine.build()
        assert len(matrices) == 3
        assert engine.profile_name == "audit"

    def test_to_dict(self, synthetic_summaries):
        engine = DecisionMatrixEngine(synthetic_summaries)
        engine.build()
        data = engine.to_dict()
        assert len(data) == 3
        assert "rank" in data[0]
        assert "scores" in data[0]

    def test_sensitivity_analysis(self, synthetic_summaries):
        engine = DecisionMatrixEngine(synthetic_summaries)
        engine.build()
        sa = engine.sensitivity_analysis("correctness", steps=5)
        assert len(sa) == 5
        assert sa[0]["weight_value"] == 0.0
        assert sa[-1]["weight_value"] == 1.0
        for step in sa:
            assert "rankings" in step
            assert "totals" in step

    def test_sensitivity_invalid_criterion(self, synthetic_summaries):
        engine = DecisionMatrixEngine(synthetic_summaries)
        engine.build()
        with pytest.raises(ValueError):
            engine.sensitivity_analysis("invalid")

    def test_different_profiles_different_rankings(
        self, synthetic_summaries, profiles_dir
    ):
        """DevOps (speed-heavy) and Budget (cost-heavy) should rank
        Gemini higher than equal-weight does."""
        equal = DecisionMatrixEngine(synthetic_summaries)
        equal.build()

        budget = DecisionMatrixEngine(
            synthetic_summaries,
            profile_name="budget",
            profiles_dir=str(profiles_dir),
        )
        budget.build()

        # Under budget profile, Gemini (cheapest) should rank better
        gemini_rank_equal = next(
            m.rank for m in equal.matrices if m.model_id == "gemini-1.5-pro"
        )
        gemini_rank_budget = next(
            m.rank for m in budget.matrices if m.model_id == "gemini-1.5-pro"
        )
        # Gemini should be ranked at least as high under budget
        assert gemini_rank_budget <= gemini_rank_equal


# =====================================================================
# ProfileSpec validation
# =====================================================================


class TestProfileSpec:
    def test_valid(self):
        spec = ProfileSpec(
            name="Test",
            description="test",
            weights={c: 0.20 for c in CRITERIA},
        )
        spec.validate()  # should not raise

    def test_invalid_sum(self):
        spec = ProfileSpec(
            name="Bad",
            description="bad",
            weights={c: 0.10 for c in CRITERIA},
        )
        with pytest.raises(ValueError):
            spec.validate()
