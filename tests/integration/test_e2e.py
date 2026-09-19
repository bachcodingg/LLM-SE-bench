"""
tests/integration/test_e2e.py — End-to-end pipeline integration test.

Validates the complete C2 → C3 → C4 → C5 data flow using:
- MockLLMClient (no real API calls)
- In-memory StatisticalSummary objects
- DecisionMatrixEngine for the final ranking

The test verifies that a valid, ranked DecisionMatrix is produced for
each model in the synthetic dataset.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from contracts import (
    DecisionMatrix,
    EvaluationResult,
    LLMResponse,
    Prompt,
    StatisticalSummary,
    Verdict,
)

# ── Helpers ───────────────────────────────────────────────────────────

def _make_prompt(problem_id: str, model_id: str) -> Prompt:
    return Prompt(
        prompt_id=f"pmt-{problem_id}-{model_id}",
        problem_id=problem_id,
        model_id=model_id,
        system_message="You are a Java expert.",
        user_message=f"Solve problem {problem_id}.",
        temperature=0.0,
        max_tokens=2048,
    )


def _make_evaluation_result(
    response: LLMResponse,
    problem_id: str,
    suite_id: str,
    tests_total: int,
    tests_passed: int,
) -> EvaluationResult:
    verdict = Verdict.PASS if tests_passed == tests_total and tests_total > 0 else Verdict.FAIL
    score = tests_passed / tests_total if tests_total > 0 else 0.0
    return EvaluationResult(
        evaluation_id=f"eval-{response.response_id}",
        response_id=response.response_id,
        problem_id=problem_id,
        suite_id=suite_id,
        verdict=verdict,
        tests_total=tests_total,
        tests_passed=tests_passed,
        weighted_score=round(score, 4),
        compile_success=True,
    )


def _make_summaries(
    model_metrics: dict[str, dict[str, float]],
) -> list[StatisticalSummary]:
    """Build StatisticalSummary objects for each model × metric pair."""
    summaries: list[StatisticalSummary] = []
    for model_id, metrics in model_metrics.items():
        for metric_name, mean_val in metrics.items():
            summaries.append(
                StatisticalSummary(
                    metric_name=metric_name,
                    model_id=model_id,
                    n=10,
                    mean=mean_val,
                    std_dev=mean_val * 0.05,
                    median=mean_val,
                    min_val=mean_val * 0.8,
                    max_val=mean_val * 1.2,
                    ci_lower_95=mean_val * 0.9,
                    ci_upper_95=mean_val * 1.1,
                    computed_at=datetime.utcnow(),
                )
            )
    return summaries


# ── Mock model metrics (synthetic benchmark data) ─────────────────────

_MOCK_MODEL_METRICS = {
    "claude-3-5-sonnet-20241022": {
        "pass_rate": 0.85,
        "cost_usd": 0.04,
        "latency_ms": 8000.0,
        "maintainability_index": 72.0,
        "weighted_score": 0.82,
    },
    "gpt-4o": {
        "pass_rate": 0.78,
        "cost_usd": 0.06,
        "latency_ms": 12000.0,
        "maintainability_index": 68.0,
        "weighted_score": 0.74,
    },
    "gemini-1.5-pro": {
        "pass_rate": 0.71,
        "cost_usd": 0.02,
        "latency_ms": 5000.0,
        "maintainability_index": 65.0,
        "weighted_score": 0.68,
    },
}


# ── C2: Mock LLM responses ────────────────────────────────────────────

class TestC2MockResponses:
    """Verify MockLLMClient produces valid LLMResponse objects."""

    def test_mock_client_produces_response(self) -> None:
        from bench.orchestrator import MockLLMClient

        client = MockLLMClient(
            default_code="public class Solution { }",
            latency_ms=0,
        )
        prompt = _make_prompt("prob-001", "mock-model")
        response = client.send_prompt(prompt)

        assert isinstance(response, LLMResponse)
        assert response.model_id == "mock-model"
        assert response.prompt_id == prompt.prompt_id
        assert "Solution" in response.extracted_code

    def test_mock_client_custom_response(self) -> None:
        from bench.orchestrator import MockLLMClient

        custom_code = "public class Custom { int x = 42; }"
        client = MockLLMClient(
            default_code="// default",
            latency_ms=0,
            custom_responses={"prob-special": custom_code},
        )
        prompt = _make_prompt("prob-special", "mock-model")
        response = client.send_prompt(prompt)

        assert custom_code in response.extracted_code

    def test_mock_client_call_count(self) -> None:
        from bench.orchestrator import MockLLMClient

        client = MockLLMClient(latency_ms=0)
        for i in range(5):
            client.send_prompt(_make_prompt(f"prob-{i}", "mock-model"))

        assert client.call_count == 5

    def test_mock_client_produces_evaluation_results(self) -> None:
        from bench.orchestrator import MockLLMClient

        client = MockLLMClient(latency_ms=0)
        results: list[EvaluationResult] = []

        for prob_id, n_pass, n_total in [("prob-001", 5, 5), ("prob-002", 1, 5)]:
            prompt = _make_prompt(prob_id, "mock-model")
            response = client.send_prompt(prompt)
            result = _make_evaluation_result(response, prob_id, "suite-001", n_total, n_pass)
            results.append(result)

        assert len(results) == 2
        assert results[0].verdict == Verdict.PASS
        assert results[1].verdict == Verdict.FAIL
        assert results[0].weighted_score == pytest.approx(1.0)
        assert results[1].weighted_score == pytest.approx(0.2)


# ── C3: Quality metrics ───────────────────────────────────────────────

class TestC3QualityMetrics:
    """Verify quality metric models can be constructed and serialised."""

    def test_quality_metrics_construction(self) -> None:
        from contracts import CKMetrics, QualityMetrics

        ck = CKMetrics(
            metrics_id="ck-001",
            response_id="resp-001",
            class_name="Solution",
            wmc=12,
            dit=2,
            noc=0,
            cbo=5,
            rfc=18,
            lcom=3,
        )
        assert ck.wmc == 12
        assert ck.class_name == "Solution"

        qm = QualityMetrics(
            metrics_id="qm-001",
            response_id="resp-001",
            problem_id="prob-001",
            cyclomatic_complexity=4.5,
            maintainability_index=72.0,
            lines_of_code=80,
            halstead_volume=450.0,
        )
        assert qm.maintainability_index == pytest.approx(72.0)
        assert qm.lines_of_code == 80

    def test_quality_metrics_serialisation(self) -> None:
        from contracts import QualityMetrics

        qm = QualityMetrics(
            metrics_id="qm-002",
            response_id="resp-002",
            problem_id="prob-001",
            cyclomatic_complexity=3.0,
            maintainability_index=65.0,
            lines_of_code=50,
            halstead_volume=300.0,
        )
        data = qm.model_dump()
        assert data["problem_id"] == "prob-001"
        assert data["maintainability_index"] == pytest.approx(65.0)
        qm2 = QualityMetrics(**data)
        assert qm2.maintainability_index == qm.maintainability_index


# ── C4: Statistical summaries ─────────────────────────────────────────

class TestC4StatisticalSummaries:
    """Verify StatisticalSummary construction and the descriptive analyser."""

    def test_summary_construction(self) -> None:
        summaries = _make_summaries(_MOCK_MODEL_METRICS)

        model_ids = {s.model_id for s in summaries}
        assert "claude-3-5-sonnet-20241022" in model_ids
        assert "gpt-4o" in model_ids
        assert "gemini-1.5-pro" in model_ids

        metric_names = {s.metric_name for s in summaries}
        assert "pass_rate" in metric_names
        assert "cost_usd" in metric_names

    def test_summary_ci_ordering(self) -> None:
        summaries = _make_summaries(_MOCK_MODEL_METRICS)
        for s in summaries:
            assert s.ci_lower_95 <= s.mean <= s.ci_upper_95, (
                f"{s.model_id}/{s.metric_name}: CI does not bracket mean"
            )

    def test_descriptive_analyzer(self) -> None:
        """DescriptiveAnalyzer can summarise evaluation results."""
        import pandas as pd

        from stats.descriptive import DescriptiveAnalyzer

        rows = []
        for model_id, metrics in _MOCK_MODEL_METRICS.items():
            for i in range(10):
                rows.append({
                    "model_id": model_id,
                    "dataset": "humaneval",
                    "problem_id": f"prob-{i:03d}",
                    "evaluation_id": f"eval-{model_id}-{i}",
                    "weighted_score": metrics["weighted_score"] + (i - 5) * 0.01,
                    "pass_rate": metrics["pass_rate"],
                    "tests_total": 5,
                    "tests_passed": round(metrics["pass_rate"] * 5),
                })
        df = pd.DataFrame(rows)

        analyser = DescriptiveAnalyzer()
        results = analyser.summarise_all_metrics(
            df, ["weighted_score"], group_by=["model_id"]
        )
        assert len(results) == 3  # one per model
        for r in results:
            assert r.mean > 0
            assert r.n > 0


# ── C5: Decision Matrix (primary integration test) ────────────────────

class TestC5DecisionMatrix:
    """Core integration: StatisticalSummaries → DecisionMatrixEngine → ranked matrices."""

    @pytest.fixture
    def summaries(self) -> list[StatisticalSummary]:
        return _make_summaries(_MOCK_MODEL_METRICS)

    def test_engine_builds_matrices(self, summaries: list[StatisticalSummary]) -> None:
        from framework.decision_matrix import DecisionMatrixEngine

        engine = DecisionMatrixEngine(summaries, profile_name="devops")
        matrices = engine.build()

        assert len(matrices) == 3
        for m in matrices:
            assert isinstance(m, DecisionMatrix)
            assert m.model_id in _MOCK_MODEL_METRICS
            assert 0.0 <= m.weighted_total <= 1.0
            assert m.rank >= 1

    def test_matrices_are_ranked(self, summaries: list[StatisticalSummary]) -> None:
        from framework.decision_matrix import DecisionMatrixEngine

        engine = DecisionMatrixEngine(summaries, profile_name="devops")
        matrices = engine.build()

        ranks = [m.rank for m in matrices]
        assert sorted(ranks) == list(range(1, len(matrices) + 1))

        scores = [m.weighted_total for m in matrices]
        assert scores == sorted(scores, reverse=True)

    def test_matrices_have_scores_for_all_criteria(
        self, summaries: list[StatisticalSummary]
    ) -> None:
        from framework.decision_matrix import CRITERIA, DecisionMatrixEngine

        engine = DecisionMatrixEngine(summaries, profile_name="devops")
        matrices = engine.build()

        for m in matrices:
            criterion_names = {s.criterion for s in m.scores}
            for c in CRITERIA:
                assert c in criterion_names, f"Criterion {c!r} missing from {m.model_id}"

    def test_all_profiles_produce_valid_matrices(
        self, summaries: list[StatisticalSummary]
    ) -> None:
        from framework.decision_matrix import DecisionMatrixEngine, ProfileLoader

        loader = ProfileLoader()
        for profile_name in loader.list_profiles():
            engine = DecisionMatrixEngine(summaries, profile_name=profile_name)
            matrices = engine.build()
            assert len(matrices) == 3, f"Profile {profile_name}: expected 3 matrices"
            top = matrices[0]
            assert top.rank == 1
            assert isinstance(top, DecisionMatrix)

    def test_recommender_produces_recommendation(
        self, summaries: list[StatisticalSummary]
    ) -> None:
        from contracts import Recommendation
        from framework.recommender import ModelRecommender

        recommender = ModelRecommender(summaries, profile_name="devops")
        rec = recommender.recommend(use_case="integration-test")

        assert isinstance(rec, Recommendation)
        assert rec.recommended_model in _MOCK_MODEL_METRICS
        assert 0.0 <= rec.confidence <= 1.0
        assert len(rec.rationale) > 0

    def test_recommender_with_constraints(
        self, summaries: list[StatisticalSummary]
    ) -> None:
        from contracts import Recommendation
        from framework.recommender import ModelRecommender

        recommender = ModelRecommender(
            summaries,
            profile_name="budget",
            constraints={"max_cost_usd": 0.05},
        )
        rec = recommender.recommend(use_case="budget-constrained")

        assert isinstance(rec, Recommendation)
        assert rec.recommended_model != ""

    def test_json_exporter(
        self, summaries: list[StatisticalSummary], tmp_path: Path
    ) -> None:
        from framework.decision_matrix import DecisionMatrixEngine
        from framework.exporters import JSONExporter

        engine = DecisionMatrixEngine(summaries, profile_name="audit")
        engine.build()

        out = tmp_path / "decision_matrix.json"
        JSONExporter(engine).export(out)

        assert out.exists()
        data = json.loads(out.read_text())
        assert "decision_matrix" in data
        assert "meta" in data
        assert data["meta"]["profile"] == "audit"

    def test_csv_exporter(
        self, summaries: list[StatisticalSummary], tmp_path: Path
    ) -> None:
        from framework.decision_matrix import DecisionMatrixEngine
        from framework.exporters import CSVExporter

        engine = DecisionMatrixEngine(summaries, profile_name="devops")
        engine.build()

        out = tmp_path / "decision_matrix.csv"
        CSVExporter(engine).export(out)

        assert out.exists()
        lines = [row for row in out.read_text().splitlines() if row.strip()]
        assert len(lines) == 4  # header + 3 models
        assert "rank" in lines[0]
        assert "model_id" in lines[0]

    def test_pareto_frontier(
        self, summaries: list[StatisticalSummary]
    ) -> None:
        from framework.tradeoffs import TradeoffAnalyzer

        analyzer = TradeoffAnalyzer(summaries)
        result = analyzer.analyze()

        assert len(result.frontier) >= 1
        for p in result.frontier:
            assert p.is_pareto is True
        for p in result.dominated:
            assert p.is_pareto is False


# ── Full pipeline smoke test ──────────────────────────────────────────

class TestFullPipelineSmoke:
    """Smoke test: write results to disk → load back → build decision matrix."""

    def test_results_written_and_read_back(self, tmp_path: Path) -> None:
        from bench.orchestrator import MockLLMClient
        from bench.results import ResultCollector

        client = MockLLMClient(latency_ms=0)
        collector = ResultCollector(tmp_path)

        models = list(_MOCK_MODEL_METRICS.keys())
        for model_id in models:
            for i in range(5):
                prob_id = f"prob-{i:03d}"
                prompt = _make_prompt(prob_id, model_id)
                response = client.send_prompt(prompt)
                n_pass = 4 if _MOCK_MODEL_METRICS[model_id]["pass_rate"] > 0.5 else 2
                result = _make_evaluation_result(response, prob_id, "suite-smoke", 5, n_pass)
                collector.record("smoke", model_id, result)

        # Load back per model
        total = 0
        for model_id in models:
            loaded = collector.load_results("smoke", model_id)
            assert len(loaded) == 5
            total += len(loaded)

        assert total == len(models) * 5

    def test_summaries_to_decision_matrix(self) -> None:
        from framework.decision_matrix import DecisionMatrixEngine

        summaries = _make_summaries(_MOCK_MODEL_METRICS)
        engine = DecisionMatrixEngine(summaries, profile_name="devops")
        matrices = engine.build()

        assert len(matrices) > 0
        top = matrices[0]
        assert isinstance(top, DecisionMatrix)
        assert top.rank == 1
        assert top.model_id in _MOCK_MODEL_METRICS
        assert top.weighted_total > 0.0

        for m in matrices:
            assert 0.0 <= m.weighted_total <= 1.0 + 1e-9, (
                f"{m.model_id}: weighted_total={m.weighted_total} out of range"
            )
