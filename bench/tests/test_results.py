"""
tests.test_results — Unit tests for ResultCollector and PassAtKCalculator.
"""

from __future__ import annotations

import json
import math
import uuid
import pytest
from pathlib import Path

from contracts import EvaluationResult, Verdict, VerificationResult, Severity
from bench.results import ResultCollector, PassAtKCalculator


# ======================================================================
# Fixtures
# ======================================================================

def _make_eval_result(
    problem_id: str = "prob_1",
    verdict: Verdict = Verdict.PASS,
    tests_total: int = 3,
    tests_passed: int = 3,
    weighted_score: float = 1.0,
) -> EvaluationResult:
    return EvaluationResult(
        evaluation_id=f"eval-{uuid.uuid4().hex[:8]}",
        response_id=f"resp-{uuid.uuid4().hex[:8]}",
        problem_id=problem_id,
        suite_id=f"suite_{problem_id}",
        verdict=verdict,
        tests_total=tests_total,
        tests_passed=tests_passed,
        weighted_score=weighted_score,
        results=[
            VerificationResult(
                test_id=f"{problem_id}_t{i}",
                passed=(i < tests_passed),
                severity=Severity.INFO if i < tests_passed else Severity.ERROR,
            )
            for i in range(tests_total)
        ],
    )


@pytest.fixture
def collector(tmp_path: Path) -> ResultCollector:
    return ResultCollector(results_dir=tmp_path / "results")


# ======================================================================
# ResultCollector Tests
# ======================================================================

class TestResultCollector:

    def test_record_creates_jsonl(self, collector: ResultCollector):
        result = _make_eval_result()
        path = collector.record("humaneval", "gpt-4", result)
        assert path.exists()
        assert path.name == "results.jsonl"

    def test_record_appends(self, collector: ResultCollector):
        r1 = _make_eval_result(problem_id="p1")
        r2 = _make_eval_result(problem_id="p2")
        collector.record("humaneval", "gpt-4", r1)
        path = collector.record("humaneval", "gpt-4", r2)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_record_valid_json(self, collector: ResultCollector):
        result = _make_eval_result()
        path = collector.record("humaneval", "gpt-4", result)
        line = path.read_text().strip()
        data = json.loads(line)
        assert data["problem_id"] == "prob_1"
        assert data["verdict"] == "pass"
        assert "recorded_at" in data

    def test_load_results(self, collector: ResultCollector):
        for i in range(5):
            collector.record("ds", "model", _make_eval_result(problem_id=f"p{i}"))
        loaded = collector.load_results("ds", "model")
        assert len(loaded) == 5
        assert all(isinstance(r, EvaluationResult) for r in loaded)

    def test_load_results_empty(self, collector: ResultCollector):
        loaded = collector.load_results("nonexistent", "model")
        assert loaded == []

    def test_get_summary(self, collector: ResultCollector):
        collector.record("ds", "m", _make_eval_result(verdict=Verdict.PASS))
        collector.record("ds", "m", _make_eval_result(verdict=Verdict.PASS))
        collector.record("ds", "m", _make_eval_result(verdict=Verdict.FAIL))
        collector.record("ds", "m", _make_eval_result(verdict=Verdict.ERROR))

        summary = collector.get_summary("ds", "m")
        assert summary["total"] == 4
        assert summary["passed"] == 2
        assert summary["failed"] == 1
        assert summary["errors"] == 1
        assert summary["pass_rate"] == pytest.approx(0.5)

    def test_get_summary_empty(self, collector: ResultCollector):
        summary = collector.get_summary("none", "none")
        assert summary["total"] == 0
        assert summary["pass_rate"] == 0.0

    def test_directory_structure(self, collector: ResultCollector, tmp_path: Path):
        collector.record("humaneval", "gpt-4", _make_eval_result())
        collector.record("humaneval", "claude-3", _make_eval_result())
        collector.record("mbpp", "gpt-4", _make_eval_result())

        results_dir = tmp_path / "results"
        assert (results_dir / "humaneval" / "gpt-4" / "results.jsonl").exists()
        assert (results_dir / "humaneval" / "claude-3" / "results.jsonl").exists()
        assert (results_dir / "mbpp" / "gpt-4" / "results.jsonl").exists()

    def test_list_datasets(self, collector: ResultCollector):
        collector.record("humaneval", "m", _make_eval_result())
        collector.record("mbpp", "m", _make_eval_result())
        datasets = collector.list_datasets()
        assert "humaneval" in datasets
        assert "mbpp" in datasets

    def test_list_models(self, collector: ResultCollector):
        collector.record("ds", "gpt-4", _make_eval_result())
        collector.record("ds", "claude-3", _make_eval_result())
        models = collector.list_models("ds")
        assert "gpt-4" in models
        assert "claude-3" in models

    def test_partial_verdict(self, collector: ResultCollector):
        result = _make_eval_result(
            verdict=Verdict.PARTIAL,
            tests_total=5,
            tests_passed=3,
            weighted_score=0.6,
        )
        collector.record("ds", "m", result)
        loaded = collector.load_results("ds", "m")
        assert loaded[0].verdict == Verdict.PARTIAL
        assert loaded[0].tests_passed == 3


# ======================================================================
# PassAtKCalculator Tests
# ======================================================================

class TestPassAtKCalculator:

    def test_pass_at_1_all_correct(self):
        # 5 samples, all correct
        p = PassAtKCalculator.pass_at_k(n=5, c=5, k=1)
        assert p == pytest.approx(1.0)

    def test_pass_at_1_none_correct(self):
        p = PassAtKCalculator.pass_at_k(n=5, c=0, k=1)
        assert p == pytest.approx(0.0)

    def test_pass_at_1_one_correct(self):
        # 5 samples, 1 correct → pass@1 = 1 - C(4,1)/C(5,1) = 1 - 4/5 = 0.2
        p = PassAtKCalculator.pass_at_k(n=5, c=1, k=1)
        assert p == pytest.approx(0.2)

    def test_pass_at_5_one_correct(self):
        # 5 samples, 1 correct → pass@5 = 1 - C(4,5)/C(5,5)
        # C(4,5) = 0, so pass@5 = 1.0
        p = PassAtKCalculator.pass_at_k(n=5, c=1, k=5)
        assert p == pytest.approx(1.0)

    def test_pass_at_k_formula(self):
        # 10 samples, 3 correct, k=1
        # pass@1 = 1 - C(7,1)/C(10,1) = 1 - 7/10 = 0.3
        p = PassAtKCalculator.pass_at_k(n=10, c=3, k=1)
        assert p == pytest.approx(0.3)

    def test_pass_at_k_medium(self):
        # 10 samples, 3 correct, k=5
        # pass@5 = 1 - C(7,5)/C(10,5)
        # C(7,5) = 21, C(10,5) = 252
        # pass@5 = 1 - 21/252 = 1 - 0.0833 = 0.9167
        p = PassAtKCalculator.pass_at_k(n=10, c=3, k=5)
        assert p == pytest.approx(1.0 - 21.0 / 252.0, abs=1e-4)

    def test_pass_at_k_n_less_than_k(self):
        # When n < k, should handle gracefully
        p = PassAtKCalculator.pass_at_k(n=3, c=1, k=5)
        assert p == 1.0  # c > 0 so we return 1.0

    def test_pass_at_k_n_less_than_k_no_correct(self):
        p = PassAtKCalculator.pass_at_k(n=3, c=0, k=5)
        assert p == 0.0

    def test_pass_at_k_c_equals_n(self):
        p = PassAtKCalculator.pass_at_k(n=10, c=10, k=1)
        assert p == 1.0

    def test_pass_at_k_n_minus_c_less_than_k(self):
        # 5 samples, 4 correct, k=3 → not enough failures to fill k
        p = PassAtKCalculator.pass_at_k(n=5, c=4, k=3)
        assert p == 1.0

    def test_compute_from_results(self):
        results = [
            _make_eval_result("p1", Verdict.PASS),
            _make_eval_result("p1", Verdict.FAIL),
            _make_eval_result("p1", Verdict.PASS),
            _make_eval_result("p2", Verdict.FAIL),
            _make_eval_result("p2", Verdict.FAIL),
            _make_eval_result("p2", Verdict.FAIL),
        ]
        pak = PassAtKCalculator.compute_from_results(results, k_values=[1])
        assert "p1" in pak
        assert "p2" in pak
        assert pak["p1"][1] == pytest.approx(
            PassAtKCalculator.pass_at_k(3, 2, 1)
        )
        assert pak["p2"][1] == pytest.approx(0.0)

    def test_compute_aggregate(self):
        results = [
            _make_eval_result("p1", Verdict.PASS),
            _make_eval_result("p1", Verdict.PASS),
            _make_eval_result("p2", Verdict.FAIL),
            _make_eval_result("p2", Verdict.FAIL),
        ]
        agg = PassAtKCalculator.compute_aggregate(results, k_values=[1])
        # p1: pass@1 = 1.0, p2: pass@1 = 0.0 → mean = 0.5
        assert agg[1] == pytest.approx(0.5)

    def test_compute_aggregate_empty(self):
        agg = PassAtKCalculator.compute_aggregate([], k_values=[1, 5])
        assert agg[1] == 0.0
        assert agg[5] == 0.0

    def test_compute_multiple_k_values(self):
        results = [
            _make_eval_result("p1", Verdict.PASS),
            _make_eval_result("p1", Verdict.FAIL),
            _make_eval_result("p1", Verdict.FAIL),
            _make_eval_result("p1", Verdict.FAIL),
            _make_eval_result("p1", Verdict.FAIL),
        ]
        pak = PassAtKCalculator.compute_from_results(results, k_values=[1, 5])
        # n=5, c=1
        assert pak["p1"][1] == pytest.approx(0.2)
        assert pak["p1"][5] == pytest.approx(1.0)

    def test_log_space_precision(self):
        """Large n values should not overflow."""
        p = PassAtKCalculator.pass_at_k(n=200, c=10, k=100)
        assert 0.0 <= p <= 1.0
        assert p > 0.99  # With 10 correct out of 200, pass@100 is very close to 1.0
