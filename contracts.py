"""
contracts.py — Single source of truth for all shared data models in llm-se-bench.

Components import from here:
    C1 (LLM Interaction) → Prompt, LLMResponse, CostRecord
    C2 (Verification)    → Problem, TestCase, TestSuite, VerificationResult, EvaluationResult
    C3 (Quality)         → QualityMetrics, CKMetrics, ExpertRating
    C4 (Statistics)      → StatisticalSummary, TestResult, ModelRanking
    C5 (Decision)        → DecisionMatrix, Recommendation
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    """Issue severity used across verification and quality models."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Verdict(str, Enum):
    """Overall pass / fail / partial verdict for evaluations."""
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"


class RankTier(str, Enum):
    """Coarse ranking bucket for model comparison."""
    S = "S"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


# ─────────────────────────────────────────────────────────────────────
# C1 — LLM Interaction
# ─────────────────────────────────────────────────────────────────────

class Prompt(BaseModel):
    """A fully-rendered prompt ready to send to an LLM."""

    prompt_id: str = Field(..., description="Unique identifier for this prompt instance.")
    problem_id: str = Field(..., description="ID of the problem this prompt targets.")
    model_id: str = Field(..., description="Target model identifier (e.g. 'gpt-4o').")
    system_message: str = Field(default="", description="System / meta prompt.")
    user_message: str = Field(..., description="Primary user-facing prompt text.")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"json_schema_extra": {"examples": [{"prompt_id": "p-001", "problem_id": "prob-fizzbuzz", "model_id": "gpt-4o", "user_message": "Implement FizzBuzz in Python."}]}}


class LLMResponse(BaseModel):
    """Raw response received from an LLM provider."""

    response_id: str = Field(..., description="Unique response identifier.")
    prompt_id: str = Field(..., description="ID of the originating Prompt.")
    model_id: str = Field(..., description="Model that produced the response.")
    raw_text: str = Field(..., description="Complete text returned by the model.")
    extracted_code: str = Field(default="", description="Code block(s) parsed from raw_text.")
    finish_reason: str = Field(default="stop", description="Provider-reported stop reason.")
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0, description="Wall-clock latency in ms.")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class CostRecord(BaseModel):
    """Tracks monetary cost for a single LLM call."""

    record_id: str = Field(..., description="Unique cost-record identifier.")
    response_id: str = Field(..., description="Linked LLMResponse.")
    model_id: str = Field(...)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    cost_per_prompt_token: float = Field(default=0.0, ge=0.0, description="USD per token.")
    cost_per_completion_token: float = Field(default=0.0, ge=0.0, description="USD per token.")
    total_cost_usd: float = Field(default=0.0, ge=0.0, description="Total cost in USD.")
    currency: str = Field(default="USD")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def recalculate(self) -> float:
        """Recompute total_cost_usd from token counts and rates."""
        self.total_cost_usd = (
            self.prompt_tokens * self.cost_per_prompt_token
            + self.completion_tokens * self.cost_per_completion_token
        )
        return self.total_cost_usd


# ─────────────────────────────────────────────────────────────────────
# C2 — Verification & Evaluation
# ─────────────────────────────────────────────────────────────────────

class Problem(BaseModel):
    """A benchmark problem definition."""

    problem_id: str = Field(..., description="Unique problem identifier.")
    title: str = Field(default="")
    description: str = Field(default="", description="Natural-language problem statement.")
    difficulty: str = Field(default="medium", description="easy | medium | hard.")
    language: str = Field(default="python", description="Expected programming language.")
    reference_solution: str = Field(default="", description="Gold-standard solution.")
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TestCase(BaseModel):
    """A single input / expected-output pair."""

    test_id: str = Field(..., description="Unique test-case identifier.")
    input_data: Any = Field(..., description="Serialisable input value(s).")
    expected_output: Any = Field(..., description="Expected result.")
    is_hidden: bool = Field(default=False, description="Hidden from the LLM prompt.")
    weight: float = Field(default=1.0, ge=0.0, description="Relative scoring weight.")
    timeout_seconds: float = Field(default=30.0, gt=0.0)


class TestSuite(BaseModel):
    """Collection of test cases for a single problem."""

    suite_id: str = Field(..., description="Unique test-suite identifier.")
    problem_id: str = Field(...)
    cases: list[TestCase] = Field(default_factory=list)
    setup_code: str = Field(default="", description="Code executed before every test case.")
    teardown_code: str = Field(default="", description="Code executed after every test case.")

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.cases)


class VerificationResult(BaseModel):
    """Outcome of running a single test case against submitted code."""

    test_id: str = Field(...)
    passed: bool = Field(...)
    actual_output: Any = Field(default=None)
    error_message: str = Field(default="")
    execution_time_ms: float = Field(default=0.0, ge=0.0)
    severity: Severity = Field(default=Severity.INFO)


class EvaluationResult(BaseModel):
    """Aggregated evaluation for one (model, problem) pair."""

    evaluation_id: str = Field(..., description="Unique evaluation identifier.")
    response_id: str = Field(..., description="Linked LLMResponse.")
    problem_id: str = Field(...)
    suite_id: str = Field(...)
    verdict: Verdict = Field(default=Verdict.FAIL)
    tests_total: int = Field(default=0, ge=0)
    tests_passed: int = Field(default=0, ge=0)
    weighted_score: float = Field(default=0.0, ge=0.0, le=1.0, description="0-1 normalised.")
    results: list[VerificationResult] = Field(default_factory=list)
    compile_success: bool = Field(default=True)
    runtime_errors: list[str] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0, ge=0.0, description="LLM wall-clock latency in ms.")
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def pass_rate(self) -> float:
        return self.tests_passed / self.tests_total if self.tests_total else 0.0


# ─────────────────────────────────────────────────────────────────────
# C3 — Quality Analysis
# ─────────────────────────────────────────────────────────────────────

class QualityMetrics(BaseModel):
    """Surface-level code-quality indicators."""

    metrics_id: str = Field(..., description="Unique metrics identifier.")
    response_id: str = Field(..., description="Linked LLMResponse.")
    problem_id: str = Field(...)
    lines_of_code: int = Field(default=0, ge=0)
    cyclomatic_complexity: float = Field(default=0.0, ge=0.0)
    maintainability_index: float = Field(default=0.0, description="0-100 scale (higher = better).")
    halstead_volume: float = Field(default=0.0, ge=0.0)
    lint_warnings: int = Field(default=0, ge=0)
    lint_errors: int = Field(default=0, ge=0)
    type_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    docstring_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    raw_lint_output: str = Field(default="", description="Full linter output for debugging.")
    computed_at: datetime = Field(default_factory=datetime.utcnow)


class CKMetrics(BaseModel):
    """Chidamber–Kemerer OO design metrics (per class or module)."""

    metrics_id: str = Field(..., description="Unique CK-metrics identifier.")
    response_id: str = Field(...)
    class_name: str = Field(default="<module>", description="Class or module name.")
    wmc: int = Field(default=0, ge=0, description="Weighted Methods per Class.")
    dit: int = Field(default=0, ge=0, description="Depth of Inheritance Tree.")
    noc: int = Field(default=0, ge=0, description="Number of Children.")
    cbo: int = Field(default=0, ge=0, description="Coupling Between Objects.")
    rfc: int = Field(default=0, ge=0, description="Response For a Class.")
    lcom: int = Field(default=0, ge=0, description="Lack of Cohesion of Methods.")
    computed_at: datetime = Field(default_factory=datetime.utcnow)


class ExpertRating(BaseModel):
    """Human (or automated-rubric) expert rating of generated code."""

    rating_id: str = Field(..., description="Unique rating identifier.")
    response_id: str = Field(...)
    problem_id: str = Field(...)
    rater_id: str = Field(default="auto", description="Human ID or 'auto' for rubric.")
    correctness: float = Field(default=0.0, ge=0.0, le=10.0)
    readability: float = Field(default=0.0, ge=0.0, le=10.0)
    efficiency: float = Field(default=0.0, ge=0.0, le=10.0)
    design: float = Field(default=0.0, ge=0.0, le=10.0)
    overall: float = Field(default=0.0, ge=0.0, le=10.0)
    comments: str = Field(default="")
    rated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def composite_score(self) -> float:
        """Weighted average (correctness counts double)."""
        weights = {"correctness": 2.0, "readability": 1.0, "efficiency": 1.0, "design": 1.0}
        total = (
            self.correctness * weights["correctness"]
            + self.readability * weights["readability"]
            + self.efficiency * weights["efficiency"]
            + self.design * weights["design"]
        )
        return total / sum(weights.values())


# ─────────────────────────────────────────────────────────────────────
# C4 — Statistical Analysis
# ─────────────────────────────────────────────────────────────────────

class StatisticalSummary(BaseModel):
    """Descriptive statistics for a single metric across runs."""

    metric_name: str = Field(..., description="Name of the metric being summarised.")
    model_id: str = Field(...)
    n: int = Field(default=0, ge=0, description="Sample count.")
    mean: float = Field(default=0.0)
    std_dev: float = Field(default=0.0, ge=0.0)
    median: float = Field(default=0.0)
    min_val: float = Field(default=0.0)
    max_val: float = Field(default=0.0)
    ci_lower_95: float = Field(default=0.0, description="95% CI lower bound.")
    ci_upper_95: float = Field(default=0.0, description="95% CI upper bound.")
    computed_at: datetime = Field(default_factory=datetime.utcnow)


class TestResult(BaseModel):
    """Result of a single statistical hypothesis test."""

    test_name: str = Field(..., description="E.g. 'mann_whitney_u', 'wilcoxon', 'kruskal'.")
    metric_name: str = Field(...)
    group_a: str = Field(..., description="First model / group label.")
    group_b: str = Field(default="", description="Second model (empty for omnibus tests).")
    statistic: float = Field(default=0.0)
    p_value: float = Field(default=1.0, ge=0.0, le=1.0)
    effect_size: float = Field(default=0.0, description="Cohen's d, rank-biserial, etc.")
    effect_size_label: str = Field(default="", description="negligible | small | medium | large.")
    significant: bool = Field(default=False, description="True when p < alpha.")
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    correction_method: str = Field(default="none", description="E.g. 'bonferroni', 'holm'.")
    computed_at: datetime = Field(default_factory=datetime.utcnow)


class ModelRanking(BaseModel):
    """Composite ranking entry for one model."""

    model_id: str = Field(...)
    rank: int = Field(..., ge=1)
    tier: RankTier = Field(default=RankTier.C)
    composite_score: float = Field(default=0.0, description="Weighted aggregate score.")
    pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_quality: float = Field(default=0.0, ge=0.0)
    avg_cost_usd: float = Field(default=0.0, ge=0.0)
    avg_latency_ms: float = Field(default=0.0, ge=0.0)
    problems_attempted: int = Field(default=0, ge=0)
    wins: int = Field(default=0, ge=0, description="Head-to-head statistical wins.")
    losses: int = Field(default=0, ge=0)
    ties: int = Field(default=0, ge=0)
    computed_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────
# C5 — Decision Support
# ─────────────────────────────────────────────────────────────────────

class CriterionScore(BaseModel):
    """Score for one criterion inside a DecisionMatrix row."""

    criterion: str = Field(..., description="E.g. 'correctness', 'cost', 'latency'.")
    raw_value: float = Field(default=0.0)
    normalised_value: float = Field(default=0.0, ge=0.0, le=1.0, description="0-1 normalised.")
    weight: float = Field(default=1.0, ge=0.0, description="Criterion weight in composite.")


class DecisionMatrix(BaseModel):
    """Multi-criteria decision matrix for a single model."""

    matrix_id: str = Field(..., description="Unique matrix identifier.")
    model_id: str = Field(...)
    scores: list[CriterionScore] = Field(default_factory=list)
    weighted_total: float = Field(default=0.0, description="Sum of weight × normalised_value.")
    rank: int = Field(default=0, ge=0)
    computed_at: datetime = Field(default_factory=datetime.utcnow)

    def recalculate(self) -> float:
        """Recompute weighted_total from criterion scores."""
        self.weighted_total = sum(s.weight * s.normalised_value for s in self.scores)
        return self.weighted_total


class Recommendation(BaseModel):
    """Final actionable recommendation produced by C5."""

    recommendation_id: str = Field(..., description="Unique recommendation identifier.")
    use_case: str = Field(default="general", description="Target scenario label.")
    recommended_model: str = Field(..., description="Top-pick model_id.")
    runner_up_model: str = Field(default="", description="Second-best model_id.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in pick.")
    rationale: str = Field(default="", description="Human-readable justification.")
    constraints_applied: list[str] = Field(default_factory=list, description="E.g. ['budget<$0.01/call'].")
    matrix_ids: list[str] = Field(default_factory=list, description="DecisionMatrix IDs used.")
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
