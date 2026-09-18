"""
mcp_servers.models — request and response models for every MCP tool.

Field descriptions here become the JSON Schema an agent reads before it
calls a tool, so they are written for a model: units, preconditions, and
what a value means when it is absent.  They are not documentation for a
human maintainer; that is what the module docstrings are for.

Two conventions hold throughout:

* Every response carries ``ok`` and ``error``.  A tool that cannot do its
  job returns ``ok=False`` with a sentence explaining why, rather than
  raising — an exception across the MCP boundary becomes an opaque
  protocol error the agent cannot reason about.
* Any field that can hold unbounded text has a sibling ``truncation``
  record.  See :mod:`mcp_servers.truncation`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from mcp_servers.truncation import Truncation

__all__ = [
    "ToolResponse",
    "SourceFile",
    "Diagnostic",
    "CompileJavaResult",
    "RunTestsResult",
    "TestOutcome",
    "ClassMetricsModel",
    "CKMetricsResult",
    "CodeSmell",
    "StaticAnalysisResult",
    "TaskSummary",
    "ListTasksResult",
    "TestSpec",
    "GetTaskResult",
    "EvaluatePatchResult",
    "RunBenchmarkResult",
    "RunResultsSummary",
    "GetRunResultsResult",
    "GodClassFinding",
    "DetectGodClassesResult",
    "ExtractedClassPlan",
    "ProposeDecompositionResult",
    "MetricDeltaModel",
    "ScoreDecompositionResult",
]


class ToolResponse(BaseModel):
    """Fields every tool response carries."""

    ok: bool = Field(
        description="True when the tool did its job. False means `error` "
                    "explains why; other fields are then unreliable and "
                    "should not be parsed."
    )
    error: str = Field(
        default="",
        description="Human-readable failure reason. Empty when ok is True.",
    )


# ──────────────────────────────────────────────────────────────────────
# jvm-sandbox
# ──────────────────────────────────────────────────────────────────────

class SourceFile(BaseModel):
    """One Java source file to stage into the sandbox."""

    path: str = Field(
        description="File name, e.g. 'Calculator.java'. Must end in .java "
                    "and must match the public class it declares. Paths are "
                    "relative and may not escape the workspace: no leading "
                    "'/', no '..' segment."
    )
    content: str = Field(description="Complete file content, not a fragment.")


class Diagnostic(BaseModel):
    """A single compiler diagnostic, parsed out of javac output."""

    file: str = Field(default="", description="Source file, if javac named one.")
    line: int = Field(default=0, description="1-indexed line; 0 when unknown.")
    column: int = Field(default=0, description="1-indexed column; 0 when unknown.")
    severity: Literal["error", "warning", "note"] = Field(default="error")
    message: str = Field(default="", description="The diagnostic text.")


class CompileJavaResult(ToolResponse):
    """Outcome of compiling a set of Java sources."""

    compiled: bool = Field(
        default=False,
        description="True when javac exited 0. Note that ok=True with "
                    "compiled=False is the normal way a compile failure is "
                    "reported: the tool worked, the code did not.",
    )
    diagnostics: list[Diagnostic] = Field(
        default_factory=list,
        description="Parsed compiler diagnostics, errors first. May be "
                    "empty even when compiled is False if javac produced "
                    "output this tool could not parse — check `log`.",
    )
    log: str = Field(default="", description="Raw javac output, truncated.")
    truncation: Truncation = Field(default_factory=Truncation)
    jdk_version: int = Field(default=17, description="JDK the sandbox used.")
    duration_ms: float = Field(default=0.0, description="Wall-clock milliseconds.")
    timed_out: bool = Field(
        default=False,
        description="True when compilation hit its timeout. `compiled` is "
                    "then meaningless, not False-because-broken.",
    )
    executed: bool = Field(
        default=True,
        description="False when Docker was unavailable and only a "
                    "structural check ran. A structural check is optimistic "
                    "by construction: do not report its result as a "
                    "compilation result.",
    )


class TestOutcome(BaseModel):
    """Result of one test method."""

    test_id: str = Field(description="Test method name or identifier.")
    passed: bool = Field(description="True when the test passed.")
    error_message: str = Field(default="", description="Failure detail, truncated.")


class RunTestsResult(ToolResponse):
    """Outcome of compiling and running a JUnit suite."""

    compiled: bool = Field(default=False, description="True when the sources compiled.")
    passed: int = Field(default=0, description="Tests that passed.")
    failed: int = Field(
        default=0,
        description="Tests that ran and failed, plus tests that errored. "
                    "passed + failed == total.",
    )
    total: int = Field(
        default=0,
        description="Tests the runner found. 0 with compiled=True usually "
                    "means no @Test method was discovered, not that the "
                    "suite passed vacuously.",
    )
    outcomes: list[TestOutcome] = Field(
        default_factory=list,
        description="Per-test results when the runner reported them. Empty "
                    "when only aggregate counts were available.",
    )
    log: str = Field(default="", description="Raw runner output, truncated.")
    truncation: Truncation = Field(default_factory=Truncation)
    duration_ms: float = Field(default=0.0, description="Wall-clock milliseconds.")
    timed_out: bool = Field(
        default=False,
        description="True when the suite hit its timeout. Counts are then "
                    "partial: a non-zero `failed` does not mean a test "
                    "assertion failed.",
    )
    executed: bool = Field(
        default=True,
        description="False when Docker was unavailable and only a "
                    "structural check ran. Such counts are optimistic and "
                    "must not be reported as test results.",
    )


class ClassMetricsModel(BaseModel):
    """Chidamber-Kemerer metrics for one class. Lower is better for all six."""

    class_name: str = Field(description="Simple class name.")
    package_name: str = Field(default="", description="Package, empty for default.")
    loc: int = Field(default=0, description="Logical lines: blanks and comments excluded.")
    wmc: int = Field(
        default=0,
        description="Weighted Methods per Class: summed cyclomatic "
                    "complexity of every method. Above 47 is a smell.",
    )
    dit: int = Field(default=0, description="Depth of Inheritance Tree; 0 means extends Object.")
    noc: int = Field(default=0, description="Number of Children within the analysed sources.")
    cbo: int = Field(
        default=0,
        description="Coupling Between Objects: distinct non-stdlib types "
                    "referenced. java.* is excluded, so coupling to String "
                    "does not count. Above 14 is a smell.",
    )
    rfc: int = Field(
        default=0,
        description="Response For a Class: own methods plus distinct methods "
                    "called. Above 50 is a smell.",
    )
    lcom: int = Field(
        default=0,
        description="Lack of Cohesion of Methods (LCOM4-style). 0 or 1 means "
                    "cohesive; above 1 means the class has that many "
                    "independent responsibilities. Above 1 is a smell.",
    )
    num_methods: int = Field(default=0)
    num_fields: int = Field(default=0)
    parent_class: str = Field(default="", description="Superclass, empty when none.")


class CKMetricsResult(ToolResponse):
    """CK metrics for every class found in the submitted source."""

    classes: list[ClassMetricsModel] = Field(
        default_factory=list,
        description="One entry per class, in declaration order. Empty with "
                    "ok=True means the source parsed but declared no class.",
    )
    files_analysed: int = Field(default=0)
    parse_failures: list[str] = Field(
        default_factory=list,
        description="Files javalang could not parse. These are silently "
                    "absent from `classes`, so check this before treating "
                    "the metrics as complete.",
    )


class CodeSmell(BaseModel):
    """A detected design problem, with the evidence that triggered it."""

    smell: str = Field(
        description="Identifier: 'god_class', 'long_method', 'large_class', "
                    "'high_coupling', 'low_cohesion', 'deep_inheritance'."
    )
    severity: Literal["moderate", "severe", "extreme"] = Field(
        description="From how far past threshold the metrics sit: moderate "
                    "under 2x, severe under 3x, extreme beyond."
    )
    class_name: str = Field(description="The class it applies to.")
    method_name: str = Field(default="", description="Method, for method-level smells.")
    detail: str = Field(description="Which metrics breached which thresholds.")
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="The metric values behind the finding, so the judgement "
                    "can be checked rather than trusted.",
    )


class StaticAnalysisResult(ToolResponse):
    """CK metrics plus detected smells for a directory of Java sources."""

    classes: list[ClassMetricsModel] = Field(default_factory=list)
    smells: list[CodeSmell] = Field(
        default_factory=list,
        description="Most severe first. Empty means nothing breached a "
                    "threshold, not that the code is good.",
    )
    files_analysed: int = Field(default=0)
    parse_failures: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# llm-se-bench
# ──────────────────────────────────────────────────────────────────────

class TaskSummary(BaseModel):
    """Enough about a task to decide whether to fetch it. No solution."""

    task_id: str = Field(description="Pass this to get_task and evaluate_patch.")
    dataset: str = Field(description="One of: humaneval-java, mbpp-java, defects4j, godclass.")
    category: str = Field(
        description="What the task asks for: 'codegen', 'bugfix' or 'refactor'."
    )
    title: str = Field(default="")
    difficulty: str = Field(default="", description="easy | medium | hard.")
    tags: list[str] = Field(default_factory=list)
    num_tests: int = Field(default=0, description="Test cases in the suite, hidden ones included.")


class ListTasksResult(ToolResponse):
    """Tasks matching the filter."""

    tasks: list[TaskSummary] = Field(default_factory=list)
    total: int = Field(default=0, description="Tasks matched, before `limit` was applied.")


class TestSpec(BaseModel):
    """What the candidate solution will be judged against."""

    suite_id: str = Field(default="")
    junit_code: str = Field(
        default="",
        description="The JUnit 4 source the patch will be compiled against. "
                    "Given so a solution can match the expected class and "
                    "method names — editing it is not possible, the harness "
                    "uses its own copy.",
    )
    num_visible_cases: int = Field(
        default=0, description="Cases whose inputs and outputs are shown below."
    )
    num_hidden_cases: int = Field(
        default=0,
        description="Cases used in scoring but not shown. Non-zero means "
                    "fitting to the visible examples will not pass.",
    )
    visible_cases: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Visible input/expected pairs, at most 5.",
    )


class GetTaskResult(ToolResponse):
    """A task, ready to attempt.

    The reference solution is never included, for any task, by design.
    """

    task_id: str = Field(default="")
    dataset: str = Field(default="")
    category: str = Field(default="")
    prompt: str = Field(
        default="",
        description="The rendered task prompt, identical to the one the "
                    "benchmark sends to a model. Use it verbatim to make "
                    "results comparable with the published run.",
    )
    repo_state: dict[str, str] = Field(
        default_factory=dict,
        description="Starting files as {filename: content}. Empty for "
                    "code-generation tasks, which start from nothing; holds "
                    "the buggy or god class for bugfix and refactor tasks.",
    )
    test_spec: TestSpec = Field(default_factory=TestSpec)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Task provenance: upstream project, bug id, original CK "
                    "metrics for refactor tasks.",
    )


class EvaluatePatchResult(ToolResponse):
    """Result of scoring a candidate solution against a task.

    Makes no API call and costs nothing, so `cost_usd` is always 0.0. It is
    present so a caller can sum cost uniformly across tools.
    """

    task_id: str = Field(default="")
    passed: bool = Field(
        default=False,
        description="True when the patch compiled and every test passed.",
    )
    verdict: str = Field(
        default="fail",
        description="'pass' (all tests), 'partial' (some), 'fail' (none), "
                    "'error' (did not compile or the sandbox failed).",
    )
    tests_passed: int = Field(default=0)
    tests_total: int = Field(default=0)
    weighted_score: float = Field(
        default=0.0, description="Weight-weighted fraction of tests passed, 0.0-1.0."
    )
    compile_success: bool = Field(default=False)
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Quality metrics of the submitted code: loc, wmc, cbo, "
                    "rfc, lcom, cyclomatic_complexity. Empty when the patch "
                    "did not parse.",
    )
    cost_usd: float = Field(default=0.0, description="Always 0.0; see the class docstring.")
    log: str = Field(default="", description="Sandbox output, truncated.")
    truncation: Truncation = Field(default_factory=Truncation)
    executed: bool = Field(
        default=True,
        description="False when Docker was unavailable and only a "
                    "structural check ran. Such a result is optimistic and "
                    "is not a benchmark score.",
    )


class RunBenchmarkResult(ToolResponse):
    """Outcome of launching a benchmark run.

    A refusal on budget grounds returns ok=False with the projection in
    `error`; it does not start and then abort.
    """

    run_id: str = Field(
        default="",
        description="Pass to get_run_results. Empty when the run did not start.",
    )
    started: bool = Field(default=False, description="True when evaluations were executed.")
    dry_run: bool = Field(
        default=True,
        description="True means mock responses were used and nothing was "
                    "spent. Results from a dry run are structural checks, "
                    "not benchmark scores.",
    )
    model: str = Field(default="")
    task_count: int = Field(default=0, description="Tasks the run covered.")
    evaluations: int = Field(default=0, description="Evaluations recorded.")
    estimated_cost_eur: float = Field(
        default=0.0,
        description="Pre-flight projection from the published run's mean "
                    "cost per evaluation for this model. An estimate, not a "
                    "measurement; compare against actual_cost_eur.",
    )
    actual_cost_eur: float = Field(
        default=0.0,
        description="Total spend recorded against this model in the cost "
                    "database, converted from USD. Cost records carry no run "
                    "id, so this is a running total across all runs of this "
                    "model, not this run's slice. 0.0 for a dry run.",
    )
    budget_eur: float = Field(default=0.0, description="Ceiling this run was checked against.")


class RunResultsSummary(BaseModel):
    """Aggregate outcome for one (model, dataset) pair."""

    model_id: str = Field(default="")
    dataset: str = Field(default="")
    tasks: int = Field(default=0)
    evaluations: int = Field(default=0)
    passed: int = Field(default=0)
    pass_rate: float = Field(default=0.0, description="passed / evaluations, 0.0-1.0.")
    mean_weighted_score: float = Field(default=0.0)
    mean_latency_ms: float = Field(default=0.0)


class GetRunResultsResult(ToolResponse):
    """Stored results for a run."""

    run_id: str = Field(default="")
    dry_run: bool = Field(
        default=False,
        description="True means these are structural-check results, not "
                    "benchmark scores.",
    )
    started_at: str = Field(default="", description="ISO 8601 UTC.")
    summaries: list[RunResultsSummary] = Field(default_factory=list)
    total_cost_eur: float = Field(default=0.0)


# ──────────────────────────────────────────────────────────────────────
# god-class-tools
# ──────────────────────────────────────────────────────────────────────

class GodClassFinding(BaseModel):
    """A class that breached the God Class thresholds."""

    class_name: str = Field(description="Simple class name.")
    file: str = Field(default="", description="File it was found in.")
    severity: Literal["moderate", "severe", "extreme"] = Field(
        description="From how far past threshold the breached metrics sit."
    )
    violations: list[str] = Field(
        description="Which thresholds were breached, e.g. ['wmc', 'lcom']. "
                    "Two or more is what makes it a God Class."
    )
    metrics: ClassMetricsModel = Field(description="The full metric set behind the finding.")


class DetectGodClassesResult(ToolResponse):
    """God Class findings for a directory of Java sources.

    A class is flagged when it breaches at least 2 of 4 thresholds: WMC > 47,
    LCOM > 1, CBO > 14, RFC > 50.
    """

    god_classes: list[GodClassFinding] = Field(
        default_factory=list, description="Most severe first."
    )
    classes_analysed: int = Field(default=0)
    files_analysed: int = Field(default=0)
    parse_failures: list[str] = Field(default_factory=list)


class ExtractedClassPlan(BaseModel):
    """One class the decomposition proposes extracting."""

    proposed_name: str = Field(
        description="Suggested name, derived from the members it would hold. "
                    "A suggestion, not a requirement."
    )
    methods: list[str] = Field(description="Methods to move into it.")
    fields: list[str] = Field(description="Fields to move with them.")
    rationale: str = Field(description="Why these members belong together.")
    estimated_wmc: int = Field(
        default=0, description="Summed complexity of the listed methods."
    )


class ProposeDecompositionResult(ToolResponse):
    """A decomposition plan for one God Class.

    Produced by static analysis of field access and call structure. It makes
    no API call, is deterministic for a given input, and does **not** write
    any code: it says what to extract, not how.
    """

    class_name: str = Field(default="")
    strategy: str = Field(
        default="",
        description="Strategy applied: 'field_clusters', 'responsibility' "
                    "or 'layered'.",
    )
    is_god_class: bool = Field(
        default=False,
        description="False means the class did not breach the thresholds. A "
                    "plan is still returned, but decomposing may not be "
                    "warranted.",
    )
    extracted_classes: list[ExtractedClassPlan] = Field(
        default_factory=list,
        description="Proposed extractions. A single entry means the analysis "
                    "found one cohesive cluster and no split is indicated.",
    )
    rationale: str = Field(default="", description="How the plan was derived.")
    warnings: list[str] = Field(
        default_factory=list,
        description="Reasons to distrust the plan: unparseable members, "
                    "static-only classes, members left unassigned.",
    )
    metrics_before: ClassMetricsModel | None = Field(default=None)


class MetricDeltaModel(BaseModel):
    """Change in one metric across a refactoring."""

    metric: str = Field(description="wmc, cbo, rfc, lcom, dit, noc or loc.")
    before: float = Field(description="Value on the original class.")
    after: float = Field(
        description="Aggregate across the decomposed classes: max for "
                    "per-class metrics, sum for size."
    )
    delta: float = Field(description="after - before. Negative is improvement for wmc/cbo/rfc/lcom.")
    pct_change: float = Field(default=0.0, description="Percent change relative to before.")
    improved: bool = Field(description="True when the change went the right way for this metric.")


class ScoreDecompositionResult(ToolResponse):
    """Score for a proposed decomposition, on structure and preservation.

    Test results are **not** part of this score. A refactoring that changes
    nothing passes every test, so behaviour preservation is necessary and
    not sufficient; run `run_tests` separately and require both.
    """

    original_class: str = Field(default="")
    decomposed_classes: list[str] = Field(default_factory=list)
    overall_score: float = Field(
        default=0.0,
        description="Composite 0-100. Read `deltas` alongside it: a "
                    "composite hides disagreement between axes.",
    )
    is_god_class: bool = Field(
        default=False, description="Whether the original breached the thresholds."
    )
    god_class_resolved: bool = Field(
        default=False,
        description="True when no decomposed class is itself a God Class.",
    )
    loc_delta: float = Field(
        default=0.0,
        description="Total logical LOC after minus before. Large negative "
                    "values are suspicious: code was deleted, not moved. "
                    "Check method_coverage.",
    )
    coupling_delta: float = Field(
        default=0.0, description="Max CBO after minus before. Negative is improvement."
    )
    method_coverage: float = Field(
        default=0.0,
        description="Fraction of the original's methods present after, "
                    "0.0-1.0. Below 1.0 means behaviour was dropped, not "
                    "moved — the main anti-gaming guard.",
    )
    field_coverage: float = Field(
        default=0.0, description="Same for fields, 0.0-1.0."
    )
    deltas: list[MetricDeltaModel] = Field(default_factory=list)
    warnings: list[str] = Field(
        default_factory=list,
        description="Reasons to distrust the score: unparseable input, empty "
                    "extracted classes, coverage below 1.0.",
    )
