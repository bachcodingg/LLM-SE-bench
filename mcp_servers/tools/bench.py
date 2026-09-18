"""
mcp_servers.tools.bench — the benchmark itself, as tool functions.

Wraps C2: dataset adapters, the orchestrator and the sandbox.

Two properties of this module are load-bearing and easy to break:

**No tool here leaks a reference solution.** ``get_task`` strips it. A
benchmark whose tasks hand out their own answers measures nothing, and an
agent that receives one has no way to know it should not look.

**Only ``run_benchmark`` can spend money, and it defaults to not.**
``dry_run`` defaults to True, the budget ceiling is checked before the
first call rather than after, and a run that projects over budget is
refused rather than started and aborted. An MCP tool is called by a model;
it must not be possible to spend real money by calling one speculatively.
"""

from __future__ import annotations

import logging
import statistics
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp_servers.models import (
    EvaluatePatchResult,
    GetRunResultsResult,
    GetTaskResult,
    ListTasksResult,
    RunBenchmarkResult,
    RunResultsSummary,
    TaskSummary,
    TestSpec,
)
from mcp_servers.registry import DATASET_CATEGORY, RunRegistry, get_registry
from mcp_servers.truncation import truncate_log

logger = logging.getLogger(__name__)

__all__ = [
    "list_tasks",
    "get_task",
    "evaluate_patch",
    "run_benchmark",
    "get_run_results",
    "MEAN_COST_USD_PER_EVALUATION",
    "MAX_PATCH_CHARS",
]

#: Mean USD per evaluation from results/2026-05-run, used for the
#: pre-flight budget projection.  An estimate from one run on 58 tasks: a
#: different task mix will differ, and an unknown model has no entry at all.
MEAN_COST_USD_PER_EVALUATION: dict[str, float] = {
    "claude-sonnet-4-6": 0.0149,
    "gpt-4o": 0.0049,
    "gemini-2.5-flash": 0.0006,
}

#: Fallback projection for a model with no measured history.  Set to the
#: most expensive model measured, so an unknown model is over-estimated
#: rather than under-estimated: a budget check should fail safe.
FALLBACK_COST_USD_PER_EVALUATION = 0.015

#: Characters accepted in a candidate patch.
MAX_PATCH_CHARS = 100_000

#: Visible test cases included in a task's test_spec.
VISIBLE_CASE_LIMIT = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_state_for(adapter: Any, problem: Any) -> dict[str, str]:
    """Starting files for a task, never including the reference solution.

    ``bugfix`` tasks start from the buggy class and ``refactor`` tasks from
    the God Class; each adapter exposes its own accessor. Code-generation
    tasks start from nothing, so an empty dict is the correct answer, not a
    failure.
    """
    metadata = problem.metadata or {}
    entry_point = metadata.get("entry_point") or problem.problem_id

    for accessor in ("get_buggy_code", "get_original_code"):
        method = getattr(adapter, accessor, None)
        if not callable(method):
            continue
        try:
            source = method(problem.problem_id)
        except Exception:  # adapters raise KeyError when a task has no such code
            continue
        if source:
            return {f"{_class_name_in(source) or entry_point}.java": source}

    return {}


def _class_name_in(source: str) -> str:
    """Public class name declared in *source*, or '' when there is none."""
    import re

    match = re.search(r"public\s+(?:final\s+|abstract\s+)?class\s+(\w+)", source)
    return match.group(1) if match else ""


# ──────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────

def list_tasks(
    category: str = "",
    dataset: str = "",
    difficulty: str = "",
    limit: int = 100,
    data_dir: str = "data",
) -> ListTasksResult:
    """List benchmark tasks, optionally filtered. Returns no solutions."""
    registry = get_registry(data_dir)
    try:
        adapters = registry.adapters()
    except Exception as exc:
        return ListTasksResult(ok=False, error=f"Could not load datasets: {exc}")

    if not adapters:
        return ListTasksResult(
            ok=False,
            error=f"No datasets loaded from {data_dir!r}. Check the path.",
        )

    known_categories = set(DATASET_CATEGORY.values())
    if category and category not in known_categories:
        return ListTasksResult(
            ok=False,
            error=f"Unknown category {category!r}; expected one of "
                  f"{sorted(known_categories)}.",
        )
    if dataset and dataset not in adapters:
        return ListTasksResult(
            ok=False,
            error=f"Unknown dataset {dataset!r}; expected one of "
                  f"{sorted(adapters)}.",
        )

    matched: list[TaskSummary] = []
    for name, adapter in adapters.items():
        task_category = DATASET_CATEGORY.get(name, "unknown")
        if dataset and name != dataset:
            continue
        if category and task_category != category:
            continue
        for task_id in adapter.list_problem_ids():
            problem = adapter.get_problem(task_id)
            if difficulty and problem.difficulty != difficulty:
                continue
            try:
                num_tests = len(adapter.get_test_suite(task_id).cases)
            except KeyError:
                num_tests = 0
            matched.append(
                TaskSummary(
                    task_id=task_id,
                    dataset=name,
                    category=task_category,
                    title=problem.title,
                    difficulty=problem.difficulty,
                    tags=list(problem.tags),
                    num_tests=num_tests,
                )
            )

    return ListTasksResult(
        ok=True,
        tasks=matched[: max(0, limit)],
        total=len(matched),
    )


def get_task(task_id: str, data_dir: str = "data") -> GetTaskResult:
    """Fetch one task: its prompt, starting files and test specification.

    The reference solution is never included.
    """
    registry = get_registry(data_dir)
    location = registry.locate(task_id)
    if location is None:
        known = registry.task_ids()
        return GetTaskResult(
            ok=False,
            error=f"Unknown task {task_id!r}. {len(known)} tasks are "
                  f"available; call list_tasks to see them.",
        )

    adapter = location.adapter
    try:
        problem = adapter.get_problem(task_id)
        prompt = adapter.format_prompt(task_id)
    except Exception as exc:
        return GetTaskResult(ok=False, error=f"Could not build task {task_id!r}: {exc}")

    try:
        suite = adapter.get_test_suite(task_id)
        cases = list(suite.cases)
        suite_id = suite.suite_id
    except KeyError:
        cases, suite_id = [], ""

    visible = [case for case in cases if not case.is_hidden]
    hidden = [case for case in cases if case.is_hidden]

    junit_code = ""
    getter = getattr(adapter, "get_junit_code", None)
    if callable(getter):
        try:
            junit_code = getter(task_id) or ""
        except Exception:  # a missing suite is not a fatal error here
            junit_code = ""

    return GetTaskResult(
        ok=True,
        task_id=task_id,
        dataset=location.dataset_name,
        category=location.category,
        prompt=prompt,
        repo_state=_repo_state_for(adapter, problem),
        test_spec=TestSpec(
            suite_id=suite_id,
            junit_code=junit_code,
            num_visible_cases=len(visible),
            num_hidden_cases=len(hidden),
            visible_cases=[
                {
                    "test_id": case.test_id,
                    "input": case.input_data,
                    "expected": case.expected_output,
                }
                for case in visible[:VISIBLE_CASE_LIMIT]
            ],
        ),
        metadata=dict(problem.metadata or {}),
    )


def evaluate_patch(
    task_id: str,
    patch: str,
    data_dir: str = "data",
    timeout_s: int = 120,
) -> EvaluatePatchResult:
    """Score a candidate solution against a task's test suite.

    Makes no API call and costs nothing.
    """
    if not patch or not patch.strip():
        return EvaluatePatchResult(ok=False, error="patch is empty.")
    if len(patch) > MAX_PATCH_CHARS:
        return EvaluatePatchResult(
            ok=False,
            error=f"patch is {len(patch)} characters; the limit is "
                  f"{MAX_PATCH_CHARS}.",
        )

    registry = get_registry(data_dir)
    location = registry.locate(task_id)
    if location is None:
        return EvaluatePatchResult(
            ok=False,
            error=f"Unknown task {task_id!r}; call list_tasks to see what exists.",
        )

    from bench.sandbox.docker_sandbox import DockerSandbox
    from quality.ck_metrics import CKMetricsExtractor

    sandbox = DockerSandbox(timeout_test=timeout_s, timeout_compile=timeout_s)
    executed = sandbox.is_docker_available()

    adapter = location.adapter
    junit_code = ""
    getter = getattr(adapter, "get_junit_code", None)
    if callable(getter):
        try:
            junit_code = getter(task_id) or ""
        except Exception:
            junit_code = ""

    result = sandbox.run(
        source_code=patch,
        junit_code=junit_code,
        problem_id=task_id,
    )

    tests_total = result.tests_total
    tests_passed = result.tests_passed
    if not result.compiled:
        verdict = "error"
    elif tests_total and tests_passed == tests_total:
        verdict = "pass"
    elif tests_passed:
        verdict = "partial"
    else:
        verdict = "fail"

    metrics: dict[str, float] = {}
    try:
        extracted = CKMetricsExtractor().extract(patch, f"{task_id}.java")
        if extracted:
            primary = extracted[0]
            metrics = {
                "loc": float(primary.loc),
                "wmc": float(primary.wmc),
                "cbo": float(primary.cbo),
                "rfc": float(primary.rfc),
                "lcom": float(primary.lcom),
            }
    except Exception as exc:
        logger.debug("CK extraction failed for %s: %s", task_id, exc)

    raw = "\n".join(
        part for part in (
            result.compile_stdout, result.compile_stderr,
            result.test_stdout, result.test_stderr, result.error_message,
        ) if part
    )
    log, truncation = truncate_log(raw)

    return EvaluatePatchResult(
        ok=True,
        task_id=task_id,
        passed=verdict == "pass",
        verdict=verdict,
        tests_passed=tests_passed,
        tests_total=tests_total,
        weighted_score=round(result.pass_rate, 4),
        compile_success=result.compiled,
        metrics=metrics,
        cost_usd=0.0,
        log=log,
        truncation=truncation,
        executed=executed,
    )


def run_benchmark(
    model: str,
    task_set: str = "",
    budget_eur: float = 0.0,
    dry_run: bool = True,
    runs_per_task: int = 1,
    limit: int = 0,
    data_dir: str = "data",
    output_dir: str = "results",
    runs_dir: str = "",
) -> RunBenchmarkResult:
    """Run the benchmark for one model over a set of tasks.

    Refuses to start when the projected spend exceeds *budget_eur*.
    """
    from settings import budget_ceiling_eur, load_dotenv, usd_to_eur

    load_dotenv()

    if runs_per_task < 1:
        return RunBenchmarkResult(ok=False, error="runs_per_task must be at least 1.")

    registry = get_registry(data_dir)
    adapters = registry.adapters()
    if not adapters:
        return RunBenchmarkResult(
            ok=False, error=f"No datasets loaded from {data_dir!r}."
        )

    dataset_name = task_set or next(iter(adapters))
    if dataset_name not in adapters:
        return RunBenchmarkResult(
            ok=False,
            error=f"Unknown task_set {task_set!r}; expected one of "
                  f"{sorted(adapters)}.",
        )
    adapter = adapters[dataset_name]

    task_ids = adapter.list_problem_ids()
    if limit > 0:
        task_ids = task_ids[:limit]
    evaluations = len(task_ids) * runs_per_task

    # Pre-flight budget check, before any client is constructed.
    ceiling = budget_eur if budget_eur > 0 else budget_ceiling_eur()
    per_eval_usd = MEAN_COST_USD_PER_EVALUATION.get(
        model, FALLBACK_COST_USD_PER_EVALUATION
    )
    projected_usd = 0.0 if dry_run else evaluations * per_eval_usd
    projected_eur = usd_to_eur(projected_usd)

    if not dry_run:
        if ceiling is None:
            return RunBenchmarkResult(
                ok=False,
                dry_run=False,
                model=model,
                task_count=len(task_ids),
                estimated_cost_eur=round(projected_eur, 4),
                error=(
                    f"A real run needs a budget ceiling. Projected spend for "
                    f"{evaluations} evaluations of {model} is about "
                    f"€{projected_eur:.2f}. Pass budget_eur, or set "
                    f"LLM_SE_BENCH_BUDGET_EUR."
                ),
            )
        if projected_eur > ceiling:
            return RunBenchmarkResult(
                ok=False,
                dry_run=False,
                model=model,
                task_count=len(task_ids),
                estimated_cost_eur=round(projected_eur, 4),
                budget_eur=ceiling,
                error=(
                    f"Refused: projected €{projected_eur:.2f} for "
                    f"{evaluations} evaluations exceeds the €{ceiling:.2f} "
                    f"ceiling. Lower `limit` or `runs_per_task`, or raise the "
                    f"ceiling deliberately. The projection uses "
                    f"${per_eval_usd:.4f}/evaluation "
                    f"({'measured' if model in MEAN_COST_USD_PER_EVALUATION else 'assumed'})."
                ),
            )

    run_id = f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"

    from bench.orchestrator import BenchmarkOrchestrator, MockLLMClient, RunConfig
    from bench.results import ResultCollector
    from bench.sandbox.docker_sandbox import DockerSandbox

    try:
        if dry_run:
            clients: dict[str, Any] = {model: MockLLMClient(latency_ms=0, model_id=model)}
        else:
            from llm_gateway.clients.base import LLMClientFactory
            from llm_gateway.config import GatewayConfig
            from llm_gateway.cost_tracker import CostTracker
            from llm_gateway.cache import ResponseCache
            from settings import cache_db_path

            config = GatewayConfig()
            db_path = str(cache_db_path())
            factory = LLMClientFactory(
                config=config,
                cost_tracker=CostTracker(config=config, db_path=db_path),
                cache=ResponseCache(db_path=db_path),
            )
            clients = {model: factory.get_client(model)}
    except Exception as exc:
        return RunBenchmarkResult(
            ok=False,
            dry_run=dry_run,
            model=model,
            error=f"Could not create a client for {model!r}: {exc}",
        )

    orchestrator = BenchmarkOrchestrator(
        dataset=adapter,
        llm_clients=clients,
        sandbox=DockerSandbox(dry_run=dry_run),
        config=RunConfig(
            models=[model],
            runs_per_problem=runs_per_task,
            results_dir=Path(output_dir),
            dry_run=dry_run,
        ),
    )
    collector = ResultCollector(output_dir)

    recorded = 0
    failures: list[str] = []
    for task_id in task_ids:
        for run_index in range(1, runs_per_task + 1):
            try:
                result = orchestrator.run_single(task_id, model, run_index)
                collector.record(dataset_name, model, result)
                recorded += 1
            except Exception as exc:
                failures.append(f"{task_id} run {run_index}: {exc}")

    actual_usd = 0.0
    if not dry_run:
        try:
            actual_usd = _recorded_spend_usd(model)
        except Exception as exc:
            logger.warning("Could not read spend for %s: %s", run_id, exc)

    record = {
        "run_id": run_id,
        "started_at": _now(),
        "model": model,
        "dataset": dataset_name,
        "dry_run": dry_run,
        "runs_per_task": runs_per_task,
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "evaluations": recorded,
        "failures": failures,
        "budget_eur": ceiling or 0.0,
        "estimated_cost_eur": round(projected_eur, 4),
        "actual_cost_eur": round(usd_to_eur(actual_usd), 4),
        "output_dir": output_dir,
    }
    RunRegistry(runs_dir or None).save(run_id, record)

    return RunBenchmarkResult(
        ok=True,
        run_id=run_id,
        started=True,
        dry_run=dry_run,
        model=model,
        task_count=len(task_ids),
        evaluations=recorded,
        estimated_cost_eur=round(projected_eur, 4),
        actual_cost_eur=round(usd_to_eur(actual_usd), 4),
        budget_eur=ceiling or 0.0,
        error="; ".join(failures[:5]) if failures else "",
    )


def _recorded_spend_usd(model: str) -> float:
    """Total USD recorded for *model*, read from the cost database.

    This is the model's whole-database total, not this run's slice: cost
    records carry no run id, so attributing spend to a run would need a
    timestamp window that concurrent runs would break. It is reported as
    ``actual_cost_eur`` with that caveat stated rather than being silently
    wrong, and the budget ceiling is enforced pre-flight and does not
    depend on it.
    """
    import sqlite3

    from settings import cache_db_path

    path = cache_db_path()
    if not path.exists():
        return 0.0
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(total_cost_usd), 0) FROM cost_records WHERE model_id = ?",
            (model,),
        ).fetchone()
        return float(row[0]) if row else 0.0
    finally:
        conn.close()


def get_run_results(
    run_id: str,
    results_dir: str = "results",
    runs_dir: str = "",
) -> GetRunResultsResult:
    """Read back the results of a run started by ``run_benchmark``."""
    registry = RunRegistry(runs_dir or None)
    record = registry.load(run_id)
    if record is None:
        known = registry.list_run_ids()[:10]
        return GetRunResultsResult(
            ok=False,
            error=f"Unknown run_id {run_id!r}."
                  + (f" Recent runs: {known}" if known else " No runs recorded yet."),
        )

    import json

    model = record.get("model", "")
    dataset = record.get("dataset", "")
    task_ids = set(record.get("task_ids", []))
    jsonl = Path(record.get("output_dir", results_dir)) / dataset / model / "results.jsonl"

    rows: list[dict[str, Any]] = []
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not task_ids or row.get("problem_id") in task_ids:
                rows.append(row)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(model, dataset)].append(row)

    summaries = [
        RunResultsSummary(
            model_id=group_model,
            dataset=group_dataset,
            tasks=len({row["problem_id"] for row in group}),
            evaluations=len(group),
            passed=sum(1 for row in group if row.get("verdict") == "pass"),
            pass_rate=round(
                sum(1 for row in group if row.get("verdict") == "pass") / len(group), 4
            ),
            mean_weighted_score=round(
                statistics.mean(row.get("weighted_score", 0.0) for row in group), 4
            ),
            mean_latency_ms=round(
                statistics.mean(row.get("latency_ms", 0.0) for row in group), 1
            ),
        )
        for (group_model, group_dataset), group in sorted(grouped.items())
    ]

    return GetRunResultsResult(
        ok=True,
        run_id=run_id,
        dry_run=bool(record.get("dry_run", False)),
        started_at=record.get("started_at", ""),
        summaries=summaries,
        total_cost_eur=float(record.get("actual_cost_eur", 0.0)),
        error="" if rows else f"No result rows found at {jsonl}.",
    )
