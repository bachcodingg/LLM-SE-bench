"""
cli.py — Root Click CLI for llm-se-bench.

Subcommands
-----------
    gateway test         Verify LLM gateway connectivity.
    datasets validate    Validate dataset integrity.
    run                  Run the benchmark pipeline.
    quality              Compute code-quality metrics.
    stats                Run the statistical analysis pipeline.
    report               Generate a decision report (text / PDF / JSON / CSV).
    dashboard            Launch the interactive Dash dashboard.
    full                 Run the complete pipeline end-to-end.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

# Auto-load .env from project root if present
from settings import cache_db_path, load_dotenv  # noqa: E402

load_dotenv()

DEFAULT_DB_PATH = str(cache_db_path())

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("llm-se-bench")


# ── Root group ────────────────────────────────────────────────────────

@click.group()
@click.version_option("0.1.0", prog_name="llm-se-bench")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    """llm-se-bench — LLM Software Engineering Benchmark Framework."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


# ── gateway test ──────────────────────────────────────────────────────

@main.group("gateway")
def gateway_group() -> None:
    """LLM gateway management commands."""


@gateway_group.command("test")
@click.option("--model", "-m", default=None,
              help="Specific model ID to test. Omit to test all 3 providers.")
@click.option("--config", "-c", default=None, type=click.Path(exists=True),
              help="Path to gateway YAML config file.")
def gateway_test(model: str | None, config: str | None) -> None:
    """Send a trivial prompt to verify gateway connectivity."""
    from contracts import Prompt
    from llm_gateway.clients.base import LLMClientFactory
    from llm_gateway.config import GatewayConfig
    from llm_gateway.cost_tracker import CostTracker

    cfg = GatewayConfig.from_yaml(config) if config else GatewayConfig()
    cost_tracker = CostTracker(config=cfg, db_path=DEFAULT_DB_PATH)
    factory = LLMClientFactory(config=cfg, cost_tracker=cost_tracker)

    # If no model specified, test one model per provider
    if model is None:
        default_models = [
            cfg.providers["claude"].default_model,
            cfg.providers["gpt4"].default_model,
            cfg.providers["gemini"].default_model,
        ]
    else:
        default_models = [model]

    import uuid
    all_passed = True
    click.echo("llm-se-bench gateway test")
    click.echo("=" * 50)

    for m in default_models:
        click.echo(f"\nModel: {m}")
        try:
            client = factory.get_client(m)
            click.echo(f"  Provider: {client.__class__.__name__}")
        except Exception as exc:
            click.secho(f"  FAIL (client init): {exc}", fg="red", err=True)
            all_passed = False
            continue

        prompt = Prompt(
            prompt_id=f"pmt-test-{uuid.uuid4().hex[:8]}",
            problem_id="gateway-test",
            model_id=m,
            system_message="You are a helpful assistant.",
            user_message="Reply with exactly: OK",
            temperature=0.0,
            max_tokens=16,
        )
        try:
            response = client.send_prompt(prompt)
            click.secho(
                f"  Response: {response.raw_text.strip()!r}  "
                f"(tokens: {response.prompt_tokens}->{response.completion_tokens}, "
                f"latency: {response.latency_ms:.0f} ms)",
                fg="green",
            )
            click.secho("  PASS", fg="green")
        except Exception as exc:
            click.secho(f"  FAIL (API call): {exc}", fg="red", err=True)
            all_passed = False

    click.echo("\n" + "=" * 50)
    if all_passed:
        click.secho("All providers: PASS", fg="green")
    else:
        click.secho("One or more providers FAILED.", fg="red", err=True)
        sys.exit(1)


@gateway_group.command("costs")
@click.option("--summary", is_flag=True, help="Show a one-line summary per model.")
@click.option("--db", "db_path", default=DEFAULT_DB_PATH, show_default=True,
              help="Path to the SQLite cache database.")
def gateway_costs(summary: bool, db_path: str) -> None:
    """Show cost tracking data from the local cache database."""
    import sqlite3
    from pathlib import Path as _Path

    p = _Path(db_path)
    if not p.exists():
        click.echo(f"No cost database found at {db_path} (no runs yet).")
        return

    try:
        conn = sqlite3.connect(str(p))
        cur = conn.cursor()
        # Try cost_records table (created by CostTracker)
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='cost_records'"
        )
        if not cur.fetchone():
            click.echo("Cost records table not found in database.")
            conn.close()
            return

        if summary:
            cur.execute(
                "SELECT model_id, COUNT(*) as calls, "
                "SUM(total_cost_usd) as total_usd, "
                "SUM(prompt_tokens) as prompt_tok, "
                "SUM(completion_tokens) as comp_tok "
                "FROM cost_records GROUP BY model_id ORDER BY total_usd DESC"
            )
            rows = cur.fetchall()
            if not rows:
                click.echo("No cost records yet.")
            else:
                click.echo(f"{'Model':<35} {'Calls':>6} {'Prompt tok':>12} {'Comp tok':>10} {'Total USD':>12}")
                click.echo("-" * 80)
                grand = 0.0
                for model_id, calls, total_usd, ptok, ctok in rows:
                    usd = total_usd or 0.0
                    grand += usd
                    click.echo(
                        f"{model_id:<35} {calls:>6} {(ptok or 0):>12,} "
                        f"{(ctok or 0):>10,} ${usd:>11.6f}"
                    )
                click.echo("-" * 80)
                click.echo(f"{'TOTAL':<35} {'':>6} {'':>12} {'':>10} ${grand:>11.6f}")
        else:
            cur.execute(
                "SELECT model_id, total_cost_usd, prompt_tokens, completion_tokens, created_at "
                "FROM cost_records ORDER BY created_at DESC LIMIT 20"
            )
            rows = cur.fetchall()
            if not rows:
                click.echo("No cost records yet.")
            else:
                click.echo(f"{'Model':<35} {'USD':>10} {'In':>8} {'Out':>8}  {'Time'}")
                click.echo("-" * 80)
                for model_id, usd, ptok, ctok, ts in rows:
                    click.echo(f"{model_id:<35} ${(usd or 0):>9.6f} {(ptok or 0):>8,} {(ctok or 0):>8,}  {ts or ''}")
        conn.close()
    except sqlite3.Error as exc:
        click.secho(f"Database error: {exc}", fg="red", err=True)


# ── datasets validate ─────────────────────────────────────────────────

@main.group("datasets")
def datasets_group() -> None:
    """Dataset management commands."""


@datasets_group.command("validate")
@click.option("--dataset", "-d",
              type=click.Choice(["humaneval", "mbpp", "defects4j", "godclass", "all"],
                                case_sensitive=False),
              default="all",
              help="Dataset to validate.")
@click.option("--data-dir", default="data", show_default=True,
              help="Root directory containing dataset files.")
def datasets_validate(dataset: str, data_dir: str) -> None:
    """Validate dataset files for completeness and schema compliance."""
    from bench.datasets.defects4j import Defects4JDataset
    from bench.datasets.godclass import GodClassDataset
    from bench.datasets.humaneval import HumanEvalDataset
    from bench.datasets.mbpp import MBPPDataset

    datasets_map = {
        "humaneval": HumanEvalDataset,
        "mbpp": MBPPDataset,
        "defects4j": Defects4JDataset,
        "godclass": GodClassDataset,
    }

    to_validate = list(datasets_map.items()) if dataset == "all" else [(dataset, datasets_map[dataset])]

    all_ok = True
    for name, cls in to_validate:
        try:
            ds = cls(data_dir=data_dir)
            problems = ds.load_problems()
            click.secho(f"  {name}: {len(problems)} problems — OK", fg="green")
        except Exception as exc:
            click.secho(f"  {name}: FAILED — {exc}", fg="red", err=True)
            all_ok = False

    if not all_ok:
        sys.exit(1)


# ── run ───────────────────────────────────────────────────────────────

_MODEL_ALIASES: dict[str, str] = {
    "claude": "claude-sonnet-4-6",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
    "haiku": "claude-haiku-4-5-20251001",
    "gpt4": "gpt-4o",
    "gpt-4": "gpt-4o",
    "gemini": "gemini-2.5-flash",
    "gemini-flash": "gemini-2.5-flash",
    "gemini-pro": "gemini-2.5-pro",
}


def _resolve_model(alias: str) -> str:
    return _MODEL_ALIASES.get(alias.lower(), alias)


@main.command("run")
@click.option("--dataset", "-d",
              type=click.Choice(["humaneval", "mbpp", "defects4j", "godclass"],
                                case_sensitive=False),
              required=True,
              help="Dataset to evaluate.")
@click.option("--models", "-m", multiple=True, required=True,
              help="Model IDs to benchmark (repeat for multiple). Aliases: claude, gpt4, gemini.")
@click.option("--runs", "-r", default=1, show_default=True,
              help="Evaluation runs per problem.")
@click.option("--limit", "-l", default=None, type=int,
              help="Limit to first N problems (default: all).")
@click.option("--output-dir", "-o", default="results", show_default=True,
              help="Directory to write result JSONL files.")
@click.option("--data-dir", default="data", show_default=True,
              help="Root directory containing dataset files.")
@click.option("--dry-run", is_flag=True,
              help="Use mock LLM responses and skip Docker (for pipeline validation).")
@click.option("--max-concurrency", default=3, show_default=True,
              help="Maximum concurrent LLM requests.")
@click.option("--max-tokens", default=16384, show_default=True,
              help="Max output tokens per LLM call (raise for thinking models).")
def run_benchmark(
    dataset: str,
    models: tuple[str, ...],
    runs: int,
    limit: int | None,
    output_dir: str,
    data_dir: str,
    dry_run: bool,
    max_concurrency: int,
    max_tokens: int,
) -> None:
    """Run the benchmark evaluation pipeline."""
    from bench.datasets.defects4j import Defects4JDataset
    from bench.datasets.godclass import GodClassDataset
    from bench.datasets.humaneval import HumanEvalDataset
    from bench.datasets.mbpp import MBPPDataset
    from bench.orchestrator import BenchmarkOrchestrator, MockLLMClient, RunConfig
    from bench.sandbox.docker_sandbox import DockerSandbox
    from llm_gateway.cache import ResponseCache
    from llm_gateway.clients.base import LLMClientFactory
    from llm_gateway.config import GatewayConfig
    from llm_gateway.cost_tracker import CostTracker

    resolved_models = [_resolve_model(m) for m in models]

    click.echo(f"Dataset:        {dataset}")
    click.echo(f"Models:         {', '.join(resolved_models)}")
    click.echo(f"Runs/problem:   {runs}")
    click.echo(f"Problem limit:  {limit or 'all'}")
    click.echo(f"Dry-run:        {dry_run}")
    click.echo()

    dataset_map = {
        "humaneval": HumanEvalDataset,
        "mbpp": MBPPDataset,
        "defects4j": Defects4JDataset,
        "godclass": GodClassDataset,
    }
    ds = dataset_map[dataset](data_dir=data_dir)

    # Build LLM clients
    if dry_run:
        llm_clients = {m: MockLLMClient(latency_ms=0, model_id=m) for m in resolved_models}
        click.echo("Using MockLLMClient (dry-run mode)")
    else:
        cfg = GatewayConfig()
        cost_tracker = CostTracker(config=cfg, db_path=DEFAULT_DB_PATH)
        response_cache = ResponseCache(db_path=DEFAULT_DB_PATH)
        factory = LLMClientFactory(config=cfg, cost_tracker=cost_tracker, cache=response_cache)
        llm_clients = {}
        for m in resolved_models:
            try:
                llm_clients[m] = factory.get_client(m)
                click.echo(f"  Client ready: {m} ({factory.get_client(m).__class__.__name__})")
            except Exception as exc:
                click.secho(f"  Cannot create client for {m}: {exc}", fg="red", err=True)
                sys.exit(1)

    sandbox = DockerSandbox(dry_run=dry_run)

    run_cfg = RunConfig(
        models=resolved_models,
        runs_per_problem=runs,
        max_concurrency=max_concurrency,
        max_tokens=max_tokens,
        results_dir=Path(output_dir),
        dry_run=dry_run,
    )

    orchestrator = BenchmarkOrchestrator(
        dataset=ds,
        llm_clients=llm_clients,
        sandbox=sandbox,
        config=run_cfg,
    )

    # Optionally restrict to first N problems
    if limit is not None:
        all_ids = ds.list_problem_ids()[:limit]
        click.echo(f"Running {len(all_ids)} problem(s): {', '.join(all_ids)}")
        click.echo()
        stats: dict[str, Any] = {"total_evaluations": 0, "successful": 0, "failed": 0, "skipped": 0}
        # run_single already writes through the orchestrator's own
        # ResultCollector; recording again here wrote every row twice.
        for model_id in resolved_models:
            for problem_id in all_ids:
                for run_id in range(1, runs + 1):
                    try:
                        result = orchestrator.run_single(problem_id, model_id, run_id)
                        verdict_str = result.verdict.value if hasattr(result.verdict, "value") else str(result.verdict)
                        score_str = f"{result.weighted_score:.2f}"
                        click.echo(
                            f"  [{model_id}] {problem_id} run#{run_id}: "
                            f"{verdict_str}  score={score_str}  "
                            f"tests={result.tests_passed}/{result.tests_total}"
                        )
                        stats["total_evaluations"] += 1
                        if result.compile_success:
                            stats["successful"] += 1
                        else:
                            stats["failed"] += 1
                    except Exception as exc:
                        click.secho(f"  [{model_id}] {problem_id} run#{run_id}: ERROR — {exc}", fg="red")
                        stats["failed"] += 1
    else:
        stats = orchestrator.run_all()

    click.echo()
    click.secho(
        f"Done — {stats.get('total_evaluations', 0)} evaluations, "
        f"{stats.get('successful', 0)} passed, "
        f"{stats.get('failed', 0)} failed, "
        f"{stats.get('skipped', 0)} skipped.",
        fg="green",
    )


# ── quality ───────────────────────────────────────────────────────────

@main.command("quality")
@click.option("--input", "-i", "input_dir", required=True,
              type=click.Path(exists=True),
              help="Directory containing generated Java source files.")
@click.option("--output", "-o", "output_dir", default="quality",
              show_default=True,
              help="Directory to write quality metric CSV files.")
@click.option("--model", "-m", default=None,
              help="Model ID label (inferred from path if omitted).")
@click.option("--dataset", "-d", default=None,
              help="Dataset label (inferred from path if omitted).")
def quality_cmd(
    input_dir: str,
    output_dir: str,
    model: str | None,
    dataset: str | None,
) -> None:
    """Compute code-quality metrics (CK, complexity, readability) for generated code."""
    from quality.reports import QualityReportGenerator

    gen = QualityReportGenerator()
    try:
        gen.analyse_results_tree(
            results_root=input_dir,
            output_root=output_dir,
        )
        click.secho(f"Quality metrics written to {output_dir}/", fg="green")
    except Exception as exc:
        click.secho(f"Quality analysis failed: {exc}", fg="red", err=True)
        logger.exception("Quality analysis error")
        sys.exit(1)


# ── stats ─────────────────────────────────────────────────────────────

@main.command("stats")
@click.option("--input", "-i", "results_dir", required=True,
              type=click.Path(exists=True),
              help="Root of results/ directory tree.")
@click.option("--quality", "-q", "quality_dir", required=True,
              type=click.Path(exists=True),
              help="Root of quality/ directory tree.")
@click.option("--output", "-o", "output_dir", default="analysis",
              show_default=True,
              help="Directory for statistical outputs.")
@click.option("--db", "db_path", default=DEFAULT_DB_PATH, show_default=True,
              type=click.Path(),
              help="SQLite file holding cost records. Without it the cost "
                   "criterion is absent from the analysis.")
@click.option("--alpha", default=0.05, show_default=True,
              help="Significance level.")
@click.option("--bootstrap", "n_bootstrap", default=10000, show_default=True,
              help="Number of bootstrap resamples.")
def stats_cmd(
    results_dir: str,
    quality_dir: str,
    output_dir: str,
    db_path: str | None,
    alpha: float,
    n_bootstrap: int,
) -> None:
    """Run the full statistical analysis pipeline (C4)."""
    from stats._pipeline import run_full_analysis

    click.echo(f"Results:  {results_dir}")
    click.echo(f"Quality:  {quality_dir}")
    click.echo(f"Output:   {output_dir}")

    try:
        summary = run_full_analysis(
            results_dir=results_dir,
            quality_dir=quality_dir,
            db_path=db_path,
            output_dir=output_dir,
            alpha=alpha,
            n_bootstrap=n_bootstrap,
        )
        n_models = len(summary.get("data_summary", {}).get("models", []))
        click.secho(f"Analysis complete — {n_models} models evaluated.", fg="green")
    except Exception as exc:
        click.secho(f"Stats pipeline failed: {exc}", fg="red", err=True)
        logger.exception("Stats pipeline error")
        sys.exit(1)


# ── report ────────────────────────────────────────────────────────────

@main.command("report")
@click.option("--input", "-i", "data_path",
              default="analysis/statistical_summary.json",
              show_default=True,
              type=click.Path(exists=True),
              help="Path to statistical_summary.json.")
@click.option("--format", "-f", "fmt",
              type=click.Choice(["text", "pdf", "json", "csv"], case_sensitive=False),
              default="text",
              show_default=True,
              help="Output format.")
@click.option("--profile", "-p",
              type=click.Choice(["devops", "audit", "budget"], case_sensitive=False),
              default="devops",
              show_default=True,
              help="Decision matrix weighting profile.")
@click.option("--output", "-o", "output_path", default=None,
              help="Output file path (defaults to reports/decision_matrix.<fmt>).")
@click.option("--use-case", default=None,
              help="Use-case label for the recommendation.")
def report_cmd(
    data_path: str,
    fmt: str,
    profile: str,
    output_path: str | None,
    use_case: str | None,
) -> None:
    """Generate a decision report from statistical summary data."""
    import json as _json

    from framework.cli_report import CLIReporter
    from framework.decision_matrix import DecisionMatrixEngine
    from framework.exporters import CSVExporter, JSONExporter
    from framework.recommender import ModelRecommender

    # Accept either a directory (auto-find statistical_summary.json) or a file
    _data_p = Path(data_path)
    if _data_p.is_dir():
        _data_p = _data_p / "statistical_summary.json"
    with open(_data_p) as fh:
        raw = _json.load(fh)

    # Load summaries — handle both a flat list and the nested dict produced by stats pipeline
    try:
        from contracts import StatisticalSummary
        rows = raw if isinstance(raw, list) else raw.get("descriptive", [])
        summaries = []
        for item in rows:
            if not isinstance(item, dict) or "metric_name" not in item:
                continue
            mapped = {
                "metric_name": item["metric_name"],
                "model_id":    item["model_id"],
                "n":           item.get("n", 0),
                "mean":        item.get("mean", 0.0),
                "std_dev":     item.get("std_dev", 0.0),
                "median":      item.get("median", 0.0),
                "min_val":     item.get("min_val", item.get("min", 0.0)),
                "max_val":     item.get("max_val", item.get("max", 0.0)),
                "ci_lower_95": item.get("ci_lower_95", 0.0),
                "ci_upper_95": item.get("ci_upper_95", 0.0),
            }
            summaries.append(StatisticalSummary(**mapped))
        if not summaries:
            click.secho("No StatisticalSummary rows found in input.", fg="red", err=True)
            sys.exit(1)
    except Exception as exc:
        click.secho(f"Failed to parse summaries: {exc}", fg="red", err=True)
        sys.exit(1)

    engine = DecisionMatrixEngine(summaries, profile_name=profile)
    engine.build()

    recommender = ModelRecommender(summaries, profile_name=profile)
    rec = recommender.recommend(use_case=use_case or profile)

    out = Path(output_path) if output_path else Path("reports") / f"decision_matrix.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "text":
        reporter = CLIReporter(engine=engine, recommendation=rec)
        text = reporter.render()
        if output_path:
            out.write_text(text)
            click.secho(f"Report written to {out}", fg="green")
        else:
            click.echo(text)

    elif fmt == "json":
        JSONExporter(engine, recommendations=[recommender.to_dict(rec)]).export(out)
        click.secho(f"JSON report written to {out}", fg="green")

    elif fmt == "csv":
        CSVExporter(engine).export(out)
        click.secho(f"CSV report written to {out}", fg="green")

    elif fmt == "pdf":
        try:
            from framework.pdf_report import PDFReporter
            reporter = PDFReporter(summaries=summaries, profile_name=profile)
            result_path = reporter.export(out)
            click.secho(f"PDF/HTML report written to {result_path}", fg="green")
        except (ImportError, OSError):
            click.secho("WeasyPrint unavailable; falling back to HTML.", fg="yellow", err=True)
            from framework.pdf_report import PDFReporter
            reporter = PDFReporter(summaries=summaries, profile_name=profile)
            result_path = reporter.export(out)
            click.secho(f"HTML report written to {result_path}", fg="green")


# ── agent ─────────────────────────────────────────────────────────────

@main.group("agent")
def agent_group() -> None:
    """Agent-mode evaluation: tool-calling loops instead of single completions."""


@agent_group.command("run")
@click.option("--task", "-t", "task_ids", multiple=True,
              help="Task id to attempt (repeat for several). Omit with --dataset.")
@click.option("--dataset", "-d", default=None,
              help="Run every task in this dataset instead of naming them.")
@click.option("--model", "-m", required=True,
              help="Model to run. Aliases: claude, gpt4, gemini.")
@click.option("--budget-eur", "-b", type=float, default=None,
              help="Hard cost ceiling per episode, in EUR. Required for a real "
                   "run; falls back to LLM_SE_BENCH_BUDGET_EUR.")
@click.option("--total-budget-eur", type=float, default=None,
              help="Ceiling for the whole run. Without it, N tasks can cost N "
                   "times the per-episode ceiling.")
@click.option("--max-steps", default=30, show_default=True,
              help="Tool calls per episode before the episode is stopped.")
@click.option("--max-tokens", default=500_000, show_default=True,
              help="Token ceiling per episode.")
@click.option("--no-progress-steps", default=5, show_default=True,
              help="Consecutive steps with no file change and no test delta "
                   "before the episode is stopped.")
@click.option("--wall-clock", default=900.0, show_default=True,
              help="Seconds before an episode is stopped.")
@click.option("--limit", "-l", default=None, type=int,
              help="Attempt only the first N tasks of --dataset.")
@click.option("--scaffold", default="react", show_default=True,
              type=click.Choice(["react", "plan_then_execute", "single_shot"]),
              help="Which agent scaffold to run. Recorded on the trajectory so "
                   "'the model is better' can be separated from 'my loop is better'.")
@click.option("--permissions", default="full", show_default=True,
              type=click.Choice(["read_only", "patch_only", "full"]),
              help="What the agent may do. Run the same tasks at each level "
                   "to measure what access actually buys.")
@click.option("--workers", default=1, show_default=True,
              help="Episodes to run concurrently. Each holds a Docker "
                   "container while its tests run.")
@click.option("--dry-run", is_flag=True,
              help="Mock the model and skip Docker. Makes no API call and "
                   "costs nothing; the results are not benchmark scores.")
@click.option("--output-dir", "-o", default="results/trajectories", show_default=True,
              help="Where trajectories are written.")
@click.option("--compare", is_flag=True,
              help="After the run, print agent versus single-shot on the same "
                   "tasks, with cost.")
def agent_run(
    task_ids: tuple[str, ...],
    dataset: str | None,
    model: str,
    budget_eur: float | None,
    total_budget_eur: float | None,
    max_steps: int,
    max_tokens: int,
    no_progress_steps: int,
    wall_clock: float,
    limit: int | None,
    scaffold: str,
    permissions: str,
    workers: int,
    dry_run: bool,
    output_dir: str,
    compare: bool,
) -> None:
    """Run the agent loop against one or more tasks."""
    from agent.runner import compare_to_single_shot, run_episodes
    from agent.termination import TerminationPolicy
    from agent.trajectory import TrajectoryStore
    from mcp_servers.registry import get_registry
    from settings import budget_ceiling_eur

    resolved_model = _resolve_model(model)

    # Resolve which tasks to attempt.
    if task_ids and dataset:
        click.secho("Use --task or --dataset, not both.", fg="red", err=True)
        sys.exit(1)
    if dataset:
        registry = get_registry("data")
        adapters = registry.adapters()
        if dataset not in adapters:
            click.secho(
                f"Unknown dataset {dataset!r}; expected one of {sorted(adapters)}.",
                fg="red", err=True,
            )
            sys.exit(1)
        selected = adapters[dataset].list_problem_ids()
    elif task_ids:
        selected = list(task_ids)
    else:
        click.secho("Supply --task or --dataset.", fg="red", err=True)
        sys.exit(1)
    if limit is not None:
        selected = selected[:limit]

    # The cost ceiling is not optional for a real run. There is no safe
    # default to assume on someone else's account.
    ceiling = budget_eur if budget_eur else budget_ceiling_eur()
    if not dry_run and not ceiling:
        click.secho(
            "A real agent run needs a cost ceiling. Pass --budget-eur, or set "
            "LLM_SE_BENCH_BUDGET_EUR. Agent episodes cost 10-100x a single-shot "
            "completion; never point an unbounded loop at a paid API.",
            fg="red", err=True,
        )
        sys.exit(1)

    policy = TerminationPolicy(
        max_cost_eur=ceiling or 1.0,
        max_steps=max_steps,
        max_tokens=max_tokens,
        no_progress_steps=no_progress_steps,
        wall_clock_seconds=wall_clock,
    )

    if dry_run:
        from agent.stub import StubAgentClient

        client: Any = StubAgentClient()
        click.echo(
            "Dry-run: scripted model, no API calls, no Docker. The stub "
            "explores and stops; it does not attempt a fix, so every episode "
            "ends unsolved by design."
        )
    else:
        from llm_gateway.cache import ResponseCache
        from llm_gateway.clients.base import LLMClientFactory
        from llm_gateway.config import GatewayConfig
        from llm_gateway.cost_tracker import CostTracker

        cfg = GatewayConfig()
        factory = LLMClientFactory(
            config=cfg,
            cost_tracker=CostTracker(config=cfg, db_path=DEFAULT_DB_PATH),
            cache=ResponseCache(db_path=DEFAULT_DB_PATH),
        )
        try:
            client = factory.get_client(resolved_model)
        except Exception as exc:
            click.secho(f"Cannot create a client for {resolved_model}: {exc}",
                        fg="red", err=True)
            sys.exit(1)

    click.echo(f"Model:          {resolved_model}")
    click.echo(f"Tasks:          {len(selected)}")
    click.echo(f"Scaffold:       {scaffold}")
    click.echo(f"Permissions:    {permissions}")
    click.echo(f"Workers:        {workers}")
    click.echo(f"Budget/episode: EUR {policy.max_cost_eur:.2f}")
    if total_budget_eur:
        click.echo(f"Budget/run:     EUR {total_budget_eur:.2f}")
    click.echo()

    store = TrajectoryStore(output_dir)
    trajectories = run_episodes(
        task_ids=selected,
        model_id=resolved_model,
        client=client,
        policy=policy,
        scaffold=scaffold,
        store=store,
        dry_run=dry_run,
        total_budget_eur=total_budget_eur,
        permissions=permissions,
        workers=workers,
    )

    if not trajectories:
        click.secho("No episodes ran.", fg="yellow")
        return

    solved = sum(1 for t in trajectories if t.solved)
    spent = sum(t.total_cost_eur for t in trajectories)
    unexecuted = sum(1 for t in trajectories if not t.tests_executed)
    for trajectory in trajectories:
        mark = "PASS" if trajectory.solved else "fail"
        click.echo(
            f"  [{mark}] {trajectory.task_id:16s} "
            f"{trajectory.num_steps:2d} steps  "
            f"EUR {trajectory.total_cost_eur:.4f}  "
            f"{trajectory.stop_condition}"
        )

    click.echo()
    click.secho(
        f"{solved}/{len(trajectories)} solved, EUR {spent:.4f} total. "
        f"Trajectories in {output_dir}/",
        fg="green" if solved else "yellow",
    )
    if unexecuted:
        click.secho(
            f"{unexecuted} episode(s) never executed their tests — the sandbox "
            f"fell back to a structural check. Those results are not benchmark "
            f"scores and no episode among them can be solved. Start Docker and "
            f"build the sandbox image to get real ones.",
            fg="yellow",
        )

    if compare:
        click.echo()
        click.echo(json.dumps(compare_to_single_shot(trajectories), indent=2))


@agent_group.command("show")
@click.argument("episode_id")
@click.option("--store", "store_dir", default="results/trajectories",
              show_default=True, help="Trajectory directory.")
@click.option("--steps/--no-steps", default=True, show_default=True,
              help="Print the step-by-step trace.")
def agent_show(episode_id: str, store_dir: str, steps: bool) -> None:
    """Print a recorded episode: its outcome, and every step."""
    from agent.trajectory import TrajectoryStore

    store = TrajectoryStore(store_dir)
    trajectory = store.load(episode_id)
    if trajectory is None:
        known = store.episode_ids()[-5:]
        click.secho(f"No episode {episode_id!r}.", fg="red", err=True)
        if known:
            click.echo(f"Recent episodes: {', '.join(known)}")
        sys.exit(1)

    click.echo(json.dumps(trajectory.summary(), indent=2))
    if not steps:
        return

    click.echo("\nSteps")
    click.echo("=" * 60)
    for step in trajectory.steps:
        click.echo(f"\n[{step.step_index}] {step.tool_name or '(no tool call)'}")
        if step.thought_text:
            click.echo(f"  thought: {step.thought_text[:300]}")
        if step.tool_args:
            click.echo(f"  args:    {json.dumps(step.tool_args, default=str)[:200]}")
        if step.result_preview:
            marker = " [error]" if step.tool_error else ""
            click.echo(f"  result{marker}: {step.result_preview[:300]}")
        if step.files_touched:
            click.echo(f"  touched: {', '.join(step.files_touched)}")
        if step.tests_passing_after is not None:
            click.echo(f"  tests:   {step.tests_passing_after}/{step.tests_total_after}")


# ── dashboard ─────────────────────────────────────────────────────────

@main.command("dashboard")
@click.option("--port", "-p", default=8050, show_default=True,
              help="Port to bind the Dash server.")
@click.option("--data", "data_path",
              default="analysis/statistical_summary.json",
              show_default=True,
              help="Path to statistical_summary.json.")
@click.option("--debug", is_flag=True,
              help="Enable Dash debug mode (auto-reload).")
def dashboard_cmd(port: int, data_path: str, debug: bool) -> None:
    """Launch the interactive Dash dashboard (C5)."""
    from framework.dashboard.app import launch_dashboard

    click.echo(f"Starting dashboard on http://0.0.0.0:{port}")
    click.echo(f"Data file: {data_path}")
    launch_dashboard(data_path=data_path, port=port, debug=debug)


# ── full ──────────────────────────────────────────────────────────────

@main.command("full")
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Path to full-pipeline YAML config file.")
@click.pass_context
def full_cmd(ctx: click.Context, config: str) -> None:
    """Run the complete pipeline: run → quality → stats → report."""
    import yaml

    with open(config) as fh:
        cfg = yaml.safe_load(fh)

    run_cfg = cfg.get("run", {})
    quality_cfg = cfg.get("quality", {})
    stats_cfg = cfg.get("stats", {})
    report_cfg = cfg.get("report", {})

    click.echo("=== Phase 1: Benchmark Run ===")
    ctx.invoke(
        run_benchmark,
        dataset=run_cfg["dataset"],
        models=tuple(run_cfg["models"]),
        runs=run_cfg.get("runs", 1),
        limit=run_cfg.get("limit"),
        output_dir=run_cfg.get("output_dir", "results"),
        data_dir=run_cfg.get("data_dir", "data"),
        dry_run=run_cfg.get("dry_run", False),
        max_concurrency=run_cfg.get("max_concurrency", 3),
    )

    click.echo("=== Phase 2: Quality Analysis ===")
    ctx.invoke(
        quality_cmd,
        input_dir=quality_cfg.get("input_dir", "results"),
        output_dir=quality_cfg.get("output_dir", "quality"),
    )

    click.echo("=== Phase 3: Statistical Analysis ===")
    ctx.invoke(
        stats_cmd,
        results_dir=stats_cfg.get("results_dir", "results"),
        quality_dir=stats_cfg.get("quality_dir", "quality"),
        output_dir=stats_cfg.get("output_dir", "analysis"),
        db_path=stats_cfg.get("db_path"),
        alpha=stats_cfg.get("alpha", 0.05),
        n_bootstrap=stats_cfg.get("n_bootstrap", 10000),
    )

    click.echo("=== Phase 4: Report ===")
    ctx.invoke(
        report_cmd,
        data_path=report_cfg.get("data_path", "analysis/statistical_summary.json"),
        fmt=report_cfg.get("format", "text"),
        profile=report_cfg.get("profile", "devops"),
        output_path=report_cfg.get("output"),
        use_case=report_cfg.get("use_case"),
    )

    click.secho("Full pipeline complete.", fg="green")


if __name__ == "__main__":
    main()
