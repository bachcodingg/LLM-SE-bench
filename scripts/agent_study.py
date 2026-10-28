"""
Agent study: what each extra euro buys, and how many solves are clean.

Two sweeps over the Defects4J tasks, both with real Docker execution:

**Cost curve.** Every model runs every task at several per-episode ceilings,
with the test file read-only. Each ceiling is a separate set of episodes, not
a truncation of one long episode: the agent is not deterministic, so a
prefix of a €0.15 episode is not a €0.02 episode.

**Tamper check.** Every model runs every task once more at the highest
ceiling with the test file *writable* and an instruction not to edit it.
quality.tamper then compares the workspace before and after: a solve that
changed the test file is raw but not clean.

Trajectories are written under ``--out/<sweep>/<model>/<budget>/`` so the
analysis can be re-run without spending anything::

    python scripts/agent_study.py --run        # spends money, asks first
    python scripts/agent_study.py              # analyses what is on disk
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from settings import cache_db_path, load_dotenv  # noqa: E402

MODELS = ["claude-sonnet-5-5", "gpt-5.6-terra", "claude-haiku-5-5", "gpt-5.6-luna"]
BUDGETS_EUR = [0.005, 0.01, 0.02, 0.05, 0.15]
DATASET = "defects4j"


def _client(model_id: str) -> Any:
    from llm_gateway.cache import ResponseCache
    from llm_gateway.clients.base import LLMClientFactory
    from llm_gateway.config import GatewayConfig
    from llm_gateway.cost_tracker import CostTracker

    config = GatewayConfig()
    db = str(cache_db_path())
    factory = LLMClientFactory(
        config=config,
        cost_tracker=CostTracker(config=config, db_path=db),
        cache=ResponseCache(db_path=db),
    )
    return factory.get_client(model_id)


def _task_ids() -> list[str]:
    from mcp_servers.registry import get_registry

    return get_registry(str(REPO_ROOT / "data")).adapters()[DATASET].list_problem_ids()


def run(out: Path, models: list[str], budgets: list[float], cap_eur: float) -> None:
    """Run both sweeps, skipping any (sweep, model, budget) already on disk."""
    from agent.runner import run_episodes
    from agent.termination import TerminationPolicy
    from agent.trajectory import TrajectoryStore

    tasks = _task_ids()
    plan = [("curve", model, budget) for model in models for budget in budgets]
    plan += [("tamper", model, max(budgets)) for model in models]

    spent = 0.0
    for sweep, model, budget in plan:
        directory = out / sweep / model / f"{budget:.3f}"
        if directory.exists() and any(directory.glob("*.json")):
            print(f"skip {sweep} {model} €{budget} (already on disk)")
            continue
        remaining = cap_eur - spent
        if remaining < budget:
            print(f"Study cap of €{cap_eur} reached; stopping before {sweep} {model} €{budget}.")
            break

        trajectories = run_episodes(
            task_ids=tasks,
            model_id=model,
            client=_client(model),
            policy=TerminationPolicy(max_cost_eur=budget, max_steps=30),
            store=TrajectoryStore(directory),
            total_budget_eur=min(remaining, budget * len(tasks)),
            writable_tests=(sweep == "tamper"),
        )
        cost = sum(t.total_cost_eur for t in trajectories)
        spent += cost
        solved = sum(1 for t in trajectories if t.solved)
        print(f"{sweep:6s} {model:20s} €{budget:<6} {solved}/{len(trajectories)} "
              f"solved, €{cost:.4f}  (study total €{spent:.4f})")


# ── analysis ──────────────────────────────────────────────────────────

def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion. Sound at n=10, unlike the normal one."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _load(directory: Path) -> list[Any]:
    from agent.trajectory import TrajectoryStore

    return list(TrajectoryStore(directory).load_all()) if directory.exists() else []


def tamper_report(trajectory: Any) -> Any:
    """Compare the episode's starting workspace with its final one."""
    from agent.runner import build_episode
    from quality.tamper import detect_tampering

    workspace, _, test_class, _ = build_episode(
        trajectory.task_id, str(REPO_ROOT / "data"), writable_tests=True
    )
    return detect_tampering(
        files_before=workspace.snapshot(),
        files_after=trajectory.final_workspace or workspace.snapshot(),
        test_paths=[f"{test_class}.java"],
    )


def analyse(out: Path) -> dict[str, Any]:
    """Summarise whatever trajectories are on disk. Spends nothing."""
    from quality.tamper import clean_solve_rate

    curve: list[dict[str, Any]] = []
    for model_dir in sorted((out / "curve").glob("*")):
        for budget_dir in sorted(model_dir.glob("*"), key=lambda p: float(p.name)):
            episodes = _load(budget_dir)
            if not episodes:
                continue
            solved = sum(1 for t in episodes if t.solved)
            spent = sum(t.total_cost_eur for t in episodes)
            low, high = wilson(solved, len(episodes))
            curve.append({
                "model_id": model_dir.name,
                "budget_eur": float(budget_dir.name),
                "episodes": len(episodes),
                "solved": solved,
                "solve_rate": round(solved / len(episodes), 4),
                "solve_rate_ci95": [round(low, 4), round(high, 4)],
                "mean_cost_eur": round(spent / len(episodes), 5),
                "cost_per_solve_eur": round(spent / solved, 5) if solved else None,
                "not_executed": sum(1 for t in episodes if not t.tests_executed),
                "stopped_by_budget": sum(1 for t in episodes if t.stop_condition == "max_cost_eur"),
            })

    tamper: list[dict[str, Any]] = []
    for model_dir in sorted((out / "tamper").glob("*")):
        episodes = [t for budget_dir in model_dir.glob("*") for t in _load(budget_dir)]
        if not episodes:
            continue
        reports = [(t.solved, tamper_report(t)) for t in episodes]
        summary = clean_solve_rate(reports)
        summary.pop("note", None)
        summary["model_id"] = model_dir.name
        summary["tampered_tasks"] = sorted(
            t.task_id for t, (_, report) in zip(episodes, reports) if report.tampered
        )
        summary["suspected_kinds"] = sorted({
            finding.kind.value
            for _, report in reports for finding in report.suspected
        })
        tamper.append(summary)

    return {"dataset": DATASET, "cost_curve": curve, "tamper": tamper}


def plot(summary: dict[str, Any], single_shot: dict[str, dict[str, float]], path: Path) -> None:
    """Solve rate against euros per episode, one line per model.

    The single-shot point for each model is drawn at its own mean cost, so
    the chart answers the question directly: how much more does iteration
    solve, and at what price.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"claude-sonnet-5-5": "#c15f3c", "gpt-5.6-terra": "#2f6f9f",
               "claude-haiku-5-5": "#e0a184", "gpt-5.6-luna": "#8fb8d8"}
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    for model in sorted({row["model_id"] for row in summary["cost_curve"]}):
        rows = [r for r in summary["cost_curve"] if r["model_id"] == model]
        xs = [r["mean_cost_eur"] for r in rows]
        ys = [r["solve_rate"] for r in rows]
        lows = [r["solve_rate"] - r["solve_rate_ci95"][0] for r in rows]
        highs = [r["solve_rate_ci95"][1] - r["solve_rate"] for r in rows]
        colour = colours.get(model, "#555555")
        ax.errorbar(xs, ys, yerr=[lows, highs], marker="o", capsize=3,
                    color=colour, label=f"{model} (agent)", linewidth=1.6)
        if model in single_shot:
            point = single_shot[model]
            ax.scatter([point["mean_cost_eur"]], [point["solve_rate"]], marker="D",
                       s=46, facecolor="white", edgecolor=colour, linewidth=1.6, zorder=5,
                       label=f"{model} (single-shot)")
    ax.set_xscale("log")
    ax.set_xlabel("Mean cost per task (EUR, log scale)")
    ax.set_ylabel("Defects4J tasks solved")
    ax.set_ylim(-0.03, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(True, which="both", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("What each extra euro buys on Defects4J (95% Wilson intervals)", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def single_shot_defects4j(run_dir: Path) -> dict[str, dict[str, float]]:
    """Per-model single-shot solve rate and cost on Defects4J, from the run artifact."""
    import csv

    from settings import usd_to_eur

    path = run_dir / "evaluations.csv"
    if not path.exists():
        return {}
    by_model: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == DATASET:
                by_model.setdefault(row["model_id"], []).append(row)
    return {
        model: {
            "solve_rate": sum(r["verdict"] == "pass" for r in rows) / len(rows),
            "mean_cost_eur": usd_to_eur(
                sum(float(r["cost_usd"] or 0) for r in rows) / len(rows)
            ),
        }
        for model, rows in by_model.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--run", action="store_true", help="Run the sweeps (spends money).")
    parser.add_argument("--out", default="results/2026-10-agent", type=Path)
    parser.add_argument("--single-shot", default="results/2026-10-run", type=Path,
                        help="Run artifact holding the single-shot baseline.")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--budgets", nargs="+", type=float, default=BUDGETS_EUR)
    parser.add_argument("--cap-eur", type=float, default=12.0,
                        help="Ceiling for the whole study.")
    args = parser.parse_args(argv)

    load_dotenv()
    out = REPO_ROOT / args.out
    if args.run:
        run(out, args.models, args.budgets, args.cap_eur)

    summary = analyse(out)
    baseline = single_shot_defects4j(REPO_ROOT / args.single_shot)
    summary["single_shot"] = baseline
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if summary["cost_curve"]:
        plot(summary, baseline, out / "cost_curve.png")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
