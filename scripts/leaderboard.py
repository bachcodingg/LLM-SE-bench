"""
Three-axis leaderboard: behaviour, structure, cost — side by side, never summed.

Reads a run artifact's ``evaluations.csv`` and writes ``leaderboard.csv``,
``leaderboard.md`` and ``leaderboard.png`` next to it::

    python scripts/leaderboard.py --run results/2026-10-run

**Behaviour** is the pass rate on the code-generation and repair tasks
(HumanEval, MBPP, Defects4J). **Structure** is measured on the God Class
tasks, where two rates are reported: the refactoring still passes its
behaviour suite, and it is also credited by ``quality.structural``. The gap
between those two rates is the finding the benchmark exists for. **Cost** is
US dollars per task, from the per-call cost recorded at call time.

Every interval is a 95% percentile bootstrap that resamples *tasks*, not
evaluations: the three samples of one task are not independent, and
resampling them separately would make the intervals too narrow.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent

BEHAVIOUR_DATASETS = {"humaneval-java", "mbpp-java", "defects4j", "humaneval", "mbpp"}
STRUCTURE_DATASET = "godclass"
BOOTSTRAP_SAMPLES = 2000
SEED = 20261027


def _truthy(value: str) -> bool:
    return str(value).strip().lower() == "true"


def load(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bootstrap(
    by_task: dict[str, list[float]],
    statistic: Callable[[list[float]], float] = lambda xs: sum(xs) / len(xs),
    samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[float, float, float]:
    """Point estimate and 95% interval, resampling whole tasks.

    Each task contributes the mean of its samples, so a task run three times
    weighs the same as one run once.
    """
    task_means = [sum(values) / len(values) for values in by_task.values() if values]
    if not task_means:
        return (float("nan"), float("nan"), float("nan"))
    point = statistic(task_means)
    rng = random.Random(SEED)
    draws = sorted(
        statistic([rng.choice(task_means) for _ in task_means]) for _ in range(samples)
    )
    return (point, draws[int(0.025 * samples)], draws[int(0.975 * samples) - 1])


def _group(rows: list[dict[str, str]], value: Callable[[dict[str, str]], float | None]
           ) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        result = value(row)
        if result is not None:
            grouped[f"{row['dataset']}/{row['problem_id']}"].append(result)
    return grouped


def leaderboard(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """One row per model. Raises if any evaluation did not execute its tests."""
    unexecuted = [r for r in rows if not _truthy(r.get("tests_executed", ""))]
    if unexecuted:
        raise SystemExit(
            f"{len(unexecuted)} evaluation(s) never executed their tests. A leaderboard "
            f"built on them would rank structural guesses. Re-run with Docker."
        )

    table: list[dict[str, Any]] = []
    for model in sorted({r["model_id"] for r in rows}):
        mine = [r for r in rows if r["model_id"] == model]
        behaviour_rows = [r for r in mine if r["dataset"] in BEHAVIOUR_DATASETS]
        structure_rows = [r for r in mine if r["dataset"] == STRUCTURE_DATASET]

        behaviour = bootstrap(_group(behaviour_rows, lambda r: float(r["verdict"] == "pass")))
        preserved = bootstrap(_group(
            structure_rows,
            lambda r: float(
                _truthy(r["compile_success"])
                and r["tests_total"] not in ("", "0")
                and r["tests_passed"] == r["tests_total"]
            ),
        ))
        credited = bootstrap(_group(
            structure_rows, lambda r: float(_truthy(r.get("structural_credited", "")))
        ))
        composite = bootstrap(_group(
            structure_rows,
            lambda r: float(r["structural_composite"]) if r.get("structural_composite") else None,
        ))
        cost = bootstrap(_group(
            mine, lambda r: float(r["cost_usd"]) if r.get("cost_usd") not in ("", None) else None
        ))
        solved = sum(1 for r in mine if r["verdict"] == "pass")
        spent = sum(float(r["cost_usd"] or 0) for r in mine)

        table.append({
            "model_id": model,
            "behaviour_pass_rate": round(behaviour[0], 4),
            "behaviour_ci95": [round(behaviour[1], 4), round(behaviour[2], 4)],
            "behaviour_tasks": len(_group(behaviour_rows, lambda r: 1.0)),
            "refactor_behaviour_preserved": round(preserved[0], 4),
            "refactor_behaviour_preserved_ci95": [round(preserved[1], 4), round(preserved[2], 4)],
            "refactor_structure_credited": round(credited[0], 4),
            "refactor_structure_credited_ci95": [round(credited[1], 4), round(credited[2], 4)],
            "refactor_mean_composite": round(composite[0], 4),
            "refactor_mean_composite_ci95": [round(composite[1], 4), round(composite[2], 4)],
            "refactor_tasks": len(_group(structure_rows, lambda r: 1.0)),
            "cost_usd_per_task": round(cost[0], 6),
            "cost_usd_per_task_ci95": [round(cost[1], 6), round(cost[2], 6)],
            "cost_usd_per_solve": round(spent / solved, 6) if solved else None,
            "evaluations": len(mine),
        })
    return table


def _pct(value: float, ci: list[float]) -> str:
    return f"{value:.0%} ({ci[0]:.0%}–{ci[1]:.0%})"


def render_markdown(table: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | Behaviour: tasks passed | Refactoring: behaviour kept | "
        "Refactoring: structure credited | Cost per task (USD) |",
        "|---|---|---|---|---|",
    ]
    for row in table:
        lines.append(
            f"| `{row['model_id']}` "
            f"| {_pct(row['behaviour_pass_rate'], row['behaviour_ci95'])} "
            f"| {_pct(row['refactor_behaviour_preserved'], row['refactor_behaviour_preserved_ci95'])} "
            f"| {_pct(row['refactor_structure_credited'], row['refactor_structure_credited_ci95'])} "
            f"| ${row['cost_usd_per_task']:.4f} "
            f"(${row['cost_usd_per_task_ci95'][0]:.4f}–${row['cost_usd_per_task_ci95'][1]:.4f}) |"
        )
    tasks = table[0] if table else {}
    lines.append("")
    lines.append(
        f"Behaviour: {tasks.get('behaviour_tasks', 0)} HumanEval, MBPP and Defects4J tasks. "
        f"Refactoring: {tasks.get('refactor_tasks', 0)} God Class tasks. Three samples per "
        f"task; intervals are 95% bootstrap over tasks. The axes are not combined."
    )
    return "\n".join(lines) + "\n"


def plot(table: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("Behaviour\n(code tasks passed)", "behaviour_pass_rate", "behaviour_ci95", True),
        ("Refactoring: behaviour kept", "refactor_behaviour_preserved",
         "refactor_behaviour_preserved_ci95", True),
        ("Refactoring: structure credited", "refactor_structure_credited",
         "refactor_structure_credited_ci95", True),
        ("Cost per task (USD)", "cost_usd_per_task", "cost_usd_per_task_ci95", False),
    ]
    models = [row["model_id"] for row in table]
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 2.9), dpi=150, sharey=True)
    for ax, (title, key, ci_key, is_rate) in zip(axes, panels):
        values = [row[key] for row in table]
        lows = [row[key] - row[ci_key][0] for row in table]
        highs = [row[ci_key][1] - row[key] for row in table]
        ax.barh(models, values, xerr=[lows, highs], color="#6b7f99", capsize=3, height=0.55)
        ax.set_title(title, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        if is_rate:
            ax.set_xlim(0, 1.0)
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def write_csv(path: Path, table: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--run", default="results/2026-10-run", type=Path,
                        help="Run artifact directory holding evaluations.csv.")
    args = parser.parse_args(argv)

    run_dir = REPO_ROOT / args.run
    table = leaderboard(load(run_dir / "evaluations.csv"))
    write_csv(run_dir / "leaderboard.csv", table)
    markdown = render_markdown(table)
    (run_dir / "leaderboard.md").write_text(markdown, encoding="utf-8")
    plot(table, run_dir / "leaderboard.png")
    sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
