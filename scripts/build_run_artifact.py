#!/usr/bin/env python3
"""
scripts/build_run_artifact.py — freeze a benchmark run into a publishable artifact.

A benchmark that cannot be rerun is an anecdote.  This script turns a local
``results/`` tree plus the cost database into a self-contained directory
that can be committed:

::

    results/<run-name>/
      manifest.json    model ids, template hashes, image digest, dates, seeds
      summary.csv      one row per (model, dataset): pass rate, cost, latency
      evaluations.csv  one row per evaluation, without generated code
      figures/         the figures the README and docs point at

The artifact carries no prompts and no model output.  Those live in the
SQLite cache, which stays out of git.

Usage
-----
::

    python scripts/build_run_artifact.py                   # defaults below
    python scripts/build_run_artifact.py --name 2026-05-run --check

``--check`` rebuilds into a temporary directory and diffs against the
committed artifact instead of overwriting it, which is what CI wants.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Figures the published artifact carries.  The full set stays in analysis/.
PUBLISHED_FIGURES = [
    "pass_at_k_bar.pdf",
    "cost_bar.pdf",
    "cost_per_correct_bar.pdf",
    "violin_latency.pdf",
    "score_boxplot.pdf",
    "heatmap_dataset_model.pdf",
    "radar_chart.pdf",
    "critical_difference.pdf",
]


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def load_evaluations(results_dir: Path) -> list[dict[str, Any]]:
    """Read every ``results/<dataset>/<model>/results.jsonl`` into flat rows."""
    rows: list[dict[str, Any]] = []
    for jsonl in sorted(results_dir.glob("*/*/results.jsonl")):
        dataset = jsonl.parent.parent.name
        model = jsonl.parent.name
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["dataset"] = dataset
            record["model_id"] = model
            rows.append(record)
    return rows


def load_cost_records(db_path: Path) -> list[dict[str, Any]]:
    """Read per-call cost records. Returns [] when the database is absent.

    The cost database is git-ignored, so a fresh clone cannot rebuild the
    cost columns.  That is intentional — it holds full prompts and model
    outputs — and it is why the artifact is committed rather than derived.
    """
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT model_id, prompt_tokens, completion_tokens, "
            "total_cost_usd, created_at FROM cost_records"
        )
        return [
            {
                "model_id": model_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_cost_usd": cost,
                "created_at": created_at,
            }
            for model_id, prompt_tokens, completion_tokens, cost, created_at in cursor
        ]
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Reproducibility fingerprints
# --------------------------------------------------------------------------

def hash_prompt_templates(templates_dir: Path) -> dict[str, str]:
    """SHA-256 every Jinja2 template.

    A prompt template is an input to the measurement.  Changing one
    invalidates comparison against these numbers, so the hashes make the
    break detectable rather than silent.
    """
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(templates_dir.glob("*.j2"))
    }


def hash_datasets(data_dir: Path) -> dict[str, str]:
    """SHA-256 every dataset file, so a silent edit shows up as a hash change."""
    return {
        str(path.relative_to(data_dir)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(data_dir.rglob("*.jsonl"))
    }


def docker_image_reference(image: str) -> dict[str, str]:
    """Resolve the sandbox image to a digest, not a tag.

    Tags move.  If the local daemon does not know the image — the normal
    case in CI — record the Dockerfile's hash instead, and say which it is.
    """
    proc = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{index .RepoDigests 0}}"],
        capture_output=True,
        text=True,
    )
    digest = proc.stdout.strip() if proc.returncode == 0 else ""
    if digest and "@" in digest:
        return {"image": image, "digest": digest, "resolved_from": "local daemon"}

    dockerfile = REPO_ROOT / "bench" / "sandbox" / "Dockerfile"
    return {
        "image": image,
        "digest": "",
        "dockerfile_sha256": hashlib.sha256(dockerfile.read_bytes()).hexdigest(),
        "resolved_from": "Dockerfile hash (image not present locally)",
        "note": (
            "No digest was recorded at run time. The image is built from the "
            "hashed Dockerfile, whose base image eclipse-temurin:17-jdk-jammy "
            "is a moving tag; a rebuild is not guaranteed bit-identical."
        ),
    }


def python_lockfile_hash() -> dict[str, str]:
    """Record whichever lockfile the repo has, and its hash."""
    for name in ("uv.lock", "poetry.lock", "requirements.lock"):
        path = REPO_ROOT / name
        if path.exists():
            return {
                "file": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    return {"file": "", "sha256": "", "note": "no lockfile committed"}


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def summarise(
    evaluations: list[dict[str, Any]],
    cost_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per (model, dataset), plus an ``ALL`` row per model.

    Cost is attributed per *call*, and calls do not map one-to-one onto
    evaluations — retries and cache misses make the counts differ. Cost is
    therefore reported per model only, and left blank on per-dataset rows
    rather than split by a ratio that would look precise and not be.
    """
    cost_by_model: dict[str, float] = defaultdict(float)
    calls_by_model: dict[str, int] = defaultdict(int)
    for record in cost_records:
        cost_by_model[record["model_id"]] += record["total_cost_usd"]
        calls_by_model[record["model_id"]] += 1

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluations:
        grouped[(row["model_id"], row["dataset"])].append(row)
        grouped[(row["model_id"], "ALL")].append(row)

    summary: list[dict[str, Any]] = []
    for (model, dataset), rows in sorted(grouped.items()):
        passed = sum(1 for r in rows if r["verdict"] == "pass")
        is_total = dataset == "ALL"
        summary.append(
            {
                "model_id": model,
                "dataset": dataset,
                "problems": len({r["problem_id"] for r in rows}),
                "evaluations": len(rows),
                "passed": passed,
                "pass_rate": round(passed / len(rows), 4),
                "mean_weighted_score": round(
                    statistics.mean(r["weighted_score"] for r in rows), 4
                ),
                "mean_latency_s": round(
                    statistics.mean(r["latency_ms"] for r in rows) / 1000, 2
                ),
                "api_calls": calls_by_model.get(model, 0) if is_total else "",
                "total_cost_usd": (
                    round(cost_by_model.get(model, 0.0), 4) if is_total else ""
                ),
                "cost_usd_per_evaluation": (
                    round(cost_by_model.get(model, 0.0) / len(rows), 6)
                    if is_total and cost_by_model.get(model)
                    else ""
                ),
            }
        )
    return summary


def flatten_evaluations(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-evaluation rows for the CSV, with generated code left out."""
    return [
        {
            "evaluation_id": row["evaluation_id"],
            "model_id": row["model_id"],
            "dataset": row["dataset"],
            "problem_id": row["problem_id"],
            "verdict": row["verdict"],
            "tests_passed": row["tests_passed"],
            "tests_total": row["tests_total"],
            "weighted_score": row["weighted_score"],
            "compile_success": row["compile_success"],
            "latency_ms": round(row["latency_ms"], 1),
            "evaluated_at": row["evaluated_at"],
        }
        for row in sorted(
            evaluations,
            key=lambda r: (r["model_id"], r["dataset"], r["problem_id"], r["evaluated_at"]),
        )
    ]


def build_manifest(
    run_name: str,
    evaluations: list[dict[str, Any]],
    cost_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Everything needed to judge whether a rerun reproduced this run."""
    models = sorted({row["model_id"] for row in evaluations})
    datasets = sorted({row["dataset"] for row in evaluations})
    eval_times = sorted(row["evaluated_at"] for row in evaluations)
    call_times = sorted(record["created_at"] for record in cost_records)

    return {
        "run_name": run_name,
        "artifact_generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/build_run_artifact.py",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        ).stdout.strip() or "<uncommitted>",
        "models": models,
        "datasets": datasets,
        "counts": {
            "problems": len({(r["dataset"], r["problem_id"]) for r in evaluations}),
            "evaluations": len(evaluations),
            "runs_per_problem": (
                len(evaluations)
                // max(1, len(models) * len({(r["dataset"], r["problem_id"]) for r in evaluations}))
            ),
            "api_calls": len(cost_records),
        },
        "wall_clock": {
            "first_api_call": call_times[0] if call_times else None,
            "last_api_call": call_times[-1] if call_times else None,
            "first_evaluation": eval_times[0] if eval_times else None,
            "last_evaluation": eval_times[-1] if eval_times else None,
        },
        "cost_usd": {
            "total": round(sum(r["total_cost_usd"] for r in cost_records), 4),
            "by_model": {
                model: round(
                    sum(r["total_cost_usd"] for r in cost_records if r["model_id"] == model), 4
                )
                for model in models
            },
        },
        "tokens": {
            model: {
                "prompt": sum(
                    r["prompt_tokens"] for r in cost_records if r["model_id"] == model
                ),
                "completion": sum(
                    r["completion_tokens"] for r in cost_records if r["model_id"] == model
                ),
            }
            for model in models
        },
        "sampling": {
            "temperature": 0.0,
            "seed": None,
            "note": (
                "Sampling is greedy (temperature 0). No provider in this run "
                "exposes a seed parameter, so decoding is not bit-reproducible "
                "even at temperature 0. This is why each problem was run 3 times."
            ),
        },
        "prompt_template_sha256": hash_prompt_templates(
            REPO_ROOT / "llm_gateway" / "templates"
        ),
        "dataset_sha256": hash_datasets(REPO_ROOT / "data"),
        "dataset_note": (
            "12 of the 58 tasks (3 per dataset) are embedded in "
            "bench/datasets/*.py as _EXAMPLE_PROBLEMS and are not covered by "
            "the hashes above; they are pinned by git_commit. See "
            "data/PROVENANCE.md."
        ),
        "sandbox": docker_image_reference("llm-se-bench-sandbox:17"),
        "python_lockfile": python_lockfile_hash(),
    }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write *rows* as CSV with a stable column order and LF line endings."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(out_dir: Path, results_dir: Path, db_path: Path, figures_dir: Path,
          run_name: str) -> dict[str, Any]:
    """Build the whole artifact into *out_dir* and return its manifest."""
    evaluations = load_evaluations(results_dir)
    if not evaluations:
        raise SystemExit(f"no evaluations found under {results_dir}")
    cost_records = load_cost_records(db_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(run_name, evaluations, cost_records)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(out_dir / "summary.csv", summarise(evaluations, cost_records))
    write_csv(out_dir / "evaluations.csv", flatten_evaluations(evaluations))

    figures_out = out_dir / "figures"
    figures_out.mkdir(exist_ok=True)
    for name in PUBLISHED_FIGURES:
        source = figures_dir / name
        if source.exists():
            shutil.copy2(source, figures_out / name)

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--name", default="2026-05-run", help="Artifact directory name.")
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--db", dest="db_path", default=None, type=Path,
                        help="Cost database. Defaults to LLM_SE_BENCH_CACHE_PATH.")
    parser.add_argument("--figures-dir", default="analysis/figures", type=Path)
    parser.add_argument("--check", action="store_true",
                        help="Rebuild into a temp dir and diff, do not overwrite.")
    args = parser.parse_args(argv)

    from settings import cache_db_path, load_dotenv

    load_dotenv()
    db_path = args.db_path or cache_db_path()
    results_dir = REPO_ROOT / args.results_dir
    figures_dir = REPO_ROOT / args.figures_dir
    target = results_dir / args.name

    if args.check:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            rebuilt = Path(tmp) / args.name
            build(rebuilt, results_dir, db_path, figures_dir, args.name)
            differences: list[str] = []
            for name in ("summary.csv", "evaluations.csv"):
                committed = target / name
                if not committed.exists():
                    differences.append(f"{name}: missing from committed artifact")
                elif committed.read_bytes() != (rebuilt / name).read_bytes():
                    differences.append(f"{name}: differs from a fresh rebuild")
            if differences:
                print("artifact is stale:")
                for line in differences:
                    print(f"  {line}")
                return 1
            print(f"artifact {args.name} matches a fresh rebuild")
            return 0

    manifest = build(target, results_dir, db_path, figures_dir, args.name)
    counts = manifest["counts"]
    print(f"wrote {target.relative_to(REPO_ROOT)}")
    print(
        f"  {counts['evaluations']} evaluations, {counts['problems']} problems, "
        f"{len(manifest['models'])} models, {counts['api_calls']} API calls, "
        f"${manifest['cost_usd']['total']:.4f}"
    )
    if not manifest["cost_usd"]["total"]:
        print("  warning: no cost records found — cost columns will be empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
