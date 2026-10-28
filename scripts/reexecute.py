#!/usr/bin/env python3
"""
scripts/reexecute.py — re-run recorded model outputs through the sandbox.

No API calls. Every response is read back from the response cache by its
``response_id`` and scored by the same code path a live run uses
(``BenchmarkOrchestrator._build_evaluation_result``), so the only thing that
changes is the execution environment.

Why it exists: the October 2026 run was executed on a host that ran out of
memory, and containers killed mid-``javac`` were recorded as compile errors
against the model. Re-executing every kept sample on a healthy host, rather
than only the ones that look broken, keeps the conditions uniform.

    python scripts/reexecute.py                 # every dataset, every model
    python scripts/reexecute.py --dataset mbpp-java --workers 2

Each ``results.jsonl`` is rewritten with one row per sample (see
``build_run_artifact.last_per_sample``); the previous file is kept beside
it as ``results.first-pass.jsonl``. The rewrite is refused if any sample
still could not be executed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bench.datasets.defects4j import Defects4JDataset  # noqa: E402
from bench.datasets.godclass import GodClassDataset  # noqa: E402
from bench.datasets.humaneval import HumanEvalDataset  # noqa: E402
from bench.datasets.mbpp import MBPPDataset  # noqa: E402
from bench.orchestrator import BenchmarkOrchestrator, RunConfig  # noqa: E402
from bench.sandbox.docker_sandbox import DockerSandbox  # noqa: E402
from contracts import LLMResponse  # noqa: E402

DATASETS = {
    cls(data_dir=str(REPO_ROOT / "data")).name: cls
    for cls in (HumanEvalDataset, MBPPDataset, Defects4JDataset, GodClassDataset)
}


def _artifact_module():
    path = REPO_ROOT / "scripts" / "build_run_artifact.py"
    spec = importlib.util.spec_from_file_location("_build_run_artifact", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cached_responses(db_path: Path) -> dict[str, dict[str, Any]]:
    """Every cached response, by response_id."""
    conn = sqlite3.connect(db_path)
    try:
        return {
            (d := json.loads(raw))["response_id"]: d
            for (raw,) in conn.execute("SELECT response_json FROM response_cache")
        }
    finally:
        conn.close()


def reexecute_row(
    orchestrator: BenchmarkOrchestrator, row: dict[str, Any], cached: dict[str, Any],
) -> dict[str, Any]:
    """Score *row*'s recorded response again; identity fields are kept."""
    response = LLMResponse(**{
        **cached, "metadata": {**cached.get("metadata", {}), "cache_hit": row["cache_hit"]},
    })
    code = response.extracted_code or response.raw_text
    dataset = orchestrator.dataset
    junit_code = ""
    try:
        junit_code = dataset.get_junit_code(row["problem_id"])  # type: ignore[attr-defined]
    except (AttributeError, KeyError):
        pass
    sandbox_result = orchestrator.sandbox.run(
        source_code=code, junit_code=junit_code, problem_id=row["problem_id"],
    )
    result = orchestrator._build_evaluation_result(
        problem_id=row["problem_id"],
        model_id=row["model_id"],
        run_id=row["sample_index"],
        response=response,
        suite=dataset.get_test_suite(row["problem_id"]),
        sandbox_result=sandbox_result,
        code=code,
    )
    fresh = result.model_dump(mode="json")
    fresh["evaluation_id"] = row["evaluation_id"]
    fresh["latency_ms"] = row["latency_ms"]
    fresh["recorded_at"] = row.get("recorded_at")
    fresh["reexecuted_at"] = datetime.now(timezone.utc).isoformat()
    return fresh


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS),
                        help="Repeatable. Default: every dataset found.")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)

    from settings import cache_db_path, load_dotenv

    load_dotenv()
    sandbox = DockerSandbox()
    if not sandbox.is_docker_available():
        raise SystemExit("Docker is not available; nothing would be executed.")
    responses = cached_responses(cache_db_path())
    artifact = _artifact_module()
    results_dir = REPO_ROOT / args.results_dir

    rows, _ = artifact.last_per_sample(artifact.load_evaluations(results_dir))
    changes: Counter[tuple[str, str, str, str]] = Counter()
    for name in args.dataset or sorted({r["dataset"] for r in rows}):
        dataset = DATASETS[name](data_dir=str(REPO_ROOT / "data"))
        dataset.load_problems()
        orchestrator = BenchmarkOrchestrator(
            dataset, {}, sandbox, RunConfig(results_dir=results_dir),
        )
        for model in sorted({r["model_id"] for r in rows if r["dataset"] == name}):
            mine = [r for r in rows if r["dataset"] == name and r["model_id"] == model]
            missing = [r["response_id"] for r in mine if r["response_id"] not in responses]
            if missing:
                raise SystemExit(f"{name}/{model}: {len(missing)} responses not in the cache")
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                fresh = list(pool.map(
                    lambda r: reexecute_row(orchestrator, r, responses[r["response_id"]]), mine,
                ))
            unexecuted = [f["problem_id"] for f in fresh if not f["tests_executed"]]
            if unexecuted:
                raise SystemExit(f"{name}/{model}: still not executed: {unexecuted[:5]}")
            for old, new in zip(mine, fresh):
                changes[(name, model, old["verdict"], new["verdict"])] += 1
            changed = sum(1 for o, n in zip(mine, fresh) if o["verdict"] != n["verdict"])

            out_dir = results_dir / name / model
            target = out_dir / "results.jsonl"
            backup = out_dir / artifact.FIRST_PASS_NAME
            if not backup.exists():
                target.replace(backup)
            fresh.sort(key=lambda f: (f["problem_id"], f["sample_index"]))
            target.write_text(
                "".join(json.dumps(f) + "\n" for f in fresh), encoding="utf-8",
            )
            print(f"{name}/{model}: {len(fresh)} re-executed, {changed} verdicts changed",
                  flush=True)

    print("\nverdict changes (dataset, model, before -> after: count):")
    for (name, model, before, after), count in sorted(changes.items()):
        if before != after:
            print(f"  {name} {model}: {before} -> {after}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
