#!/usr/bin/env python3
"""
scripts/regression_gate.py — fail the build when results get worse (M10).

A benchmark harness is code, and code regresses. The difference here is that
a regression does not crash: it produces a slightly lower resolve rate,
nobody notices for three months, and every number published in between is
wrong.

Two gates:

**Resolve rate.** Compare a run against a committed baseline and fail when
it drops by more than a threshold. The threshold matters: agent results are
stochastic, so a gate set too tight fails on noise and gets disabled, which
is worse than not having one. The default allows a 5-point drop and the
reasoning is in ``DEFAULT_TOLERANCE``.

**Prompt and tool versioning.** A changed prompt template or tool schema
invalidates comparison with previously published numbers. The gate does not
block the change — changing a prompt is legitimate — it requires that the
change be *declared*, so a silent edit cannot quietly invalidate a year of
results.

Usage
-----
::

    python scripts/regression_gate.py --baseline results/2026-05-run \\
                                      --candidate results/nightly
    python scripts/regression_gate.py --check-prompts --baseline results/2026-05-run

Exit codes: 0 clean, 1 regression, 2 could not run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: Allowed drop in resolve rate before the gate fails, in percentage points.
#: Five is not arbitrary: with 58 tasks a single flipped task is 1.7 points,
#: and observed run-to-run variation on this suite is about 3. A tighter
#: gate fails on noise, and a gate that fails on noise gets turned off.
DEFAULT_TOLERANCE = 0.05

#: A drop this large is a real regression whatever the noise level.
HARD_FAIL_TOLERANCE = 0.15


def load_summary(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Read ``summary.csv`` into ``{model: row}`` for the ALL rows."""
    path = run_dir / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"no summary.csv in {run_dir}")

    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset") == "ALL":
                rows[row["model_id"]] = row
    if not rows:
        raise ValueError(f"{path} has no ALL rows; nothing to compare")
    return rows


def compare_resolve_rates(
    baseline_dir: Path,
    candidate_dir: Path,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Compare two runs and decide whether the build should fail."""
    baseline = load_summary(baseline_dir)
    candidate = load_summary(candidate_dir)

    comparisons: list[dict[str, Any]] = []
    failures: list[str] = []

    for model in sorted(set(baseline) | set(candidate)):
        if model not in candidate:
            failures.append(f"{model}: present in the baseline, absent from the run")
            continue
        if model not in baseline:
            comparisons.append({
                "model": model, "status": "new",
                "candidate_pass_rate": float(candidate[model]["pass_rate"]),
            })
            continue

        before = float(baseline[model]["pass_rate"])
        after = float(candidate[model]["pass_rate"])
        delta = after - before

        if delta < -HARD_FAIL_TOLERANCE:
            status = "hard_regression"
            failures.append(
                f"{model}: pass rate fell {abs(delta):.1%} "
                f"({before:.1%} -> {after:.1%}); beyond any plausible noise"
            )
        elif delta < -tolerance:
            status = "regression"
            failures.append(
                f"{model}: pass rate fell {abs(delta):.1%} "
                f"({before:.1%} -> {after:.1%}); tolerance is {tolerance:.1%}"
            )
        elif delta > tolerance:
            status = "improvement"
        else:
            status = "within_tolerance"

        comparisons.append({
            "model": model,
            "status": status,
            "baseline_pass_rate": before,
            "candidate_pass_rate": after,
            "delta": round(delta, 4),
        })

    return {
        "passed": not failures,
        "tolerance": tolerance,
        "comparisons": comparisons,
        "failures": failures,
    }


def prompt_fingerprints() -> dict[str, str]:
    """SHA-256 of every prompt template, as they are now."""
    templates = REPO_ROOT / "llm_gateway" / "templates"
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(templates.glob("*.j2"))
    }


def check_prompt_drift(baseline_dir: Path) -> dict[str, Any]:
    """Compare current prompt hashes against the baseline manifest.

    A changed template is not an error. An *undeclared* changed template is:
    it silently invalidates every number the baseline holds, and the whole
    point of hashing the templates was to make that visible.
    """
    manifest_path = baseline_dir / "manifest.json"
    if not manifest_path.exists():
        return {
            "passed": False,
            "reason": f"no manifest.json in {baseline_dir}; cannot compare prompts",
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("prompt_template_sha256", {})
    current = prompt_fingerprints()

    changed = sorted(
        name for name in set(recorded) & set(current)
        if recorded[name] != current[name]
    )
    added = sorted(set(current) - set(recorded))
    removed = sorted(set(recorded) - set(current))

    return {
        "passed": not (changed or removed),
        "changed": changed,
        "added": added,
        "removed": removed,
        "baseline_run": manifest.get("run_name", str(baseline_dir)),
        "advice": (
            "A changed template is a different measurement. Re-run the smoke "
            "subset, state in the PR which published results it makes stale, "
            "and regenerate the baseline manifest."
            if changed or removed else
            "Templates match the baseline; published comparisons remain valid."
        ),
    }


def render_markdown(resolve: dict[str, Any] | None, prompts: dict[str, Any] | None) -> str:
    """A PR comment. Short, and leads with the verdict."""
    lines: list[str] = ["## Regression gate", ""]

    if resolve is not None:
        verdict = "PASSED" if resolve["passed"] else "FAILED"
        lines.append(f"**Resolve rate: {verdict}** (tolerance {resolve['tolerance']:.1%})")
        lines.append("")
        lines.append("| Model | Baseline | This run | Delta | |")
        lines.append("|---|---|---|---|---|")
        for row in resolve["comparisons"]:
            if row["status"] == "new":
                lines.append(
                    f"| {row['model']} | — | {row['candidate_pass_rate']:.1%} | — | new |"
                )
                continue
            marker = {
                "regression": "REGRESSION",
                "hard_regression": "HARD REGRESSION",
                "improvement": "improved",
                "within_tolerance": "ok",
            }[row["status"]]
            lines.append(
                f"| {row['model']} | {row['baseline_pass_rate']:.1%} | "
                f"{row['candidate_pass_rate']:.1%} | {row['delta']:+.1%} | {marker} |"
            )
        if resolve["failures"]:
            lines.extend(["", "**Failures**"])
            lines.extend(f"- {failure}" for failure in resolve["failures"])
        lines.append("")

    if prompts is not None:
        verdict = "unchanged" if prompts["passed"] else "CHANGED"
        lines.append(f"**Prompt templates: {verdict}**")
        if prompts.get("changed"):
            lines.append(f"- changed: {', '.join(prompts['changed'])}")
        if prompts.get("added"):
            lines.append(f"- added: {', '.join(prompts['added'])}")
        if prompts.get("removed"):
            lines.append(f"- removed: {', '.join(prompts['removed'])}")
        lines.extend(["", prompts.get("advice", prompts.get("reason", ""))])

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--baseline", type=Path, default=Path("results/2026-05-run"),
                        help="Committed baseline run directory.")
    parser.add_argument("--candidate", type=Path, default=None,
                        help="Run to check. Omit to check prompts only.")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="Allowed drop in resolve rate, as a fraction.")
    parser.add_argument("--check-prompts", action="store_true",
                        help="Compare prompt hashes against the baseline manifest.")
    parser.add_argument("--markdown", type=Path, default=None,
                        help="Write a PR comment to this file.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    baseline = REPO_ROOT / args.baseline if not args.baseline.is_absolute() else args.baseline

    resolve: dict[str, Any] | None = None
    prompts: dict[str, Any] | None = None

    try:
        if args.candidate is not None:
            candidate = (
                REPO_ROOT / args.candidate
                if not args.candidate.is_absolute() else args.candidate
            )
            resolve = compare_resolve_rates(baseline, candidate, args.tolerance)
        if args.check_prompts or args.candidate is None:
            prompts = check_prompt_drift(baseline)
    except (FileNotFoundError, ValueError) as exc:
        print(f"regression_gate: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"resolve": resolve, "prompts": prompts}, indent=2))
    else:
        print(render_markdown(resolve, prompts))

    if args.markdown:
        args.markdown.write_text(render_markdown(resolve, prompts), encoding="utf-8")

    failed = (resolve is not None and not resolve["passed"]) or (
        prompts is not None and not prompts["passed"]
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
