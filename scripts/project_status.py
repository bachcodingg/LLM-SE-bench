#!/usr/bin/env python3
"""
scripts/project_status.py — what is built, tested, and actually run.

Part 7's first risk is scope collapse: modules accumulate, none of them get
finished, and nobody notices because "built" and "working" look the same
from inside. This reports the difference mechanically, so the answer to
"where is this project really" does not depend on how the last week felt.

Three columns, and the third is the one that matters:

``source``   the module exists
``tested``   it has tests, and how many
``exercised``  it has been run end to end against something real

A module that is built and tested but has never been exercised is not
finished. Saying so here is cheaper than discovering it in a README six
months from now.

Also checks the things that block publication: placeholder markers, an
unpushed repository, a stale run artifact.

Usage
-----
::

    python scripts/project_status.py
    python scripts/project_status.py --json

Exit codes: 0 when nothing blocks publication, 1 when something does.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Modules, and the evidence that each has been exercised for real. The
#: evidence is a path that only exists once a module has *produced*
#: something — not one that exists because the module was written.
#:
#: An empty evidence path means the module is a library of pure functions
#: with nothing to exercise separately. Those report as ``library`` rather
#: than ``not run``, because lumping them together would dilute the signal
#: from the module that genuinely has never run.
MODULES: list[tuple[str, str, list[str], str]] = [
    ("M1 agent harness", "agent", ["agent/tests"], "results/trajectories"),
    ("M2 tool layer + MCP", "mcp_servers", ["mcp_servers/tests"], ""),
    ("M3 task factory", "taskfactory", ["taskfactory/tests"], "data/mined/instances.jsonl"),
    ("M4 trajectory analysis", "agent/analysis.py", ["agent/tests/test_analysis.py"],
     "results/trajectories"),
    ("M5 structural scoring", "quality/structural.py", ["quality/tests/test_structural.py"], ""),
    ("M6 tamper detection", "quality/tamper.py", ["quality/tests/test_tamper.py"], ""),
    ("M7 cost governance", "llm_gateway/budget.py", ["llm_gateway/tests/test_budget.py"], ""),
    ("M8 agent statistics", "stats/agents.py", ["stats/tests/test_agents.py"], ""),
    ("M9 dashboard v2", "framework/dashboard/pages/trajectory.py",
     ["framework/tests/test_trajectory_page.py"], "results/trajectories"),
    ("M10 CI + regression", "scripts/regression_gate.py", ["tests/test_scripts.py"],
     ".github/workflows/regression.yml"),
    ("C1 LLM gateway", "llm_gateway", ["llm_gateway/tests"], "results/2026-05-run"),
    ("C2 benchmark engine", "bench", ["bench/tests"], "results/2026-05-run"),
    ("C3 quality analyser", "quality", ["quality/tests"], "results/2026-05-run"),
    ("C4 statistics", "stats", ["stats/tests"], "analysis/statistical_summary.json"),
    ("C5 decision framework", "framework", ["framework/tests"],
     "analysis/statistical_summary.json"),
]

#: Placeholders that must not survive to a public repository.
PLACEHOLDERS = {
    "<OWNER>": "replace with the GitHub account before pushing",
    "TODO(before first push)": "an action the repository says is outstanding",
}

#: Files that are allowed to mention a placeholder, because explaining it is
#: their job.
PLACEHOLDER_EXEMPT = {"scripts/project_status.py"}


@dataclass
class ModuleStatus:
    """What is known about one module."""

    name: str
    source_exists: bool
    test_files: int
    test_count: int
    exercised: bool
    evidence: str

    #: True when the module has no end-to-end artifact to point at, because
    #: it is a library of pure functions rather than something that runs.
    library: bool = False

    @property
    def state(self) -> str:
        if not self.source_exists:
            return "missing"
        if self.test_count == 0:
            return "untested"
        if self.library:
            return "library"
        if not self.exercised:
            return "not run"
        return "working"

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.name,
            "state": self.state,
            "tests": self.test_count,
            "test_files": self.test_files,
            "exercised": self.exercised,
            "evidence": self.evidence,
        }


def _count_tests(paths: list[str]) -> tuple[int, int]:
    """(test files, test functions) under *paths*. Counted by reading, not
    by running: a status check that needs a 90-second test run will not be
    run, and an uncounted status is no status."""
    files = 0
    functions = 0
    pattern = re.compile(r"^\s*def (test_\w+)", re.MULTILINE)

    for entry in paths:
        target = REPO_ROOT / entry
        candidates = (
            sorted(target.rglob("test_*.py")) if target.is_dir()
            else [target] if target.is_file() else []
        )
        for path in candidates:
            files += 1
            functions += len(pattern.findall(path.read_text(encoding="utf-8", errors="replace")))
    return files, functions


def suite_test_count() -> int:
    """Test functions in the repository, counted once each.

    The per-module counts overlap — ``agent/tests`` is evidence for both M1
    and M4 — so summing them overstates the suite. This scans the tree once
    instead, which is the number that should match ``pytest``.
    """
    pattern = re.compile(r"^\s*def (test_\w+)", re.MULTILINE)
    seen: set[Path] = set()
    total = 0
    for path in REPO_ROOT.rglob("test_*.py"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        total += len(pattern.findall(path.read_text(encoding="utf-8", errors="replace")))
    return total


def module_statuses() -> list[ModuleStatus]:
    """Assess every module."""
    statuses: list[ModuleStatus] = []
    for name, source, tests, evidence in MODULES:
        test_files, test_count = _count_tests(tests)
        exercised = bool(evidence) and (REPO_ROOT / evidence).exists()
        statuses.append(ModuleStatus(
            name=name,
            source_exists=(REPO_ROOT / source).exists(),
            test_files=test_files,
            test_count=test_count,
            exercised=exercised,
            evidence=evidence or "pure functions, nothing to exercise",
            library=not evidence,
        ))
    return statuses


def find_placeholders() -> dict[str, list[str]]:
    """Placeholder markers still present in tracked files."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.split()

    found: dict[str, list[str]] = {}
    for rel in tracked:
        if rel in PLACEHOLDER_EXEMPT:
            continue
        path = REPO_ROOT / rel
        if not path.is_file() or path.suffix in {".db", ".pdf", ".png", ".jpg"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for marker in PLACEHOLDERS:
            if marker in text:
                found.setdefault(marker, []).append(rel)
    return found


def publication_checks() -> list[dict[str, Any]]:
    """The things that block a first push."""
    checks: list[dict[str, Any]] = []

    remotes = subprocess.run(
        ["git", "remote"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.split()
    checks.append({
        "check": "published",
        "passed": bool(remotes),
        "detail": (
            f"remotes: {', '.join(remotes)}" if remotes else
            "No git remote. An unpublished repository is a claim, not "
            "evidence, and it is no defence against someone publishing "
            "first. This is risk 6 and it is the highest-value hour "
            "available."
        ),
    })

    placeholders = find_placeholders()
    checks.append({
        "check": "placeholders",
        "passed": not placeholders,
        "detail": (
            "No placeholder markers in tracked files." if not placeholders else
            "; ".join(
                f"{marker} in {len(files)} file(s) ({PLACEHOLDERS[marker]})"
                for marker, files in placeholders.items()
            )
        ),
        "files": {marker: files for marker, files in placeholders.items()},
    })

    artifact = REPO_ROOT / "results" / "2026-05-run" / "manifest.json"
    checks.append({
        "check": "run_artifact",
        "passed": artifact.exists(),
        "detail": (
            "results/2026-05-run present. Verify it with "
            "`python scripts/build_run_artifact.py --check`."
            if artifact.exists() else
            "No published run artifact. The results directory is what makes "
            "the repository credible."
        ),
    })

    gif = any((REPO_ROOT / "docs" / "assets").glob("*.gif")) if (
        REPO_ROOT / "docs" / "assets"
    ).exists() else False
    checks.append({
        "check": "demo_gif",
        "passed": gif,
        "detail": (
            "A demo GIF is present." if gif else
            "No demo GIF. Not blocking, but it is the highest-return 15 "
            "minutes in the README."
        ),
        "blocking": False,
    })

    return checks


def render(statuses: list[ModuleStatus], checks: list[dict[str, Any]]) -> str:
    """A plain-text report."""
    lines = ["llm-se-bench project status", "=" * 58, ""]

    width = max(len(status.name) for status in statuses)
    lines.append(f"{'MODULE'.ljust(width)}  {'STATE':<10} {'TESTS':>6}  EXERCISED BY")
    lines.append("-" * 58)
    for status in statuses:
        evidence = (
            status.evidence if (status.exercised or status.library)
            else f"-- {status.evidence}"
        )
        lines.append(
            f"{status.name.ljust(width)}  {status.state:<10} "
            f"{status.test_count:>6}  {evidence}"
        )

    not_run = [s for s in statuses if s.state == "not run"]
    untested = [s for s in statuses if s.state == "untested"]
    lines.extend(["", "-" * 58])
    lines.append(
        f"{len(statuses)} modules, {suite_test_count()} test functions, "
        f"{len(not_run)} never run end to end."
    )
    lines.append(
        "  (Per-module counts overlap where modules share a test directory. "
        "The total is deduplicated, and is lower than pytest's count because "
        "parametrised tests expand at collection.)"
    )
    if not_run:
        lines.append("")
        lines.append("Built and tested, but never exercised for real:")
        for status in not_run:
            lines.append(f"  - {status.name}")
        lines.append(
            "  A module in this list is not finished. Describing it as "
            "finished is how risk 1 happens."
        )
    if untested:
        lines.append("")
        lines.append("No tests: " + ", ".join(s.name for s in untested))

    lines.extend(["", "PUBLICATION", "-" * 58])
    for check in checks:
        mark = "ok  " if check["passed"] else ("warn" if check.get("blocking") is False else "BLOCK")
        lines.append(f"  [{mark}] {check['check']}: {check['detail']}")

    blocking = [
        c for c in checks if not c["passed"] and c.get("blocking", True)
    ]
    lines.append("")
    lines.append(
        "Nothing blocks publication." if not blocking else
        f"{len(blocking)} thing(s) block publication. See docs/risks.md."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    statuses = module_statuses()
    checks = publication_checks()

    if args.json:
        print(json.dumps({
            "modules": [status.to_dict() for status in statuses],
            "publication": checks,
        }, indent=2))
    else:
        print(render(statuses, checks))

    blocking = [c for c in checks if not c["passed"] and c.get("blocking", True)]
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
