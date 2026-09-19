"""
taskfactory.validation — the gates a task must pass to be admitted.

A benchmark is only as good as its worst task. A task that fails 30% of the
time for environmental reasons looks exactly like a model capability
difference, and once it is in the dataset every result computed from that
dataset is contaminated by it — quietly, and in a way no amount of
statistical care downstream can undo.

So nothing is admitted without passing every gate:

``golden_fixes_target``
    The reference fix makes FAIL_TO_PASS pass. If it does not, the task is
    mis-derived and nothing can solve it.
``target_fails_before``
    FAIL_TO_PASS actually fails against the broken code. If it already
    passes, there is nothing to solve and the task will show a 100% resolve
    rate that means nothing.
``guard_passes_before``
    PASS_TO_PASS passes before the fix, so it can serve as a guard rail.
``not_flaky``
    Five runs, identical results. This is the gate the guide calls not
    optional, and it is the one most likely to be skipped because it costs
    five times as much as the others put together.
``solvable_from_statement``
    The problem statement names enough to work from. A task whose fix is
    unreachable from its description measures luck.
``environment_builds``
    The image builds and the suite runs at all.

Every rejection carries its reason. A rejection log is as useful as the
dataset: it says what the mining pipeline cannot yet handle.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from taskfactory.patches import TaskInstance

logger = logging.getLogger(__name__)

__all__ = [
    "Gate",
    "GateResult",
    "ValidationReport",
    "FLAKE_RUNS",
    "check_flakiness",
    "check_solvability",
    "validate_instance",
    "summarise_rejections",
]

#: Repeats used by the flake gate. Five is the guide's number. Three would
#: miss a test that fails one time in four, which is exactly the rate that
#: does the most damage: frequent enough to matter, rare enough to look like
#: a model difference.
FLAKE_RUNS = 5


class Gate(str, Enum):
    """The admission gates, in the order they are applied."""

    ENVIRONMENT_BUILDS = "environment_builds"
    TARGET_FAILS_BEFORE = "target_fails_before"
    GUARD_PASSES_BEFORE = "guard_passes_before"
    GOLDEN_FIXES_TARGET = "golden_fixes_target"
    NOT_FLAKY = "not_flaky"
    SOLVABLE_FROM_STATEMENT = "solvable_from_statement"

    @property
    def description(self) -> str:
        return {
            Gate.ENVIRONMENT_BUILDS: "The image builds and the suite runs.",
            Gate.TARGET_FAILS_BEFORE: "FAIL_TO_PASS fails against the broken code.",
            Gate.GUARD_PASSES_BEFORE: "PASS_TO_PASS passes before the fix.",
            Gate.GOLDEN_FIXES_TARGET: "The reference fix makes FAIL_TO_PASS pass.",
            Gate.NOT_FLAKY: f"{FLAKE_RUNS} runs give identical results.",
            Gate.SOLVABLE_FROM_STATEMENT: "The statement is enough to work from.",
        }[self]


@dataclass
class GateResult:
    """Whether one gate passed, and why not if it did not."""

    gate: Gate
    passed: bool
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate.value,
            "description": self.gate.description,
            "passed": self.passed,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass
class ValidationReport:
    """Every gate's result for one candidate task."""

    instance_id: str
    results: list[GateResult] = field(default_factory=list)

    @property
    def admitted(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    @property
    def failed_gates(self) -> list[Gate]:
        return [result.gate for result in self.results if not result.passed]

    @property
    def rejection_reason(self) -> str:
        failures = [r for r in self.results if not r.passed]
        if not failures:
            return ""
        return f"{failures[0].gate.value}: {failures[0].detail}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "admitted": self.admitted,
            "failed_gates": [gate.value for gate in self.failed_gates],
            "rejection_reason": self.rejection_reason,
            "gates": [result.to_dict() for result in self.results],
        }


# ──────────────────────────────────────────────────────────────────────
# Individual gates
# ──────────────────────────────────────────────────────────────────────

def check_flakiness(
    run_tests: Callable[[], dict[str, bool]],
    runs: int = FLAKE_RUNS,
) -> GateResult:
    """Run the suite *runs* times and require identical results.

    *run_tests* returns ``{test id: passed}``. Any test whose result varies
    across runs makes the task flaky and the task is rejected — not
    down-weighted, not flagged, rejected. A flaky task in the dataset
    contaminates every aggregate computed from it.

    This gate costs five times what the others cost together. Skipping it is
    the single most tempting shortcut in the pipeline and the most expensive
    one, because its damage is invisible until someone tries to reproduce a
    result and cannot.
    """
    if runs < 2:
        raise ValueError("flakiness cannot be assessed from fewer than 2 runs")

    observations: list[dict[str, bool]] = []
    for index in range(runs):
        try:
            observations.append(run_tests())
        except Exception as exc:
            return GateResult(
                Gate.NOT_FLAKY,
                passed=False,
                detail=f"Run {index + 1} of {runs} raised: {exc}",
                evidence={"completed_runs": index},
            )

    all_tests = sorted({test for run in observations for test in run})
    unstable: dict[str, list[bool]] = {}
    for test in all_tests:
        seen = [run.get(test) for run in observations]
        if len(set(seen)) > 1:
            unstable[test] = [bool(value) for value in seen]

    if unstable:
        return GateResult(
            Gate.NOT_FLAKY,
            passed=False,
            detail=(
                f"{len(unstable)} test(s) gave different results across "
                f"{runs} runs. A flaky task is indistinguishable from a model "
                f"capability difference once it is in the dataset."
            ),
            evidence={
                "unstable_tests": dict(list(unstable.items())[:10]),
                "runs": runs,
            },
        )

    return GateResult(
        Gate.NOT_FLAKY,
        passed=True,
        detail=f"{len(all_tests)} test(s) gave identical results across {runs} runs.",
        evidence={"runs": runs, "tests": len(all_tests)},
    )


#: A statement shorter than this cannot describe a defect.
MIN_STATEMENT_CHARS = 80

#: Words that indicate a statement describes a concrete, locatable problem
#: rather than a vague aspiration.
_CONCRETE = (
    "error", "exception", "fail", "incorrect", "wrong", "should", "expected",
    "returns", "throws", "null", "crash", "bug", "regression", "instead",
)


def check_solvability(instance: TaskInstance) -> GateResult:
    """Whether the statement gives enough to work from.

    Heuristic and stated as such. It catches the two clear cases — a
    statement that is empty, and one that gives the answer away — and it
    cannot judge the middle. A task that passes this gate is not proven
    solvable; a task that fails it is proven not worth including.
    """
    statement = (instance.problem_statement or "").strip()

    if len(statement) < MIN_STATEMENT_CHARS:
        return GateResult(
            Gate.SOLVABLE_FROM_STATEMENT,
            passed=False,
            detail=(
                f"Statement is {len(statement)} characters; below "
                f"{MIN_STATEMENT_CHARS} it cannot describe a defect. The task "
                f"would measure guessing."
            ),
            evidence={"statement": statement[:200]},
        )

    lowered = statement.lower()
    if not any(word in lowered for word in _CONCRETE):
        return GateResult(
            Gate.SOLVABLE_FROM_STATEMENT,
            passed=False,
            detail=(
                "Statement contains no word describing a concrete failure "
                "(error, exception, expected, returns, ...). Probably a "
                "feature request rather than a defect."
            ),
            evidence={"statement": statement[:200]},
        )

    # A statement that contains the fix is worse than one that contains too
    # little: it turns the task into transcription and inflates every score.
    for path in instance.golden_files:
        filename = path.rsplit("/", 1)[-1]
        if filename and filename in statement:
            snippet_leaked = any(
                line.strip() and line.strip() in statement
                for content in instance.golden_patch.values()
                for line in content.splitlines()[:200]
                if len(line.strip()) > 40
            )
            if snippet_leaked:
                return GateResult(
                    Gate.SOLVABLE_FROM_STATEMENT,
                    passed=False,
                    detail=(
                        f"The statement names {filename} and quotes the fix. "
                        f"The task would measure transcription."
                    ),
                    evidence={"leaked_file": path},
                )

    return GateResult(
        Gate.SOLVABLE_FROM_STATEMENT,
        passed=True,
        detail=f"Statement is {len(statement)} characters and describes a failure.",
        evidence={"length": len(statement)},
    )


def validate_instance(
    instance: TaskInstance,
    run_before: Callable[[], dict[str, bool]] | None = None,
    run_after: Callable[[], dict[str, bool]] | None = None,
    flake_runs: int = FLAKE_RUNS,
) -> ValidationReport:
    """Apply every gate to *instance*.

    Parameters
    ----------
    run_before
        Runs the suite against the broken repository. ``{test id: passed}``.
    run_after
        Runs it with the golden patch applied.

    When the runners are omitted, only the checks that need no execution are
    applied and the execution gates are recorded as failed with the reason —
    never as passed. A gate that did not run is not a gate that passed.
    """
    report = ValidationReport(instance_id=instance.instance_id)

    environment = instance.environment or {}
    report.results.append(GateResult(
        Gate.ENVIRONMENT_BUILDS,
        passed=bool(environment.get("buildable")),
        detail=(
            "Build spec is buildable." if environment.get("buildable")
            else f"Not buildable: {environment.get('build_system', 'unknown')} "
                 f"{'; '.join(environment.get('warnings', []))}"
        ),
        evidence={"build_system": environment.get("build_system", "unknown")},
    ))

    sets = instance.test_sets
    if not sets.usable:
        report.results.append(GateResult(
            Gate.TARGET_FAILS_BEFORE,
            passed=False,
            detail="No FAIL_TO_PASS test was derived; there is nothing to solve.",
            evidence={"warnings": sets.warnings},
        ))
        return report

    if run_before is None:
        report.results.append(GateResult(
            Gate.TARGET_FAILS_BEFORE,
            passed=False,
            detail=(
                "Not verified: no runner supplied. An unrun gate is not a "
                "passed gate."
            ),
        ))
    else:
        before = run_before()
        still_passing = [t for t in sets.fail_to_pass if before.get(t, False)]
        report.results.append(GateResult(
            Gate.TARGET_FAILS_BEFORE,
            passed=not still_passing,
            detail=(
                "Every FAIL_TO_PASS test fails against the broken code."
                if not still_passing else
                f"{len(still_passing)} target test(s) already pass, so the "
                f"task is partly or wholly solved before it starts."
            ),
            evidence={"already_passing": still_passing[:10]},
        ))

        failing_guards = [t for t in sets.pass_to_pass if not before.get(t, True)]
        report.results.append(GateResult(
            Gate.GUARD_PASSES_BEFORE,
            passed=not failing_guards,
            detail=(
                f"{len(sets.pass_to_pass)} guard test(s) pass before the fix."
                if not failing_guards else
                f"{len(failing_guards)} guard test(s) already fail, so they "
                f"cannot guard against anything."
            ),
            evidence={"already_failing": failing_guards[:10]},
        ))

    if run_after is None:
        report.results.append(GateResult(
            Gate.GOLDEN_FIXES_TARGET,
            passed=False,
            detail="Not verified: no runner supplied.",
        ))
    else:
        after = run_after()
        unfixed = [t for t in sets.fail_to_pass if not after.get(t, False)]
        broken = [t for t in sets.pass_to_pass if not after.get(t, True)]
        report.results.append(GateResult(
            Gate.GOLDEN_FIXES_TARGET,
            passed=not unfixed and not broken,
            detail=(
                "The reference fix makes every target test pass and breaks "
                "no guard test."
                if not unfixed and not broken else
                f"{len(unfixed)} target test(s) still fail and "
                f"{len(broken)} guard test(s) break under the reference fix. "
                f"The task is mis-derived: nothing can solve it."
            ),
            evidence={"unfixed": unfixed[:10], "broken_guards": broken[:10]},
        ))

    if run_before is None:
        report.results.append(GateResult(
            Gate.NOT_FLAKY,
            passed=False,
            detail=(
                "Not verified: no runner supplied. This gate is the one the "
                "guide calls not optional."
            ),
        ))
    else:
        report.results.append(check_flakiness(run_before, runs=flake_runs))

    report.results.append(check_solvability(instance))
    return report


def summarise_rejections(reports: Iterable[ValidationReport]) -> dict[str, Any]:
    """Which gates reject the most candidates.

    As useful as the dataset itself: the distribution says what the mining
    pipeline cannot yet handle. A pipeline rejecting 70% of candidates at
    ``environment_builds`` is a pipeline with a build-system problem, not a
    pipeline that found bad repositories.
    """
    reports = list(reports)
    if not reports:
        return {"candidates": 0}

    admitted = [report for report in reports if report.admitted]
    rejections = Counter(
        gate.value for report in reports for gate in report.failed_gates
    )
    first_failures = Counter(
        report.failed_gates[0].value
        for report in reports if report.failed_gates
    )

    return {
        "candidates": len(reports),
        "admitted": len(admitted),
        "admission_rate": round(len(admitted) / len(reports), 4),
        "rejected_by_gate": dict(rejections.most_common()),
        "first_failure_by_gate": dict(first_failures.most_common()),
        "admitted_ids": [report.instance_id for report in admitted],
        "note": (
            "first_failure_by_gate is the actionable one: it says where the "
            "pipeline gives up first. A large environment_builds share means "
            "a build-system gap, not a shortage of good repositories."
        ),
    }
