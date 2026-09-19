"""
agent.analysis — process metrics, failure taxonomy, trajectory diffing (M4).

Outcome metrics tell you *what* happened: solved, or not. Process metrics
tell you *why*, and they are the difference between "Gemini scores lower"
and "Gemini never opens the right file".

Three things live here.

**Process metrics.** Localisation accuracy, steps to first edit, edit churn,
backtracking, dead-end depth, context utilisation, cache hit rate. Computed
from the trajectory table, which is why that table's schema was fixed before
any of this was written.

**Failure taxonomy.** Nine classes, assigned by rule. Rules first, an
LLM-as-judge fallback only for what the rules cannot reach, and the
agreement between the two reported — because a taxonomy that a model
assigns is a taxonomy nobody can reproduce, and using a judge where a
deterministic rule exists is how a benchmark quietly becomes unfalsifiable.

**Trajectory diffing.** Two models, one task, aligned side by side, with the
step where they diverged. This is the most compelling single view in the
dashboard and it is not hard; it is just rarely built.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from agent.trajectory import Trajectory, TrajectoryStep

logger = logging.getLogger(__name__)

__all__ = [
    "FailureClass",
    "FailureVerdict",
    "ProcessMetrics",
    "LocalisationAccuracy",
    "classify_failure",
    "compute_process_metrics",
    "compute_localisation",
    "diff_trajectories",
    "TrajectoryDiff",
    "aggregate_failures",
]


class FailureClass(str, Enum):
    """Why an episode did not succeed.

    Ordered by how early in the pipeline the failure occurred, because that
    is the order the rules test them in: a run that never built cannot have
    failed at the semantic level.
    """

    SOLVED = "solved"
    LOCALISATION = "localisation_failure"
    ENVIRONMENT = "environment_failure"
    SYNTACTIC = "syntactic_failure"
    SEMANTIC = "semantic_failure"
    REGRESSION = "regression"
    BUDGET = "budget_exhaustion"
    LOOP = "loop"
    ABANDONMENT = "refusal_or_abandonment"
    TAMPER = "tamper"
    UNKNOWN = "unknown"

    @property
    def description(self) -> str:
        return {
            FailureClass.SOLVED: "Target tests pass and nothing else broke.",
            FailureClass.LOCALISATION: "Never opened the file that needed changing.",
            FailureClass.ENVIRONMENT: "The build never succeeded at all.",
            FailureClass.SYNTACTIC: "Produced a patch that does not compile.",
            FailureClass.SEMANTIC: "Compiles, tests still fail.",
            FailureClass.REGRESSION: "Target tests pass, others broke.",
            FailureClass.BUDGET: "Ran out of steps, tokens, euros or time.",
            FailureClass.LOOP: "Repeated itself without progress.",
            FailureClass.ABANDONMENT: "Stopped working before solving.",
            FailureClass.TAMPER: "Made the tests pass by cheating.",
            FailureClass.UNKNOWN: "No rule matched; needs a human or a judge.",
        }[self]


@dataclass
class FailureVerdict:
    """A classification, with the rule that produced it."""

    failure_class: FailureClass
    rule: str
    confidence: str = "rule"  # "rule" | "judge" | "ambiguous"
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.failure_class.value,
            "description": self.failure_class.description,
            "rule": self.rule,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


# ──────────────────────────────────────────────────────────────────────
# Failure taxonomy — rule-based
# ──────────────────────────────────────────────────────────────────────

#: Substrings in tool output that mean the build itself never worked, as
#: opposed to the patch not compiling. The distinction matters: one is the
#: harness's problem and one is the model's.
_ENVIRONMENT_MARKERS = (
    "EXECUTION SKIPPED",
    "docker",
    "could not resolve dependencies",
    "no compiler",
    "sandbox exception",
    "timed out after",
)

#: Substrings that mean the model's own patch failed to compile.
_SYNTACTIC_MARKERS = (
    "compile failed",
    "error: ",
    "cannot find symbol",
    "illegal start of",
    "';' expected",
    "incompatible types",
)


def classify_failure(
    trajectory: Trajectory,
    golden_files: Iterable[str] | None = None,
    tampered: bool = False,
) -> FailureVerdict:
    """Assign a failure class to *trajectory* by rule.

    Parameters
    ----------
    golden_files
        Paths the reference fix touches. Needed for the localisation rule —
        without them, "never opened the right file" is unanswerable and the
        rule is skipped rather than guessed at.
    tampered
        Whether :mod:`quality.tamper` found mechanical tampering.

    Returns a verdict whose ``confidence`` is ``"rule"`` when a
    deterministic rule fired and ``"ambiguous"`` when none did. Only the
    ambiguous ones are worth sending to a judge.
    """
    # Tampering first: a "solve" that cheated is not a solve, and a failure
    # that cheated is mis-diagnosed by every later rule.
    if tampered:
        return FailureVerdict(
            FailureClass.TAMPER,
            rule="mechanical tamper finding present",
            evidence="see the tamper report",
        )

    if trajectory.solved:
        return FailureVerdict(FailureClass.SOLVED, rule="stop_condition == success")

    if trajectory.regressed:
        return FailureVerdict(
            FailureClass.REGRESSION,
            rule="final_passed < baseline_passed",
            evidence=f"{trajectory.baseline_passed} -> {trajectory.final_passed}",
        )

    if not trajectory.tests_executed:
        return FailureVerdict(
            FailureClass.ENVIRONMENT,
            rule="tests never executed",
            evidence="the sandbox fell back to a structural check",
        )

    if trajectory.stop_condition == "error":
        return FailureVerdict(
            FailureClass.ENVIRONMENT,
            rule="stop_condition == error",
            evidence=trajectory.error[:200],
        )

    test_output = " ".join(
        step.result_preview.lower()
        for step in trajectory.steps
        if step.tool_name == "run_tests"
    )

    if any(marker in test_output for marker in _ENVIRONMENT_MARKERS):
        return FailureVerdict(
            FailureClass.ENVIRONMENT,
            rule="environment marker in run_tests output",
            evidence=next(m for m in _ENVIRONMENT_MARKERS if m in test_output),
        )

    # Localisation: did it ever open or write the file the fix belongs in?
    if golden_files:
        touched = set(trajectory.files_changed) | {
            step.tool_args.get("path", "")
            for step in trajectory.steps
            if step.tool_name in ("read_file", "apply_patch")
        }
        golden = set(golden_files)
        if golden and not (touched & golden):
            return FailureVerdict(
                FailureClass.LOCALISATION,
                rule="never read or wrote any golden file",
                evidence=f"golden={sorted(golden)}, touched={sorted(p for p in touched if p)}",
            )

    if trajectory.stop_condition == "no_progress":
        return FailureVerdict(
            FailureClass.LOOP,
            rule="stop_condition == no_progress",
            evidence=f"{trajectory.repeated_calls} repeated call(s)",
        )

    # Syntactic before semantic: a patch that does not compile has not been
    # given the chance to be semantically wrong.
    if any(marker in test_output for marker in _SYNTACTIC_MARKERS):
        return FailureVerdict(
            FailureClass.SYNTACTIC,
            rule="compiler diagnostic in run_tests output",
            evidence=next(m for m in _SYNTACTIC_MARKERS if m in test_output),
        )

    if trajectory.stop_condition in ("max_steps", "max_tokens", "max_cost_eur", "wall_clock"):
        return FailureVerdict(
            FailureClass.BUDGET,
            rule=f"stop_condition == {trajectory.stop_condition}",
            evidence=(
                f"{trajectory.num_steps} steps, {trajectory.total_tokens} tokens, "
                f"EUR {trajectory.total_cost_eur:.4f}"
            ),
        )

    if trajectory.stop_condition == "abandoned":
        # An agent that abandoned without editing anything did not attempt
        # the task; one that edited and gave up attempted and failed.
        if trajectory.steps_to_first_edit is None:
            return FailureVerdict(
                FailureClass.ABANDONMENT,
                rule="stopped calling tools, never edited a file",
                evidence="no apply_patch call succeeded",
            )
        return FailureVerdict(
            FailureClass.SEMANTIC,
            rule="edited, compiled, stopped with tests failing",
            evidence=f"{trajectory.final_passed}/{trajectory.final_total} passing",
        )

    if trajectory.final_total and trajectory.final_passed is not None:
        return FailureVerdict(
            FailureClass.SEMANTIC,
            rule="compiled, tests still failing",
            evidence=f"{trajectory.final_passed}/{trajectory.final_total} passing",
        )

    return FailureVerdict(
        FailureClass.UNKNOWN,
        rule="no rule matched",
        confidence="ambiguous",
        evidence=f"stop_condition={trajectory.stop_condition}",
    )


def aggregate_failures(verdicts: list[FailureVerdict]) -> dict[str, Any]:
    """Distribution over failure classes, plus how much needed a judge."""
    counts = Counter(verdict.failure_class.value for verdict in verdicts)
    by_confidence = Counter(verdict.confidence for verdict in verdicts)
    total = len(verdicts) or 1
    return {
        "episodes": len(verdicts),
        "distribution": dict(counts.most_common()),
        "shares": {
            name: round(count / total, 4) for name, count in counts.most_common()
        },
        "rule_coverage": round(by_confidence.get("rule", 0) / total, 4),
        "ambiguous": by_confidence.get("ambiguous", 0),
        "note": (
            "rule_coverage is the share classified deterministically. Only "
            "the ambiguous remainder should ever reach an LLM judge; a "
            "taxonomy assigned by a model is one nobody can reproduce."
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# Process metrics
# ──────────────────────────────────────────────────────────────────────

@dataclass
class LocalisationAccuracy:
    """Whether, and how quickly, the agent found the right file.

    Precision and recall over files. Recall is the one that matters: an
    agent that never opens the right file cannot fix it, and that is a
    different problem from one that opens it and writes the wrong patch.
    """

    golden_files: list[str] = field(default_factory=list)
    opened_files: list[str] = field(default_factory=list)
    edited_files: list[str] = field(default_factory=list)
    step_first_golden_opened: int | None = None
    step_first_golden_edited: int | None = None

    @property
    def recall(self) -> float:
        """Share of golden files the agent opened at all."""
        if not self.golden_files:
            return 0.0
        found = set(self.golden_files) & set(self.opened_files)
        return len(found) / len(self.golden_files)

    @property
    def precision(self) -> float:
        """Share of opened files that were golden. Low means wandering."""
        if not self.opened_files:
            return 0.0
        return len(set(self.golden_files) & set(self.opened_files)) / len(set(self.opened_files))

    @property
    def edit_precision(self) -> float:
        """Share of edited files that were golden. Low means collateral edits."""
        if not self.edited_files:
            return 0.0
        return len(set(self.golden_files) & set(self.edited_files)) / len(set(self.edited_files))

    @property
    def found(self) -> bool:
        return self.recall > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "edit_precision": round(self.edit_precision, 4),
            "step_first_golden_opened": self.step_first_golden_opened,
            "step_first_golden_edited": self.step_first_golden_edited,
            "golden_files": self.golden_files,
        }


def compute_localisation(
    trajectory: Trajectory,
    golden_files: Iterable[str],
) -> LocalisationAccuracy:
    """Measure how well the agent found the files the fix belongs in."""
    accuracy = LocalisationAccuracy(golden_files=sorted(golden_files))
    golden = set(accuracy.golden_files)

    opened: list[str] = []
    edited: list[str] = []
    for step in trajectory.steps:
        if step.tool_name == "read_file":
            path = step.tool_args.get("path", "")
            if path:
                opened.append(path)
                if path in golden and accuracy.step_first_golden_opened is None:
                    accuracy.step_first_golden_opened = step.step_index
        for path in step.files_touched:
            edited.append(path)
            if path in golden and accuracy.step_first_golden_edited is None:
                accuracy.step_first_golden_edited = step.step_index

    accuracy.opened_files = sorted(set(opened))
    accuracy.edited_files = sorted(set(edited))
    return accuracy


@dataclass
class ProcessMetrics:
    """How the agent worked, independent of whether it succeeded."""

    steps: int = 0
    steps_to_first_edit: int | None = None
    steps_to_solution: int | None = None
    edit_steps: int = 0
    read_steps: int = 0
    test_runs: int = 0

    repeated_calls: int = 0
    backtracking_rate: float = 0.0
    dead_end_depth: int = 0
    edit_churn: int = 0

    tool_distribution: dict[str, int] = field(default_factory=dict)
    tool_error_rate: float = 0.0
    malformed_call_rate: float = 0.0

    cache_hit_rate: float = 0.0
    context_growth: list[int] = field(default_factory=list)
    mean_tokens_per_step: float = 0.0

    cost_per_step_eur: float = 0.0
    cost_to_solution_eur: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "steps_to_first_edit": self.steps_to_first_edit,
            "steps_to_solution": self.steps_to_solution,
            "edit_steps": self.edit_steps,
            "read_steps": self.read_steps,
            "test_runs": self.test_runs,
            "repeated_calls": self.repeated_calls,
            "backtracking_rate": round(self.backtracking_rate, 4),
            "dead_end_depth": self.dead_end_depth,
            "edit_churn": self.edit_churn,
            "tool_distribution": self.tool_distribution,
            "tool_error_rate": round(self.tool_error_rate, 4),
            "malformed_call_rate": round(self.malformed_call_rate, 4),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "mean_tokens_per_step": round(self.mean_tokens_per_step, 1),
            "cost_per_step_eur": round(self.cost_per_step_eur, 6),
            "cost_to_solution_eur": self.cost_to_solution_eur,
        }


def compute_process_metrics(trajectory: Trajectory) -> ProcessMetrics:
    """Derive every process metric from a trajectory."""
    metrics = ProcessMetrics(
        steps=trajectory.num_steps,
        steps_to_first_edit=trajectory.steps_to_first_edit,
        repeated_calls=trajectory.repeated_calls,
        edit_churn=trajectory.edit_churn,
        tool_distribution=trajectory.tool_call_distribution,
        tool_error_rate=trajectory.tool_error_rate,
        malformed_call_rate=trajectory.malformed_call_rate,
        cache_hit_rate=trajectory.cache_hit_rate,
        cost_per_step_eur=trajectory.cost_per_step(),
    )
    if not trajectory.steps:
        return metrics

    metrics.edit_steps = sum(1 for step in trajectory.steps if step.is_edit)
    metrics.read_steps = sum(1 for step in trajectory.steps if step.tool_name == "read_file")
    metrics.test_runs = sum(1 for step in trajectory.steps if step.tool_name == "run_tests")

    if trajectory.solved:
        metrics.steps_to_solution = trajectory.num_steps
        metrics.cost_to_solution_eur = round(trajectory.total_cost_eur, 6)

    # Backtracking: a step that follows a *worse* test result than the one
    # before it. Going backwards is not the same as making no progress, and
    # an agent that oscillates looks fine on the no-progress counter.
    regressions = 0
    comparisons = 0
    previous: int | None = None
    for step in trajectory.steps:
        if step.tests_passing_after is None:
            continue
        if previous is not None:
            comparisons += 1
            if step.tests_passing_after < previous:
                regressions += 1
        previous = step.tests_passing_after
    metrics.backtracking_rate = regressions / comparisons if comparisons else 0.0

    metrics.dead_end_depth = _dead_end_depth(trajectory.steps)

    metrics.context_growth = _context_growth(trajectory.steps)
    total_tokens = sum(step.total_tokens for step in trajectory.steps)
    metrics.mean_tokens_per_step = total_tokens / trajectory.num_steps

    return metrics


def _dead_end_depth(steps: list[TrajectoryStep]) -> int:
    """Longest run of consecutive steps that changed nothing.

    The signature of an agent stuck in an unproductive line of enquiry. The
    no-progress rule caps this in a live episode; on a recorded one it
    measures how deep the agent got before the cap fired.
    """
    longest = 0
    current = 0
    previous_tests: int | None = None
    for step in steps:
        changed = bool(step.files_touched) or (
            step.tests_passing_after is not None
            and step.tests_passing_after != previous_tests
        )
        if step.tests_passing_after is not None:
            previous_tests = step.tests_passing_after
        if changed:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _context_growth(steps: list[TrajectoryStep]) -> list[int]:
    """Cumulative input tokens per step: how fast the window fills.

    A curve that flattens means prompt caching is working. One that grows
    linearly means every turn re-sends the whole conversation at full price,
    which is the dominant cost in a long episode.
    """
    growth: list[int] = []
    running = 0
    for step in steps:
        running += step.tokens_in + step.cache_read_tokens
        growth.append(running)
    return growth


# ──────────────────────────────────────────────────────────────────────
# Trajectory diffing
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryDiff:
    """Two episodes on the same task, aligned, with the divergence point."""

    task_id: str = ""
    left_model: str = ""
    right_model: str = ""
    aligned: list[dict[str, Any]] = field(default_factory=list)
    divergence_step: int | None = None
    left_outcome: str = ""
    right_outcome: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "left_model": self.left_model,
            "right_model": self.right_model,
            "divergence_step": self.divergence_step,
            "left_outcome": self.left_outcome,
            "right_outcome": self.right_outcome,
            "steps": self.aligned,
        }

    def render(self, width: int = 38) -> str:
        """A side-by-side text rendering, for a terminal or a report."""
        lines = [
            f"{self.task_id}: {self.left_model} vs {self.right_model}",
            f"{'=' * (width * 2 + 7)}",
            f"{'#':>3}  {self.left_model[:width]:<{width}}  {self.right_model[:width]:<{width}}",
            f"{'-' * (width * 2 + 7)}",
        ]
        for row in self.aligned:
            marker = " *" if row["step"] == self.divergence_step else "  "
            lines.append(
                f"{row['step']:>3}{marker}{row['left'][:width]:<{width}}  "
                f"{row['right'][:width]:<{width}}"
            )
        lines.append(f"{'-' * (width * 2 + 7)}")
        lines.append(f"     {self.left_outcome:<{width}}  {self.right_outcome:<{width}}")
        if self.divergence_step is not None:
            lines.append(f"\n* diverged at step {self.divergence_step}")
        return "\n".join(lines)


def diff_trajectories(left: Trajectory, right: Trajectory) -> TrajectoryDiff:
    """Align two episodes on the same task and find where they diverged.

    Alignment is positional: step 1 against step 1. A semantic alignment
    would be better and is much harder; positional alignment is honest about
    what it does and is enough to see that one model started editing at step
    3 while the other was still reading at step 9.
    """
    diff = TrajectoryDiff(
        task_id=left.task_id or right.task_id,
        left_model=left.model_id or "left",
        right_model=right.model_id or "right",
        left_outcome=f"{left.stop_condition} ({left.num_steps} steps)",
        right_outcome=f"{right.stop_condition} ({right.num_steps} steps)",
    )

    for index in range(max(len(left.steps), len(right.steps))):
        left_step = left.steps[index] if index < len(left.steps) else None
        right_step = right.steps[index] if index < len(right.steps) else None

        left_label = _label(left_step)
        right_label = _label(right_step)
        same = (
            left_step is not None
            and right_step is not None
            and left_step.tool_name == right_step.tool_name
            and left_step.tool_args == right_step.tool_args
        )
        if not same and diff.divergence_step is None:
            diff.divergence_step = index + 1

        diff.aligned.append({
            "step": index + 1,
            "left": left_label,
            "right": right_label,
            "same": same,
            "left_tests": left_step.tests_passing_after if left_step else None,
            "right_tests": right_step.tests_passing_after if right_step else None,
        })

    return diff


def _label(step: TrajectoryStep | None) -> str:
    """One-line description of a step, for the aligned view."""
    if step is None:
        return "—"
    if not step.tool_name:
        return "(no tool call)"
    target = step.tool_args.get("path") or step.tool_args.get("pattern") or ""
    label = f"{step.tool_name}({target})" if target else step.tool_name
    if step.tool_error:
        label += " [error]"
    return label
