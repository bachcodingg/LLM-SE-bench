"""
agent.termination — every way an episode can end, and the budget governor.

Six conditions, all enforced, none optional:

``success``
    The target tests pass **and** no previously-passing test broke. Passing
    the target while breaking something else is a regression, not a
    solution, and scoring it as success is how a benchmark rewards
    vandalism.
``max_steps``
``max_tokens``
``max_cost_eur``
``no_progress``
    N consecutive steps with no effective file mutation and no change in
    the test result. An agent re-reading the same file forever is a failure
    mode worth naming, not waiting out.
``wall_clock``

The money one is different in kind from the rest. Steps and tokens are
counted after a call; euros must be checked *before* one, because the call
is what spends them. :class:`BudgetGovernor` therefore refuses a step whose
worst-case cost would cross the ceiling, and the ceiling has no default:
there is no safe number to assume on someone else's account.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "StopCondition",
    "TerminationPolicy",
    "EpisodeState",
    "BudgetGovernor",
    "BudgetExceeded",
    "check_termination",
]


class StopCondition(str, Enum):
    """Why an episode ended."""

    SUCCESS = "success"
    MAX_STEPS = "max_steps"
    MAX_TOKENS = "max_tokens"
    MAX_COST = "max_cost_eur"
    NO_PROGRESS = "no_progress"
    WALL_CLOCK = "wall_clock"
    #: The model stopped calling tools without the tests passing.
    ABANDONED = "abandoned"
    #: The provider or the harness failed. Not the model's fault, and kept
    #: separate so it cannot be counted as a capability result.
    ERROR = "error"

    @property
    def is_success(self) -> bool:
        return self is StopCondition.SUCCESS

    @property
    def is_budget(self) -> bool:
        """True for the four resource ceilings."""
        return self in {
            StopCondition.MAX_STEPS,
            StopCondition.MAX_TOKENS,
            StopCondition.MAX_COST,
            StopCondition.WALL_CLOCK,
        }


@dataclass
class TerminationPolicy:
    """The ceilings for one episode.

    ``max_cost_eur`` has no default worth calling safe, so it is required
    and :class:`~agent.loop.AgentLoop` refuses to run a real client without
    it.
    """

    max_cost_eur: float
    max_steps: int = 30
    max_tokens: int = 500_000
    no_progress_steps: int = 5
    wall_clock_seconds: float = 900.0

    def __post_init__(self) -> None:
        if self.max_cost_eur <= 0:
            raise ValueError("max_cost_eur must be positive; there is no safe default.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if self.no_progress_steps < 1:
            raise ValueError("no_progress_steps must be at least 1.")


@dataclass
class EpisodeState:
    """Everything the termination checks look at.

    Updated by the loop after each step; read by :func:`check_termination`.
    Kept as plain data so a recorded episode can be replayed through the
    same checks without the loop.
    """

    steps: int = 0
    total_tokens: int = 0
    cost_eur: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    #: Test outcome after the most recent run_tests, or None if never run.
    tests_passed: int | None = None
    tests_total: int | None = None
    #: Whether that outcome came from real execution. False means the
    #: sandbox fell back to a structural check, whose counts are optimistic
    #: by construction — it credits every ``@Test`` it can see as passing.
    #: Success must not be declared on those numbers.
    tests_executed: bool = True
    #: The baseline: tests passing before the agent touched anything.
    baseline_passed: int | None = None
    baseline_total: int | None = None

    effective_edits: int = 0
    steps_without_progress: int = 0
    model_stopped_calling_tools: bool = False
    error: str = ""

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def tests_all_pass(self) -> bool:
        """True when the tests really ran and everything in them passed.

        ``tests_executed`` is part of the condition, not a footnote. The
        sandbox's structural fallback reports every ``@Test`` it can see as
        passing, so without this an episode that never compiled anything
        would terminate as a success — and a dry run would produce a
        100% resolve rate that someone would eventually quote.
        """
        return (
            self.tests_executed
            and self.tests_total is not None
            and self.tests_total > 0
            and self.tests_passed == self.tests_total
        )

    @property
    def regressed(self) -> bool:
        """True when fewer tests pass now than before the agent started.

        A solution that fixes the target by breaking something else is not
        a solution. Without a baseline nothing can be said, so this is
        False rather than optimistically assumed.
        """
        if self.baseline_passed is None or self.tests_passed is None:
            return False
        return self.tests_passed < self.baseline_passed

    def record_progress(self, made_progress: bool) -> None:
        """Advance or reset the no-progress counter."""
        self.steps_without_progress = 0 if made_progress else self.steps_without_progress + 1


class BudgetExceeded(RuntimeError):
    """Raised when a step would cross the cost ceiling.

    Raised rather than returned because this is the one condition that must
    stop the loop *before* the action, and a return value can be ignored by
    a caller that forgets to check it.
    """


class BudgetGovernor:
    """Refuses a step whose worst-case cost would cross the ceiling.

    Steps and tokens are counted after the fact; money cannot be, because
    the API call is the thing that spends it. So the governor projects the
    worst case for the next call — every output token at the output rate,
    every input token at the input rate — and refuses if that would cross.

    The projection is deliberately pessimistic. An agent that stops one step
    early has cost a result; an agent that overruns a budget on a paid
    account has cost trust.

    Parameters
    ----------
    ceiling_eur
        Hard limit for the episode.
    worst_case_call_eur
        Projected cost of one more call. When None, it is estimated from
        the model's published per-evaluation cost.
    """

    def __init__(self, ceiling_eur: float, worst_case_call_eur: float | None = None) -> None:
        if ceiling_eur <= 0:
            raise ValueError("ceiling_eur must be positive.")
        self.ceiling_eur = ceiling_eur
        self.worst_case_call_eur = worst_case_call_eur or 0.0
        self.spent_eur = 0.0
        self.calls = 0

    @property
    def remaining_eur(self) -> float:
        return max(0.0, self.ceiling_eur - self.spent_eur)

    def check(self) -> None:
        """Raise if the next call could cross the ceiling.

        Raises:
            BudgetExceeded: with the numbers, so the trajectory records why.
        """
        if self.spent_eur >= self.ceiling_eur:
            raise BudgetExceeded(
                f"Budget exhausted: spent €{self.spent_eur:.4f} of "
                f"€{self.ceiling_eur:.4f}."
            )
        projected = self.spent_eur + self.worst_case_call_eur
        if self.worst_case_call_eur and projected > self.ceiling_eur:
            raise BudgetExceeded(
                f"Refusing the next call: €{self.spent_eur:.4f} spent, the "
                f"call could cost up to €{self.worst_case_call_eur:.4f}, and "
                f"the ceiling is €{self.ceiling_eur:.4f}."
            )

    def record(self, cost_eur: float) -> float:
        """Add *cost_eur* to the running total and return the new total."""
        self.spent_eur += max(0.0, cost_eur)
        self.calls += 1
        return self.spent_eur


def check_termination(
    state: EpisodeState,
    policy: TerminationPolicy,
) -> StopCondition | None:
    """Return the condition that ends the episode, or None to continue.

    Order matters. Errors first, because nothing else is meaningful after
    one. Success next, so an episode that solves the task on its final
    permitted step is a success rather than a budget exhaustion. Then the
    ceilings, then the behavioural conditions.
    """
    if state.error:
        return StopCondition.ERROR

    if state.tests_all_pass and not state.regressed:
        return StopCondition.SUCCESS

    if state.cost_eur >= policy.max_cost_eur:
        return StopCondition.MAX_COST
    if state.steps >= policy.max_steps:
        return StopCondition.MAX_STEPS
    if state.total_tokens >= policy.max_tokens:
        return StopCondition.MAX_TOKENS
    if state.elapsed_seconds >= policy.wall_clock_seconds:
        return StopCondition.WALL_CLOCK

    if state.steps_without_progress >= policy.no_progress_steps:
        return StopCondition.NO_PROGRESS

    if state.model_stopped_calling_tools:
        return StopCondition.ABANDONED

    return None
