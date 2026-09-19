"""
llm_gateway.budget — cost governance v2 (M7).

Small module, disproportionate impact. Four things live here.

**Correct prompt-cache accounting.** Cache writes and cache reads are priced
differently from base input tokens — a read is usually a tenth of the input
rate, a write a premium above it. Charging all three at the input rate, as
the v1 pricing table does, gets the *direction* of the error right for a
budget check and the *magnitude* wrong for a comparison. A model whose
answers are mostly cache reads then looks far more expensive than it is, and
a leaderboard sorted on that figure is sorted wrongly.

**Cost-normalised scoring.** Resolve rate per euro, per million tokens, per
hour. Raw resolve rate answers "which is most capable"; for anyone with a
budget the question is "which is most capable per euro", and those have
different answers.

**Budget simulation.** Given €250, which configurations are affordable, and
what statistical power does each buy? Answering this before a run is much
cheaper than discovering it after.

**Pareto frontier.** Over (quality, cost, latency), with the dominated
configurations marked. A configuration that is worse on every axis than
another is not a trade-off, it is a mistake, and it should be visibly so.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger(__name__)

__all__ = [
    "CachePricing",
    "CACHE_PRICING",
    "price_call",
    "CostNormalised",
    "normalise_by_cost",
    "BudgetSimulator",
    "Configuration",
    "pareto_frontier",
]


@dataclass(frozen=True)
class CachePricing:
    """Multipliers on the base input rate for cached tokens.

    Published rates as of the 2026 price lists. They are multipliers rather
    than absolute prices so that a change to the base rate flows through
    without touching this table.

    ``write`` above 1.0 is not an error: writing a cache entry costs *more*
    than a fresh input token, which is why caching a prefix used twice loses
    money and caching one used twenty times is the largest lever available.
    """

    read: float = 0.1
    write: float = 1.25

    #: Batch/asynchronous processing discount, where a provider offers it.
    #: Applies to the whole call, input and output.
    batch: float = 0.5


#: Per provider. A model with no entry falls back to the conservative
#: default, which prices a cache read at the full input rate — overstating
#: cost, which is the safe direction for a ceiling and the wrong direction
#: for a comparison. Add an entry rather than relying on it.
CACHE_PRICING: dict[str, CachePricing] = {
    "claude": CachePricing(read=0.1, write=1.25, batch=0.5),
    "gpt4": CachePricing(read=0.5, write=1.0, batch=0.5),
    "gemini": CachePricing(read=0.25, write=1.0, batch=0.5),
}

#: No discount and no premium: cached tokens cost what fresh ones cost.
CONSERVATIVE_PRICING = CachePricing(read=1.0, write=1.0, batch=1.0)


def price_call(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cost_per_input_token: float = 0.0,
    cost_per_output_token: float = 0.0,
    provider: str = "",
    batch: bool = False,
) -> dict[str, float]:
    """Price one call, with cache tiers charged at their own rates.

    Returns a breakdown rather than a single number, so a caller can see
    *where* the money went. In a long agent episode the cache-read line is
    usually the largest by token count and among the smallest by cost, and
    a single total hides that entirely.
    """
    pricing = CACHE_PRICING.get(provider, CONSERVATIVE_PRICING)
    multiplier = pricing.batch if batch else 1.0

    fresh_input = input_tokens * cost_per_input_token
    cached_read = cache_read_tokens * cost_per_input_token * pricing.read
    cached_write = cache_write_tokens * cost_per_input_token * pricing.write
    output = output_tokens * cost_per_output_token

    total = (fresh_input + cached_read + cached_write + output) * multiplier
    uncached_equivalent = (
        (input_tokens + cache_read_tokens + cache_write_tokens) * cost_per_input_token
        + output
    )

    return {
        "input_usd": round(fresh_input * multiplier, 8),
        "cache_read_usd": round(cached_read * multiplier, 8),
        "cache_write_usd": round(cached_write * multiplier, 8),
        "output_usd": round(output * multiplier, 8),
        "total_usd": round(total, 8),
        "batch_discount_applied": batch,
        "cache_saving_usd": round(max(0.0, uncached_equivalent - total), 8),
        "pricing_source": provider if provider in CACHE_PRICING else "conservative-default",
    }


# ──────────────────────────────────────────────────────────────────────
# Cost-normalised scoring
# ──────────────────────────────────────────────────────────────────────

@dataclass
class CostNormalised:
    """Quality per unit of each resource spent."""

    label: str = ""
    resolve_rate: float = 0.0
    solved: int = 0
    attempted: int = 0
    cost_eur: float = 0.0
    tokens: int = 0
    wall_clock_hours: float = 0.0

    @property
    def solves_per_eur(self) -> float | None:
        """Solves per euro. ``None`` when nothing was spent.

        Not zero: a run with no recorded cost has an *unknown* efficiency,
        and zero would sort it last among configurations that are actually
        measured.
        """
        return round(self.solved / self.cost_eur, 4) if self.cost_eur > 0 else None

    @property
    def eur_per_solve(self) -> float | None:
        """The number a team with a budget actually asks for.

        ``None`` when nothing was solved *or* nothing was spent. A zero cost
        means the run was not costed, not that it was free, and returning
        0.0 would sort an unmeasured run ahead of every measured one.
        """
        if not self.solved or self.cost_eur <= 0:
            return None
        return round(self.cost_eur / self.solved, 6)

    @property
    def solves_per_million_tokens(self) -> float | None:
        return round(self.solved / (self.tokens / 1_000_000), 4) if self.tokens else None

    @property
    def solves_per_hour(self) -> float | None:
        return (
            round(self.solved / self.wall_clock_hours, 4)
            if self.wall_clock_hours > 0 else None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "attempted": self.attempted,
            "solved": self.solved,
            "resolve_rate": round(self.resolve_rate, 4),
            "cost_eur": round(self.cost_eur, 4),
            "eur_per_solve": self.eur_per_solve,
            "solves_per_eur": self.solves_per_eur,
            "solves_per_million_tokens": self.solves_per_million_tokens,
            "solves_per_hour": self.solves_per_hour,
        }


def normalise_by_cost(
    runs: Iterable[dict[str, Any]],
) -> list[CostNormalised]:
    """Turn raw run totals into cost-normalised rows, best value first.

    Each input row needs ``label``, ``solved``, ``attempted``, ``cost_eur``,
    and optionally ``tokens`` and ``wall_clock_seconds``.

    Sorted by euros per solve, ascending — which is a different order from
    resolve rate, and the point of the exercise.
    """
    rows: list[CostNormalised] = []
    for run in runs:
        attempted = int(run.get("attempted", 0))
        solved = int(run.get("solved", 0))
        rows.append(CostNormalised(
            label=str(run.get("label", "")),
            solved=solved,
            attempted=attempted,
            resolve_rate=solved / attempted if attempted else 0.0,
            cost_eur=float(run.get("cost_eur", 0.0)),
            tokens=int(run.get("tokens", 0)),
            wall_clock_hours=float(run.get("wall_clock_seconds", 0.0)) / 3600.0,
        ))

    # Unmeasurable efficiency sorts last rather than first.
    return sorted(
        rows,
        key=lambda row: (row.eur_per_solve is None, row.eur_per_solve or 0.0),
    )


# ──────────────────────────────────────────────────────────────────────
# Budget simulation
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Configuration:
    """One thing you could spend a budget on."""

    label: str
    cost_per_episode_eur: float
    tasks: int
    repeats: int = 1
    expected_resolve_rate: float = 0.5
    mean_latency_s: float = 0.0

    @property
    def episodes(self) -> int:
        return self.tasks * self.repeats

    @property
    def total_cost_eur(self) -> float:
        return self.cost_per_episode_eur * self.episodes

    @property
    def wall_clock_hours(self) -> float:
        return self.episodes * self.mean_latency_s / 3600.0


class BudgetSimulator:
    """Answers "given €X, what can I afford and what does it buy me?".

    The second half is the part usually skipped. Affording a run is not the
    same as affording a run that can detect anything, and a study that
    spends its whole budget on a comparison with no power to resolve the
    difference it is looking for has wasted all of it.
    """

    def __init__(self, budget_eur: float) -> None:
        if budget_eur <= 0:
            raise ValueError("budget_eur must be positive.")
        self.budget_eur = budget_eur

    def affordable(self, configurations: Iterable[Configuration]) -> list[dict[str, Any]]:
        """Which configurations fit, and what power each gives.

        Power is for detecting a 5-percentage-point difference in resolve
        rate between two configurations at α = 0.05, via the normal
        approximation for two proportions. It is an approximation, and it is
        stated as one — but it is the difference between "I can afford 300
        episodes" and "300 episodes cannot see the effect I am looking for".
        """
        rows: list[dict[str, Any]] = []
        for configuration in configurations:
            total = configuration.total_cost_eur
            fits = total <= self.budget_eur
            max_episodes = (
                int(self.budget_eur / configuration.cost_per_episode_eur)
                if configuration.cost_per_episode_eur > 0 else configuration.episodes
            )
            rows.append({
                "label": configuration.label,
                "episodes": configuration.episodes,
                "total_cost_eur": round(total, 4),
                "affordable": fits,
                "budget_headroom_eur": round(self.budget_eur - total, 4),
                "max_episodes_within_budget": max_episodes,
                "wall_clock_hours": round(configuration.wall_clock_hours, 2),
                "power_for_5pp": round(
                    power_two_proportions(
                        n_per_group=configuration.episodes,
                        p1=configuration.expected_resolve_rate,
                        effect=0.05,
                    ), 3
                ),
                "episodes_for_80_percent_power": required_n_two_proportions(
                    p1=configuration.expected_resolve_rate, effect=0.05, power=0.8
                ),
            })
        return sorted(rows, key=lambda row: (not row["affordable"], row["total_cost_eur"]))

    def allocate(
        self,
        cost_per_episode_eur: float,
        tasks: int,
        min_repeats: int = 1,
    ) -> dict[str, Any]:
        """How to split a budget between more tasks and more repeats.

        The trade-off is real and usually decided by accident. More tasks
        reduce task-sampling variance; more repeats reduce the model's own
        stochastic variance. Which matters more depends on which is larger,
        and the honest answer is that you will not know until you have run
        something.
        """
        if cost_per_episode_eur <= 0:
            raise ValueError("cost_per_episode_eur must be positive.")
        affordable_episodes = int(self.budget_eur / cost_per_episode_eur)

        options: list[dict[str, Any]] = []
        for repeats in (1, 2, 3, 5):
            if repeats < min_repeats:
                continue
            coverable = affordable_episodes // repeats
            options.append({
                "repeats": repeats,
                "tasks_covered": min(tasks, coverable),
                "coverage": round(min(tasks, coverable) / tasks, 4) if tasks else 0.0,
                "episodes": min(tasks, coverable) * repeats,
                "cost_eur": round(min(tasks, coverable) * repeats * cost_per_episode_eur, 4),
                "detects_flakiness": repeats >= 3,
            })

        return {
            "budget_eur": self.budget_eur,
            "cost_per_episode_eur": cost_per_episode_eur,
            "affordable_episodes": affordable_episodes,
            "options": options,
            "guidance": (
                "Repeats below 3 cannot distinguish a flaky task from a model "
                "difference. Coverage below about 0.5 makes any per-dataset "
                "claim a claim about the half you happened to run."
            ),
        }


def power_two_proportions(
    n_per_group: int,
    p1: float,
    effect: float,
    alpha: float = 0.05,
) -> float:
    """Approximate power to detect *effect* between two proportions.

    Normal approximation, two-sided. Adequate for planning and not a
    substitute for the exact test; it is here to stop a run being
    commissioned that cannot answer its own question.
    """
    if n_per_group < 2:
        return 0.0
    p1 = min(max(p1, 0.001), 0.999)
    p2 = min(max(p1 + effect, 0.001), 0.999)
    pooled = (p1 + p2) / 2

    standard_error = math.sqrt(2 * pooled * (1 - pooled) / n_per_group)
    if standard_error == 0:
        return 0.0

    # Two-sided critical value at alpha, via the normal quantile.
    z_alpha = _normal_quantile(1 - alpha / 2)
    z_beta = abs(p2 - p1) / standard_error - z_alpha
    return max(0.0, min(1.0, _normal_cdf(z_beta)))


def required_n_two_proportions(
    p1: float,
    effect: float,
    power: float = 0.8,
    alpha: float = 0.05,
) -> int:
    """Episodes per group needed for *power* to detect *effect*."""
    p1 = min(max(p1, 0.001), 0.999)
    p2 = min(max(p1 + effect, 0.001), 0.999)
    pooled = (p1 + p2) / 2
    z_alpha = _normal_quantile(1 - alpha / 2)
    z_power = _normal_quantile(power)
    if p2 == p1:
        return 0
    n = 2 * pooled * (1 - pooled) * (z_alpha + z_power) ** 2 / (p2 - p1) ** 2
    return int(math.ceil(n))


def _normal_cdf(z: float) -> float:
    """Standard normal CDF, via the error function."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _normal_quantile(p: float) -> float:
    """Standard normal inverse CDF (Acklam's rational approximation).

    Accurate to about 1e-9, which is far beyond what a power calculation
    needs, and avoids a SciPy import in a module that is otherwise stdlib.
    """
    if not 0 < p < 1:
        raise ValueError("p must be strictly between 0 and 1")

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    low, high = 0.02425, 1 - 0.02425
    if p < low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# ──────────────────────────────────────────────────────────────────────
# Pareto frontier
# ──────────────────────────────────────────────────────────────────────

def pareto_frontier(
    points: Iterable[dict[str, Any]],
    maximise: tuple[str, ...] = ("quality",),
    minimise: tuple[str, ...] = ("cost", "latency"),
) -> dict[str, Any]:
    """Split *points* into the frontier and the configurations it dominates.

    A point is dominated when another is at least as good on every axis and
    strictly better on one. A dominated configuration is not a trade-off —
    there is no budget and no deadline under which it is the right choice —
    and saying so is more useful than ranking it fourth.

    Each point needs ``label`` plus the named axes.
    """
    rows = list(points)
    if not rows:
        return {"frontier": [], "dominated": [], "axes": {}}

    def dominates(better: dict[str, Any], worse: dict[str, Any]) -> bool:
        at_least_as_good = all(
            float(better.get(axis, 0)) >= float(worse.get(axis, 0)) for axis in maximise
        ) and all(
            float(better.get(axis, 0)) <= float(worse.get(axis, 0)) for axis in minimise
        )
        strictly_better = any(
            float(better.get(axis, 0)) > float(worse.get(axis, 0)) for axis in maximise
        ) or any(
            float(better.get(axis, 0)) < float(worse.get(axis, 0)) for axis in minimise
        )
        return at_least_as_good and strictly_better

    frontier: list[dict[str, Any]] = []
    dominated: list[dict[str, Any]] = []
    for candidate in rows:
        dominators = [
            other["label"] for other in rows
            if other is not candidate and dominates(other, candidate)
        ]
        if dominators:
            dominated.append({**candidate, "dominated_by": dominators})
        else:
            frontier.append(candidate)

    primary = maximise[0] if maximise else (minimise[0] if minimise else "label")
    frontier.sort(key=lambda row: float(row.get(primary, 0)), reverse=bool(maximise))

    return {
        "frontier": frontier,
        "dominated": dominated,
        "axes": {"maximise": list(maximise), "minimise": list(minimise)},
        "note": (
            "A dominated configuration is beaten on every axis at once. "
            "There is no budget or deadline under which it is the right "
            "choice, which is a stronger statement than 'it ranked lower'."
        ),
    }
