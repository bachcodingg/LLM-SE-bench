"""
stats.agents — statistics for stochastic, clustered agent outcomes (M8).

The v1 engine assumes deterministic paired outcomes: one number per (model,
problem), compared pairwise. Agents are not that. The same agent on the same
task gives different answers, several episodes share a task, and the
variance has at least four sources that get conflated if you let them.

Five things live here.

**Unbiased pass@k.** The combinatorial estimator, not "ran it k times and
took the max", which is biased upward and increasingly so as k approaches n.

**Variance decomposition.** How much of the spread comes from the task, the
model, the seed, the scaffold. A benchmark where task variance dominates is
not measuring models, and you cannot tell without decomposing.

**Power analysis from observed variance.** Given what you have seen, how
many tasks and repeats does it take to detect a 5-point difference. Usually
more than anyone budgeted for, which is the point of asking.

**Clustered bootstrap.** Episodes on the same task are correlated. Resampling
episodes independently treats 3 repeats of 58 tasks as 174 independent
observations, and produces confidence intervals roughly √3 too narrow.
Resample *tasks*, carry their episodes along.

**Sequential testing with alpha spending.** Stop a run early when the answer
is already clear, without inflating the false-positive rate from repeated
looking. This saves money directly, which is the only reason it is here.
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "pass_at_k",
    "pass_at_k_from_episodes",
    "VarianceDecomposition",
    "decompose_variance",
    "required_episodes",
    "clustered_bootstrap_ci",
    "SequentialTest",
    "AlphaSpending",
]


# ──────────────────────────────────────────────────────────────────────
# Unbiased pass@k
# ──────────────────────────────────────────────────────────────────────

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for *c* successes out of *n* samples.

    ``1 - C(n-c, k) / C(n, k)``: the probability that a random draw of *k*
    from the *n* samples contains at least one success.

    The naive alternative — run k times, report whether any passed — is
    biased upward, badly so when k is close to n, because it conditions on
    the particular k you drew. Chen et al. (2021), §2.1.

    Raises:
        ValueError: when k > n, which has no defined answer rather than a
            convenient one.
    """
    if n < 0 or c < 0 or k < 1:
        raise ValueError("n and c must be non-negative and k at least 1")
    if c > n:
        raise ValueError(f"successes ({c}) cannot exceed samples ({n})")
    if k > n:
        raise ValueError(
            f"pass@{k} needs at least {k} samples; only {n} were drawn. "
            f"Reporting it anyway would be extrapolation dressed as a "
            f"measurement."
        )
    if n - c < k:
        return 1.0
    # Product form avoids building large binomials.
    return 1.0 - math.prod(
        (n - c - i) / (n - i) for i in range(k)
    )


def pass_at_k_from_episodes(
    outcomes: dict[str, list[bool]],
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """pass@k across tasks, from ``{task_id: [solved, solved, ...]}``.

    Tasks with fewer than *k* samples are excluded from that k and counted,
    rather than silently padded. A pass@5 computed over the three tasks that
    happened to have five samples is not a pass@5 for the benchmark.
    """
    results: dict[str, Any] = {"tasks": len(outcomes), "estimates": {}}

    for k in k_values:
        eligible = {
            task: samples for task, samples in outcomes.items() if len(samples) >= k
        }
        if not eligible:
            results["estimates"][f"pass@{k}"] = {
                "value": None,
                "tasks_included": 0,
                "tasks_excluded": len(outcomes),
                "reason": f"no task has {k} or more samples",
            }
            continue

        per_task = [
            pass_at_k(n=len(samples), c=sum(samples), k=k)
            for samples in eligible.values()
        ]
        results["estimates"][f"pass@{k}"] = {
            "value": round(sum(per_task) / len(per_task), 4),
            "tasks_included": len(eligible),
            "tasks_excluded": len(outcomes) - len(eligible),
        }

    results["note"] = (
        "Unbiased combinatorial estimator (Chen et al., 2021). Tasks with "
        "fewer than k samples are excluded from that k, not padded."
    )
    return results


# ──────────────────────────────────────────────────────────────────────
# Variance decomposition
# ──────────────────────────────────────────────────────────────────────

@dataclass
class VarianceDecomposition:
    """Where the spread in outcomes comes from."""

    total: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    residual: float = 0.0
    n_observations: int = 0

    @property
    def shares(self) -> dict[str, float]:
        """Each component as a share of total variance."""
        if self.total <= 0:
            return {}
        shares = {
            name: round(value / self.total, 4)
            for name, value in self.components.items()
        }
        shares["residual"] = round(self.residual / self.total, 4)
        return shares

    @property
    def dominant(self) -> str:
        """The largest single source."""
        candidates = {**self.components, "residual": self.residual}
        return max(candidates, key=lambda name: candidates[name]) if candidates else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_observations": self.n_observations,
            "total_variance": round(self.total, 6),
            "components": {k: round(v, 6) for k, v in self.components.items()},
            "residual": round(self.residual, 6),
            "shares": self.shares,
            "dominant_source": self.dominant,
            "interpretation": self._interpretation(),
        }

    def _interpretation(self) -> str:
        shares = self.shares
        if not shares:
            return "No variance to decompose: every outcome was identical."
        task = shares.get("task", 0.0)
        model = shares.get("model", 0.0)
        seed = shares.get("seed", 0.0)
        parts = []
        if task > 0.5:
            parts.append(
                f"Task variance dominates ({task:.0%}). The benchmark is "
                f"measuring which tasks are hard more than which models are "
                f"good; comparisons need pairing on task, which the "
                f"Wilcoxon and Friedman tests already do."
            )
        if model < 0.1 and self.n_observations > 20:
            parts.append(
                f"Model variance is {model:.0%}. On this task set the models "
                f"are close to indistinguishable."
            )
        if seed > 0.2:
            parts.append(
                f"Seed variance is {seed:.0%}: repeated runs of the same "
                f"model on the same task disagree often. Single-run results "
                f"are not reliable here."
            )
        return " ".join(parts) or "No single source dominates."


def decompose_variance(
    observations: Iterable[dict[str, Any]],
    value_key: str = "score",
    factors: Sequence[str] = ("task", "model", "scaffold", "seed"),
) -> VarianceDecomposition:
    """Attribute variance in *value_key* to each factor.

    A one-way decomposition per factor: between-group variance for each,
    with whatever is left as residual. Not a full crossed ANOVA — that needs
    a balanced design this data rarely has — and the shares are therefore
    indicative rather than exact, which is stated rather than implied.

    Each observation is a dict holding *value_key* and the factor columns.
    """
    rows = [row for row in observations if value_key in row]
    decomposition = VarianceDecomposition(n_observations=len(rows))
    if len(rows) < 2:
        return decomposition

    values = [float(row[value_key]) for row in rows]
    grand_mean = sum(values) / len(values)
    decomposition.total = sum((v - grand_mean) ** 2 for v in values) / (len(values) - 1)
    if decomposition.total <= 0:
        return decomposition

    explained = 0.0
    for factor in factors:
        groups: dict[Any, list[float]] = defaultdict(list)
        for row in rows:
            if factor in row:
                groups[row[factor]].append(float(row[value_key]))
        if len(groups) < 2:
            continue

        between = sum(
            len(group) * (sum(group) / len(group) - grand_mean) ** 2
            for group in groups.values()
        ) / (len(values) - 1)
        decomposition.components[factor] = between
        explained += between

    decomposition.residual = max(0.0, decomposition.total - explained)
    return decomposition


# ──────────────────────────────────────────────────────────────────────
# Power analysis from observed variance
# ──────────────────────────────────────────────────────────────────────

def required_episodes(
    observed_variance: float,
    effect: float = 0.05,
    power: float = 0.8,
    alpha: float = 0.05,
    repeats: int = 1,
    intraclass_correlation: float = 0.0,
) -> dict[str, Any]:
    """Tasks needed to detect *effect*, given the variance you measured.

    Accounts for clustering: repeated episodes on one task are correlated,
    so k repeats are worth less than k independent observations. The design
    effect ``1 + (k-1)·ICC`` is the standard correction, and ignoring it is
    how a study convinces itself that tripling the repeats tripled its
    sample.

    Parameters
    ----------
    observed_variance
        Variance of the per-task outcome, from a pilot run.
    intraclass_correlation
        How alike repeats of the same task are, 0.0-1.0. At 1.0 extra
        repeats add nothing at all.
    """
    from llm_gateway.budget import _normal_quantile

    if observed_variance <= 0:
        return {
            "tasks_required": 0,
            "note": "Observed variance is zero: every outcome was identical.",
        }
    if effect <= 0:
        raise ValueError("effect must be positive")

    z_alpha = _normal_quantile(1 - alpha / 2)
    z_power = _normal_quantile(power)

    base = 2 * observed_variance * (z_alpha + z_power) ** 2 / effect ** 2
    design_effect = 1 + (repeats - 1) * intraclass_correlation
    required = base * design_effect / repeats

    return {
        "tasks_required": int(math.ceil(required)),
        "episodes_required": int(math.ceil(required)) * repeats,
        "effect_size": effect,
        "power": power,
        "alpha": alpha,
        "repeats": repeats,
        "design_effect": round(design_effect, 3),
        "effective_sample_per_task": round(repeats / design_effect, 3),
        "note": (
            "Normal approximation. The design effect accounts for repeats on "
            "the same task being correlated: at ICC 1.0 extra repeats add "
            "nothing, and treating them as independent observations produces "
            "intervals that are too narrow."
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# Clustered bootstrap
# ──────────────────────────────────────────────────────────────────────

def clustered_bootstrap_ci(
    clusters: dict[str, list[float]],
    statistic: str = "mean",
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap CI that resamples *clusters*, not observations.

    Episodes on the same task are correlated. Resampling episodes
    independently treats 3 repeats of 58 tasks as 174 independent
    observations and produces an interval roughly √3 too narrow — an
    interval that will be wrong about three times as often as it claims.

    Resampling whole tasks, carrying their episodes along, preserves the
    correlation structure. This is the only correct way to bootstrap this
    data, and it is easy to get wrong because the incorrect version runs
    fine and gives a nicer-looking answer.
    """
    if not clusters:
        return {"error": "no clusters supplied"}
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    keys = list(clusters)
    observed = _statistic([v for values in clusters.values() for v in values], statistic)

    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_resamples):
        drawn = [rng.choice(keys) for _ in keys]
        pooled = [value for key in drawn for value in clusters[key]]
        if pooled:
            estimates.append(_statistic(pooled, statistic))

    if not estimates:
        return {"error": "every resample was empty"}

    estimates.sort()
    lower_index = int((1 - confidence) / 2 * len(estimates))
    upper_index = int((1 + confidence) / 2 * len(estimates)) - 1

    naive_n = sum(len(values) for values in clusters.values())
    return {
        "statistic": statistic,
        "observed": round(observed, 6),
        "ci_lower": round(estimates[max(0, lower_index)], 6),
        "ci_upper": round(estimates[min(len(estimates) - 1, upper_index)], 6),
        "confidence": confidence,
        "n_clusters": len(keys),
        "n_observations": naive_n,
        "n_resamples": len(estimates),
        "note": (
            f"Resampled {len(keys)} clusters, not {naive_n} observations. "
            f"Resampling observations would treat correlated repeats as "
            f"independent and give an interval that is too narrow."
        ),
    }


def _statistic(values: list[float], name: str) -> float:
    if not values:
        return 0.0
    if name == "mean":
        return sum(values) / len(values)
    if name == "median":
        ordered = sorted(values)
        mid = len(ordered) // 2
        return (
            ordered[mid] if len(ordered) % 2
            else (ordered[mid - 1] + ordered[mid]) / 2
        )
    if name == "sum":
        return sum(values)
    raise ValueError(f"unknown statistic {name!r}; expected mean, median or sum")


# ──────────────────────────────────────────────────────────────────────
# Sequential testing with alpha spending
# ──────────────────────────────────────────────────────────────────────

class AlphaSpending:
    """O'Brien-Fleming alpha spending for interim analyses.

    Looking at your data repeatedly and stopping when it looks significant
    inflates the false-positive rate badly — five looks at α = 0.05 gives a
    real error rate near 14%. An alpha spending function fixes this by
    making early looks conservative and spending the remaining budget later.

    O'Brien-Fleming is the right default here because it is very strict
    early: an agent run's first few tasks are the least representative, and
    stopping on them is the mistake this exists to prevent.
    """

    def __init__(self, alpha: float = 0.05, total_looks: int = 5) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if total_looks < 1:
            raise ValueError("total_looks must be at least 1")
        self.alpha = alpha
        self.total_looks = total_looks

    def spent_by(self, look: int) -> float:
        """Cumulative alpha spent by look number *look* (1-based)."""
        from llm_gateway.budget import _normal_cdf, _normal_quantile

        if look < 1:
            return 0.0
        if look >= self.total_looks:
            return self.alpha
        fraction = look / self.total_looks
        z = _normal_quantile(1 - self.alpha / 2)
        return round(2 * (1 - _normal_cdf(z / math.sqrt(fraction))), 6)

    def threshold_at(self, look: int) -> float:
        """The p-value threshold for look *look*."""
        return round(self.spent_by(look) - self.spent_by(look - 1), 6)

    def schedule(self) -> list[dict[str, float]]:
        return [
            {
                "look": look,
                "cumulative_alpha": self.spent_by(look),
                "threshold": self.threshold_at(look),
            }
            for look in range(1, self.total_looks + 1)
        ]


@dataclass
class SequentialTest:
    """Run a comparison in stages, stopping early when the answer is clear.

    This saves money directly: a run that can be stopped at look 2 of 5 has
    spent 40% of its budget. That is the only reason to accept the extra
    machinery, and it is enough of one.

    The discipline it enforces: the number of looks is declared *before*
    the run, not chosen once the data looks promising. Adding a sixth look
    because the fifth was nearly significant is exactly the practice alpha
    spending exists to prevent.
    """

    alpha: float = 0.05
    total_looks: int = 5
    looks: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.spending = AlphaSpending(self.alpha, self.total_looks)

    def look(self, p_value: float, n_so_far: int, note: str = "") -> dict[str, Any]:
        """Record an interim analysis and say whether to stop."""
        index = len(self.looks) + 1
        if index > self.total_looks:
            raise ValueError(
                f"look {index} exceeds the {self.total_looks} declared in "
                f"advance. Adding a look after seeing the data is the "
                f"practice alpha spending exists to prevent."
            )

        threshold = self.spending.threshold_at(index)
        stop = p_value <= threshold
        record = {
            "look": index,
            "n": n_so_far,
            "p_value": round(p_value, 6),
            "threshold": threshold,
            "cumulative_alpha": self.spending.spent_by(index),
            "stop": stop,
            "decision": (
                "stop: significant at this look's threshold" if stop
                else "continue: not significant yet"
            ),
            "note": note,
        }
        self.looks.append(record)
        return record

    def summary(self) -> dict[str, Any]:
        stopped = next((look for look in self.looks if look["stop"]), None)
        return {
            "alpha": self.alpha,
            "total_looks_declared": self.total_looks,
            "looks_taken": len(self.looks),
            "stopped_early": stopped is not None,
            "stopped_at_look": stopped["look"] if stopped else None,
            "budget_saved": (
                round(1 - stopped["look"] / self.total_looks, 4) if stopped else 0.0
            ),
            "schedule": self.spending.schedule(),
            "looks": self.looks,
        }
