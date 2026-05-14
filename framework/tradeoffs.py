"""
tradeoffs.py — Pareto frontier computation and trade-off analysis.

This module identifies the set of *Pareto-optimal* models — models for
which no other model is strictly better on every criterion.  It also
provides utilities for computing marginal trade-offs between criterion
pairs and for generating trade-off summary tables.

Pareto optimality
-----------------
A model *A* **dominates** model *B* if:

*   A is at least as good as B on *every* criterion, AND
*   A is strictly better than B on *at least one* criterion.

The **Pareto frontier** is the set of models that are not dominated by
any other model.

For cost criteria (speed, cost) "better" means *lower*; for benefit
criteria (correctness, quality, consistency) "better" means *higher*.
We handle this by working on *normalised* values (0–1, higher = better)
produced by :class:`framework.decision_matrix.Normaliser`.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from framework.decision_matrix import (
    CRITERIA,
    COST_CRITERIA,
    DecisionMatrixEngine,
    MetricAggregator,
    ModelMetrics,
    Normaliser,
)

try:
    from contracts import StatisticalSummary
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, Field as PField
    from datetime import datetime

    class StatisticalSummary(BaseModel):  # type: ignore[no-redef]
        metric_name: str = ""
        model_id: str = ""
        mean: float = 0.0
        std_dev: float = 0.0
        median: float = 0.0
        min_val: float = 0.0
        max_val: float = 0.0
        ci_lower_95: float = 0.0
        ci_upper_95: float = 0.0
        computed_at: datetime = PField(default_factory=datetime.utcnow)

logger = logging.getLogger(__name__)


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class ParetoPoint:
    """A single point on (or off) the Pareto frontier."""

    model_id: str
    values: dict[str, float]  # criterion → normalised value (higher = better)
    is_pareto: bool = False

    @property
    def criteria_vector(self) -> list[float]:
        """Return values as a list in canonical ``CRITERIA`` order."""
        return [self.values.get(c, 0.0) for c in CRITERIA]


@dataclass
class TradeoffPair:
    """Quantified trade-off between two models on two criteria."""

    model_a: str
    model_b: str
    criterion_x: str
    criterion_y: str
    delta_x: float  # positive means A is better on x
    delta_y: float  # positive means A is better on y
    marginal_rate: float | None = None  # |Δy / Δx| if Δx ≠ 0

    @property
    def summary(self) -> str:
        if self.marginal_rate is not None:
            return (
                f"Switching from {self.model_b} to {self.model_a}: "
                f"gain {self.delta_x:+.4f} on {self.criterion_x}, "
                f"lose {self.delta_y:+.4f} on {self.criterion_y} "
                f"(MRS = {self.marginal_rate:.4f})"
            )
        return (
            f"{self.model_a} vs {self.model_b}: "
            f"Δ{self.criterion_x}={self.delta_x:+.4f}, "
            f"Δ{self.criterion_y}={self.delta_y:+.4f}"
        )


@dataclass
class ParetoResult:
    """Complete result of a Pareto analysis."""

    points: list[ParetoPoint]
    frontier: list[ParetoPoint]
    dominated: list[ParetoPoint]
    tradeoffs: list[TradeoffPair]

    @property
    def frontier_ids(self) -> list[str]:
        return [p.model_id for p in self.frontier]

    @property
    def dominated_ids(self) -> list[str]:
        return [p.model_id for p in self.dominated]


# ======================================================================
# Core algorithms
# ======================================================================

def _dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    """Return True if *a* Pareto-dominates *b*.

    Both dicts map criterion names to *normalised* values where
    higher is always better.
    """
    at_least_as_good = True
    strictly_better = False
    for c in CRITERIA:
        va = a.get(c, 0.0)
        vb = b.get(c, 0.0)
        if va < vb:
            at_least_as_good = False
            break
        if va > vb:
            strictly_better = True
    return at_least_as_good and strictly_better


def compute_pareto_frontier(
    normalised: dict[str, dict[str, float]],
) -> list[ParetoPoint]:
    """Identify the Pareto frontier from normalised scores.

    Parameters
    ----------
    normalised : dict[str, dict[str, float]]
        ``{model_id: {criterion: normalised_value}}``.

    Returns
    -------
    list[ParetoPoint]
        All points with ``is_pareto`` flag set appropriately.
    """
    model_ids = sorted(normalised.keys())
    points: list[ParetoPoint] = []

    for mid in model_ids:
        vals = normalised[mid]
        dominated = False
        for other_id in model_ids:
            if other_id == mid:
                continue
            if _dominates(normalised[other_id], vals):
                dominated = True
                break
        points.append(
            ParetoPoint(
                model_id=mid,
                values=dict(vals),
                is_pareto=not dominated,
            )
        )
    return points


def compute_pairwise_tradeoffs(
    normalised: dict[str, dict[str, float]],
    criteria_pairs: list[tuple[str, str]] | None = None,
) -> list[TradeoffPair]:
    """Compute pairwise trade-offs between all model pairs.

    Parameters
    ----------
    normalised : dict[str, dict[str, float]]
        Normalised values (higher = better).
    criteria_pairs : list of (str, str) | None
        Which criterion pairs to analyse.  Defaults to all unique pairs.
    """
    if criteria_pairs is None:
        criteria_pairs = list(itertools.combinations(CRITERIA, 2))

    model_ids = sorted(normalised.keys())
    pairs: list[TradeoffPair] = []

    for mid_a, mid_b in itertools.combinations(model_ids, 2):
        for cx, cy in criteria_pairs:
            dx = normalised[mid_a].get(cx, 0.0) - normalised[mid_b].get(cx, 0.0)
            dy = normalised[mid_a].get(cy, 0.0) - normalised[mid_b].get(cy, 0.0)
            mrs: float | None = None
            if abs(dx) > 1e-9:
                mrs = abs(dy / dx)
            pairs.append(
                TradeoffPair(
                    model_a=mid_a,
                    model_b=mid_b,
                    criterion_x=cx,
                    criterion_y=cy,
                    delta_x=round(dx, 6),
                    delta_y=round(dy, 6),
                    marginal_rate=round(mrs, 6) if mrs is not None else None,
                )
            )
    return pairs


def compute_2d_frontier(
    normalised: dict[str, dict[str, float]],
    criterion_x: str,
    criterion_y: str,
) -> list[ParetoPoint]:
    """Compute 2-D Pareto frontier for a specific criterion pair.

    Useful for scatter-plot overlays in the dashboard.
    """
    model_ids = sorted(normalised.keys())
    points: list[ParetoPoint] = []

    for mid in model_ids:
        vx = normalised[mid].get(criterion_x, 0.0)
        vy = normalised[mid].get(criterion_y, 0.0)
        dominated = False
        for other_id in model_ids:
            if other_id == mid:
                continue
            ox = normalised[other_id].get(criterion_x, 0.0)
            oy = normalised[other_id].get(criterion_y, 0.0)
            if ox >= vx and oy >= vy and (ox > vx or oy > vy):
                dominated = True
                break
        points.append(
            ParetoPoint(
                model_id=mid,
                values={criterion_x: vx, criterion_y: vy},
                is_pareto=not dominated,
            )
        )
    return points


# ======================================================================
# TradeoffAnalyzer — high-level API
# ======================================================================

class TradeoffAnalyzer:
    """High-level API for Pareto frontier and trade-off analysis.

    Parameters
    ----------
    summaries : list[StatisticalSummary]
        Flat list of statistical summaries for all models.

    Usage
    -----
    >>> analyzer = TradeoffAnalyzer(summaries)
    >>> result = analyzer.analyze()
    >>> result.frontier_ids
    ['claude-3.5-sonnet', 'gemini-1.5-pro']
    """

    def __init__(self, summaries: Sequence[StatisticalSummary]) -> None:
        self._summaries = list(summaries)
        self._normalised: dict[str, dict[str, float]] = {}
        self._model_metrics: list[ModelMetrics] = []

    # ------------------------------------------------------------------
    def analyze(
        self,
        criteria_pairs: list[tuple[str, str]] | None = None,
    ) -> ParetoResult:
        """Run full Pareto + trade-off analysis.

        Returns
        -------
        ParetoResult
            Contains frontier, dominated, and pairwise trade-offs.
        """
        agg = MetricAggregator(self._summaries)
        self._model_metrics = agg.aggregate()
        self._normalised = Normaliser.normalise(self._model_metrics)

        points = compute_pareto_frontier(self._normalised)
        frontier = [p for p in points if p.is_pareto]
        dominated = [p for p in points if not p.is_pareto]
        tradeoffs = compute_pairwise_tradeoffs(
            self._normalised, criteria_pairs
        )

        return ParetoResult(
            points=points,
            frontier=frontier,
            dominated=dominated,
            tradeoffs=tradeoffs,
        )

    # ------------------------------------------------------------------
    def frontier_2d(
        self, criterion_x: str, criterion_y: str
    ) -> list[ParetoPoint]:
        """Return 2-D frontier for a single criterion pair."""
        if not self._normalised:
            agg = MetricAggregator(self._summaries)
            self._model_metrics = agg.aggregate()
            self._normalised = Normaliser.normalise(self._model_metrics)
        return compute_2d_frontier(
            self._normalised, criterion_x, criterion_y
        )

    # ------------------------------------------------------------------
    def tradeoff_table(
        self,
        criteria_pairs: list[tuple[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Return trade-offs as a list of dicts (for CSV/JSON export)."""
        if not self._normalised:
            agg = MetricAggregator(self._summaries)
            self._model_metrics = agg.aggregate()
            self._normalised = Normaliser.normalise(self._model_metrics)

        toffs = compute_pairwise_tradeoffs(self._normalised, criteria_pairs)
        return [
            {
                "model_a": t.model_a,
                "model_b": t.model_b,
                "criterion_x": t.criterion_x,
                "criterion_y": t.criterion_y,
                "delta_x": t.delta_x,
                "delta_y": t.delta_y,
                "marginal_rate": t.marginal_rate,
                "summary": t.summary,
            }
            for t in toffs
        ]

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialise the full analysis to a dict."""
        result = self.analyze()
        return {
            "frontier": [
                {"model_id": p.model_id, "values": p.values}
                for p in result.frontier
            ],
            "dominated": [
                {"model_id": p.model_id, "values": p.values}
                for p in result.dominated
            ],
            "tradeoffs": self.tradeoff_table(),
        }

    # ── properties ────────────────────────────────────────────────────

    @property
    def normalised_scores(self) -> dict[str, dict[str, float]]:
        return dict(self._normalised)

    @property
    def raw_metrics(self) -> list[ModelMetrics]:
        return list(self._model_metrics)
