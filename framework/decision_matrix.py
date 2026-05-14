"""
decision_matrix.py — Weighted Multi-Criteria Decision Analysis (MCDA) engine.

Reads ``analysis/statistical_summary.json`` (a list of
:class:`contracts.StatisticalSummary` objects), normalises each criterion
to [0, 1], applies profile weights, and produces ranked
:class:`contracts.DecisionMatrix` entries per model.

Normalisation
-------------
*   **Benefit criteria** (correctness, quality, consistency): higher is
    better → ``(val - min) / (max - min)``  (min-max scaling).
*   **Cost criteria** (cost, speed/latency): lower is better →
    ``1 - (val - min) / (max - min)``  (inverted min-max).

When all values are identical the normalised score is set to 1.0 to
avoid division-by-zero and to reward determinism.

Profile loading
---------------
:class:`ProfileLoader` reads YAML files from ``framework/profiles/`` and
returns weight dictionaries keyed by criterion name.  Three built-in
profiles ship with the project:

*   **DevOps** — Speed 35 %, Cost 25 %, Correctness 25 %, Quality 10 %,
    Consistency 5 %.
*   **Audit** — Quality 30 %, Correctness 30 %, Consistency 20 %,
    Cost 15 %, Speed 5 %.
*   **Budget** — Cost 40 %, Correctness 30 %, Speed 15 %, Quality 10 %,
    Consistency 5 %.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

# ---------------------------------------------------------------------------
# Contracts — imported when available, otherwise lightweight stand-ins
# ---------------------------------------------------------------------------
try:
    from contracts import (
        CriterionScore,
        DecisionMatrix,
        StatisticalSummary,
    )
except ImportError:  # pragma: no cover – standalone testing
    from pydantic import BaseModel, Field as PField

    class StatisticalSummary(BaseModel):  # type: ignore[no-redef]
        metric_name: str = ""
        model_id: str = ""
        n: int = 0
        mean: float = 0.0
        std_dev: float = 0.0
        median: float = 0.0
        min_val: float = 0.0
        max_val: float = 0.0
        ci_lower_95: float = 0.0
        ci_upper_95: float = 0.0
        computed_at: datetime = PField(default_factory=datetime.utcnow)

    class CriterionScore(BaseModel):  # type: ignore[no-redef]
        criterion: str = ""
        raw_value: float = 0.0
        normalised_value: float = 0.0
        weight: float = 1.0

    class DecisionMatrix(BaseModel):  # type: ignore[no-redef]
        matrix_id: str = ""
        model_id: str = ""
        scores: list = PField(default_factory=list)
        weighted_total: float = 0.0
        rank: int = 0
        computed_at: datetime = PField(default_factory=datetime.utcnow)

        def recalculate(self) -> float:
            self.weighted_total = sum(
                s.weight * s.normalised_value for s in self.scores
            )
            return self.weighted_total


logger = logging.getLogger(__name__)

# ── Canonical criterion names ──────────────────────────────────────────
CRITERIA: list[str] = [
    "correctness",
    "quality",
    "speed",
    "cost",
    "consistency",
]

# Mapping from criterion → metric_name expected in StatisticalSummary
CRITERION_METRIC_MAP: dict[str, str] = {
    "correctness": "pass_rate",
    "quality": "maintainability_index",
    "speed": "latency_ms",
    "cost": "cost_usd",
    "consistency": "consistency_score",
}

# Which criteria are *cost* criteria (lower-is-better)?
COST_CRITERIA: set[str] = {"speed", "cost"}

# Default equal-weight profile
DEFAULT_WEIGHTS: dict[str, float] = {c: 0.20 for c in CRITERIA}

# ── helpers ────────────────────────────────────────────────────────────

_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"


def _uid() -> str:
    return uuid.uuid4().hex[:12]


# ======================================================================
# ProfileLoader
# ======================================================================

@dataclass
class ProfileSpec:
    """Parsed content of a single YAML profile."""

    name: str
    description: str
    weights: dict[str, float]
    constraints: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Raise ``ValueError`` if weights don't sum to ≈1 or are negative."""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Profile '{self.name}': weights sum to {total:.4f}, expected 1.0"
            )
        for k, v in self.weights.items():
            if v < 0:
                raise ValueError(
                    f"Profile '{self.name}': negative weight {v} for '{k}'"
                )
            if k not in CRITERIA:
                raise ValueError(
                    f"Profile '{self.name}': unknown criterion '{k}'"
                )


class ProfileLoader:
    """Load and validate YAML weighting profiles.

    Parameters
    ----------
    profiles_dir : Path | str | None
        Directory containing ``*.yaml`` files.  Defaults to the
        ``framework/profiles/`` directory shipped with this package.

    Usage
    -----
    >>> loader = ProfileLoader()
    >>> spec = loader.get("devops")
    >>> spec.weights
    {'speed': 0.35, 'cost': 0.25, ...}
    """

    def __init__(self, profiles_dir: Path | str | None = None) -> None:
        self._dir = Path(profiles_dir) if profiles_dir else _PROFILES_DIR
        self._cache: dict[str, ProfileSpec] = {}

    # ------------------------------------------------------------------
    def list_profiles(self) -> list[str]:
        """Return sorted names of available YAML profiles."""
        return sorted(
            p.stem for p in self._dir.glob("*.yaml") if p.is_file()
        )

    # ------------------------------------------------------------------
    def get(self, name: str) -> ProfileSpec:
        """Return the parsed :class:`ProfileSpec` for *name*.

        *name* is matched case-insensitively against file stems in the
        profiles directory (e.g. ``"DevOps"`` → ``devops.yaml``).

        Raises
        ------
        FileNotFoundError
            If no matching YAML file is found.
        """
        key = name.lower()
        if key in self._cache:
            return self._cache[key]

        path = self._dir / f"{key}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"Profile '{name}' not found in {self._dir}"
            )

        with open(path) as fh:
            raw: dict = yaml.safe_load(fh)

        spec = ProfileSpec(
            name=raw.get("name", key),
            description=raw.get("description", ""),
            weights=raw.get("weights", {}),
            constraints=raw.get("constraints", {}),
            tags=raw.get("tags", []),
        )
        spec.validate()
        self._cache[key] = spec
        return spec

    # ------------------------------------------------------------------
    def get_weights(self, name: str) -> dict[str, float]:
        """Convenience: return just the weight dict for *name*."""
        return dict(self.get(name).weights)

    # ------------------------------------------------------------------
    def get_constraints(self, name: str) -> dict[str, float]:
        """Convenience: return just the constraint dict for *name*."""
        return dict(self.get(name).constraints)


# ======================================================================
# MetricAggregator — extract per-model raw values from summaries
# ======================================================================

@dataclass
class ModelMetrics:
    """Raw metric values for a single model, keyed by criterion."""

    model_id: str
    values: dict[str, float] = field(default_factory=dict)


class MetricAggregator:
    """Aggregate a flat list of :class:`StatisticalSummary` into per-model metrics.

    Each summary has a ``metric_name`` (e.g. ``"pass_rate"``) and a
    ``model_id``.  We look up the canonical criterion that maps to that
    metric name and store the summary's ``mean`` value.

    For ``consistency_score`` we derive the value as ``1 − coeff_of_variation``
    (capped to [0, 1]) when not provided directly, using
    ``std_dev / mean`` from any available metric (preferring pass_rate).
    """

    def __init__(self, summaries: Sequence[StatisticalSummary]) -> None:
        self._summaries = list(summaries)
        self._reverse_map: dict[str, str] = {
            v: k for k, v in CRITERION_METRIC_MAP.items()
        }

    # ------------------------------------------------------------------
    def aggregate(self) -> list[ModelMetrics]:
        """Return one :class:`ModelMetrics` per model found in summaries."""
        models: dict[str, ModelMetrics] = {}
        # Accumulate all non-zero means per (model, criterion) for averaging
        criterion_values: dict[str, dict[str, list[float]]] = {}
        per_model_summaries: dict[str, list[StatisticalSummary]] = {}

        for s in self._summaries:
            if s.model_id not in models:
                models[s.model_id] = ModelMetrics(model_id=s.model_id)
                criterion_values[s.model_id] = {}
            if s.model_id not in per_model_summaries:
                per_model_summaries[s.model_id] = []
            per_model_summaries[s.model_id].append(s)

            criterion = self._reverse_map.get(s.metric_name)
            if criterion is not None and s.n > 0:
                criterion_values[s.model_id].setdefault(criterion, []).append(s.mean)

        # Use mean of non-zero-n summaries per criterion
        for mid, mm in models.items():
            for criterion, vals in criterion_values.get(mid, {}).items():
                if vals:
                    mm.values[criterion] = sum(vals) / len(vals)

        # Derive consistency where missing
        for mid, mm in models.items():
            if "consistency" not in mm.values:
                mm.values["consistency"] = self._derive_consistency(
                    per_model_summaries.get(mid, [])
                )

        return list(models.values())

    # ------------------------------------------------------------------
    @staticmethod
    def _derive_consistency(
        summaries: list[StatisticalSummary],
    ) -> float:
        """Derive a 0-1 consistency score from coefficient of variation."""
        # Prefer pass_rate; fall back to first numeric metric
        target: StatisticalSummary | None = None
        for s in summaries:
            if s.metric_name == "pass_rate":
                target = s
                break
        if target is None and summaries:
            target = summaries[0]
        if target is None or target.mean == 0:
            return 0.5  # no data → neutral
        cv = target.std_dev / abs(target.mean)
        return max(0.0, min(1.0, 1.0 - cv))


# ======================================================================
# Normaliser — min-max scaling with benefit/cost awareness
# ======================================================================

class Normaliser:
    """Min-max normalise raw values across models per criterion."""

    @staticmethod
    def normalise(
        model_metrics: list[ModelMetrics],
    ) -> dict[str, dict[str, float]]:
        """Return ``{model_id: {criterion: normalised_value}}``."""
        if not model_metrics:
            return {}

        # Collect all criterion values across models
        all_criteria = set()
        for mm in model_metrics:
            all_criteria.update(mm.values.keys())

        result: dict[str, dict[str, float]] = {
            mm.model_id: {} for mm in model_metrics
        }

        for criterion in all_criteria:
            vals = [
                mm.values.get(criterion, 0.0) for mm in model_metrics
            ]
            lo, hi = min(vals), max(vals)
            span = hi - lo

            is_cost = criterion in COST_CRITERIA

            for mm in model_metrics:
                raw = mm.values.get(criterion, 0.0)
                if span == 0:
                    norm = 1.0  # all equal → max score
                elif is_cost:
                    norm = 1.0 - (raw - lo) / span
                else:
                    norm = (raw - lo) / span

                result[mm.model_id][criterion] = round(
                    max(0.0, min(1.0, norm)), 6
                )

        return result


# ======================================================================
# DecisionMatrixEngine — the main entry point
# ======================================================================

class DecisionMatrixEngine:
    """Build and rank :class:`DecisionMatrix` objects from summaries.

    Parameters
    ----------
    summaries : list[StatisticalSummary]
        Flat list of statistical summaries (all models, all metrics).
    weights : dict[str, float] | None
        Criterion weights.  Defaults to equal weights (0.20 each).
    profile_name : str | None
        If given, loads weights from the named YAML profile
        (overrides *weights*).
    profiles_dir : Path | str | None
        Custom profiles directory; passed to :class:`ProfileLoader`.
    """

    def __init__(
        self,
        summaries: Sequence[StatisticalSummary],
        weights: dict[str, float] | None = None,
        profile_name: str | None = None,
        profiles_dir: Path | str | None = None,
    ) -> None:
        self._summaries = list(summaries)
        self._loader = ProfileLoader(profiles_dir)

        if profile_name:
            self._weights = self._loader.get_weights(profile_name)
            self._profile_name = profile_name
        else:
            self._weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
            self._profile_name = "custom" if weights else "default"

        self._model_metrics: list[ModelMetrics] = []
        self._normalised: dict[str, dict[str, float]] = {}
        self._matrices: list[DecisionMatrix] = []

    # ── properties ────────────────────────────────────────────────────

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    @property
    def profile_name(self) -> str:
        return self._profile_name

    @property
    def model_metrics(self) -> list[ModelMetrics]:
        return list(self._model_metrics)

    @property
    def normalised_scores(self) -> dict[str, dict[str, float]]:
        return dict(self._normalised)

    @property
    def matrices(self) -> list[DecisionMatrix]:
        return list(self._matrices)

    # ── core pipeline ─────────────────────────────────────────────────

    def build(self) -> list[DecisionMatrix]:
        """Run the full MCDA pipeline and return ranked matrices.

        Steps
        -----
        1.  Aggregate summaries → per-model raw metrics.
        2.  Normalise to [0, 1] (benefit / cost aware).
        3.  Build :class:`DecisionMatrix` per model with weighted scores.
        4.  Rank by ``weighted_total`` (descending).
        """
        agg = MetricAggregator(self._summaries)
        self._model_metrics = agg.aggregate()
        self._normalised = Normaliser.normalise(self._model_metrics)
        self._matrices = self._build_matrices()
        self._rank()
        return list(self._matrices)

    # ------------------------------------------------------------------
    def rebuild_with_weights(
        self, weights: dict[str, float]
    ) -> list[DecisionMatrix]:
        """Re-score existing normalised data with new weights.

        Useful for the dashboard's interactive weight sliders: avoids
        re-reading summaries and re-normalising.
        """
        self._weights = dict(weights)
        self._profile_name = "custom"
        if not self._normalised:
            return self.build()
        self._matrices = self._build_matrices()
        self._rank()
        return list(self._matrices)

    # ── private helpers ───────────────────────────────────────────────

    def _build_matrices(self) -> list[DecisionMatrix]:
        matrices: list[DecisionMatrix] = []
        for model_id, norms in self._normalised.items():
            scores: list[CriterionScore] = []
            for criterion in CRITERIA:
                nv = norms.get(criterion, 0.0)
                w = self._weights.get(criterion, 0.0)
                raw = 0.0
                for mm in self._model_metrics:
                    if mm.model_id == model_id:
                        raw = mm.values.get(criterion, 0.0)
                        break
                scores.append(
                    CriterionScore(
                        criterion=criterion,
                        raw_value=raw,
                        normalised_value=nv,
                        weight=w,
                    )
                )

            dm = DecisionMatrix(
                matrix_id=f"dm-{_uid()}",
                model_id=model_id,
                scores=scores,
                weighted_total=0.0,
                rank=0,
            )
            dm.recalculate()
            matrices.append(dm)
        return matrices

    def _rank(self) -> None:
        self._matrices.sort(key=lambda m: m.weighted_total, reverse=True)
        for i, m in enumerate(self._matrices, start=1):
            m.rank = i

    # ── I/O convenience ───────────────────────────────────────────────

    @classmethod
    def from_json_file(
        cls,
        path: str | Path,
        *,
        weights: dict[str, float] | None = None,
        profile_name: str | None = None,
        profiles_dir: Path | str | None = None,
    ) -> "DecisionMatrixEngine":
        """Construct an engine from a ``statistical_summary.json`` file."""
        path = Path(path)
        with open(path) as fh:
            raw = json.load(fh)

        summaries = [StatisticalSummary(**item) for item in raw]
        return cls(
            summaries,
            weights=weights,
            profile_name=profile_name,
            profiles_dir=profiles_dir,
        )

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialise the ranked matrices to plain dicts."""
        out: list[dict] = []
        for dm in self._matrices:
            out.append(
                {
                    "matrix_id": dm.matrix_id,
                    "model_id": dm.model_id,
                    "rank": dm.rank,
                    "weighted_total": round(dm.weighted_total, 6),
                    "profile": self._profile_name,
                    "scores": [
                        {
                            "criterion": s.criterion,
                            "raw_value": round(s.raw_value, 6),
                            "normalised_value": round(s.normalised_value, 6),
                            "weight": round(s.weight, 4),
                        }
                        for s in dm.scores
                    ],
                }
            )
        return out

    # ------------------------------------------------------------------
    def sensitivity_analysis(
        self,
        criterion: str,
        steps: int = 11,
    ) -> list[dict[str, Any]]:
        """Vary one criterion weight from 0→1 and record rank changes.

        Other weights are proportionally rescaled so the total remains 1.

        Returns a list of dicts with keys ``weight_value``,
        ``rankings`` (model_id → rank), ``totals`` (model_id → score).
        """
        if criterion not in CRITERIA:
            raise ValueError(f"Unknown criterion: {criterion}")
        if not self._normalised:
            self.build()

        results: list[dict[str, Any]] = []
        others = [c for c in CRITERIA if c != criterion]
        base_other_sum = sum(self._weights.get(c, 0.0) for c in others)

        for i in range(steps):
            w = i / (steps - 1)
            remaining = 1.0 - w
            trial_weights: dict[str, float] = {criterion: w}
            for c in others:
                if base_other_sum > 0:
                    trial_weights[c] = (
                        self._weights.get(c, 0.0) / base_other_sum * remaining
                    )
                else:
                    trial_weights[c] = remaining / len(others)

            # Score
            totals: dict[str, float] = {}
            for mid, norms in self._normalised.items():
                score = sum(
                    trial_weights.get(cr, 0.0) * norms.get(cr, 0.0)
                    for cr in CRITERIA
                )
                totals[mid] = round(score, 6)

            ranked = sorted(totals, key=totals.get, reverse=True)  # type: ignore[arg-type]
            rankings = {mid: rank + 1 for rank, mid in enumerate(ranked)}

            results.append(
                {
                    "weight_value": round(w, 4),
                    "rankings": rankings,
                    "totals": totals,
                }
            )
        return results
