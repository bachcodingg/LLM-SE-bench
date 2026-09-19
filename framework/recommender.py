"""
recommender.py — Constraint-based model recommendation engine.

Given a task type (codegen / bugfix / refactor) and a set of hard
constraints (max cost, min accuracy, max latency), the recommender:

1.  Filters out models that violate *any* hard constraint.
2.  Ranks survivors by weighted MCDA score (from :class:`DecisionMatrixEngine`).
3.  Computes a confidence value based on the margin between #1 and #2.
4.  Generates a human-readable rationale explaining the pick.
5.  Returns a :class:`contracts.Recommendation` object.

Constraint specification
------------------------
Constraints are plain dicts (also loadable from YAML profiles)::

    {
        "max_cost_usd": 0.05,
        "min_pass_rate": 0.70,
        "max_latency_ms": 15000,
    }

Any key not present is treated as "unconstrained".
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

try:
    from contracts import (
        DecisionMatrix,
        Recommendation,
        StatisticalSummary,
    )
except ImportError:  # pragma: no cover
    from pydantic import BaseModel
    from pydantic import Field as PField

    class DecisionMatrix(BaseModel):  # type: ignore[no-redef]
        matrix_id: str = ""
        model_id: str = ""
        scores: list = PField(default_factory=list)
        weighted_total: float = 0.0
        rank: int = 0

    class Recommendation(BaseModel):  # type: ignore[no-redef]
        recommendation_id: str = ""
        use_case: str = "general"
        recommended_model: str = ""
        runner_up_model: str = ""
        confidence: float = 0.0
        rationale: str = ""
        constraints_applied: list = PField(default_factory=list)
        matrix_ids: list = PField(default_factory=list)
        warnings: list = PField(default_factory=list)
        created_at: datetime = PField(default_factory=datetime.utcnow)

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

from framework.decision_matrix import (
    DecisionMatrixEngine,
)

logger = logging.getLogger(__name__)

# ── Constraint keys and their relationship to metrics ─────────────────
_CONSTRAINT_METRIC_MAP: dict[str, tuple[str, str]] = {
    # constraint_key → (metric_name, comparison)
    "max_cost_usd": ("cost_usd", "le"),
    "min_pass_rate": ("pass_rate", "ge"),
    "max_latency_ms": ("latency_ms", "le"),
    "min_quality": ("maintainability_index", "ge"),
    "max_std_dev": ("pass_rate", "le_std"),
}


def _uid() -> str:
    return uuid.uuid4().hex[:12]


# ======================================================================
# ConstraintChecker
# ======================================================================

@dataclass
class ConstraintViolation:
    """Details of a single constraint violated by a model."""

    model_id: str
    constraint: str
    limit: float
    actual: float
    metric_name: str

    def __str__(self) -> str:
        return (
            f"{self.model_id}: {self.constraint}={self.limit} "
            f"violated (actual={self.actual:.4f}, metric={self.metric_name})"
        )


class ConstraintChecker:
    """Evaluate hard constraints against statistical summaries.

    Parameters
    ----------
    summaries : list[StatisticalSummary]
        Flat list of summaries for all models.
    constraints : dict[str, float]
        Constraint specification (see module docstring).
    """

    def __init__(
        self,
        summaries: Sequence[StatisticalSummary],
        constraints: dict[str, float],
    ) -> None:
        self._summaries = list(summaries)
        self._constraints = dict(constraints)
        # Index: (model_id, metric_name) → summary
        self._index: dict[tuple[str, str], StatisticalSummary] = {}
        for s in self._summaries:
            self._index[(s.model_id, s.metric_name)] = s

    # ------------------------------------------------------------------
    @property
    def model_ids(self) -> list[str]:
        return sorted({s.model_id for s in self._summaries})

    # ------------------------------------------------------------------
    def check(self, model_id: str) -> list[ConstraintViolation]:
        """Return list of violations for *model_id* (empty = feasible)."""
        violations: list[ConstraintViolation] = []
        for ckey, limit in self._constraints.items():
            entry = _CONSTRAINT_METRIC_MAP.get(ckey)
            if entry is None:
                continue
            metric_name, cmp = entry

            # Handle special std_dev constraint
            if cmp == "le_std":
                summary = self._index.get((model_id, metric_name))
                actual = summary.std_dev if summary else 0.0
            else:
                summary = self._index.get((model_id, metric_name))
                actual = summary.mean if summary else 0.0

            ok = True
            if cmp == "le" and actual > limit:
                ok = False
            elif cmp == "ge" and actual < limit:
                ok = False
            elif cmp == "le_std" and actual > limit:
                ok = False

            if not ok:
                violations.append(
                    ConstraintViolation(
                        model_id=model_id,
                        constraint=ckey,
                        limit=limit,
                        actual=actual,
                        metric_name=metric_name,
                    )
                )
        return violations

    # ------------------------------------------------------------------
    def feasible_models(self) -> list[str]:
        """Return model IDs that satisfy all constraints."""
        return [
            mid for mid in self.model_ids if not self.check(mid)
        ]

    # ------------------------------------------------------------------
    def infeasible_models(self) -> dict[str, list[ConstraintViolation]]:
        """Return ``{model_id: [violations]}`` for models that fail."""
        out: dict[str, list[ConstraintViolation]] = {}
        for mid in self.model_ids:
            vs = self.check(mid)
            if vs:
                out[mid] = vs
        return out


# ======================================================================
# RationaleGenerator
# ======================================================================

class RationaleGenerator:
    """Build human-readable rationale text for a recommendation."""

    @staticmethod
    def generate(
        top: DecisionMatrix,
        runner_up: DecisionMatrix | None,
        use_case: str,
        constraints_applied: list[str],
        infeasible: dict[str, list[ConstraintViolation]],
        confidence: float,
    ) -> str:
        parts: list[str] = []

        parts.append(
            f"For the '{use_case}' use-case, {top.model_id} is recommended "
            f"with a weighted MCDA score of {top.weighted_total:.4f} "
            f"(rank #{top.rank})."
        )

        if runner_up:
            gap = top.weighted_total - runner_up.weighted_total
            parts.append(
                f"The runner-up is {runner_up.model_id} "
                f"(score {runner_up.weighted_total:.4f}, "
                f"gap {gap:.4f})."
            )

        if constraints_applied:
            parts.append(
                "Hard constraints applied: "
                + ", ".join(constraints_applied)
                + "."
            )

        if infeasible:
            excluded = ", ".join(
                f"{mid} ({len(vs)} violation{'s' if len(vs) > 1 else ''})"
                for mid, vs in infeasible.items()
            )
            parts.append(f"Excluded by constraints: {excluded}.")

        parts.append(f"Confidence: {confidence:.0%}.")

        return " ".join(parts)


# ======================================================================
# ModelRecommender — the main public class
# ======================================================================

class ModelRecommender:
    """Constraint-based model recommendation engine.

    Parameters
    ----------
    summaries : list[StatisticalSummary]
        All statistical summaries from C4.
    weights : dict[str, float] | None
        MCDA criterion weights.  If *profile_name* is given it takes
        precedence.
    profile_name : str | None
        Load weights from a YAML profile.
    constraints : dict[str, float] | None
        Hard constraints to apply before ranking.  If *profile_name* is
        given and no explicit constraints are passed, the profile's
        constraints are used.
    profiles_dir : str | None
        Custom path for profile YAML files.
    """

    def __init__(
        self,
        summaries: Sequence[StatisticalSummary],
        *,
        weights: dict[str, float] | None = None,
        profile_name: str | None = None,
        constraints: dict[str, float] | None = None,
        profiles_dir: str | None = None,
    ) -> None:
        self._summaries = list(summaries)
        self._profile_name = profile_name
        self._profiles_dir = profiles_dir

        # Resolve weights and constraints from profile if needed
        self._engine = DecisionMatrixEngine(
            summaries,
            weights=weights,
            profile_name=profile_name,
            profiles_dir=profiles_dir,
        )

        if constraints is not None:
            self._constraints = dict(constraints)
        elif profile_name:
            from framework.decision_matrix import ProfileLoader

            loader = ProfileLoader(profiles_dir)
            self._constraints = loader.get_constraints(profile_name)
        else:
            self._constraints = {}

    # ------------------------------------------------------------------
    @property
    def constraints(self) -> dict[str, float]:
        return dict(self._constraints)

    # ------------------------------------------------------------------
    def recommend(
        self,
        use_case: str = "general",
        task_type: str | None = None,
    ) -> Recommendation:
        """Produce a :class:`Recommendation`.

        Parameters
        ----------
        use_case : str
            Label for the scenario (e.g. ``"devops"``, ``"audit"``).
        task_type : str | None
            If provided (``"codegen"``/``"bugfix"``/``"refactor"``), it
            is recorded in the recommendation metadata but does not
            currently alter weights.
        """
        # Step 1: build MCDA matrices
        matrices = self._engine.build()
        if not matrices:
            return self._empty_recommendation(use_case)

        # Step 2: apply constraints
        checker = ConstraintChecker(self._summaries, self._constraints)
        feasible_ids = set(checker.feasible_models())
        infeasible = checker.infeasible_models()

        feasible_matrices = [
            m for m in matrices if m.model_id in feasible_ids
        ]

        constraints_applied = [
            f"{k}={v}" for k, v in self._constraints.items()
        ]

        # If no models survive constraints, fall back to best overall
        warnings: list[str] = []
        if not feasible_matrices:
            warnings.append(
                "No model satisfies all constraints; "
                "returning best overall model."
            )
            feasible_matrices = matrices

        # Step 3: pick top and runner-up
        top = feasible_matrices[0]
        runner_up = feasible_matrices[1] if len(feasible_matrices) > 1 else None

        # Step 4: compute confidence
        confidence = self._compute_confidence(top, runner_up)

        # Step 5: generate rationale
        rationale = RationaleGenerator.generate(
            top=top,
            runner_up=runner_up,
            use_case=use_case,
            constraints_applied=constraints_applied,
            infeasible=infeasible,
            confidence=confidence,
        )

        return Recommendation(
            recommendation_id=f"rec-{_uid()}",
            use_case=use_case,
            recommended_model=top.model_id,
            runner_up_model=runner_up.model_id if runner_up else "",
            confidence=confidence,
            rationale=rationale,
            constraints_applied=constraints_applied,
            matrix_ids=[m.matrix_id for m in feasible_matrices],
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    def recommend_all_profiles(
        self,
        use_case_prefix: str = "",
        profiles: Sequence[str] | None = None,
    ) -> list[Recommendation]:
        """Run recommendations for multiple profiles.

        Parameters
        ----------
        use_case_prefix : str
            Prepended to the profile name for the ``use_case`` field.
        profiles : list[str] | None
            Profile names to iterate.  Defaults to all available.
        """
        from framework.decision_matrix import ProfileLoader

        loader = ProfileLoader(self._profiles_dir)
        names = list(profiles) if profiles else loader.list_profiles()

        recs: list[Recommendation] = []
        for pname in names:
            spec = loader.get(pname)
            sub = ModelRecommender(
                self._summaries,
                weights=spec.weights,
                constraints=spec.constraints,
                profiles_dir=self._profiles_dir,
            )
            uc = f"{use_case_prefix}{pname}" if use_case_prefix else pname
            recs.append(sub.recommend(use_case=uc))
        return recs

    # ── private ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(
        top: DecisionMatrix,
        runner_up: DecisionMatrix | None,
    ) -> float:
        """Confidence ∈ [0, 1] based on margin between top two.

        * No runner-up → 1.0 (only option).
        * Margin ≥ 0.20 → 0.95.
        * Margin ~ 0 → 0.50.
        * Linear interpolation in between.
        """
        if runner_up is None:
            return 1.0
        gap = top.weighted_total - runner_up.weighted_total
        if gap >= 0.20:
            return 0.95
        # Linear from 0.50 (gap=0) to 0.95 (gap=0.20)
        return round(0.50 + gap * (0.45 / 0.20), 4)

    @staticmethod
    def _empty_recommendation(use_case: str) -> Recommendation:
        return Recommendation(
            recommendation_id=f"rec-{_uid()}",
            use_case=use_case,
            recommended_model="none",
            runner_up_model="",
            confidence=0.0,
            rationale="No models available for evaluation.",
            warnings=["No statistical summaries provided."],
        )

    # ── serialisation ─────────────────────────────────────────────────

    def to_dict(self, rec: Recommendation) -> dict[str, Any]:
        """Serialise a recommendation to a plain dict."""
        return {
            "recommendation_id": rec.recommendation_id,
            "use_case": rec.use_case,
            "recommended_model": rec.recommended_model,
            "runner_up_model": rec.runner_up_model,
            "confidence": rec.confidence,
            "rationale": rec.rationale,
            "constraints_applied": rec.constraints_applied,
            "warnings": rec.warnings,
        }
