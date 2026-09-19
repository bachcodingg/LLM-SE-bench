"""
stats/cost_model.py — Cost modelling engine for llm-se-bench.

Computes:
    * Per-task and per-model aggregate costs.
    * Cost-effectiveness ratios (cost per correct solution, per quality point).
    * Scaling projections to realistic codebase sizes.
    * Token-usage analysis (input vs. output distribution).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── result containers ────────────────────────────────────────────────

@dataclass
class ModelCostProfile:
    """Aggregated cost profile for a single model."""

    model_id: str
    total_cost_usd: float = 0.0
    mean_cost_per_task: float = 0.0
    median_cost_per_task: float = 0.0
    std_cost_per_task: float = 0.0
    min_cost: float = 0.0
    max_cost: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    mean_prompt_tokens: float = 0.0
    mean_completion_tokens: float = 0.0
    n_tasks: int = 0
    cost_per_correct: float = 0.0
    cost_per_quality_point: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 6) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


@dataclass
class ScalingProjection:
    """Cost projection for a hypothetical codebase size."""

    model_id: str
    n_files: int
    projected_cost_usd: float = 0.0
    projected_tokens: int = 0
    projected_time_hours: float = 0.0
    cost_per_file: float = 0.0
    confidence_lower: float = 0.0
    confidence_upper: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 6) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


@dataclass
class CostEffectivenessRatio:
    """Cost-effectiveness ratio between two models."""

    model_a: str
    model_b: str
    metric_name: str
    ratio: float = 0.0  # cost_a / cost_b
    delta_cost: float = 0.0
    delta_metric: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return {k: round(v, 6) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


# ── engine ───────────────────────────────────────────────────────────

class CostModel:
    """Cost analysis engine.

    Parameters
    ----------
    scaling_sizes : list[int]
        Target codebase sizes for scaling projections
        (default: 100, 500, 1000, 5000, 10000 files).
    """

    DEFAULT_SCALING_SIZES = [100, 500, 1_000, 5_000, 10_000]

    def __init__(
        self,
        scaling_sizes: list[int] | None = None,
    ) -> None:
        self.scaling_sizes = scaling_sizes or self.DEFAULT_SCALING_SIZES

    # ── per-model cost profiles ──────────────────────────────────────

    def compute_profiles(
        self,
        cost_df: pd.DataFrame,
        results_df: pd.DataFrame | None = None,
        quality_df: pd.DataFrame | None = None,
    ) -> list[ModelCostProfile]:
        """Compute cost profiles for each model.

        Parameters
        ----------
        cost_df : pd.DataFrame
            Must have ``model_id``, ``total_cost_usd``, ``prompt_tokens``,
            ``completion_tokens``.
        results_df : pd.DataFrame, optional
            Evaluation results for computing cost-per-correct.
        quality_df : pd.DataFrame, optional
            Quality metrics for computing cost-per-quality-point.
        """
        profiles: list[ModelCostProfile] = []

        for model_id, grp in cost_df.groupby("model_id"):
            costs = grp["total_cost_usd"].values
            prompt_tok = grp["prompt_tokens"].values
            compl_tok = grp["completion_tokens"].values

            n = len(costs)
            profile = ModelCostProfile(
                model_id=str(model_id),
                total_cost_usd=float(np.sum(costs)),
                mean_cost_per_task=float(np.mean(costs)) if n > 0 else 0.0,
                median_cost_per_task=float(np.median(costs)) if n > 0 else 0.0,
                std_cost_per_task=float(np.std(costs, ddof=1)) if n > 1 else 0.0,
                min_cost=float(np.min(costs)) if n > 0 else 0.0,
                max_cost=float(np.max(costs)) if n > 0 else 0.0,
                total_prompt_tokens=int(np.sum(prompt_tok)),
                total_completion_tokens=int(np.sum(compl_tok)),
                mean_prompt_tokens=float(np.mean(prompt_tok)) if n > 0 else 0.0,
                mean_completion_tokens=float(np.mean(compl_tok)) if n > 0 else 0.0,
                n_tasks=n,
            )

            # Cost per correct solution
            if results_df is not None and "is_pass" in results_df.columns:
                model_results = results_df[results_df["model_id"] == model_id]
                n_correct = model_results["is_pass"].sum()
                if n_correct > 0:
                    profile.cost_per_correct = profile.total_cost_usd / n_correct

            # Cost per quality point
            if quality_df is not None and "maintainability_index" in quality_df.columns:
                model_quality = quality_df[quality_df["model_id"] == model_id]
                avg_quality = model_quality["maintainability_index"].mean()
                if avg_quality > 0:
                    profile.cost_per_quality_point = (
                        profile.total_cost_usd / avg_quality
                    )

            profiles.append(profile)

        return profiles

    # ── cost per dataset × model ─────────────────────────────────────

    def cost_by_dataset(
        self,
        cost_df: pd.DataFrame,
        results_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute cost breakdown by (dataset, model).

        Requires ``results_df`` to have ``response_id`` or a join key to
        link costs to datasets.

        Returns
        -------
        pd.DataFrame
            Columns: dataset, model_id, total_cost, mean_cost, n_tasks,
            n_correct, cost_per_correct.
        """
        if "response_id" in cost_df.columns and "response_id" in results_df.columns:
            merged = results_df.merge(
                cost_df[["response_id", "total_cost_usd", "prompt_tokens",
                          "completion_tokens"]],
                on="response_id",
                how="left",
            )
        elif "model_id" in cost_df.columns and "model_id" in results_df.columns:
            # Fallback: aggregate by model only
            merged = results_df.copy()
            agg_cost = (
                cost_df.groupby("model_id")["total_cost_usd"]
                .mean()
                .reset_index()
                .rename(columns={"total_cost_usd": "est_cost"})
            )
            merged = merged.merge(agg_cost, on="model_id", how="left")
            merged["total_cost_usd"] = merged.get("est_cost", 0.0)
        else:
            logger.warning("Cannot link costs to results — no common key")
            return pd.DataFrame()

        group_cols = ["dataset", "model_id"]
        if not all(c in merged.columns for c in group_cols):
            return pd.DataFrame()

        agg = merged.groupby(group_cols).agg(
            total_cost=("total_cost_usd", "sum"),
            mean_cost=("total_cost_usd", "mean"),
            n_tasks=("total_cost_usd", "count"),
            n_correct=("is_pass", "sum") if "is_pass" in merged.columns else ("total_cost_usd", "count"),
        ).reset_index()

        if "is_pass" in merged.columns:
            agg["cost_per_correct"] = np.where(
                agg["n_correct"] > 0,
                agg["total_cost"] / agg["n_correct"],
                np.nan,
            )
        return agg

    # ── scaling projections ──────────────────────────────────────────

    def project_scaling(
        self,
        cost_df: pd.DataFrame,
        sizes: list[int] | None = None,
    ) -> list[ScalingProjection]:
        """Extrapolate costs to larger codebase sizes.

        Assumes per-file cost is roughly constant (linear scaling).

        Parameters
        ----------
        cost_df : pd.DataFrame
        sizes : list[int], optional

        Returns
        -------
        list[ScalingProjection]
        """
        sizes = sizes or self.scaling_sizes
        projections: list[ScalingProjection] = []

        for model_id, grp in cost_df.groupby("model_id"):
            costs = grp["total_cost_usd"].values
            tokens = (grp["prompt_tokens"] + grp["completion_tokens"]).values
            n = len(costs)
            if n == 0:
                continue

            mean_cost = float(np.mean(costs))
            std_cost = float(np.std(costs, ddof=1)) if n > 1 else 0.0
            mean_tokens = float(np.mean(tokens))

            for size in sizes:
                projected_cost = mean_cost * size
                ci_lo = max(0, (mean_cost - 1.96 * std_cost / np.sqrt(n)) * size) if n > 1 else projected_cost
                ci_hi = (mean_cost + 1.96 * std_cost / np.sqrt(n)) * size if n > 1 else projected_cost

                # Rough time estimate: ~2 seconds per API call
                est_time_h = (size * 2.0) / 3600.0

                projections.append(
                    ScalingProjection(
                        model_id=str(model_id),
                        n_files=size,
                        projected_cost_usd=projected_cost,
                        projected_tokens=int(mean_tokens * size),
                        projected_time_hours=est_time_h,
                        cost_per_file=mean_cost,
                        confidence_lower=ci_lo,
                        confidence_upper=ci_hi,
                    )
                )
        return projections

    # ── token-usage analysis ─────────────────────────────────────────

    @staticmethod
    def token_analysis(cost_df: pd.DataFrame) -> pd.DataFrame:
        """Summarise token usage by model.

        Returns
        -------
        pd.DataFrame
            Columns: model_id, mean_prompt, mean_completion, median_prompt,
            median_completion, total_prompt, total_completion,
            input_output_ratio.
        """
        agg = cost_df.groupby("model_id").agg(
            mean_prompt=("prompt_tokens", "mean"),
            mean_completion=("completion_tokens", "mean"),
            median_prompt=("prompt_tokens", "median"),
            median_completion=("completion_tokens", "median"),
            total_prompt=("prompt_tokens", "sum"),
            total_completion=("completion_tokens", "sum"),
            n_calls=("prompt_tokens", "count"),
        ).reset_index()
        agg["input_output_ratio"] = np.where(
            agg["total_completion"] > 0,
            agg["total_prompt"] / agg["total_completion"],
            0.0,
        )
        return agg

    # ── cost-effectiveness comparison ────────────────────────────────

    @staticmethod
    def compare_cost_effectiveness(
        profiles: list[ModelCostProfile],
    ) -> list[CostEffectivenessRatio]:
        """Pairwise cost-effectiveness ratios between models."""
        from itertools import combinations

        ratios: list[CostEffectivenessRatio] = []
        for a, b in combinations(profiles, 2):
            if b.mean_cost_per_task > 0:
                ratio = a.mean_cost_per_task / b.mean_cost_per_task
            else:
                ratio = float("inf")

            ratios.append(
                CostEffectivenessRatio(
                    model_a=a.model_id,
                    model_b=b.model_id,
                    metric_name="mean_cost_per_task",
                    ratio=ratio,
                    delta_cost=a.mean_cost_per_task - b.mean_cost_per_task,
                )
            )

            # Also for cost_per_correct
            if a.cost_per_correct > 0 and b.cost_per_correct > 0:
                ratios.append(
                    CostEffectivenessRatio(
                        model_a=a.model_id,
                        model_b=b.model_id,
                        metric_name="cost_per_correct",
                        ratio=a.cost_per_correct / b.cost_per_correct,
                        delta_cost=a.cost_per_correct - b.cost_per_correct,
                    )
                )
        return ratios
