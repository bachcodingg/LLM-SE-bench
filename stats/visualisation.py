"""
stats/visualisation.py — Publication-quality figure generator for llm-se-bench.

Produces 25+ figure types using Matplotlib/Seaborn, saved as PDF.
All figures follow a consistent academic style with Times font and
muted colour palette suitable for colour-blind readers.

Figure catalogue
----------------
 1. pass_at_1_bar           — pass@1 by model × dataset
 2. pass_at_k_bar           — pass@k comparison
 3. score_boxplot           — weighted-score distributions
 4. quality_boxplot         — quality-metric distributions
 5. heatmap_difficulty      — model × difficulty performance
 6. heatmap_dataset_model   — dataset × model pass-rate
 7. radar_chart             — multi-dimensional model profile
 8. scatter_cost_accuracy   — cost vs. accuracy trade-off
 9. scatter_latency_quality — latency vs. quality
10. cumulative_pass         — cumulative pass rate by attempts
11. critical_difference     — Nemenyi post-hoc CD diagram
12. violin_latency          — latency distributions
13. violin_score            — score distributions
14. stacked_error           — error-type breakdown
15. cost_bar                — total cost per model
16. cost_per_correct_bar    — cost per correct solution
17. token_distribution      — prompt vs. completion tokens
18. consistency_heatmap     — cross-run agreement
19. flip_bar                — flip-rate per model
20. correlation_heatmap     — quality-metric correlations
21. effect_size_forest      — forest plot of effect sizes
22. ci_forest               — forest plot of CIs
23. scaling_line            — cost scaling projections
24. ck_metrics_radar        — CK-metric profiles per model
25. difficulty_stacked      — performance by difficulty level
26. boxplot_by_dataset      — per-dataset score comparison
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ── global style ─────────────────────────────────────────────────────

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
_FIG_DPI = 300
_DEFAULT_FIGSIZE = (8, 5)


def _apply_style() -> None:
    """Apply consistent publication style."""
    plt.rcParams.update({
        "figure.dpi": _FIG_DPI,
        "figure.figsize": _DEFAULT_FIGSIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "font.size": 10,
    })
    sns.set_palette(_PALETTE)


class VisualisationEngine:
    """Generate publication-quality figures.

    Parameters
    ----------
    output_dir : str | Path
        Directory where figure PDFs are saved.
    fmt : str
        File format (default ``"pdf"``).
    dpi : int
        Resolution (default 300).
    """

    def __init__(
        self,
        output_dir: str | Path = "analysis/figures",
        fmt: str = "pdf",
        dpi: int = 300,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fmt = fmt
        self.dpi = dpi
        _apply_style()

    def _save(self, fig: plt.Figure, name: str) -> Path:
        path = self.output_dir / f"{name}.{self.fmt}"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved figure: %s", path)
        return path

    # ────────────────────────────────────────────────────────────────
    # 1. pass@1 bar chart
    # ────────────────────────────────────────────────────────────────
    def pass_at_1_bar(
        self,
        df: pd.DataFrame,
        model_col: str = "model_id",
        dataset_col: str = "dataset",
        score_col: str = "is_pass",
    ) -> Path:
        """Bar chart of pass@1 rates by model grouped by dataset."""
        agg = (
            df.groupby([dataset_col, model_col])[score_col]
            .mean()
            .reset_index()
            .rename(columns={score_col: "pass@1"})
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        datasets = sorted(agg[dataset_col].unique())
        models = sorted(agg[model_col].unique())
        x = np.arange(len(datasets))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            vals = [
                agg.loc[(agg[dataset_col] == d) & (agg[model_col] == model), "pass@1"]
                .values
                for d in datasets
            ]
            heights = [v[0] if len(v) > 0 else 0 for v in vals]
            ax.bar(x + i * width, heights, width, label=model, color=_PALETTE[i % len(_PALETTE)])

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(datasets, rotation=15)
        ax.set_ylabel("pass@1")
        ax.set_title("pass@1 by Model and Dataset")
        ax.legend()
        ax.set_ylim(0, 1.05)
        return self._save(fig, "pass_at_1_bar")

    # ────────────────────────────────────────────────────────────────
    # 2. pass@k comparison
    # ────────────────────────────────────────────────────────────────
    def pass_at_k_bar(
        self,
        pass_at_k_df: pd.DataFrame,
        model_col: str = "model_id",
    ) -> Path:
        """Bar chart comparing pass@1 and pass@5 (or any k columns present)."""
        fig, ax = plt.subplots(figsize=(8, 5))
        k_cols = [c for c in pass_at_k_df.columns if c.startswith("pass@")]
        models = pass_at_k_df[model_col].tolist()
        x = np.arange(len(models))
        width = 0.8 / max(len(k_cols), 1)

        for i, col in enumerate(k_cols):
            ax.bar(x + i * width, pass_at_k_df[col], width, label=col,
                   color=_PALETTE[i % len(_PALETTE)])

        ax.set_xticks(x + width * (len(k_cols) - 1) / 2)
        ax.set_xticklabels(models)
        ax.set_ylabel("Pass rate")
        ax.set_title("pass@k Comparison")
        ax.legend()
        ax.set_ylim(0, 1.05)
        return self._save(fig, "pass_at_k_bar")

    # ────────────────────────────────────────────────────────────────
    # 3. Score box plot
    # ────────────────────────────────────────────────────────────────
    def score_boxplot(
        self,
        df: pd.DataFrame,
        score_col: str = "weighted_score",
        model_col: str = "model_id",
    ) -> Path:
        """Box plot of weighted scores by model."""
        fig, ax = plt.subplots(figsize=(8, 5))
        models = sorted(df[model_col].unique())
        data = [df.loc[df[model_col] == m, score_col].dropna().values for m in models]
        bp = ax.boxplot(data, labels=models, patch_artist=True, showmeans=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(_PALETTE[i % len(_PALETTE)])
            patch.set_alpha(0.7)
        ax.set_ylabel(score_col)
        ax.set_title("Score Distribution by Model")
        return self._save(fig, "score_boxplot")

    # ────────────────────────────────────────────────────────────────
    # 4. Quality metric box plots
    # ────────────────────────────────────────────────────────────────
    def quality_boxplot(
        self,
        df: pd.DataFrame,
        metric_cols: Sequence[str] = (
            "cyclomatic_complexity",
            "maintainability_index",
            "halstead_volume",
        ),
        model_col: str = "model_id",
    ) -> list[Path]:
        """One box plot per quality metric, faceted by model."""
        paths: list[Path] = []
        for col in metric_cols:
            if col not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            models = sorted(df[model_col].unique())
            data = [df.loc[df[model_col] == m, col].dropna().values for m in models]
            bp = ax.boxplot(data, labels=models, patch_artist=True, showmeans=True)
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(_PALETTE[i % len(_PALETTE)])
                patch.set_alpha(0.7)
            ax.set_ylabel(col.replace("_", " ").title())
            ax.set_title(f"{col.replace('_', ' ').title()} by Model")
            paths.append(self._save(fig, f"quality_boxplot_{col}"))
        return paths

    # ────────────────────────────────────────────────────────────────
    # 5. Heatmap: model × difficulty
    # ────────────────────────────────────────────────────────────────
    def heatmap_difficulty(
        self,
        df: pd.DataFrame,
        score_col: str = "weighted_score",
        model_col: str = "model_id",
        difficulty_col: str = "difficulty",
    ) -> Path:
        """Heatmap of mean score by model and difficulty level."""
        pivot = df.pivot_table(
            index=model_col, columns=difficulty_col, values=score_col, aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax,
                    vmin=0, vmax=1, linewidths=0.5)
        ax.set_title("Performance by Model × Difficulty")
        return self._save(fig, "heatmap_difficulty")

    # ────────────────────────────────────────────────────────────────
    # 6. Heatmap: dataset × model
    # ────────────────────────────────────────────────────────────────
    def heatmap_dataset_model(
        self,
        df: pd.DataFrame,
        score_col: str = "is_pass",
        model_col: str = "model_id",
        dataset_col: str = "dataset",
    ) -> Path:
        """Heatmap of pass rate by dataset and model."""
        pivot = df.pivot_table(
            index=dataset_col, columns=model_col, values=score_col, aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
                    vmin=0, vmax=1, linewidths=0.5)
        ax.set_title("Pass Rate: Dataset × Model")
        return self._save(fig, "heatmap_dataset_model")

    # ────────────────────────────────────────────────────────────────
    # 7. Radar chart
    # ────────────────────────────────────────────────────────────────
    def radar_chart(
        self,
        model_profiles: dict[str, dict[str, float]],
        dimensions: Sequence[str] | None = None,
    ) -> Path:
        """Radar chart comparing models across multiple dimensions.

        Parameters
        ----------
        model_profiles : dict
            ``{model_id: {dimension: normalised_score}}``.
        dimensions : list of str, optional
            Dimension names (keys of inner dicts if not given).
        """
        models = list(model_profiles.keys())
        if not dimensions:
            dimensions = list(next(iter(model_profiles.values())).keys())

        n_dims = len(dimensions)
        angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for i, model in enumerate(models):
            values = [model_profiles[model].get(d, 0) for d in dimensions]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=model,
                    color=_PALETTE[i % len(_PALETTE)])
            ax.fill(angles, values, alpha=0.15, color=_PALETTE[i % len(_PALETTE)])

        ax.set_thetagrids(np.degrees(angles[:-1]), dimensions)
        ax.set_ylim(0, 1)
        ax.set_title("Model Profile Comparison", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        return self._save(fig, "radar_chart")

    # ────────────────────────────────────────────────────────────────
    # 8. Cost vs. accuracy scatter
    # ────────────────────────────────────────────────────────────────
    def scatter_cost_accuracy(
        self,
        df: pd.DataFrame,
        cost_col: str = "total_cost_usd",
        accuracy_col: str = "pass_rate",
        model_col: str = "model_id",
    ) -> Path:
        """Scatter plot of cost vs. accuracy per model."""
        fig, ax = plt.subplots(figsize=(8, 6))
        models = sorted(df[model_col].unique())
        for i, model in enumerate(models):
            sub = df[df[model_col] == model]
            ax.scatter(sub[cost_col], sub[accuracy_col], label=model, s=60,
                       color=_PALETTE[i % len(_PALETTE)], alpha=0.7)
        ax.set_xlabel("Cost (USD)")
        ax.set_ylabel("Pass Rate")
        ax.set_title("Cost vs. Accuracy Trade-off")
        ax.legend()
        return self._save(fig, "scatter_cost_accuracy")

    # ────────────────────────────────────────────────────────────────
    # 9. Latency vs. quality scatter
    # ────────────────────────────────────────────────────────────────
    def scatter_latency_quality(
        self,
        df: pd.DataFrame,
        latency_col: str = "latency_ms",
        quality_col: str = "maintainability_index",
        model_col: str = "model_id",
    ) -> Path:
        """Scatter plot of latency vs. quality per model."""
        fig, ax = plt.subplots(figsize=(8, 6))
        models = sorted(df[model_col].unique())
        for i, model in enumerate(models):
            sub = df[df[model_col] == model]
            if latency_col in sub.columns and quality_col in sub.columns:
                ax.scatter(sub[latency_col], sub[quality_col], label=model, s=60,
                           color=_PALETTE[i % len(_PALETTE)], alpha=0.7)
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Maintainability Index")
        ax.set_title("Latency vs. Code Quality")
        ax.legend()
        return self._save(fig, "scatter_latency_quality")

    # ────────────────────────────────────────────────────────────────
    # 10. Cumulative pass rate
    # ────────────────────────────────────────────────────────────────
    def cumulative_pass(
        self,
        df: pd.DataFrame,
        model_col: str = "model_id",
        problem_col: str = "problem_id",
        binary_col: str = "is_pass",
    ) -> Path:
        """Line plot of cumulative pass rate by number of attempts."""
        fig, ax = plt.subplots(figsize=(8, 5))
        models = sorted(df[model_col].unique())

        for i, model in enumerate(models):
            sub = df[df[model_col] == model].sort_values(problem_col)
            cumpass = sub[binary_col].astype(int).cumsum()
            cumrate = cumpass / (np.arange(len(cumpass)) + 1)
            ax.plot(range(1, len(cumrate) + 1), cumrate, label=model,
                    color=_PALETTE[i % len(_PALETTE)], linewidth=2)

        ax.set_xlabel("Number of Problems Attempted")
        ax.set_ylabel("Cumulative Pass Rate")
        ax.set_title("Cumulative Pass Rate by Model")
        ax.legend()
        ax.set_ylim(0, 1.05)
        return self._save(fig, "cumulative_pass")

    # ────────────────────────────────────────────────────────────────
    # 11. Critical difference diagram (Nemenyi)
    # ────────────────────────────────────────────────────────────────
    def critical_difference_diagram(
        self,
        avg_ranks: dict[str, float],
        cd: float,
        n_subjects: int,
    ) -> Path:
        """Demšar (2006) style critical-difference diagram.

        Parameters
        ----------
        avg_ranks : dict
            ``{model_id: average_rank}`` (lower = better).
        cd : float
            Critical difference from Nemenyi.
        n_subjects : int
            Number of matched subjects.
        """
        models = sorted(avg_ranks, key=lambda m: avg_ranks[m])
        ranks = [avg_ranks[m] for m in models]
        k = len(models)

        fig, ax = plt.subplots(figsize=(10, 2 + k * 0.4))
        ax.set_xlim(0.5, k + 0.5)

        # Draw axis
        for r in range(1, k + 1):
            ax.axvline(r, color="grey", linewidth=0.5, alpha=0.3)

        # Draw models
        for i, (model, rank) in enumerate(zip(models, ranks)):
            y = i
            ax.plot(rank, y, "o", color=_PALETTE[i % len(_PALETTE)], markersize=8)
            side = "left" if rank < (k + 1) / 2 else "right"
            offset = -0.15 if side == "left" else 0.15
            ax.text(rank + offset, y, f"  {model} ({rank:.2f})",
                    va="center", ha=side, fontsize=9)

        # Draw CD bar
        ax.plot([1, 1 + cd], [-0.8, -0.8], "k-", linewidth=2)
        ax.text((1 + 1 + cd) / 2, -1.1, f"CD = {cd:.3f}",
                ha="center", fontsize=9)

        # Connect non-significant pairs
        for i in range(k):
            for j in range(i + 1, k):
                if abs(ranks[i] - ranks[j]) < cd:
                    y_mid = (i + j) / 2
                    ax.plot([ranks[i], ranks[j]], [y_mid + 0.2, y_mid + 0.2],
                            "k-", linewidth=2.5, alpha=0.4)

        ax.set_xlabel("Average Rank")
        ax.set_yticks([])
        ax.set_title(f"Critical Difference Diagram (n = {n_subjects})")
        ax.invert_yaxis()
        return self._save(fig, "critical_difference")

    # ────────────────────────────────────────────────────────────────
    # 12. Violin plot: latency
    # ────────────────────────────────────────────────────────────────
    def violin_latency(
        self,
        df: pd.DataFrame,
        latency_col: str = "latency_ms",
        model_col: str = "model_id",
    ) -> Path:
        """Violin plot of latency distributions."""
        if latency_col not in df.columns:
            logger.warning("Column %s missing — skipping violin_latency", latency_col)
            return Path()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(data=df, x=model_col, y=latency_col, ax=ax,
                       palette=_PALETTE, inner="quartile", cut=0)
        ax.set_title("Latency Distribution by Model")
        ax.set_ylabel("Latency (ms)")
        return self._save(fig, "violin_latency")

    # ────────────────────────────────────────────────────────────────
    # 13. Violin plot: scores
    # ────────────────────────────────────────────────────────────────
    def violin_score(
        self,
        df: pd.DataFrame,
        score_col: str = "weighted_score",
        model_col: str = "model_id",
    ) -> Path:
        """Violin plot of score distributions."""
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(data=df, x=model_col, y=score_col, ax=ax,
                       palette=_PALETTE, inner="quartile", cut=0)
        ax.set_title("Score Distribution by Model")
        ax.set_ylabel("Weighted Score")
        return self._save(fig, "violin_score")

    # ────────────────────────────────────────────────────────────────
    # 14. Stacked error-type bar chart
    # ────────────────────────────────────────────────────────────────
    def stacked_error(
        self,
        df: pd.DataFrame,
        model_col: str = "model_id",
    ) -> Path:
        """Stacked bar chart of error-type breakdown per model."""
        # Derive error categories from data
        cats = {}
        for model, grp in df.groupby(model_col):
            n = len(grp)
            cats[model] = {
                "Pass": int(grp.get("is_pass", pd.Series(dtype=bool)).sum()) if "is_pass" in grp.columns else 0,
                "Compile Error": int((~grp.get("compile_success", pd.Series(True, index=grp.index))).sum()) if "compile_success" in grp.columns else 0,
                "Wrong Answer": 0,
                "Runtime Error": 0,
            }
            if "verdict" in grp.columns:
                cats[model]["Wrong Answer"] = int((grp["verdict"] == "fail").sum())
                cats[model]["Runtime Error"] = int((grp["verdict"] == "error").sum())
            # Adjust so they sum to n
            accounted = sum(cats[model].values())
            if accounted < n:
                cats[model]["Wrong Answer"] += n - accounted

        cat_df = pd.DataFrame(cats).T
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#55A868", "#C44E52", "#DD8452", "#8172B3"]
        cat_df.plot.bar(stacked=True, ax=ax, color=colors[:len(cat_df.columns)])
        ax.set_ylabel("Count")
        ax.set_title("Error Type Breakdown by Model")
        ax.legend(title="Outcome")
        plt.xticks(rotation=0)
        return self._save(fig, "stacked_error")

    # ────────────────────────────────────────────────────────────────
    # 15. Total cost bar
    # ────────────────────────────────────────────────────────────────
    def cost_bar(
        self,
        profiles: list[dict],
    ) -> Path:
        """Bar chart of total cost per model."""
        fig, ax = plt.subplots(figsize=(7, 5))
        models = [p["model_id"] for p in profiles]
        costs = [p["total_cost_usd"] for p in profiles]
        bars = ax.bar(models, costs, color=_PALETTE[:len(models)])
        ax.set_ylabel("Total Cost (USD)")
        ax.set_title("Total API Cost by Model")
        for bar, cost in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"${cost:.2f}", ha="center", va="bottom", fontsize=9)
        return self._save(fig, "cost_bar")

    # ────────────────────────────────────────────────────────────────
    # 16. Cost per correct solution
    # ────────────────────────────────────────────────────────────────
    def cost_per_correct_bar(
        self,
        profiles: list[dict],
    ) -> Path:
        """Bar chart of cost per correct solution."""
        fig, ax = plt.subplots(figsize=(7, 5))
        models = [p["model_id"] for p in profiles]
        costs = [p.get("cost_per_correct", 0) for p in profiles]
        bars = ax.bar(models, costs, color=_PALETTE[:len(models)])
        ax.set_ylabel("Cost per Correct Solution (USD)")
        ax.set_title("Cost-Effectiveness by Model")
        for bar, cost in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"${cost:.3f}", ha="center", va="bottom", fontsize=9)
        return self._save(fig, "cost_per_correct_bar")

    # ────────────────────────────────────────────────────────────────
    # 17. Token distribution
    # ────────────────────────────────────────────────────────────────
    def token_distribution(
        self,
        token_df: pd.DataFrame,
        model_col: str = "model_id",
    ) -> Path:
        """Grouped bar chart of mean prompt vs. completion tokens."""
        fig, ax = plt.subplots(figsize=(8, 5))
        models = token_df[model_col].tolist()
        x = np.arange(len(models))
        w = 0.35
        ax.bar(x - w / 2, token_df["mean_prompt"], w, label="Prompt", color=_PALETTE[0])
        ax.bar(x + w / 2, token_df["mean_completion"], w, label="Completion", color=_PALETTE[1])
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel("Mean Tokens")
        ax.set_title("Token Usage Distribution")
        ax.legend()
        return self._save(fig, "token_distribution")

    # ────────────────────────────────────────────────────────────────
    # 18. Consistency heatmap
    # ────────────────────────────────────────────────────────────────
    def consistency_heatmap(
        self,
        profiles: list[dict],
    ) -> Path:
        """Heatmap of consistency metrics across models."""
        df = pd.DataFrame(profiles)
        if df.empty:
            return Path()
        metrics = ["agreement_rate", "mean_cv", "icc", "flip_rate"]
        available = [m for m in metrics if m in df.columns]
        if not available:
            return Path()
        pivot = df.set_index("model_id")[available]
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd_r", ax=ax,
                    linewidths=0.5)
        ax.set_title("Cross-Run Consistency Metrics")
        return self._save(fig, "consistency_heatmap")

    # ────────────────────────────────────────────────────────────────
    # 19. Flip-rate bar
    # ────────────────────────────────────────────────────────────────
    def flip_bar(
        self,
        profiles: list[dict],
    ) -> Path:
        """Bar chart of flip rates per model."""
        fig, ax = plt.subplots(figsize=(7, 5))
        models = [p["model_id"] for p in profiles]
        rates = [p.get("flip_rate", 0) for p in profiles]
        ax.bar(models, rates, color=_PALETTE[:len(models)])
        ax.set_ylabel("Flip Rate")
        ax.set_title("Pass/Fail Flip Rate by Model")
        ax.set_ylim(0, max(rates) * 1.3 if rates and max(rates) > 0 else 1)
        return self._save(fig, "flip_bar")

    # ────────────────────────────────────────────────────────────────
    # 20. Correlation heatmap
    # ────────────────────────────────────────────────────────────────
    def correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        pval_matrix: pd.DataFrame | None = None,
    ) -> Path:
        """Heatmap of correlation matrix with significance markers."""
        fig, ax = plt.subplots(figsize=(8, 7))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax, linewidths=0.5,
            vmin=-1, vmax=1,
        )
        ax.set_title("Quality Metric Correlations (Spearman)")
        return self._save(fig, "correlation_heatmap")

    # ────────────────────────────────────────────────────────────────
    # 21. Effect-size forest plot
    # ────────────────────────────────────────────────────────────────
    def effect_size_forest(
        self,
        results: list[dict],
    ) -> Path:
        """Forest plot of effect sizes with CIs."""
        fig, ax = plt.subplots(figsize=(8, max(3, len(results) * 0.5)))
        labels = [f"{r['group_a']} vs {r['group_b']}" for r in results]
        values = [r["value"] for r in results]
        ci_lo = [r.get("ci_lower", r["value"]) for r in results]
        ci_hi = [r.get("ci_upper", r["value"]) for r in results]
        y = range(len(results))

        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        for i in range(len(results)):
            color = _PALETTE[0] if values[i] >= 0 else _PALETTE[3]
            ax.plot(values[i], i, "o", color=color, markersize=8)
            ax.plot([ci_lo[i], ci_hi[i]], [i, i], "-", color=color, linewidth=2)

        ax.set_yticks(list(y))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Effect Size (Cliff's δ)")
        ax.set_title("Effect Sizes with 95% CI")
        ax.invert_yaxis()
        return self._save(fig, "effect_size_forest")

    # ────────────────────────────────────────────────────────────────
    # 22. CI forest plot
    # ────────────────────────────────────────────────────────────────
    def ci_forest(
        self,
        ci_results: list[dict],
    ) -> Path:
        """Forest plot of bootstrap confidence intervals."""
        fig, ax = plt.subplots(figsize=(8, max(3, len(ci_results) * 0.5)))
        labels = [r.get("statistic", "") for r in ci_results]
        points = [r["point_estimate"] for r in ci_results]
        lo = [r["ci_lower"] for r in ci_results]
        hi = [r["ci_upper"] for r in ci_results]
        y = range(len(ci_results))

        for i in range(len(ci_results)):
            ax.plot(points[i], i, "s", color=_PALETTE[0], markersize=7)
            ax.plot([lo[i], hi[i]], [i, i], "-", color=_PALETTE[0], linewidth=2)

        ax.set_yticks(list(y))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Value")
        ax.set_title("Bootstrap 95% Confidence Intervals")
        ax.invert_yaxis()
        return self._save(fig, "ci_forest")

    # ────────────────────────────────────────────────────────────────
    # 23. Scaling projection line plot
    # ────────────────────────────────────────────────────────────────
    def scaling_line(
        self,
        projections: list[dict],
    ) -> Path:
        """Line plot of projected costs at different codebase sizes."""
        fig, ax = plt.subplots(figsize=(9, 5))
        proj_df = pd.DataFrame(projections)
        if proj_df.empty:
            plt.close(fig)
            return Path()

        models = sorted(proj_df["model_id"].unique())
        for i, model in enumerate(models):
            sub = proj_df[proj_df["model_id"] == model].sort_values("n_files")
            ax.plot(sub["n_files"], sub["projected_cost_usd"], "o-",
                    label=model, color=_PALETTE[i % len(_PALETTE)], linewidth=2)
            ax.fill_between(
                sub["n_files"],
                sub["confidence_lower"],
                sub["confidence_upper"],
                alpha=0.15, color=_PALETTE[i % len(_PALETTE)],
            )

        ax.set_xlabel("Number of Files")
        ax.set_ylabel("Projected Cost (USD)")
        ax.set_title("Cost Scaling Projections")
        ax.legend()
        ax.set_xscale("log")
        return self._save(fig, "scaling_line")

    # ────────────────────────────────────────────────────────────────
    # 24. CK metrics radar
    # ────────────────────────────────────────────────────────────────
    def ck_metrics_radar(
        self,
        df: pd.DataFrame,
        model_col: str = "model_id",
        ck_cols: Sequence[str] = ("wmc", "dit", "noc", "cbo", "rfc", "lcom"),
    ) -> Path:
        """Radar chart of CK metric profiles per model."""
        available = [c for c in ck_cols if c in df.columns]
        if not available:
            return Path()

        # Normalise to 0-1
        profiles: dict[str, dict[str, float]] = {}
        for model, grp in df.groupby(model_col):
            vals = {}
            for col in available:
                raw = grp[col].mean()
                col_max = df[col].max()
                vals[col.upper()] = raw / col_max if col_max > 0 else 0.0
            profiles[str(model)] = vals

        return self.radar_chart(profiles, [c.upper() for c in available])

    # ────────────────────────────────────────────────────────────────
    # 25. Difficulty stacked bar
    # ────────────────────────────────────────────────────────────────
    def difficulty_stacked(
        self,
        df: pd.DataFrame,
        model_col: str = "model_id",
        difficulty_col: str = "difficulty",
        binary_col: str = "is_pass",
    ) -> Path:
        """Stacked bar of pass rates by difficulty level per model."""
        if difficulty_col not in df.columns:
            return Path()

        agg = df.groupby([model_col, difficulty_col])[binary_col].mean().unstack(difficulty_col)
        fig, ax = plt.subplots(figsize=(8, 5))
        agg.plot.bar(ax=ax, color=_PALETTE[:agg.shape[1]])
        ax.set_ylabel("Pass Rate")
        ax.set_title("Pass Rate by Difficulty Level")
        ax.legend(title="Difficulty")
        plt.xticks(rotation=0)
        return self._save(fig, "difficulty_stacked")

    # ────────────────────────────────────────────────────────────────
    # 26. Box plot by dataset
    # ────────────────────────────────────────────────────────────────
    def boxplot_by_dataset(
        self,
        df: pd.DataFrame,
        score_col: str = "weighted_score",
        model_col: str = "model_id",
        dataset_col: str = "dataset",
    ) -> Path:
        """Faceted box plots per dataset."""
        datasets = sorted(df[dataset_col].unique())
        n_ds = len(datasets)
        fig, axes = plt.subplots(1, n_ds, figsize=(4 * n_ds, 5), sharey=True)
        if n_ds == 1:
            axes = [axes]

        for ax, ds in zip(axes, datasets):
            sub = df[df[dataset_col] == ds]
            models = sorted(sub[model_col].unique())
            data = [sub.loc[sub[model_col] == m, score_col].dropna().values
                    for m in models]
            bp = ax.boxplot(data, labels=models, patch_artist=True)
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(_PALETTE[i % len(_PALETTE)])
                patch.set_alpha(0.7)
            ax.set_title(ds)
            ax.set_ylim(0, 1.05)

        fig.suptitle("Score Distribution by Dataset", y=1.02)
        fig.tight_layout()
        return self._save(fig, "boxplot_by_dataset")

    # ── generate all figures ─────────────────────────────────────────

    def generate_all(
        self,
        results_df: pd.DataFrame,
        quality_df: pd.DataFrame | None = None,
        cost_profiles: list[dict] | None = None,
        token_df: pd.DataFrame | None = None,
        consistency_profiles: list[dict] | None = None,
        corr_matrix: pd.DataFrame | None = None,
        pval_matrix: pd.DataFrame | None = None,
        effect_sizes: list[dict] | None = None,
        ci_results: list[dict] | None = None,
        projections: list[dict] | None = None,
        nemenyi_ranks: dict[str, float] | None = None,
        nemenyi_cd: float | None = None,
        model_profiles: dict[str, dict[str, float]] | None = None,
        pass_at_k_df: pd.DataFrame | None = None,
    ) -> list[Path]:
        """Generate all applicable figures, returning paths to saved files."""
        paths: list[Path] = []

        # Always available from results
        try:
            paths.append(self.pass_at_1_bar(results_df))
        except Exception as e:
            logger.warning("pass_at_1_bar failed: %s", e)
        try:
            paths.append(self.score_boxplot(results_df))
        except Exception as e:
            logger.warning("score_boxplot failed: %s", e)
        try:
            paths.append(self.violin_score(results_df))
        except Exception as e:
            logger.warning("violin_score failed: %s", e)
        try:
            paths.append(self.heatmap_dataset_model(results_df))
        except Exception as e:
            logger.warning("heatmap_dataset_model failed: %s", e)
        try:
            paths.append(self.cumulative_pass(results_df))
        except Exception as e:
            logger.warning("cumulative_pass failed: %s", e)
        try:
            paths.append(self.stacked_error(results_df))
        except Exception as e:
            logger.warning("stacked_error failed: %s", e)

        if "difficulty" in results_df.columns:
            try:
                paths.append(self.heatmap_difficulty(results_df))
            except Exception as e:
                logger.warning("heatmap_difficulty failed: %s", e)
            try:
                paths.append(self.difficulty_stacked(results_df))
            except Exception as e:
                logger.warning("difficulty_stacked failed: %s", e)

        if "dataset" in results_df.columns:
            try:
                paths.append(self.boxplot_by_dataset(results_df))
            except Exception as e:
                logger.warning("boxplot_by_dataset failed: %s", e)

        if pass_at_k_df is not None:
            try:
                paths.append(self.pass_at_k_bar(pass_at_k_df))
            except Exception as e:
                logger.warning("pass_at_k_bar failed: %s", e)

        # Quality metrics
        if quality_df is not None and not quality_df.empty:
            try:
                paths.extend(self.quality_boxplot(quality_df))
            except Exception as e:
                logger.warning("quality_boxplot failed: %s", e)

        # Cost
        if cost_profiles:
            try:
                paths.append(self.cost_bar(cost_profiles))
            except Exception as e:
                logger.warning("cost_bar failed: %s", e)
            try:
                paths.append(self.cost_per_correct_bar(cost_profiles))
            except Exception as e:
                logger.warning("cost_per_correct_bar failed: %s", e)

        if token_df is not None:
            try:
                paths.append(self.token_distribution(token_df))
            except Exception as e:
                logger.warning("token_distribution failed: %s", e)

        # Consistency
        if consistency_profiles:
            try:
                paths.append(self.consistency_heatmap(consistency_profiles))
            except Exception as e:
                logger.warning("consistency_heatmap failed: %s", e)
            try:
                paths.append(self.flip_bar(consistency_profiles))
            except Exception as e:
                logger.warning("flip_bar failed: %s", e)

        # Correlations
        if corr_matrix is not None:
            try:
                paths.append(self.correlation_heatmap(corr_matrix, pval_matrix))
            except Exception as e:
                logger.warning("correlation_heatmap failed: %s", e)

        # Effect sizes
        if effect_sizes:
            try:
                paths.append(self.effect_size_forest(effect_sizes))
            except Exception as e:
                logger.warning("effect_size_forest failed: %s", e)

        # CIs
        if ci_results:
            try:
                paths.append(self.ci_forest(ci_results))
            except Exception as e:
                logger.warning("ci_forest failed: %s", e)

        # Scaling
        if projections:
            try:
                paths.append(self.scaling_line(projections))
            except Exception as e:
                logger.warning("scaling_line failed: %s", e)

        # Nemenyi CD
        if nemenyi_ranks and nemenyi_cd is not None:
            try:
                n_sub = len(results_df["problem_id"].unique()) if "problem_id" in results_df.columns else 0
                paths.append(self.critical_difference_diagram(nemenyi_ranks, nemenyi_cd, n_sub))
            except Exception as e:
                logger.warning("critical_difference_diagram failed: %s", e)

        # Radar
        if model_profiles:
            try:
                paths.append(self.radar_chart(model_profiles))
            except Exception as e:
                logger.warning("radar_chart failed: %s", e)

        # Latency violin
        if "latency_ms" in results_df.columns:
            try:
                paths.append(self.violin_latency(results_df))
            except Exception as e:
                logger.warning("violin_latency failed: %s", e)

        # Scatter plots
        if "total_cost_usd" in results_df.columns and "pass_rate" in results_df.columns:
            try:
                paths.append(self.scatter_cost_accuracy(results_df))
            except Exception as e:
                logger.warning("scatter_cost_accuracy failed: %s", e)

        paths = [p for p in paths if p and p != Path()]
        logger.info("Generated %d figures in %s", len(paths), self.output_dir)
        return paths
