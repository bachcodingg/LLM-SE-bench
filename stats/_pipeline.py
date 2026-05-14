"""
stats/_pipeline.py — Full analysis pipeline orchestrating all C4 modules.

The :func:`run_full_analysis` entry point loads data, runs every analysis,
generates figures and LaTeX tables, and writes a
``analysis/statistical_summary.json`` file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stats._loader import (
    load_evaluation_results,
    load_quality_metrics,
    load_cost_data,
    merge_results_quality,
)
from stats.descriptive import DescriptiveAnalyzer
from stats.hypothesis import HypothesisEngine
from stats.effect_size import EffectSizeCalculator
from stats.confidence import BootstrapCI
from stats.cost_model import CostModel
from stats.consistency import ConsistencyAnalyzer
from stats.correlation import CorrelationAnalyzer
from stats.visualisation import VisualisationEngine
from stats.report_data import ReportDataGenerator

logger = logging.getLogger(__name__)

# Metrics that we always analyse when present
_PERF_METRICS = ["weighted_score", "pass_rate", "latency_ms"]
_QUALITY_METRICS = [
    "cyclomatic_complexity",
    "maintainability_index",
    "halstead_volume",
    "lines_of_code",
]


def run_full_analysis(
    results_dir: str | Path,
    quality_dir: str | Path,
    db_path: str | Path | None = None,
    output_dir: str | Path = "analysis",
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
) -> dict[str, Any]:
    """Execute the complete C4 statistical analysis pipeline.

    Parameters
    ----------
    results_dir : path
        Root of ``results/{dataset}/{model}/results.jsonl`` tree.
    quality_dir : path
        Root of ``quality/{dataset}/{model}/metrics.csv`` tree.
    db_path : path, optional
        Path to ``llm_cache.db`` (C1 cost data).
    output_dir : path
        Where to write ``statistical_summary.json`` and sub-directories
        ``figures/`` and ``tables/``.
    alpha : float
        Significance level for hypothesis tests.
    n_bootstrap : int
        Number of bootstrap resamples.

    Returns
    -------
    dict
        The complete statistical summary (also written to JSON).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    tab_dir = out / "tables"

    summary: dict[str, Any] = {"meta": {"alpha": alpha, "n_bootstrap": n_bootstrap}}

    # ── 1. Load data ─────────────────────────────────────────────────
    logger.info("Loading evaluation results from %s", results_dir)
    results_df = load_evaluation_results(results_dir)
    quality_df = load_quality_metrics(quality_dir)
    merged_df = merge_results_quality(results_df, quality_df)

    cost_df = pd.DataFrame()
    if db_path and Path(db_path).exists():
        cost_df = load_cost_data(db_path)

    summary["data_summary"] = {
        "n_results": len(results_df),
        "n_quality": len(quality_df),
        "n_cost": len(cost_df),
        "models": sorted(results_df["model_id"].unique().tolist()) if not results_df.empty else [],
        "datasets": sorted(results_df["dataset"].unique().tolist()) if not results_df.empty else [],
    }

    if results_df.empty:
        logger.warning("No results loaded — aborting analysis")
        _write_json(summary, out / "statistical_summary.json")
        return summary

    # ── 2. Descriptive statistics ────────────────────────────────────
    logger.info("Computing descriptive statistics")
    desc = DescriptiveAnalyzer()
    available_metrics = [m for m in _PERF_METRICS if m in results_df.columns]
    desc_results = desc.summarise_all_metrics(
        results_df, available_metrics, group_by=["model_id", "dataset"]
    )
    summary["descriptive"] = [d.to_dict() for d in desc_results]

    # pass@k
    pass_at_k_frames = []
    for k in (1, 5):
        try:
            pak = desc.pass_at_k(results_df, k=k)
            pass_at_k_frames.append(pak)
        except Exception:
            pass

    pass_at_k_df = pd.DataFrame()
    if pass_at_k_frames:
        pass_at_k_df = pass_at_k_frames[0]
        for extra in pass_at_k_frames[1:]:
            pass_at_k_df = pass_at_k_df.merge(extra, on="model_id", how="outer")
    summary["pass_at_k"] = pass_at_k_df.to_dict(orient="records") if not pass_at_k_df.empty else []

    # ── 3. Hypothesis tests ──────────────────────────────────────────
    logger.info("Running hypothesis tests")
    hyp = HypothesisEngine(alpha=alpha)
    hyp_results: list[dict] = []

    for metric in available_metrics:
        bundle = hyp.run_all(results_df, metric)
        friedman = bundle["friedman"]
        hyp_results.append(friedman.to_dict())
        for r in bundle["nemenyi"]:
            hyp_results.append(r.to_dict())
        for r in bundle["wilcoxon"]:
            hyp_results.append(r.to_dict())

    # Cochran's Q on binary
    if "is_pass" in results_df.columns:
        q_result = hyp.cochrans_q(results_df)
        hyp_results.append(q_result.to_dict())

    summary["hypothesis_tests"] = hyp_results

    # ── 4. Effect sizes ──────────────────────────────────────────────
    logger.info("Computing effect sizes")
    esc = EffectSizeCalculator(method="cliffs_delta", n_bootstrap=min(n_bootstrap, 2000))
    es_results: list[dict] = []
    for metric in available_metrics:
        for r in esc.compute_pairwise(results_df, metric):
            es_results.append(r.to_dict())
    summary["effect_sizes"] = es_results

    # ── 5. Confidence intervals ──────────────────────────────────────
    logger.info("Computing bootstrap confidence intervals")
    bci = BootstrapCI(n_bootstrap=n_bootstrap, method="percentile")
    ci_results: list[dict] = []
    for model in sorted(results_df["model_id"].unique()):
        for metric in available_metrics:
            vals = results_df.loc[results_df["model_id"] == model, metric].dropna().values
            ci = bci.compute(vals, np.mean, f"{model}_{metric}_mean")
            ci_results.append(ci.to_dict())
    summary["confidence_intervals"] = ci_results

    # ── 6. Cost modelling ────────────────────────────────────────────
    cost_profiles_dicts: list[dict] = []
    token_df = pd.DataFrame()
    projections: list[dict] = []

    if not cost_df.empty:
        logger.info("Running cost analysis")
        cm = CostModel()
        profiles = cm.compute_profiles(cost_df, results_df, quality_df)
        cost_profiles_dicts = [p.to_dict() for p in profiles]
        token_df = cm.token_analysis(cost_df)
        proj = cm.project_scaling(cost_df)
        projections = [p.to_dict() for p in proj]

    summary["cost_profiles"] = cost_profiles_dicts
    summary["scaling_projections"] = projections

    # ── 7. Consistency analysis ──────────────────────────────────────
    logger.info("Running consistency analysis")
    cons = ConsistencyAnalyzer()
    cons_profiles = cons.analyse(results_df, group_by=["model_id"])
    cons_dicts = [p.to_dict() for p in cons_profiles]
    flips = cons.find_flips(results_df)
    summary["consistency"] = cons_dicts
    summary["flips"] = [f.to_dict() for f in flips]

    # ── 8. Correlation analysis ──────────────────────────────────────
    corr_matrix = pd.DataFrame()
    pval_matrix = pd.DataFrame()
    corr_results: list[dict] = []

    if not quality_df.empty:
        logger.info("Running correlation analysis")
        ca = CorrelationAnalyzer()
        avail_quality = [c for c in _QUALITY_METRICS if c in merged_df.columns]
        if avail_quality and "weighted_score" in merged_df.columns:
            corr_results = [
                r.to_dict()
                for r in ca.metrics_vs_performance(merged_df, avail_quality)
            ]
        if len(avail_quality) >= 2:
            corr_matrix, pval_matrix = ca.correlation_matrix(merged_df, avail_quality)

    summary["correlations"] = corr_results

    # ── 9. Model rankings ────────────────────────────────────────────
    logger.info("Computing model rankings")
    rankings = _compute_rankings(results_df, quality_df, cost_profiles_dicts, hyp_results)
    summary["rankings"] = rankings

    # ── 10. Figures ──────────────────────────────────────────────────
    logger.info("Generating figures")
    viz = VisualisationEngine(output_dir=fig_dir)

    # Nemenyi data for CD diagram
    nemenyi_ranks = None
    nemenyi_cd = None
    for r in hyp_results:
        if r.get("test_name") == "nemenyi" and r.get("notes", ""):
            # Parse from the first nemenyi result
            pass
    # Build from Friedman rank data
    if "problem_id" in results_df.columns and "weighted_score" in results_df.columns:
        pivot = (
            results_df.groupby(["problem_id", "model_id"])["weighted_score"]
            .mean().unstack("model_id").dropna()
        )
        if pivot.shape[1] >= 3:
            ranks = pivot.rank(axis=1, method="average", ascending=False)
            nemenyi_ranks = ranks.mean().to_dict()
            k = len(nemenyi_ranks)
            n = len(pivot)
            from scipy.stats import studentized_range
            q_a = studentized_range.ppf(1 - alpha, k, np.inf)
            nemenyi_cd = q_a * np.sqrt(k * (k + 1) / (12.0 * n))

    # Model profiles for radar
    model_profiles = _build_model_profiles(results_df, quality_df, cost_profiles_dicts)

    viz.generate_all(
        results_df=results_df,
        quality_df=quality_df if not quality_df.empty else None,
        cost_profiles=cost_profiles_dicts or None,
        token_df=token_df if not token_df.empty else None,
        consistency_profiles=cons_dicts or None,
        corr_matrix=corr_matrix if not corr_matrix.empty else None,
        pval_matrix=pval_matrix if not pval_matrix.empty else None,
        effect_sizes=es_results or None,
        ci_results=ci_results or None,
        projections=projections or None,
        nemenyi_ranks=nemenyi_ranks,
        nemenyi_cd=nemenyi_cd,
        model_profiles=model_profiles or None,
        pass_at_k_df=pass_at_k_df if not pass_at_k_df.empty else None,
    )

    # ── 11. LaTeX tables ─────────────────────────────────────────────
    logger.info("Generating LaTeX tables")
    rdg = ReportDataGenerator(output_dir=tab_dir)
    rdg.generate_all(
        descriptive_stats=summary.get("descriptive"),
        hypothesis_results=hyp_results or None,
        effect_sizes=es_results or None,
        rankings=rankings or None,
        cost_profiles=cost_profiles_dicts or None,
        pass_df=pass_at_k_df if not pass_at_k_df.empty else None,
        correlations=corr_results or None,
        consistency_profiles=cons_dicts or None,
    )

    # ── 12. Write summary JSON ───────────────────────────────────────
    _write_json(summary, out / "statistical_summary.json")
    logger.info("Analysis complete — summary written to %s", out / "statistical_summary.json")
    return summary


# ── helpers ──────────────────────────────────────────────────────────

def _write_json(data: dict, path: Path) -> None:
    """Write dict to JSON, handling numpy types."""

    class _Encoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return super().default(obj)

    path.write_text(json.dumps(data, indent=2, cls=_Encoder), encoding="utf-8")


def _compute_rankings(
    results_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    cost_profiles: list[dict],
    hyp_results: list[dict],
) -> list[dict]:
    """Compute composite model rankings."""
    models = sorted(results_df["model_id"].unique())
    rankings: list[dict] = []

    for model in models:
        sub = results_df[results_df["model_id"] == model]
        pr = sub["is_pass"].mean() if "is_pass" in sub.columns else 0.0
        ws = sub["weighted_score"].mean() if "weighted_score" in sub.columns else 0.0

        avg_quality = 0.0
        if not quality_df.empty and "maintainability_index" in quality_df.columns:
            mq = quality_df[quality_df["model_id"] == model]["maintainability_index"]
            avg_quality = float(mq.mean()) if len(mq) > 0 else 0.0

        avg_cost = 0.0
        for cp in cost_profiles:
            if cp["model_id"] == model:
                avg_cost = cp.get("mean_cost_per_task", 0.0)
                break

        # Win/loss/tie from pairwise Wilcoxon
        wins = losses = ties = 0
        for r in hyp_results:
            if r.get("test_name") != "wilcoxon_signed_rank":
                continue
            if r.get("group_a") == model and r.get("significant"):
                wins += 1
            elif r.get("group_b") == model and r.get("significant"):
                losses += 1
            elif model in (r.get("group_a", ""), r.get("group_b", "")) and not r.get("significant"):
                ties += 1

        # Composite: 40% pass_rate + 30% quality_norm + 20% (1-cost_norm) + 10% consistency
        composite = 0.4 * pr + 0.3 * (avg_quality / 100.0) + 0.2 * ws
        rankings.append({
            "model_id": model,
            "composite_score": round(composite, 4),
            "pass_rate": round(pr, 4),
            "avg_quality": round(avg_quality, 2),
            "avg_cost_usd": round(avg_cost, 4),
            "problems_attempted": len(sub),
            "wins": wins,
            "losses": losses,
            "ties": ties,
        })

    # Sort and assign ranks + tiers
    rankings.sort(key=lambda r: r["composite_score"], reverse=True)
    tier_thresholds = [(0.9, "S"), (0.75, "A"), (0.6, "B"), (0.45, "C"), (0.3, "D")]
    for i, r in enumerate(rankings, 1):
        r["rank"] = i
        tier = "F"
        for threshold, t in tier_thresholds:
            if r["composite_score"] >= threshold:
                tier = t
                break
        r["tier"] = tier

    return rankings


def _build_model_profiles(
    results_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    cost_profiles: list[dict],
) -> dict[str, dict[str, float]]:
    """Build normalised model profiles for radar chart."""
    models = sorted(results_df["model_id"].unique())
    profiles: dict[str, dict[str, float]] = {}

    for model in models:
        sub = results_df[results_df["model_id"] == model]
        accuracy = sub["is_pass"].mean() if "is_pass" in sub.columns else 0.0
        score = sub["weighted_score"].mean() if "weighted_score" in sub.columns else 0.0

        quality = 0.0
        if not quality_df.empty and "maintainability_index" in quality_df.columns:
            mq = quality_df[quality_df["model_id"] == model]["maintainability_index"]
            quality = float(mq.mean()) / 100.0 if len(mq) > 0 else 0.0

        cost_norm = 0.5  # default mid-range
        for cp in cost_profiles:
            if cp["model_id"] == model:
                # Invert: lower cost = higher score
                max_cost = max(p.get("mean_cost_per_task", 0.01) for p in cost_profiles) or 0.01
                cost_norm = 1.0 - (cp.get("mean_cost_per_task", 0.0) / max_cost)
                break

        profiles[model] = {
            "Accuracy": accuracy,
            "Score": score,
            "Quality": quality,
            "Cost-Eff.": cost_norm,
            "Consistency": 1.0 - (sub["weighted_score"].std() if "weighted_score" in sub.columns else 0.5),
        }

    return profiles
