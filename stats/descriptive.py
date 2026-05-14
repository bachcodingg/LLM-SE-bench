"""
stats/descriptive.py — Descriptive statistics for llm-se-bench.

Computes mean, median, standard deviation, IQR, min, max, skewness,
kurtosis, and percentile profiles for any numeric column, grouped by
model and/or dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class DescriptiveStats:
    """Container for descriptive statistics of a single metric slice."""

    metric_name: str
    model_id: str
    dataset: str = ""
    n: int = 0
    mean: float = 0.0
    std_dev: float = 0.0
    median: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    q1: float = 0.0
    q3: float = 0.0
    iqr: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    ci_lower_95: float = 0.0
    ci_upper_95: float = 0.0

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "model_id": self.model_id,
            "dataset": self.dataset,
            "n": self.n,
            "mean": round(self.mean, 6),
            "std_dev": round(self.std_dev, 6),
            "median": round(self.median, 6),
            "min": round(self.min_val, 6),
            "max": round(self.max_val, 6),
            "q1": round(self.q1, 6),
            "q3": round(self.q3, 6),
            "iqr": round(self.iqr, 6),
            "skewness": round(self.skewness, 6),
            "kurtosis": round(self.kurtosis, 6),
            "ci_lower_95": round(self.ci_lower_95, 6),
            "ci_upper_95": round(self.ci_upper_95, 6),
        }


class DescriptiveAnalyzer:
    """Compute descriptive statistics for benchmark metrics.

    Parameters
    ----------
    ci_method : str
        ``"t"`` for Student-*t* interval (default), ``"bootstrap"`` for
        percentile bootstrap (delegates to :class:`BootstrapCI`).
    """

    def __init__(self, ci_method: str = "t") -> None:
        self.ci_method = ci_method

    # ── core computation ─────────────────────────────────────────────

    @staticmethod
    def compute(
        values: np.ndarray | Sequence[float],
        metric_name: str = "",
        model_id: str = "",
        dataset: str = "",
    ) -> DescriptiveStats:
        """Compute descriptive statistics for a 1-D array of values.

        Parameters
        ----------
        values : array-like
            Numeric sample.  NaN values are silently dropped.
        metric_name, model_id, dataset : str
            Labels stored on the returned :class:`DescriptiveStats`.

        Returns
        -------
        DescriptiveStats
        """
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        n = len(arr)

        if n == 0:
            return DescriptiveStats(
                metric_name=metric_name,
                model_id=model_id,
                dataset=dataset,
            )

        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        med = float(np.median(arr))
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))

        # 95 % CI via Student-t
        if n > 1:
            se = std / np.sqrt(n)
            t_crit = sp_stats.t.ppf(0.975, df=n - 1)
            ci_lo = mean - t_crit * se
            ci_hi = mean + t_crit * se
        else:
            ci_lo = ci_hi = mean

        # Skewness / kurtosis (need n ≥ 3 for meaningful results)
        if n >= 3:
            skew = float(sp_stats.skew(arr, bias=False))
            kurt = float(sp_stats.kurtosis(arr, bias=False))
        else:
            skew = kurt = 0.0

        return DescriptiveStats(
            metric_name=metric_name,
            model_id=model_id,
            dataset=dataset,
            n=n,
            mean=mean,
            std_dev=std,
            median=med,
            min_val=float(np.min(arr)),
            max_val=float(np.max(arr)),
            q1=q1,
            q3=q3,
            iqr=q3 - q1,
            skewness=skew,
            kurtosis=kurt,
            ci_lower_95=ci_lo,
            ci_upper_95=ci_hi,
        )

    # ── DataFrame-level helpers ──────────────────────────────────────

    def summarise_metric(
        self,
        df: pd.DataFrame,
        metric_col: str,
        group_by: str | list[str] = "model_id",
    ) -> list[DescriptiveStats]:
        """Compute descriptive stats for *metric_col* grouped by *group_by*.

        Parameters
        ----------
        df : pd.DataFrame
        metric_col : str
            Column containing the numeric values.
        group_by : str | list[str]
            Grouping column(s), typically ``"model_id"`` and/or ``"dataset"``.

        Returns
        -------
        list[DescriptiveStats]
        """
        if isinstance(group_by, str):
            group_by = [group_by]

        results: list[DescriptiveStats] = []
        for keys, grp in df.groupby(group_by, sort=True):
            if isinstance(keys, str):
                keys = (keys,)
            labels = dict(zip(group_by, keys))
            vals = grp[metric_col].dropna().values
            ds = self.compute(
                vals,
                metric_name=metric_col,
                model_id=labels.get("model_id", ""),
                dataset=labels.get("dataset", ""),
            )
            results.append(ds)
        return results

    def summarise_all_metrics(
        self,
        df: pd.DataFrame,
        metric_cols: Sequence[str],
        group_by: str | list[str] = "model_id",
    ) -> list[DescriptiveStats]:
        """Compute descriptive stats for multiple metrics at once."""
        all_results: list[DescriptiveStats] = []
        for col in metric_cols:
            if col not in df.columns:
                logger.warning("Column '%s' not in DataFrame — skipped", col)
                continue
            all_results.extend(self.summarise_metric(df, col, group_by))
        return all_results

    # ── pass-rate convenience ────────────────────────────────────────

    def pass_at_k(
        self,
        df: pd.DataFrame,
        k: int = 1,
        group_by: str | list[str] = "model_id",
    ) -> pd.DataFrame:
        """Compute pass@k per group using the unbiased estimator.

        Uses the formula from Chen et al. (2021)::

            pass@k = 1 − C(n − c, k) / C(n, k)

        where *n* = total attempts, *c* = correct attempts.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``problem_id``, ``is_pass`` (bool), and group columns.
        k : int
        group_by : str | list[str]

        Returns
        -------
        pd.DataFrame
            Columns: group keys + ``pass_at_k``.
        """
        if isinstance(group_by, str):
            group_by = [group_by]

        agg = (
            df.groupby(group_by + ["problem_id"])
            .agg(n=("is_pass", "count"), c=("is_pass", "sum"))
            .reset_index()
        )
        agg["pass_at_k"] = agg.apply(
            lambda r: _pass_at_k_single(int(r["n"]), int(r["c"]), k), axis=1
        )
        return (
            agg.groupby(group_by)["pass_at_k"]
            .mean()
            .reset_index()
            .rename(columns={"pass_at_k": f"pass@{k}"})
        )

    # ── percentile profile ───────────────────────────────────────────

    @staticmethod
    def percentile_profile(
        values: np.ndarray | Sequence[float],
        percentiles: Sequence[int] = (5, 10, 25, 50, 75, 90, 95),
    ) -> dict[str, float]:
        """Return a dict mapping ``"pN"`` → value for each percentile."""
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        return {f"p{p}": float(np.percentile(arr, p)) for p in percentiles}


# ── private helpers ──────────────────────────────────────────────────

def _pass_at_k_single(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator for one problem.

    Handles edge cases where k > n gracefully (returns 1.0 if c > 0).
    """
    if n - c < k:
        return 1.0
    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k)
