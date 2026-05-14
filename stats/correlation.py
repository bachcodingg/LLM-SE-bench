"""
stats/correlation.py — Correlation analysis for llm-se-bench.

Computes Spearman rank correlations between code-quality metrics and
functional performance, plus optional Kendall's tau and point-biserial
for binary outcomes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result of a single correlation computation."""

    method: str  # "spearman", "kendall", "point_biserial"
    variable_a: str
    variable_b: str
    coefficient: float = 0.0
    p_value: float = 1.0
    n: int = 0
    significant: bool = False
    alpha: float = 0.05

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "variable_a": self.variable_a,
            "variable_b": self.variable_b,
            "coefficient": round(self.coefficient, 6),
            "p_value": round(self.p_value, 6),
            "n": self.n,
            "significant": self.significant,
            "alpha": self.alpha,
        }


class CorrelationAnalyzer:
    """Compute correlations between metrics and performance.

    Parameters
    ----------
    method : str
        Default correlation method: ``"spearman"`` (default) or ``"kendall"``.
    alpha : float
        Significance level (default 0.05).
    min_n : int
        Minimum sample size to attempt a correlation (default 5).
    """

    def __init__(
        self,
        method: str = "spearman",
        alpha: float = 0.05,
        min_n: int = 5,
    ) -> None:
        self.method = method
        self.alpha = alpha
        self.min_n = min_n

    # ── single pair ──────────────────────────────────────────────────

    def correlate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        var_a: str = "",
        var_b: str = "",
        method: str | None = None,
    ) -> CorrelationResult:
        """Compute correlation between two arrays.

        Parameters
        ----------
        x, y : array-like
            Numeric data of the same length.
        var_a, var_b : str
            Variable names for labelling.
        method : str, optional
            Override the instance default.

        Returns
        -------
        CorrelationResult
        """
        method = method or self.method
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # Drop NaN pairs
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        n = len(x)

        result = CorrelationResult(
            method=method,
            variable_a=var_a,
            variable_b=var_b,
            n=n,
            alpha=self.alpha,
        )

        if n < self.min_n:
            return result

        if method == "kendall":
            coef, pval = sp_stats.kendalltau(x, y)
        elif method == "point_biserial":
            coef, pval = sp_stats.pointbiserialr(x, y)
        else:  # spearman
            coef, pval = sp_stats.spearmanr(x, y)

        result.coefficient = float(coef) if not np.isnan(coef) else 0.0
        result.p_value = float(pval) if not np.isnan(pval) else 1.0
        result.significant = result.p_value < self.alpha
        return result

    # ── correlation matrix ───────────────────────────────────────────

    def correlation_matrix(
        self,
        df: pd.DataFrame,
        columns: Sequence[str],
        method: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute pairwise correlations and p-values for selected columns.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Correlation matrix and p-value matrix, both square and
            symmetric with column/index = *columns*.
        """
        method = method or self.method
        n = len(columns)
        corr_mat = np.eye(n)
        pval_mat = np.zeros((n, n))

        for i, j in combinations(range(n), 2):
            r = self.correlate(
                df[columns[i]].values,
                df[columns[j]].values,
                var_a=columns[i],
                var_b=columns[j],
                method=method,
            )
            corr_mat[i, j] = corr_mat[j, i] = r.coefficient
            pval_mat[i, j] = pval_mat[j, i] = r.p_value

        corr_df = pd.DataFrame(corr_mat, index=columns, columns=columns)
        pval_df = pd.DataFrame(pval_mat, index=columns, columns=columns)
        return corr_df, pval_df

    # ── metrics vs. performance ──────────────────────────────────────

    def metrics_vs_performance(
        self,
        df: pd.DataFrame,
        metric_cols: Sequence[str],
        performance_col: str = "weighted_score",
        binary_col: str | None = "is_pass",
    ) -> list[CorrelationResult]:
        """Correlate each quality metric against performance.

        Computes Spearman against the continuous performance column, plus
        point-biserial against the binary column if provided.

        Parameters
        ----------
        df : pd.DataFrame
        metric_cols : sequence of str
            Quality metric column names.
        performance_col : str
        binary_col : str | None

        Returns
        -------
        list[CorrelationResult]
        """
        results: list[CorrelationResult] = []

        for col in metric_cols:
            if col not in df.columns:
                logger.warning("Column '%s' not found, skipping", col)
                continue

            # Spearman vs continuous
            if performance_col in df.columns:
                r = self.correlate(
                    df[col].values,
                    df[performance_col].values,
                    var_a=col,
                    var_b=performance_col,
                    method="spearman",
                )
                results.append(r)

            # Point-biserial vs binary
            if binary_col and binary_col in df.columns:
                r = self.correlate(
                    df[col].values,
                    df[binary_col].astype(float).values,
                    var_a=col,
                    var_b=binary_col,
                    method="point_biserial",
                )
                results.append(r)

        return results

    # ── per-model correlations ───────────────────────────────────────

    def per_model_correlations(
        self,
        df: pd.DataFrame,
        metric_cols: Sequence[str],
        performance_col: str = "weighted_score",
        model_col: str = "model_id",
    ) -> dict[str, list[CorrelationResult]]:
        """Run metrics-vs-performance correlations per model.

        Returns
        -------
        dict[str, list[CorrelationResult]]
            Keyed by model_id.
        """
        out: dict[str, list[CorrelationResult]] = {}
        for model_id, grp in df.groupby(model_col):
            out[str(model_id)] = self.metrics_vs_performance(
                grp, metric_cols, performance_col
            )
        return out
