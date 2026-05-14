"""
stats/effect_size.py — Effect-size computation for llm-se-bench.

Implements:
    * **Cliff's delta** — non-parametric ordinal effect size.
    * **Cohen's d** — standardised mean difference.
    * **Rank-biserial correlation** — effect size for Wilcoxon.
    * **Vargha–Delaney A** — probability of superiority.

All functions accept raw arrays; the :class:`EffectSizeCalculator` adds
DataFrame-level convenience and automatic labelling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── result container ─────────────────────────────────────────────────

@dataclass
class EffectSizeResult:
    """Container for a single effect-size computation."""

    measure: str  # e.g. "cliffs_delta", "cohens_d"
    metric_name: str
    group_a: str
    group_b: str
    value: float = 0.0
    label: str = "negligible"
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    n_a: int = 0
    n_b: int = 0

    def to_dict(self) -> dict:
        return {
            "measure": self.measure,
            "metric_name": self.metric_name,
            "group_a": self.group_a,
            "group_b": self.group_b,
            "value": round(self.value, 6),
            "label": self.label,
            "ci_lower": round(self.ci_lower, 6),
            "ci_upper": round(self.ci_upper, 6),
            "n_a": self.n_a,
            "n_b": self.n_b,
        }


# ── Cliff's delta ────────────────────────────────────────────────────

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Compute Cliff's delta between two independent samples.

    Returns
    -------
    delta : float
        Value in [-1, 1].
    label : str
        One of ``"negligible"``, ``"small"``, ``"medium"``, ``"large"``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0, "negligible"

    # Vectorised pairwise comparison
    # delta = (# concordant - # discordant) / (n_x * n_y)
    more = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    delta = (more - less) / (n_x * n_y)

    label = _cliffs_delta_label(abs(delta))
    return float(delta), label


def _cliffs_delta_label(abs_d: float) -> str:
    """Interpret |Cliff's delta| per Romano et al. (2006)."""
    if abs_d < 0.147:
        return "negligible"
    elif abs_d < 0.33:
        return "small"
    elif abs_d < 0.474:
        return "medium"
    return "large"


# ── Cohen's d ────────────────────────────────────────────────────────

def cohens_d(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Compute Cohen's d (pooled standard deviation).

    Returns
    -------
    d : float
    label : str
        ``"negligible"`` / ``"small"`` / ``"medium"`` / ``"large"``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    n_x, n_y = len(x), len(y)
    if n_x < 2 or n_y < 2:
        return 0.0, "negligible"

    mean_diff = np.mean(x) - np.mean(y)
    # Pooled SD
    pooled_var = ((n_x - 1) * np.var(x, ddof=1) + (n_y - 1) * np.var(y, ddof=1)) / (
        n_x + n_y - 2
    )
    pooled_sd = np.sqrt(pooled_var)

    if pooled_sd == 0:
        return 0.0, "negligible"

    d = mean_diff / pooled_sd
    label = _cohens_d_label(abs(d))
    return float(d), label


def _cohens_d_label(abs_d: float) -> str:
    """Interpret |Cohen's d| per Cohen (1988)."""
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    return "large"


# ── Rank-biserial correlation ────────────────────────────────────────

def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation (matched-pairs version).

    This is the effect size r for the Wilcoxon signed-rank test,
    computed as r = 1 - (2T) / (n(n+1)/2) where T is the smaller
    of the positive/negative rank sums.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return 0.0

    abs_diff = np.abs(diff)
    from scipy.stats import rankdata

    ranks = rankdata(abs_diff)

    pos_sum = np.sum(ranks[diff > 0])
    neg_sum = np.sum(ranks[diff < 0])

    T = min(pos_sum, neg_sum)
    r = 1.0 - (2.0 * T) / (n * (n + 1) / 2.0)
    return float(r)


# ── Vargha–Delaney A ────────────────────────────────────────────────

def vargha_delaney_a(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Vargha–Delaney A: probability that x > y.

    Returns
    -------
    A : float
        Value in [0, 1]; 0.5 = no effect.
    label : str
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.5, "negligible"

    more = np.sum(x[:, None] > y[None, :])
    equal = np.sum(x[:, None] == y[None, :])
    A = (more + 0.5 * equal) / (n_x * n_y)

    label = _vd_label(A)
    return float(A), label


def _vd_label(A: float) -> str:
    """Interpret Vargha–Delaney A."""
    diff = abs(A - 0.5)
    if diff < 0.06:
        return "negligible"
    elif diff < 0.14:
        return "small"
    elif diff < 0.21:
        return "medium"
    return "large"


# ── DataFrame-level calculator ───────────────────────────────────────

class EffectSizeCalculator:
    """Compute effect sizes between models across a DataFrame.

    Parameters
    ----------
    method : str
        ``"cliffs_delta"`` (default) or ``"cohens_d"``.
    bootstrap_ci : bool
        If True, compute bootstrap 95 % CIs for the effect size.
    n_bootstrap : int
        Number of bootstrap samples (default 2000).
    """

    def __init__(
        self,
        method: str = "cliffs_delta",
        bootstrap_ci: bool = True,
        n_bootstrap: int = 2000,
    ) -> None:
        self.method = method
        self.bootstrap_ci = bootstrap_ci
        self.n_bootstrap = n_bootstrap

    def compute_pairwise(
        self,
        df: pd.DataFrame,
        metric_col: str,
        model_col: str = "model_id",
    ) -> list[EffectSizeResult]:
        """Compute effect sizes for every pair of models.

        Parameters
        ----------
        df : pd.DataFrame
        metric_col : str
        model_col : str

        Returns
        -------
        list[EffectSizeResult]
        """
        models = sorted(df[model_col].unique())
        results: list[EffectSizeResult] = []

        for a, b in combinations(models, 2):
            x = df.loc[df[model_col] == a, metric_col].dropna().values
            y = df.loc[df[model_col] == b, metric_col].dropna().values

            if self.method == "cohens_d":
                value, label = cohens_d(x, y)
            else:
                value, label = cliffs_delta(x, y)

            ci_lo, ci_hi = 0.0, 0.0
            if self.bootstrap_ci and len(x) > 1 and len(y) > 1:
                ci_lo, ci_hi = self._bootstrap_ci(x, y)

            results.append(
                EffectSizeResult(
                    measure=self.method,
                    metric_name=metric_col,
                    group_a=a,
                    group_b=b,
                    value=value,
                    label=label,
                    ci_lower=ci_lo,
                    ci_upper=ci_hi,
                    n_a=len(x),
                    n_b=len(y),
                )
            )
        return results

    def _bootstrap_ci(
        self,
        x: np.ndarray,
        y: np.ndarray,
        ci: float = 0.95,
    ) -> tuple[float, float]:
        """Percentile bootstrap CI for the chosen effect size."""
        rng = np.random.default_rng(42)
        boot_values: list[float] = []
        func = cohens_d if self.method == "cohens_d" else cliffs_delta

        for _ in range(self.n_bootstrap):
            xi = rng.choice(x, size=len(x), replace=True)
            yi = rng.choice(y, size=len(y), replace=True)
            val, _ = func(xi, yi)
            boot_values.append(val)

        lower_pct = (1 - ci) / 2 * 100
        upper_pct = (1 + ci) / 2 * 100
        lo = float(np.percentile(boot_values, lower_pct))
        hi = float(np.percentile(boot_values, upper_pct))
        return lo, hi
