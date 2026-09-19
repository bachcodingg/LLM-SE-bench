"""
stats/hypothesis.py — Hypothesis testing for llm-se-bench.

Implements non-parametric tests appropriate for LLM benchmark data:

* **Wilcoxon signed-rank** — pairwise matched-sample comparison.
* **Friedman test** — omnibus test for ≥3 related groups.
* **Nemenyi post-hoc** — pairwise follow-up after a significant Friedman.
* **Cochran's Q** — for binary (pass/fail) matched data.

Every test returns a :class:`HypothesisTestResult` that maps directly to
the ``TestResult`` contract (see contracts.py).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ── result container ─────────────────────────────────────────────────

@dataclass
class HypothesisTestResult:
    """Outcome of a single hypothesis test."""

    test_name: str
    metric_name: str
    group_a: str
    group_b: str = ""
    statistic: float = 0.0
    p_value: float = 1.0
    effect_size: float = 0.0
    effect_size_label: str = ""
    significant: bool = False
    alpha: float = 0.05
    correction_method: str = "none"
    n_samples: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "metric_name": self.metric_name,
            "group_a": self.group_a,
            "group_b": self.group_b,
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6),
            "effect_size": round(self.effect_size, 6),
            "effect_size_label": self.effect_size_label,
            "significant": self.significant,
            "alpha": self.alpha,
            "correction_method": self.correction_method,
            "n_samples": self.n_samples,
            "notes": self.notes,
        }


# ── multiple-testing correction ──────────────────────────────────────

def _holm_bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> list[tuple[float, bool]]:
    """Apply Holm–Bonferroni step-down correction.

    Returns
    -------
    list[tuple[float, bool]]
        Adjusted p-values and significance flags in the *original* order.
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    sig = [False] * m
    cummax = 0.0
    for rank, (orig_idx, pval) in enumerate(indexed):
        adj = pval * (m - rank)
        adj = min(adj, 1.0)
        cummax = max(cummax, adj)
        adjusted[orig_idx] = cummax
        sig[orig_idx] = cummax < alpha
    return list(zip(adjusted, sig))


# ── engine ───────────────────────────────────────────────────────────

class HypothesisEngine:
    """Run hypothesis tests on benchmark data.

    Parameters
    ----------
    alpha : float
        Significance level (default 0.05).
    correction : str
        Multiple-testing correction: ``"holm"`` (default) or ``"bonferroni"``.
    min_samples : int
        Minimum sample size to attempt a test (default 5).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        correction: str = "holm",
        min_samples: int = 5,
    ) -> None:
        self.alpha = alpha
        self.correction = correction
        self.min_samples = min_samples

    # ── Wilcoxon signed-rank (pairwise) ──────────────────────────────

    def wilcoxon_signed_rank(
        self,
        x: np.ndarray,
        y: np.ndarray,
        metric_name: str = "",
        group_a: str = "",
        group_b: str = "",
    ) -> HypothesisTestResult:
        """Wilcoxon signed-rank test on two matched samples.

        Parameters
        ----------
        x, y : array-like
            Paired observations (same length, same problem order).

        Returns
        -------
        HypothesisTestResult
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # Drop pairs where either is NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        n = len(x)

        result = HypothesisTestResult(
            test_name="wilcoxon_signed_rank",
            metric_name=metric_name,
            group_a=group_a,
            group_b=group_b,
            alpha=self.alpha,
            n_samples=n,
        )

        if n < self.min_samples:
            result.notes = f"Insufficient samples ({n} < {self.min_samples})"
            return result

        diff = x - y
        if np.all(diff == 0):
            result.notes = "All differences are zero"
            result.p_value = 1.0
            return result

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stat, pval = sp_stats.wilcoxon(x, y, alternative="two-sided")
            except ValueError as exc:
                result.notes = f"Test failed: {exc}"
                return result

        result.statistic = float(stat)
        result.p_value = float(pval)
        result.significant = pval < self.alpha
        return result

    # ── all-pairs Wilcoxon with correction ───────────────────────────

    def pairwise_wilcoxon(
        self,
        df: pd.DataFrame,
        metric_col: str,
        model_col: str = "model_id",
        problem_col: str = "problem_id",
    ) -> list[HypothesisTestResult]:
        """Run Wilcoxon signed-rank on every pair of models.

        The DataFrame is pivoted so that each problem is a row and each
        model is a column, producing matched samples.

        Parameters
        ----------
        df : pd.DataFrame
        metric_col : str
            Column with the numeric metric.
        model_col, problem_col : str
            Columns identifying models and problems.

        Returns
        -------
        list[HypothesisTestResult]
            One per pair, with Holm-corrected p-values.
        """
        pivot = (
            df.groupby([problem_col, model_col])[metric_col]
            .mean()
            .unstack(model_col)
            .dropna()
        )
        models = sorted(pivot.columns)
        if len(models) < 2:
            logger.warning("Need ≥2 models for pairwise tests, found %d", len(models))
            return []

        raw_results: list[HypothesisTestResult] = []
        for a, b in combinations(models, 2):
            r = self.wilcoxon_signed_rank(
                pivot[a].values,
                pivot[b].values,
                metric_name=metric_col,
                group_a=a,
                group_b=b,
            )
            raw_results.append(r)

        # Apply multiple-testing correction
        self._apply_correction(raw_results)
        return raw_results

    # ── Friedman test (omnibus) ──────────────────────────────────────

    def friedman_test(
        self,
        df: pd.DataFrame,
        metric_col: str,
        model_col: str = "model_id",
        problem_col: str = "problem_id",
    ) -> HypothesisTestResult:
        """Friedman test for ≥3 matched groups.

        Parameters
        ----------
        df : pd.DataFrame
        metric_col : str
        model_col, problem_col : str

        Returns
        -------
        HypothesisTestResult
        """
        pivot = (
            df.groupby([problem_col, model_col])[metric_col]
            .mean()
            .unstack(model_col)
            .dropna()
        )
        models = sorted(pivot.columns)
        n = len(pivot)

        result = HypothesisTestResult(
            test_name="friedman",
            metric_name=metric_col,
            group_a=",".join(models),
            alpha=self.alpha,
            n_samples=n,
        )

        if len(models) < 3:
            result.notes = f"Need ≥3 groups, found {len(models)}"
            return result
        if n < self.min_samples:
            result.notes = f"Insufficient subjects ({n} < {self.min_samples})"
            return result

        groups = [pivot[m].values for m in models]
        try:
            stat, pval = sp_stats.friedmanchisquare(*groups)
        except ValueError as exc:
            result.notes = f"Test failed: {exc}"
            return result

        result.statistic = float(stat)
        result.p_value = float(pval)
        result.significant = pval < self.alpha

        # Kendall's W as effect size for Friedman
        k = len(models)
        w = (12 * stat) / (n * k * (k ** 2 - 1)) if n > 0 and k > 1 else 0.0
        result.effect_size = float(w)
        result.effect_size_label = _kendall_w_label(w)
        return result

    # ── Nemenyi post-hoc ─────────────────────────────────────────────

    def nemenyi_posthoc(
        self,
        df: pd.DataFrame,
        metric_col: str,
        model_col: str = "model_id",
        problem_col: str = "problem_id",
    ) -> list[HypothesisTestResult]:
        """Nemenyi post-hoc test after a significant Friedman.

        Uses the mean-rank approach and Studentized range distribution
        to obtain p-values.

        Parameters
        ----------
        df : pd.DataFrame
        metric_col : str
        model_col, problem_col : str

        Returns
        -------
        list[HypothesisTestResult]
        """
        pivot = (
            df.groupby([problem_col, model_col])[metric_col]
            .mean()
            .unstack(model_col)
            .dropna()
        )
        models = sorted(pivot.columns)
        n = len(pivot)
        k = len(models)

        if k < 3 or n < self.min_samples:
            return []

        # Rank within each problem (row), average ties
        ranks = pivot.rank(axis=1, method="average")
        mean_ranks = ranks.mean(axis=0)

        # Critical difference (CD) for Nemenyi
        # q_alpha from Studentized range distribution
        from scipy.stats import studentized_range

        q_alpha = studentized_range.ppf(1 - self.alpha, k, np.inf)
        cd = q_alpha * np.sqrt(k * (k + 1) / (12.0 * n))

        results: list[HypothesisTestResult] = []
        for a, b in combinations(models, 2):
            diff = abs(mean_ranks[a] - mean_ranks[b])
            sig = diff > cd

            # Approximate p-value via Studentized range
            q_stat = diff / np.sqrt(k * (k + 1) / (12.0 * n))
            pval = float(studentized_range.sf(q_stat, k, np.inf))
            pval = min(pval, 1.0)

            results.append(
                HypothesisTestResult(
                    test_name="nemenyi",
                    metric_name=metric_col,
                    group_a=a,
                    group_b=b,
                    statistic=float(diff),
                    p_value=pval,
                    significant=sig,
                    alpha=self.alpha,
                    n_samples=n,
                    notes=f"CD={cd:.4f}, rank_a={mean_ranks[a]:.3f}, rank_b={mean_ranks[b]:.3f}",
                )
            )
        return results

    # ── Cochran's Q (binary outcomes) ────────────────────────────────

    def cochrans_q(
        self,
        df: pd.DataFrame,
        binary_col: str = "is_pass",
        model_col: str = "model_id",
        problem_col: str = "problem_id",
    ) -> HypothesisTestResult:
        """Cochran's Q test for binary matched data.

        Parameters
        ----------
        df : pd.DataFrame
        binary_col : str
            Boolean column (True/False or 1/0).
        model_col, problem_col : str

        Returns
        -------
        HypothesisTestResult
        """
        pivot = (
            df.groupby([problem_col, model_col])[binary_col]
            .max()  # use max so any-pass counts
            .astype(int)
            .unstack(model_col)
            .dropna()
        )
        models = sorted(pivot.columns)
        n = len(pivot)
        k = len(models)

        result = HypothesisTestResult(
            test_name="cochrans_q",
            metric_name=binary_col,
            group_a=",".join(models),
            alpha=self.alpha,
            n_samples=n,
        )

        if k < 2:
            result.notes = "Need ≥2 groups"
            return result
        if n < self.min_samples:
            result.notes = f"Insufficient subjects ({n} < {self.min_samples})"
            return result

        X = pivot.values.astype(float)
        row_sums = X.sum(axis=1)
        col_sums = X.sum(axis=0)
        N = X.sum()

        if N == 0 or np.var(col_sums) == 0:
            result.notes = "All-zero or constant columns"
            return result

        numerator = (k - 1) * (k * np.sum(col_sums ** 2) - N ** 2)
        denominator = k * N - np.sum(row_sums ** 2)

        if denominator == 0:
            result.notes = "Denominator is zero (degenerate data)"
            return result

        Q = numerator / denominator
        pval = float(sp_stats.chi2.sf(Q, df=k - 1))

        result.statistic = float(Q)
        result.p_value = pval
        result.significant = pval < self.alpha
        return result

    # ── multiple-testing correction ──────────────────────────────────

    def _apply_correction(self, results: list[HypothesisTestResult]) -> None:
        """In-place Holm or Bonferroni correction on a list of results."""
        if not results:
            return
        p_values = [r.p_value for r in results]

        if self.correction == "bonferroni":
            m = len(p_values)
            for r in results:
                r.p_value = min(r.p_value * m, 1.0)
                r.significant = r.p_value < self.alpha
                r.correction_method = "bonferroni"
        else:  # holm (default)
            adjusted = _holm_bonferroni(p_values, self.alpha)
            for r, (adj_p, sig) in zip(results, adjusted):
                r.p_value = adj_p
                r.significant = sig
                r.correction_method = "holm"

    # ── convenience: run all tests on a metric ───────────────────────

    def run_all(
        self,
        df: pd.DataFrame,
        metric_col: str,
        model_col: str = "model_id",
        problem_col: str = "problem_id",
    ) -> dict[str, list[HypothesisTestResult] | HypothesisTestResult]:
        """Run Friedman + Nemenyi + pairwise Wilcoxon on one metric.

        Returns
        -------
        dict
            Keys: ``"friedman"``, ``"nemenyi"``, ``"wilcoxon"``.
        """
        friedman = self.friedman_test(df, metric_col, model_col, problem_col)
        nemenyi = (
            self.nemenyi_posthoc(df, metric_col, model_col, problem_col)
            if friedman.significant
            else []
        )
        wilcoxon = self.pairwise_wilcoxon(df, metric_col, model_col, problem_col)
        return {
            "friedman": friedman,
            "nemenyi": nemenyi,
            "wilcoxon": wilcoxon,
        }


# ── helpers ──────────────────────────────────────────────────────────

def _kendall_w_label(w: float) -> str:
    """Interpret Kendall's W concordance coefficient."""
    if w < 0.1:
        return "negligible"
    elif w < 0.3:
        return "weak"
    elif w < 0.5:
        return "moderate"
    elif w < 0.7:
        return "strong"
    return "very_strong"
