"""
stats/confidence.py — Bootstrap confidence interval estimation.

Provides percentile, BCa (bias-corrected and accelerated), and basic
bootstrap CIs for arbitrary statistics.  The default is 10,000 resamples
with BCa correction, per project plan §4.3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class CIResult:
    """Bootstrap confidence-interval result."""

    statistic_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float = 0.95
    method: str = "percentile"
    n_bootstrap: int = 10_000
    n_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "statistic": self.statistic_name,
            "point_estimate": round(self.point_estimate, 6),
            "ci_lower": round(self.ci_lower, 6),
            "ci_upper": round(self.ci_upper, 6),
            "ci_level": self.ci_level,
            "method": self.method,
            "n_bootstrap": self.n_bootstrap,
            "n_samples": self.n_samples,
        }


class BootstrapCI:
    """Bootstrap confidence-interval engine.

    Parameters
    ----------
    n_bootstrap : int
        Number of resamples (default 10 000).
    ci_level : float
        Confidence level (default 0.95).
    method : str
        ``"percentile"`` (default) or ``"bca"`` (bias-corrected accelerated).
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_bootstrap: int = 10_000,
        ci_level: float = 0.95,
        method: str = "percentile",
        seed: int | None = 42,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.method = method
        self.rng = np.random.default_rng(seed)

    # ── public API ───────────────────────────────────────────────────

    def compute(
        self,
        data: np.ndarray | Sequence[float],
        statistic_fn: Callable[[np.ndarray], float] = np.mean,
        statistic_name: str = "mean",
    ) -> CIResult:
        """Compute a bootstrap CI for *statistic_fn* applied to *data*.

        Parameters
        ----------
        data : array-like
            1-D numeric sample.
        statistic_fn : callable
            Maps array → scalar.  Default: ``np.mean``.
        statistic_name : str
            Label for the statistic being estimated.

        Returns
        -------
        CIResult
        """
        arr = np.asarray(data, dtype=float)
        arr = arr[~np.isnan(arr)]
        n = len(arr)

        if n == 0:
            return CIResult(
                statistic_name=statistic_name,
                point_estimate=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                ci_level=self.ci_level,
                method=self.method,
                n_bootstrap=0,
                n_samples=0,
            )

        point = float(statistic_fn(arr))

        if n == 1:
            return CIResult(
                statistic_name=statistic_name,
                point_estimate=point,
                ci_lower=point,
                ci_upper=point,
                ci_level=self.ci_level,
                method=self.method,
                n_bootstrap=0,
                n_samples=1,
            )

        # Draw bootstrap samples
        boot_stats = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            sample = self.rng.choice(arr, size=n, replace=True)
            boot_stats[i] = statistic_fn(sample)

        if self.method == "bca":
            lo, hi = self._bca(arr, boot_stats, statistic_fn, point)
        else:
            lo, hi = self._percentile(boot_stats)

        return CIResult(
            statistic_name=statistic_name,
            point_estimate=point,
            ci_lower=float(lo),
            ci_upper=float(hi),
            ci_level=self.ci_level,
            method=self.method,
            n_bootstrap=self.n_bootstrap,
            n_samples=n,
        )

    def ci_mean(self, data: np.ndarray | Sequence[float]) -> CIResult:
        """Shortcut for bootstrap CI of the mean."""
        return self.compute(data, np.mean, "mean")

    def ci_median(self, data: np.ndarray | Sequence[float]) -> CIResult:
        """Shortcut for bootstrap CI of the median."""
        return self.compute(data, np.median, "median")

    def ci_proportion(self, data: np.ndarray | Sequence[float]) -> CIResult:
        """Bootstrap CI for a proportion (binary data)."""
        return self.compute(data, np.mean, "proportion")

    def ci_difference(
        self,
        x: np.ndarray | Sequence[float],
        y: np.ndarray | Sequence[float],
        statistic_fn: Callable[[np.ndarray], float] = np.mean,
        statistic_name: str = "mean_difference",
    ) -> CIResult:
        """Bootstrap CI for the difference in a statistic between two groups.

        Parameters
        ----------
        x, y : array-like
            Two independent samples.
        statistic_fn : callable
            Applied to each bootstrap sample independently.
        statistic_name : str

        Returns
        -------
        CIResult
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        x_arr = x_arr[~np.isnan(x_arr)]
        y_arr = y_arr[~np.isnan(y_arr)]

        if len(x_arr) == 0 or len(y_arr) == 0:
            return CIResult(
                statistic_name=statistic_name,
                point_estimate=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                ci_level=self.ci_level,
                method=self.method,
                n_bootstrap=0,
                n_samples=0,
            )

        point = float(statistic_fn(x_arr) - statistic_fn(y_arr))

        boot_diffs = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            sx = self.rng.choice(x_arr, size=len(x_arr), replace=True)
            sy = self.rng.choice(y_arr, size=len(y_arr), replace=True)
            boot_diffs[i] = statistic_fn(sx) - statistic_fn(sy)

        lo, hi = self._percentile(boot_diffs)
        return CIResult(
            statistic_name=statistic_name,
            point_estimate=point,
            ci_lower=float(lo),
            ci_upper=float(hi),
            ci_level=self.ci_level,
            method="percentile",
            n_bootstrap=self.n_bootstrap,
            n_samples=len(x_arr) + len(y_arr),
        )

    # ── internal CI methods ──────────────────────────────────────────

    def _percentile(self, boot_stats: np.ndarray) -> tuple[float, float]:
        alpha = 1 - self.ci_level
        lo = np.percentile(boot_stats, alpha / 2 * 100)
        hi = np.percentile(boot_stats, (1 - alpha / 2) * 100)
        return float(lo), float(hi)

    def _bca(
        self,
        data: np.ndarray,
        boot_stats: np.ndarray,
        statistic_fn: Callable[[np.ndarray], float],
        point: float,
    ) -> tuple[float, float]:
        """Bias-corrected and accelerated bootstrap interval."""
        n = len(data)
        alpha = 1 - self.ci_level

        # Bias correction factor z0
        prop_below = np.mean(boot_stats < point)
        prop_below = np.clip(prop_below, 1e-10, 1 - 1e-10)
        z0 = sp_stats.norm.ppf(prop_below)

        # Acceleration factor a (jackknife)
        jackknife_stats = np.empty(n)
        for i in range(n):
            jack_sample = np.delete(data, i)
            jackknife_stats[i] = statistic_fn(jack_sample)
        jack_mean = np.mean(jackknife_stats)
        diffs = jack_mean - jackknife_stats
        num = np.sum(diffs ** 3)
        den = 6.0 * (np.sum(diffs ** 2)) ** 1.5
        a = num / den if den != 0 else 0.0

        # Adjusted percentiles
        z_lo = sp_stats.norm.ppf(alpha / 2)
        z_hi = sp_stats.norm.ppf(1 - alpha / 2)

        def _adj(z: float) -> float:
            num = z0 + z
            denom = 1 - a * num
            if denom == 0:
                return 0.5
            return float(sp_stats.norm.cdf(z0 + num / denom))

        p_lo = _adj(z_lo) * 100
        p_hi = _adj(z_hi) * 100
        p_lo = np.clip(p_lo, 0, 100)
        p_hi = np.clip(p_hi, 0, 100)

        return float(np.percentile(boot_stats, p_lo)), float(
            np.percentile(boot_stats, p_hi)
        )
