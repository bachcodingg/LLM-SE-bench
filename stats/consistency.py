"""
stats/consistency.py — Cross-run consistency analysis for llm-se-bench.

Quantifies how stable each model's outputs are across repeated runs of the
same problem.  Key metrics:

    * **Intra-problem variance** — SD of scores across runs for each problem.
    * **Agreement rate** — proportion of problems where all runs agree on
      pass/fail.
    * **Coefficient of variation** — SD / mean, normalised instability.
    * **ICC (intra-class correlation)** — reliability across runs.
    * **Flip analysis** — problems that flip between pass and fail across runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyProfile:
    """Consistency summary for one model (optionally per dataset)."""

    model_id: str
    dataset: str = ""
    n_problems: int = 0
    n_runs_per_problem: float = 0.0
    agreement_rate: float = 0.0
    mean_intra_sd: float = 0.0
    median_intra_sd: float = 0.0
    mean_cv: float = 0.0
    icc: float = 0.0
    n_flips: int = 0
    flip_rate: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 6) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


@dataclass
class FlipRecord:
    """A problem that flips pass/fail across runs."""

    problem_id: str
    model_id: str
    dataset: str
    n_pass: int
    n_fail: int
    n_total: int
    scores: list[float]

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "model_id": self.model_id,
            "dataset": self.dataset,
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "n_total": self.n_total,
            "scores": [round(s, 4) for s in self.scores],
        }


class ConsistencyAnalyzer:
    """Analyse cross-run consistency of benchmark results.

    Parameters
    ----------
    score_col : str
        Column used for continuous consistency (default ``"weighted_score"``).
    binary_col : str
        Column used for binary agreement (default ``"is_pass"``).
    run_col : str
        Column identifying repeated runs (default ``"evaluation_id"``).
    """

    def __init__(
        self,
        score_col: str = "weighted_score",
        binary_col: str = "is_pass",
        run_col: str = "evaluation_id",
    ) -> None:
        self.score_col = score_col
        self.binary_col = binary_col
        self.run_col = run_col

    # ── main entry point ─────────────────────────────────────────────

    def analyse(
        self,
        df: pd.DataFrame,
        group_by: str | list[str] = "model_id",
    ) -> list[ConsistencyProfile]:
        """Compute consistency profiles grouped by *group_by*.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``problem_id``, the score and binary columns, and
            the grouping column(s).
        group_by : str | list[str]
            Typically ``"model_id"`` or ``["model_id", "dataset"]``.

        Returns
        -------
        list[ConsistencyProfile]
        """
        if isinstance(group_by, str):
            group_by = [group_by]

        profiles: list[ConsistencyProfile] = []
        for keys, grp in df.groupby(group_by, sort=True):
            if isinstance(keys, str):
                keys = (keys,)
            labels = dict(zip(group_by, keys))

            profile = self._compute_profile(
                grp,
                model_id=labels.get("model_id", ""),
                dataset=labels.get("dataset", ""),
            )
            profiles.append(profile)

        return profiles

    # ── flip analysis ────────────────────────────────────────────────

    def find_flips(
        self,
        df: pd.DataFrame,
    ) -> list[FlipRecord]:
        """Identify problems that flip between pass and fail across runs.

        Returns
        -------
        list[FlipRecord]
        """
        if self.binary_col not in df.columns:
            return []

        flips: list[FlipRecord] = []
        group_cols = ["model_id", "dataset", "problem_id"] if "dataset" in df.columns \
            else ["model_id", "problem_id"]

        for keys, grp in df.groupby(group_cols, sort=True):
            if isinstance(keys, str):
                keys = (keys,)
            labels = dict(zip(group_cols, keys))
            binary = grp[self.binary_col].astype(bool)
            n_pass = int(binary.sum())
            n_fail = int((~binary).sum())

            if n_pass > 0 and n_fail > 0:
                scores = grp[self.score_col].tolist() if self.score_col in grp.columns else []
                flips.append(
                    FlipRecord(
                        problem_id=labels.get("problem_id", ""),
                        model_id=labels.get("model_id", ""),
                        dataset=labels.get("dataset", ""),
                        n_pass=n_pass,
                        n_fail=n_fail,
                        n_total=len(grp),
                        scores=scores,
                    )
                )
        return flips

    # ── intra-class correlation (ICC) ────────────────────────────────

    @staticmethod
    def compute_icc(
        df: pd.DataFrame,
        subject_col: str = "problem_id",
        rater_col: str = "evaluation_id",
        value_col: str = "weighted_score",
    ) -> float:
        """Compute ICC(3,1) — two-way mixed, single measures.

        Uses one-way ANOVA decomposition as a simplified estimator.
        """
        pivot = df.pivot_table(
            index=subject_col,
            columns=rater_col,
            values=value_col,
            aggfunc="mean",
        ).dropna()

        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            return 0.0

        n = pivot.shape[0]  # subjects
        k = pivot.shape[1]  # raters

        grand_mean = pivot.values.mean()
        ss_between = k * np.sum((pivot.mean(axis=1).values - grand_mean) ** 2)
        ss_within = np.sum((pivot.values - pivot.mean(axis=1).values[:, None]) ** 2)

        ms_between = ss_between / (n - 1) if n > 1 else 0.0
        ms_within = ss_within / (n * (k - 1)) if (n * (k - 1)) > 0 else 0.0

        if (ms_between + (k - 1) * ms_within) == 0:
            return 0.0

        icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
        return float(np.clip(icc, -1.0, 1.0))

    # ── private helpers ──────────────────────────────────────────────

    def _compute_profile(
        self,
        grp: pd.DataFrame,
        model_id: str,
        dataset: str,
    ) -> ConsistencyProfile:
        """Build a ConsistencyProfile from a single model/dataset group."""
        problems = grp.groupby("problem_id")
        n_problems = problems.ngroups
        runs_per = grp.groupby("problem_id").size()
        mean_runs = float(runs_per.mean()) if len(runs_per) > 0 else 0.0

        # Intra-problem SD and CV
        intra_sds: list[float] = []
        cvs: list[float] = []
        n_agree = 0
        n_flips = 0

        for _, pgrp in problems:
            if self.score_col in pgrp.columns:
                vals = pgrp[self.score_col].dropna().values
                if len(vals) > 1:
                    sd = float(np.std(vals, ddof=1))
                    mn = float(np.mean(vals))
                    intra_sds.append(sd)
                    if mn > 0:
                        cvs.append(sd / mn)

            if self.binary_col in pgrp.columns:
                binary = pgrp[self.binary_col].astype(bool)
                if binary.all() or (~binary).all():
                    n_agree += 1
                else:
                    n_flips += 1

        agreement_rate = n_agree / n_problems if n_problems > 0 else 0.0
        flip_rate = n_flips / n_problems if n_problems > 0 else 0.0

        # ICC
        icc = 0.0
        if self.score_col in grp.columns and self.run_col in grp.columns:
            try:
                icc = self.compute_icc(
                    grp, "problem_id", self.run_col, self.score_col
                )
            except Exception:
                pass

        return ConsistencyProfile(
            model_id=model_id,
            dataset=dataset,
            n_problems=n_problems,
            n_runs_per_problem=mean_runs,
            agreement_rate=agreement_rate,
            mean_intra_sd=float(np.mean(intra_sds)) if intra_sds else 0.0,
            median_intra_sd=float(np.median(intra_sds)) if intra_sds else 0.0,
            mean_cv=float(np.mean(cvs)) if cvs else 0.0,
            icc=icc,
            n_flips=n_flips,
            flip_rate=flip_rate,
        )
