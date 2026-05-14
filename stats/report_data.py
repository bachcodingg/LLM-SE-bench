"""
stats/report_data.py — LaTeX table and thesis-ready data generator.

Produces:
    * Descriptive-statistics summary tables (LaTeX ``tabular``).
    * Hypothesis-test result tables.
    * Effect-size summary tables.
    * Model ranking tables.
    * Cost comparison tables.
    * All formatted for direct inclusion in a LaTeX thesis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ReportDataGenerator:
    """Generate LaTeX tables and structured report data.

    Parameters
    ----------
    output_dir : str | Path
        Directory for LaTeX ``.tex`` files.
    float_fmt : str
        Default float format (default ``".3f"``).
    """

    def __init__(
        self,
        output_dir: str | Path = "analysis/tables",
        float_fmt: str = ".3f",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.float_fmt = float_fmt

    def _save(self, content: str, name: str) -> Path:
        path = self.output_dir / f"{name}.tex"
        path.write_text(content, encoding="utf-8")
        logger.info("Saved table: %s", path)
        return path

    @staticmethod
    def _escape(text: str) -> str:
        """Escape LaTeX special characters."""
        for char in ("&", "%", "$", "#", "_", "{", "}"):
            text = text.replace(char, f"\\{char}")
        return text

    def _fmt(self, val: Any) -> str:
        if isinstance(val, float):
            return format(val, self.float_fmt)
        return str(val)

    # ── descriptive statistics table ─────────────────────────────────

    def descriptive_table(
        self,
        stats: list[dict],
        caption: str = "Descriptive statistics by model",
        label: str = "tab:descriptive",
    ) -> Path:
        """Generate a LaTeX table of descriptive statistics.

        Parameters
        ----------
        stats : list[dict]
            Each dict should have keys: metric_name, model_id, n, mean,
            std_dev, median, q1, q3.
        """
        cols = ["Metric", "Model", "N", "Mean", "SD", "Median", "Q1", "Q3"]
        header = " & ".join(cols) + r" \\"
        rows: list[str] = []
        for s in stats:
            row = (
                f"{self._escape(s.get('metric_name', ''))} & "
                f"{self._escape(s.get('model_id', ''))} & "
                f"{s.get('n', 0)} & "
                f"{self._fmt(s.get('mean', 0))} & "
                f"{self._fmt(s.get('std_dev', 0))} & "
                f"{self._fmt(s.get('median', 0))} & "
                f"{self._fmt(s.get('q1', 0))} & "
                f"{self._fmt(s.get('q3', 0))} \\\\"
            )
            rows.append(row)

        body = "\n".join(rows)
        col_spec = "ll" + "r" * (len(cols) - 2)
        tex = _wrap_table(header, body, col_spec, caption, label)
        return self._save(tex, "descriptive_stats")

    # ── hypothesis-test results ──────────────────────────────────────

    def hypothesis_table(
        self,
        results: list[dict],
        caption: str = "Hypothesis test results",
        label: str = "tab:hypothesis",
    ) -> Path:
        """LaTeX table of hypothesis-test outcomes."""
        cols = ["Test", "Metric", "Groups", "Statistic", "$p$", "Sig.", "Correction"]
        header = " & ".join(cols) + r" \\"
        rows: list[str] = []
        for r in results:
            groups = r.get("group_a", "")
            if r.get("group_b"):
                groups += f" vs. {r['group_b']}"
            sig = r"$\checkmark$" if r.get("significant") else "---"
            row = (
                f"{self._escape(r.get('test_name', ''))} & "
                f"{self._escape(r.get('metric_name', ''))} & "
                f"{self._escape(groups)} & "
                f"{self._fmt(r.get('statistic', 0))} & "
                f"{self._fmt(r.get('p_value', 1))} & "
                f"{sig} & "
                f"{self._escape(r.get('correction_method', 'none'))} \\\\"
            )
            rows.append(row)

        body = "\n".join(rows)
        tex = _wrap_table(header, body, "llllrcc", caption, label)
        return self._save(tex, "hypothesis_tests")

    # ── effect-size table ────────────────────────────────────────────

    def effect_size_table(
        self,
        results: list[dict],
        caption: str = "Effect sizes (pairwise)",
        label: str = "tab:effect_size",
    ) -> Path:
        """LaTeX table of pairwise effect sizes."""
        cols = ["Metric", "Comparison", "Measure", "Value", "Label", "95\\% CI"]
        header = " & ".join(cols) + r" \\"
        rows: list[str] = []
        for r in results:
            comparison = f"{r.get('group_a', '')} vs. {r.get('group_b', '')}"
            ci = f"[{self._fmt(r.get('ci_lower', 0))}, {self._fmt(r.get('ci_upper', 0))}]"
            row = (
                f"{self._escape(r.get('metric_name', ''))} & "
                f"{self._escape(comparison)} & "
                f"{self._escape(r.get('measure', ''))} & "
                f"{self._fmt(r.get('value', 0))} & "
                f"{self._escape(r.get('label', ''))} & "
                f"{ci} \\\\"
            )
            rows.append(row)

        body = "\n".join(rows)
        tex = _wrap_table(header, body, "llllrl", caption, label)
        return self._save(tex, "effect_sizes")

    # ── model ranking table ──────────────────────────────────────────

    def ranking_table(
        self,
        rankings: list[dict],
        caption: str = "Model rankings",
        label: str = "tab:ranking",
    ) -> Path:
        """LaTeX table of model rankings."""
        cols = ["Rank", "Model", "Tier", "Composite", "Pass Rate",
                "Avg Quality", "Avg Cost", "W--L--T"]
        header = " & ".join(cols) + r" \\"
        rows: list[str] = []
        for r in rankings:
            wlt = f"{r.get('wins', 0)}--{r.get('losses', 0)}--{r.get('ties', 0)}"
            row = (
                f"{r.get('rank', 0)} & "
                f"{self._escape(r.get('model_id', ''))} & "
                f"{self._escape(r.get('tier', ''))} & "
                f"{self._fmt(r.get('composite_score', 0))} & "
                f"{self._fmt(r.get('pass_rate', 0))} & "
                f"{self._fmt(r.get('avg_quality', 0))} & "
                f"\\${self._fmt(r.get('avg_cost_usd', 0))} & "
                f"{wlt} \\\\"
            )
            rows.append(row)

        body = "\n".join(rows)
        tex = _wrap_table(header, body, "clcrrrrr", caption, label)
        return self._save(tex, "model_rankings")

    # ── cost comparison table ────────────────────────────────────────

    def cost_table(
        self,
        profiles: list[dict],
        caption: str = "Cost comparison by model",
        label: str = "tab:cost",
    ) -> Path:
        """LaTeX table of cost profiles."""
        cols = ["Model", "Total (\\$)", "Mean/Task (\\$)",
                "Cost/Correct (\\$)", "Prompt Tok.", "Compl. Tok."]
        header = " & ".join(cols) + r" \\"
        rows: list[str] = []
        for p in profiles:
            row = (
                f"{self._escape(p.get('model_id', ''))} & "
                f"{self._fmt(p.get('total_cost_usd', 0))} & "
                f"{self._fmt(p.get('mean_cost_per_task', 0))} & "
                f"{self._fmt(p.get('cost_per_correct', 0))} & "
                f"{int(p.get('total_prompt_tokens', 0)):,} & "
                f"{int(p.get('total_completion_tokens', 0)):,} \\\\"
            )
            rows.append(row)

        body = "\n".join(rows)
        tex = _wrap_table(header, body, "lrrrrr", caption, label)
        return self._save(tex, "cost_comparison")

    # ── pass@k table ─────────────────────────────────────────────────

    def pass_at_k_table(
        self,
        pass_df: pd.DataFrame,
        caption: str = "pass@k results by model",
        label: str = "tab:pass_at_k",
    ) -> Path:
        """LaTeX table of pass@k values."""
        k_cols = [c for c in pass_df.columns if c.startswith("pass@")]
        cols = ["Model"] + k_cols
        header = " & ".join(cols) + r" \\"
        rows: list[str] = []
        for _, row_data in pass_df.iterrows():
            vals = " & ".join(self._fmt(row_data[c]) for c in k_cols)
            rows.append(f"{self._escape(str(row_data['model_id']))} & {vals} \\\\")

        body = "\n".join(rows)
        col_spec = "l" + "r" * len(k_cols)
        tex = _wrap_table(header, body, col_spec, caption, label)
        return self._save(tex, "pass_at_k")

    # ── correlation table ────────────────────────────────────────────

    def correlation_table(
        self,
        results: list[dict],
        caption: str = "Quality–performance correlations",
        label: str = "tab:correlations",
    ) -> Path:
        """LaTeX table of correlation results."""
        cols = ["Variable A", "Variable B", "Method", "$\\rho$", "$p$", "Sig."]
        header = " & ".join(cols) + r" \\"
        rows: list[str] = []
        for r in results:
            sig = r"$\checkmark$" if r.get("significant") else "---"
            row = (
                f"{self._escape(r.get('variable_a', ''))} & "
                f"{self._escape(r.get('variable_b', ''))} & "
                f"{self._escape(r.get('method', ''))} & "
                f"{self._fmt(r.get('coefficient', 0))} & "
                f"{self._fmt(r.get('p_value', 1))} & "
                f"{sig} \\\\"
            )
            rows.append(row)

        body = "\n".join(rows)
        tex = _wrap_table(header, body, "lllrrc", caption, label)
        return self._save(tex, "correlations")

    # ── consistency table ────────────────────────────────────────────

    def consistency_table(
        self,
        profiles: list[dict],
        caption: str = "Cross-run consistency analysis",
        label: str = "tab:consistency",
    ) -> Path:
        """LaTeX table of consistency profiles."""
        cols = ["Model", "Agreement", "Mean CV", "ICC", "Flip Rate"]
        header = " & ".join(cols) + r" \\"
        rows: list[str] = []
        for p in profiles:
            row = (
                f"{self._escape(p.get('model_id', ''))} & "
                f"{self._fmt(p.get('agreement_rate', 0))} & "
                f"{self._fmt(p.get('mean_cv', 0))} & "
                f"{self._fmt(p.get('icc', 0))} & "
                f"{self._fmt(p.get('flip_rate', 0))} \\\\"
            )
            rows.append(row)

        body = "\n".join(rows)
        tex = _wrap_table(header, body, "lrrrr", caption, label)
        return self._save(tex, "consistency")

    # ── generate all tables ──────────────────────────────────────────

    def generate_all(
        self,
        descriptive_stats: list[dict] | None = None,
        hypothesis_results: list[dict] | None = None,
        effect_sizes: list[dict] | None = None,
        rankings: list[dict] | None = None,
        cost_profiles: list[dict] | None = None,
        pass_df: pd.DataFrame | None = None,
        correlations: list[dict] | None = None,
        consistency_profiles: list[dict] | None = None,
    ) -> list[Path]:
        """Generate all applicable LaTeX tables."""
        paths: list[Path] = []
        if descriptive_stats:
            paths.append(self.descriptive_table(descriptive_stats))
        if hypothesis_results:
            paths.append(self.hypothesis_table(hypothesis_results))
        if effect_sizes:
            paths.append(self.effect_size_table(effect_sizes))
        if rankings:
            paths.append(self.ranking_table(rankings))
        if cost_profiles:
            paths.append(self.cost_table(cost_profiles))
        if pass_df is not None and not pass_df.empty:
            paths.append(self.pass_at_k_table(pass_df))
        if correlations:
            paths.append(self.correlation_table(correlations))
        if consistency_profiles:
            paths.append(self.consistency_table(consistency_profiles))
        logger.info("Generated %d LaTeX tables in %s", len(paths), self.output_dir)
        return paths


# ── helper ───────────────────────────────────────────────────────────

def _wrap_table(
    header: str,
    body: str,
    col_spec: str,
    caption: str,
    label: str,
) -> str:
    """Wrap rows in a complete LaTeX table environment."""
    return (
        f"\\begin{{table}}[htbp]\n"
        f"\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        f"\\toprule\n"
        f"{header}\n"
        f"\\midrule\n"
        f"{body}\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}\n"
        f"\\end{{table}}\n"
    )
