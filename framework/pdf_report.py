"""
pdf_report.py — WeasyPrint-based PDF report generator.

Renders the decision matrix, recommendation, and Pareto analysis into
a professionally styled PDF document.

The report is generated from an HTML template string (no external
template files required) using ``weasyprint.HTML(string=…).write_pdf()``.

Output
------
``reports/decision_matrix.pdf``

Dependencies
------------
- weasyprint (rendering)
- framework.decision_matrix
- framework.recommender
- framework.tradeoffs
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from framework.decision_matrix import CRITERIA, DecisionMatrixEngine
from framework.recommender import ModelRecommender
from framework.tradeoffs import TradeoffAnalyzer

logger = logging.getLogger(__name__)


# ── HTML template ─────────────────────────────────────────────────────

_CSS = """
@page {
    size: A4;
    margin: 2cm;
    @bottom-center { content: "Page " counter(page) " of " counter(pages); }
}
body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.5;
    color: #222;
}
h1 { color: #1a365d; font-size: 18pt; margin-bottom: 0.5em; }
h2 { color: #2c5282; font-size: 14pt; border-bottom: 1px solid #bee3f8;
     padding-bottom: 4px; margin-top: 1.5em; }
h3 { color: #2b6cb0; font-size: 11pt; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th { background: #ebf8ff; color: #2c5282; text-align: left;
     padding: 6px 8px; font-size: 9pt; border-bottom: 2px solid #90cdf4; }
td { padding: 5px 8px; border-bottom: 1px solid #e2e8f0; font-size: 9pt; }
tr:nth-child(even) td { background: #f7fafc; }
.rank-1 td { background: #f0fff4; font-weight: 600; }
.meta { color: #718096; font-size: 8pt; }
.rec-box { background: #f0fff4; border: 1px solid #68d391;
           border-radius: 6px; padding: 12px; margin: 1em 0; }
.rec-box strong { color: #22543d; }
.warn-box { background: #fffff0; border: 1px solid #ecc94b;
            border-radius: 6px; padding: 10px; margin: 0.8em 0; }
.pareto-tag { display: inline-block; background: #bee3f8;
              border-radius: 3px; padding: 2px 6px; font-size: 8pt;
              color: #2a4365; margin: 2px; }
.footer { margin-top: 2em; border-top: 1px solid #e2e8f0;
          padding-top: 0.5em; font-size: 8pt; color: #a0aec0; }
"""


def _score_cell(norm: float, weight: float) -> str:
    """Format a score cell with normalised value and weight."""
    bar_width = int(norm * 60)
    return (
        f'<td>{norm:.3f} '
        f'<span style="color:#a0aec0">({weight:.0%})</span>'
        f'<div style="background:#bee3f8;height:3px;width:{bar_width}px;'
        f'border-radius:2px;margin-top:2px"></div></td>'
    )


# ======================================================================
# PDFReporter
# ======================================================================

class PDFReporter:
    """Generate a PDF decision report.

    Parameters
    ----------
    summaries : list[StatisticalSummary]
        All statistical summaries from C4.
    profile_name : str
        Profile to use for MCDA weights.
    profiles_dir : str | None
        Custom profiles directory.
    """

    def __init__(
        self,
        summaries: Sequence[Any],
        profile_name: str = "devops",
        profiles_dir: str | None = None,
    ) -> None:
        self._summaries = list(summaries)
        self._profile = profile_name
        self._profiles_dir = profiles_dir

        # Run analysis
        self._engine = DecisionMatrixEngine(
            summaries, profile_name=profile_name, profiles_dir=profiles_dir
        )
        self._engine.build()

        self._recommender = ModelRecommender(
            summaries,
            profile_name=profile_name,
            profiles_dir=profiles_dir,
        )
        self._rec = self._recommender.recommend(use_case=profile_name)

        self._analyzer = TradeoffAnalyzer(summaries)
        self._pareto = self._analyzer.analyze()

    # ------------------------------------------------------------------
    def render_html(self) -> str:
        """Return the full HTML document as a string."""
        parts: list[str] = []
        parts.append(self._html_head())
        parts.append(self._title_section())
        parts.append(self._matrix_section())
        parts.append(self._recommendation_section())
        parts.append(self._pareto_section())
        parts.append(self._weights_section())
        parts.append(self._footer())
        parts.append("</body></html>")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    def export(self, path: str | Path = "reports/decision_matrix.pdf") -> Path:
        """Write the PDF to *path*.

        Returns the resolved path.
        """
        try:
            from weasyprint import HTML  # type: ignore[import-untyped]
        except (ImportError, OSError):
            logger.warning(
                "weasyprint not available (missing native libs) — writing HTML fallback instead"
            )
            p = Path(path).with_suffix(".html")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(self.render_html(), encoding="utf-8")
            return p

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        HTML(string=self.render_html()).write_pdf(str(p))
        logger.info("PDF exported to %s", p)
        return p

    # ── HTML builders ─────────────────────────────────────────────────

    def _html_head(self) -> str:
        return (
            "<!DOCTYPE html><html><head>"
            "<meta charset='utf-8'>"
            f"<style>{_CSS}</style>"
            "</head><body>"
        )

    def _title_section(self) -> str:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        return (
            "<h1>llm-se-bench — Decision Report</h1>"
            f"<p class='meta'>Profile: <strong>{self._profile}</strong> "
            f"| Generated: {now} | Models: "
            f"{len(self._engine.matrices)}</p>"
        )

    def _matrix_section(self) -> str:
        rows: list[str] = []
        rows.append("<h2>Ranked Decision Matrix</h2>")
        rows.append("<table><thead><tr>")
        rows.append("<th>#</th><th>Model</th><th>Score</th>")
        for c in CRITERIA:
            rows.append(f"<th>{c.title()}</th>")
        rows.append("</tr></thead><tbody>")

        for dm in self._engine.matrices:
            cls = ' class="rank-1"' if dm.rank == 1 else ""
            rows.append(f"<tr{cls}>")
            rows.append(f"<td>{dm.rank}</td>")
            rows.append(f"<td>{dm.model_id}</td>")
            rows.append(f"<td><strong>{dm.weighted_total:.4f}</strong></td>")
            scores_map = {s.criterion: s for s in dm.scores}
            for c in CRITERIA:
                s = scores_map.get(c)
                if s:
                    rows.append(_score_cell(s.normalised_value, s.weight))
                else:
                    rows.append("<td>—</td>")
            rows.append("</tr>")

        rows.append("</tbody></table>")
        return "\n".join(rows)

    def _recommendation_section(self) -> str:
        rec = self._rec
        parts = ['<h2>Recommendation</h2>', '<div class="rec-box">']
        parts.append(
            f"<strong>Recommended model:</strong> {rec.recommended_model} "
            f"(confidence {rec.confidence:.0%})<br>"
        )
        if rec.runner_up_model:
            parts.append(f"<strong>Runner-up:</strong> {rec.runner_up_model}<br>")
        parts.append(f"<p>{rec.rationale}</p>")

        if rec.constraints_applied:
            parts.append(
                "<p><strong>Constraints:</strong> "
                + ", ".join(rec.constraints_applied)
                + "</p>"
            )

        parts.append("</div>")

        if rec.warnings:
            parts.append('<div class="warn-box">')
            for w in rec.warnings:
                parts.append(f"<p>⚠ {w}</p>")
            parts.append("</div>")

        return "\n".join(parts)

    def _pareto_section(self) -> str:
        parts = ["<h2>Pareto Frontier</h2>"]
        parts.append("<p><strong>Frontier models:</strong> ")
        for mid in self._pareto.frontier_ids:
            parts.append(f'<span class="pareto-tag">{mid}</span>')
        parts.append("</p>")

        if self._pareto.dominated_ids:
            parts.append("<p><strong>Dominated:</strong> ")
            parts.append(", ".join(self._pareto.dominated_ids))
            parts.append("</p>")

        # Trade-off summary for frontier pairs
        frontier_ids = set(self._pareto.frontier_ids)
        if len(frontier_ids) >= 2:
            parts.append("<h3>Key Trade-offs</h3><table>")
            parts.append(
                "<thead><tr><th>Model A</th><th>Model B</th>"
                "<th>Criterion X</th><th>Criterion Y</th>"
                "<th>ΔX</th><th>ΔY</th><th>MRS</th></tr></thead><tbody>"
            )
            count = 0
            for t in self._pareto.tradeoffs:
                if t.model_a in frontier_ids and t.model_b in frontier_ids:
                    if count >= 10:
                        break
                    mrs_str = f"{t.marginal_rate:.3f}" if t.marginal_rate else "—"
                    parts.append(
                        f"<tr><td>{t.model_a}</td><td>{t.model_b}</td>"
                        f"<td>{t.criterion_x}</td><td>{t.criterion_y}</td>"
                        f"<td>{t.delta_x:+.4f}</td><td>{t.delta_y:+.4f}</td>"
                        f"<td>{mrs_str}</td></tr>"
                    )
                    count += 1
            parts.append("</tbody></table>")

        return "\n".join(parts)

    def _weights_section(self) -> str:
        parts = ["<h2>Weight Configuration</h2>"]
        parts.append("<table><thead><tr><th>Criterion</th><th>Weight</th>")
        parts.append("<th>Visual</th></tr></thead><tbody>")
        weights = self._engine.weights
        for c in CRITERIA:
            w = weights.get(c, 0.0)
            bar = int(w * 200)
            parts.append(
                f"<tr><td>{c.title()}</td><td>{w:.0%}</td>"
                f'<td><div style="background:#63b3ed;height:12px;'
                f'width:{bar}px;border-radius:3px"></div></td></tr>'
            )
        parts.append("</tbody></table>")
        return "\n".join(parts)

    def _footer(self) -> str:
        return (
            '<div class="footer">'
            "Generated by llm-se-bench Decision Framework (Component 5). "
            "Weights and constraints are configurable via YAML profiles."
            "</div>"
        )
