"""
cli_report.py — Terminal summary generator for llm-se-bench decisions.

Produces coloured, aligned text output suitable for ``llm-se-bench report``
or piping to a file.  Requires no external rendering libraries.

Layout
------
::

    ╔══════════════════════════════════════════════╗
    ║      llm-se-bench  Decision Report          ║
    ║      Profile: DevOps                        ║
    ╚══════════════════════════════════════════════╝

    ── Ranked Decision Matrix ───────────────────────
    #  Model                Score   Correctness  Quality  Speed   Cost    Consistency
    1  claude-3.5-sonnet    0.8234  0.92 (0.25)  0.78 …  0.85 …  0.71 …  0.90 …
    2  gpt-4-turbo          0.7891  …
    3  gemini-1.5-pro       0.6543  …

    ── Recommendation ──────────────────────────────
    Recommended: claude-3.5-sonnet (confidence 87%)
    Runner-up:   gpt-4-turbo
    Rationale:   …

    ── Pareto Frontier ─────────────────────────────
    Frontier models: claude-3.5-sonnet, gpt-4-turbo
    Dominated:       gemini-1.5-pro

    ── Sensitivity ─────────────────────────────────
    …
"""

from __future__ import annotations

import sys
from io import StringIO
from typing import IO, Any, Sequence

from framework.decision_matrix import CRITERIA, DecisionMatrixEngine
from framework.recommender import ModelRecommender
from framework.tradeoffs import TradeoffAnalyzer

try:
    from contracts import DecisionMatrix, Recommendation, StatisticalSummary
except ImportError:  # pragma: no cover
    pass


# ── ANSI colours (disabled when not a tty) ────────────────────────────

class _Colors:
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls) -> None:
        for attr in ("BOLD", "GREEN", "YELLOW", "CYAN", "RED", "DIM", "RESET"):
            setattr(cls, attr, "")


C = _Colors


def _supports_color(stream: IO) -> bool:
    return hasattr(stream, "isatty") and stream.isatty()


# ======================================================================
# CLIReporter
# ======================================================================

class CLIReporter:
    """Generate a rich terminal report from decision analysis results.

    Parameters
    ----------
    summaries : list[StatisticalSummary]
        All statistical summaries.
    profile_name : str
        MCDA profile to use (``"devops"`` / ``"audit"`` / ``"budget"``).
    profiles_dir : str | None
        Custom profiles directory.
    width : int
        Maximum line width.
    color : bool | None
        Force colour on/off.  ``None`` = auto-detect.
    """

    def __init__(
        self,
        summaries: Sequence[Any],
        profile_name: str = "devops",
        profiles_dir: str | None = None,
        width: int = 80,
        color: bool | None = None,
    ) -> None:
        self._summaries = list(summaries)
        self._profile = profile_name
        self._profiles_dir = profiles_dir
        self._width = width

        if color is False or (color is None and not _supports_color(sys.stdout)):
            C.disable()

        # Build analysis
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
    def render(self) -> str:
        """Return the full report as a string."""
        buf = StringIO()
        self._header(buf)
        self._matrix_table(buf)
        self._recommendation(buf)
        self._pareto_section(buf)
        self._sensitivity(buf)
        return buf.getvalue()

    # ------------------------------------------------------------------
    def print(self, file: IO | None = None) -> None:
        """Print the report to *file* (default: stdout)."""
        print(self.render(), file=file or sys.stdout)

    # ── sections ──────────────────────────────────────────────────────

    def _header(self, buf: StringIO) -> None:
        w = self._width
        buf.write(f"\n{C.CYAN}{'═' * w}{C.RESET}\n")
        title = "llm-se-bench  Decision Report"
        buf.write(f"{C.BOLD}{C.CYAN}{title:^{w}}{C.RESET}\n")
        sub = f"Profile: {self._profile}"
        buf.write(f"{C.CYAN}{sub:^{w}}{C.RESET}\n")
        buf.write(f"{C.CYAN}{'═' * w}{C.RESET}\n\n")

    def _matrix_table(self, buf: StringIO) -> None:
        buf.write(f"{C.BOLD}── Ranked Decision Matrix ──{C.RESET}\n\n")

        # Header
        hdr = f"{'#':>2}  {'Model':<22} {'Score':>7}"
        for c in CRITERIA:
            hdr += f"  {c[:5]:>7}"
        buf.write(f"{C.DIM}{hdr}{C.RESET}\n")
        buf.write(f"{C.DIM}{'-' * len(hdr)}{C.RESET}\n")

        # Rows
        for dm in self._engine.matrices:
            color = C.GREEN if dm.rank == 1 else ""
            line = f"{dm.rank:>2}  {dm.model_id:<22} {dm.weighted_total:>7.4f}"
            scores_map = {s.criterion: s for s in dm.scores}
            for c in CRITERIA:
                s = scores_map.get(c)
                if s:
                    line += f"  {s.normalised_value:>5.2f}×{s.weight:.0%}"
                else:
                    line += f"  {'—':>7}"
            buf.write(f"{color}{line}{C.RESET}\n")

        buf.write("\n")

    def _recommendation(self, buf: StringIO) -> None:
        buf.write(f"{C.BOLD}── Recommendation ──{C.RESET}\n\n")

        rec = self._rec
        buf.write(
            f"  {C.GREEN}Recommended:{C.RESET} {rec.recommended_model} "
            f"(confidence {rec.confidence:.0%})\n"
        )
        if rec.runner_up_model:
            buf.write(f"  Runner-up:   {rec.runner_up_model}\n")
        if rec.constraints_applied:
            buf.write(
                f"  Constraints: {', '.join(rec.constraints_applied)}\n"
            )
        buf.write(f"\n  {C.DIM}Rationale:{C.RESET}\n")
        # Wrap rationale
        words = rec.rationale.split()
        line = "  "
        for w in words:
            if len(line) + len(w) + 1 > self._width:
                buf.write(f"{line}\n")
                line = "  "
            line += w + " "
        if line.strip():
            buf.write(f"{line}\n")

        if rec.warnings:
            buf.write(f"\n  {C.YELLOW}Warnings:{C.RESET}\n")
            for w in rec.warnings:
                buf.write(f"  ⚠ {w}\n")

        buf.write("\n")

    def _pareto_section(self, buf: StringIO) -> None:
        buf.write(f"{C.BOLD}── Pareto Frontier ──{C.RESET}\n\n")

        frontier = self._pareto.frontier_ids
        dominated = self._pareto.dominated_ids

        buf.write(f"  Frontier:  {', '.join(frontier) or 'none'}\n")
        buf.write(f"  Dominated: {', '.join(dominated) or 'none'}\n\n")

    def _sensitivity(self, buf: StringIO) -> None:
        buf.write(f"{C.BOLD}── Sensitivity Analysis ──{C.RESET}\n\n")
        buf.write(
            f"  {C.DIM}Showing rank-1 model as each criterion weight "
            f"varies 0→1:{C.RESET}\n"
        )

        for criterion in CRITERIA:
            sa = self._engine.sensitivity_analysis(criterion, steps=5)
            winners = []
            for step in sa:
                w = step["weight_value"]
                ranked = step["rankings"]
                top = min(ranked, key=ranked.get)  # type: ignore[arg-type]
                winners.append(f"{w:.0%}→{top}")
            buf.write(f"  {criterion:<14} {' | '.join(winners)}\n")

        buf.write("\n")
