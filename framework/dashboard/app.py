"""
app.py — Multi-page Plotly Dash application for llm-se-bench.

Pages
-----
1.  **Overview** — Model comparison radar chart + summary statistics.
2.  **Deep Dive** — Per-model, per-criterion drill-down with bar charts.
3.  **Decision Tool** — Interactive weight sliders → live matrix update.
4.  **Cost Calculator** — Project-size cost projection tool.

Launch
------
::

    python -m framework.dashboard.app                 # default port 8050
    python -m framework.dashboard.app --port 8051     # custom port
    python -m framework.dashboard.app --data path.json

The application reads ``analysis/statistical_summary.json`` by default.
Pass ``--data`` to use a different file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Dash import — so the module can be imported even when dash is missing
# (e.g. for unit-testing the data layer)
# ---------------------------------------------------------------------------

_DEFAULT_DATA_PATH = "analysis/statistical_summary.json"


def _load_summaries(path: str | Path) -> list[Any]:
    """Load StatisticalSummary objects from a JSON file."""
    try:
        from contracts import StatisticalSummary
    except ImportError:
        from pydantic import BaseModel, Field as PField
        from datetime import datetime

        class StatisticalSummary(BaseModel):  # type: ignore[no-redef]
            metric_name: str = ""
            model_id: str = ""
            n: int = 0
            mean: float = 0.0
            std_dev: float = 0.0
            median: float = 0.0
            min_val: float = 0.0
            max_val: float = 0.0
            ci_lower_95: float = 0.0
            ci_upper_95: float = 0.0
            computed_at: datetime = PField(default_factory=datetime.utcnow)

    p = Path(path)
    if not p.exists():
        logger.warning("Data file %s not found — using empty dataset", p)
        return []
    with open(p) as fh:
        raw = json.load(fh)
    items = raw.get("descriptive", raw) if isinstance(raw, dict) else raw
    return [StatisticalSummary(**item) for item in items if isinstance(item, dict)]


def create_app(
    data_path: str | Path = _DEFAULT_DATA_PATH,
    debug: bool = False,
) -> Any:
    """Build and return the Dash ``app`` instance.

    Importing Dash at function-call time keeps the module importable
    without ``pip install dash`` (e.g. for tests of non-dashboard code).
    """
    try:
        import dash  # type: ignore[import-untyped]
        from dash import Dash, dcc, html  # type: ignore[import-untyped]
        import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
    except ImportError as exc:
        logger.error("Dash is required for the dashboard: %s", exc)
        raise SystemExit(
            "Install Dash: pip install dash dash-bootstrap-components plotly"
        ) from exc

    summaries = _load_summaries(data_path)

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,
        title="llm-se-bench Dashboard",
    )

    # ── Navigation bar ────────────────────────────────────────────────
    navbar = dbc.NavbarSimple(
        brand="llm-se-bench",
        brand_href="/",
        color="primary",
        dark=True,
        children=[
            dbc.NavItem(dbc.NavLink("Overview", href="/")),
            dbc.NavItem(dbc.NavLink("Deep Dive", href="/deep-dive")),
            dbc.NavItem(dbc.NavLink("Decision Tool", href="/decision-tool")),
            dbc.NavItem(dbc.NavLink("Cost Calculator", href="/cost-calculator")),
        ],
    )

    # ── Layout ────────────────────────────────────────────────────────
    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            navbar,
            dbc.Container(
                id="page-content",
                className="mt-4",
                fluid=True,
            ),
            # Hidden store for summaries (serialised)
            dcc.Store(
                id="summaries-store",
                data=[s.model_dump() if hasattr(s, "model_dump") else s.dict()
                      for s in summaries],
            ),
        ]
    )

    # ── Page routing ──────────────────────────────────────────────────
    from framework.dashboard.pages.overview import register_overview
    from framework.dashboard.pages.deep_dive import register_deep_dive
    from framework.dashboard.pages.decision_tool import register_decision_tool
    from framework.dashboard.pages.cost_calculator import register_cost_calculator

    register_overview(app)
    register_deep_dive(app)
    register_decision_tool(app)
    register_cost_calculator(app)

    @app.callback(
        dash.Output("page-content", "children"),
        [dash.Input("url", "pathname")],
    )
    def display_page(pathname: str) -> Any:
        from framework.dashboard.pages.overview import overview_layout
        from framework.dashboard.pages.deep_dive import deep_dive_layout
        from framework.dashboard.pages.decision_tool import decision_tool_layout
        from framework.dashboard.pages.cost_calculator import cost_calculator_layout

        if pathname == "/deep-dive":
            return deep_dive_layout()
        elif pathname == "/decision-tool":
            return decision_tool_layout()
        elif pathname == "/cost-calculator":
            return cost_calculator_layout()
        return overview_layout()

    return app


def launch_dashboard(
    data_path: str | Path = _DEFAULT_DATA_PATH,
    port: int = 8050,
    debug: bool = False,
) -> None:
    """Create and run the dashboard server."""
    app = create_app(data_path, debug=debug)
    app.run(host="0.0.0.0", port=port, debug=debug)


# ── CLI entry point ───────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch llm-se-bench interactive dashboard"
    )
    parser.add_argument(
        "--data", default=_DEFAULT_DATA_PATH,
        help="Path to statistical_summary.json"
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Port (default 8050)"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    launch_dashboard(args.data, args.port, args.debug)


if __name__ == "__main__":
    main()
