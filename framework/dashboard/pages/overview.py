"""
pages/overview.py — Page 1: Model comparison radar chart + summary statistics.

Shows a radar (spider) chart comparing all models across the five
criteria, plus a summary stats table and profile selector.
"""

from __future__ import annotations

from typing import Any

_DASH_AVAILABLE = True
try:
    import dash  # type: ignore[import-untyped]
    from dash import Input, Output, State, dcc, html  # type: ignore[import-untyped]
    import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
    import plotly.graph_objects as go  # type: ignore[import-untyped]
except ImportError:
    _DASH_AVAILABLE = False


def overview_layout() -> Any:
    """Return the layout component for the overview page."""
    if not _DASH_AVAILABLE:
        return None

    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Model Comparison Overview", className="mb-3"),
                html.P(
                    "Radar chart comparing models across all five "
                    "decision criteria.  Select a profile to adjust weights.",
                    className="text-muted",
                ),
            ]),
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Label("Profile"),
                dcc.Dropdown(
                    id="overview-profile",
                    options=[
                        {"label": "Default (Equal)", "value": "default"},
                        {"label": "DevOps", "value": "devops"},
                        {"label": "Audit", "value": "audit"},
                        {"label": "Budget", "value": "budget"},
                    ],
                    value="default",
                    clearable=False,
                ),
            ], md=3),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dcc.Loading(dcc.Graph(id="radar-chart")),
            ], md=7),
            dbc.Col([
                html.H5("Summary Statistics"),
                html.Div(id="summary-stats-table"),
            ], md=5),
        ]),

        dbc.Row([
            dbc.Col([
                html.H5("Ranking Table", className="mt-4"),
                html.Div(id="ranking-table"),
            ]),
        ]),
    ])


def register_overview(app: Any) -> None:
    """Register overview callbacks on *app*."""
    if not _DASH_AVAILABLE:
        return

    @app.callback(
        [
            Output("radar-chart", "figure"),
            Output("summary-stats-table", "children"),
            Output("ranking-table", "children"),
        ],
        [Input("overview-profile", "value")],
        [State("summaries-store", "data")],
    )
    def update_overview(
        profile: str, summaries_data: list[dict],
    ) -> tuple:
        from framework.decision_matrix import (
            CRITERIA,
            DecisionMatrixEngine,
            DEFAULT_WEIGHTS,
        )

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

        summaries = [StatisticalSummary(**d) for d in (summaries_data or [])]

        if profile == "default":
            engine = DecisionMatrixEngine(summaries)
        else:
            try:
                engine = DecisionMatrixEngine(
                    summaries, profile_name=profile
                )
            except FileNotFoundError:
                engine = DecisionMatrixEngine(summaries)

        matrices = engine.build()
        normalised = engine.normalised_scores

        # ── Radar chart ───────────────────────────────────────────────
        categories = [c.title() for c in CRITERIA] + [CRITERIA[0].title()]
        fig = go.Figure()

        colors = ["#3182ce", "#e53e3e", "#38a169", "#d69e2e", "#805ad5"]
        for i, dm in enumerate(matrices):
            scores_map = {s.criterion: s.normalised_value for s in dm.scores}
            vals = [scores_map.get(c, 0) for c in CRITERIA]
            vals.append(vals[0])  # close the polygon
            fig.add_trace(
                go.Scatterpolar(
                    r=vals,
                    theta=categories,
                    fill="toself",
                    name=dm.model_id,
                    line=dict(color=colors[i % len(colors)]),
                    opacity=0.7,
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title=f"Model Comparison — {profile.title()} Profile",
            height=450,
            margin=dict(t=60, b=40, l=60, r=60),
        )

        # ── Summary stats table ───────────────────────────────────────
        summary_rows = []
        for dm in matrices:
            scores_map = {s.criterion: s for s in dm.scores}
            row = {"Model": dm.model_id, "Rank": dm.rank,
                   "Score": f"{dm.weighted_total:.4f}"}
            for c in CRITERIA:
                s = scores_map.get(c)
                row[c.title()] = f"{s.raw_value:.4f}" if s else "—"
            summary_rows.append(row)

        cols = ["Model", "Rank", "Score"] + [c.title() for c in CRITERIA]
        stats_table = dbc.Table(
            [
                html.Thead(html.Tr([html.Th(c) for c in cols])),
                html.Tbody([
                    html.Tr([html.Td(r.get(c, "")) for c in cols])
                    for r in summary_rows
                ]),
            ],
            bordered=True, hover=True, responsive=True, size="sm",
        )

        # ── Ranking table ─────────────────────────────────────────────
        rank_rows = []
        for dm in matrices:
            scores_map = {s.criterion: s for s in dm.scores}
            cells = [
                html.Td(html.Strong(str(dm.rank))),
                html.Td(dm.model_id),
                html.Td(f"{dm.weighted_total:.4f}"),
            ]
            for c in CRITERIA:
                s = scores_map.get(c)
                if s:
                    cells.append(
                        html.Td(f"{s.normalised_value:.3f}")
                    )
                else:
                    cells.append(html.Td("—"))
            cls = "table-success" if dm.rank == 1 else ""
            rank_rows.append(html.Tr(cells, className=cls))

        rank_cols = ["#", "Model", "Score"] + [c.title() for c in CRITERIA]
        ranking = dbc.Table(
            [
                html.Thead(html.Tr([html.Th(c) for c in rank_cols])),
                html.Tbody(rank_rows),
            ],
            bordered=True, hover=True, responsive=True, striped=True,
        )

        return fig, stats_table, ranking
