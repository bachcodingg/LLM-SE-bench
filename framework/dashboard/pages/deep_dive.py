"""
pages/deep_dive.py — Page 2: Per-model, per-criterion drill-down.

Shows grouped bar charts, box-plots of raw distributions, and a
Pareto frontier scatter plot for any user-selected criterion pair.
"""

from __future__ import annotations

from typing import Any

_DASH_AVAILABLE = True
try:
    import dash  # type: ignore[import-untyped]
    from dash import Input, Output, State, dcc, html  # type: ignore[import-untyped]
    import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    import plotly.express as px  # type: ignore[import-untyped]
except ImportError:
    _DASH_AVAILABLE = False


def deep_dive_layout() -> Any:
    """Return the layout component for the deep-dive page."""
    if not _DASH_AVAILABLE:
        return None

    criteria = ["correctness", "quality", "speed", "cost", "consistency"]

    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Deep Dive Analysis", className="mb-3"),
                html.P(
                    "Drill into per-model performance on each criterion. "
                    "Select two criteria for the Pareto frontier overlay.",
                    className="text-muted",
                ),
            ]),
        ]),

        # Bar chart section
        dbc.Row([
            dbc.Col([
                html.H5("Normalised Scores by Model"),
                dcc.Loading(dcc.Graph(id="deep-bar-chart")),
            ]),
        ], className="mb-4"),

        # Raw values heatmap
        dbc.Row([
            dbc.Col([
                html.H5("Raw Metric Values"),
                dcc.Loading(dcc.Graph(id="deep-heatmap")),
            ]),
        ], className="mb-4"),

        # Pareto scatter
        dbc.Row([
            dbc.Col([
                dbc.Label("X axis criterion"),
                dcc.Dropdown(
                    id="pareto-x",
                    options=[{"label": c.title(), "value": c} for c in criteria],
                    value="correctness",
                    clearable=False,
                ),
            ], md=3),
            dbc.Col([
                dbc.Label("Y axis criterion"),
                dcc.Dropdown(
                    id="pareto-y",
                    options=[{"label": c.title(), "value": c} for c in criteria],
                    value="cost",
                    clearable=False,
                ),
            ], md=3),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dcc.Loading(dcc.Graph(id="pareto-scatter")),
            ]),
        ]),
    ])


def register_deep_dive(app: Any) -> None:
    """Register deep-dive callbacks on *app*."""
    if not _DASH_AVAILABLE:
        return

    @app.callback(
        [
            Output("deep-bar-chart", "figure"),
            Output("deep-heatmap", "figure"),
        ],
        [Input("summaries-store", "data")],
    )
    def update_bar_and_heatmap(summaries_data: list[dict]) -> tuple:
        from framework.decision_matrix import CRITERIA, DecisionMatrixEngine

        try:
            from contracts import StatisticalSummary
        except ImportError:
            from pydantic import BaseModel, Field as PField
            from datetime import datetime

            class StatisticalSummary(BaseModel):  # type: ignore[no-redef]
                metric_name: str = ""
                model_id: str = ""
                mean: float = 0.0
                std_dev: float = 0.0
                computed_at: datetime = PField(default_factory=datetime.utcnow)

        summaries = [StatisticalSummary(**d) for d in (summaries_data or [])]
        engine = DecisionMatrixEngine(summaries)
        engine.build()

        # ── Grouped bar chart ─────────────────────────────────────────
        fig_bar = go.Figure()
        colors = ["#3182ce", "#e53e3e", "#38a169", "#d69e2e", "#805ad5"]
        for i, dm in enumerate(engine.matrices):
            scores_map = {s.criterion: s.normalised_value for s in dm.scores}
            fig_bar.add_trace(
                go.Bar(
                    name=dm.model_id,
                    x=[c.title() for c in CRITERIA],
                    y=[scores_map.get(c, 0) for c in CRITERIA],
                    marker_color=colors[i % len(colors)],
                )
            )
        fig_bar.update_layout(
            barmode="group",
            yaxis=dict(title="Normalised Score", range=[0, 1.05]),
            height=350,
            margin=dict(t=30, b=40),
        )

        # ── Heatmap of raw values ─────────────────────────────────────
        model_ids = [dm.model_id for dm in engine.matrices]
        z_vals = []
        for dm in engine.matrices:
            scores_map = {s.criterion: s.raw_value for s in dm.scores}
            z_vals.append([scores_map.get(c, 0) for c in CRITERIA])

        fig_heat = go.Figure(
            go.Heatmap(
                z=z_vals,
                x=[c.title() for c in CRITERIA],
                y=model_ids,
                colorscale="Blues",
                text=[[f"{v:.3f}" for v in row] for row in z_vals],
                texttemplate="%{text}",
            )
        )
        fig_heat.update_layout(
            height=250,
            margin=dict(t=20, b=40),
        )

        return fig_bar, fig_heat

    # ── Pareto scatter ────────────────────────────────────────────────
    @app.callback(
        Output("pareto-scatter", "figure"),
        [
            Input("pareto-x", "value"),
            Input("pareto-y", "value"),
        ],
        [State("summaries-store", "data")],
    )
    def update_pareto(
        cx: str, cy: str, summaries_data: list[dict]
    ) -> go.Figure:
        from framework.tradeoffs import TradeoffAnalyzer

        try:
            from contracts import StatisticalSummary
        except ImportError:
            from pydantic import BaseModel, Field as PField
            from datetime import datetime

            class StatisticalSummary(BaseModel):  # type: ignore[no-redef]
                metric_name: str = ""
                model_id: str = ""
                mean: float = 0.0
                std_dev: float = 0.0
                computed_at: datetime = PField(default_factory=datetime.utcnow)

        summaries = [StatisticalSummary(**d) for d in (summaries_data or [])]
        analyzer = TradeoffAnalyzer(summaries)
        points = analyzer.frontier_2d(cx, cy)

        fig = go.Figure()
        for p in points:
            color = "#38a169" if p.is_pareto else "#e53e3e"
            symbol = "star" if p.is_pareto else "circle"
            fig.add_trace(
                go.Scatter(
                    x=[p.values.get(cx, 0)],
                    y=[p.values.get(cy, 0)],
                    mode="markers+text",
                    marker=dict(size=14, color=color, symbol=symbol),
                    text=[p.model_id],
                    textposition="top center",
                    name=f"{p.model_id} ({'frontier' if p.is_pareto else 'dominated'})",
                    showlegend=True,
                )
            )

        # Draw frontier line
        frontier = sorted(
            [p for p in points if p.is_pareto],
            key=lambda p: p.values.get(cx, 0),
        )
        if len(frontier) > 1:
            fig.add_trace(
                go.Scatter(
                    x=[p.values.get(cx, 0) for p in frontier],
                    y=[p.values.get(cy, 0) for p in frontier],
                    mode="lines",
                    line=dict(dash="dash", color="#38a169", width=1),
                    showlegend=False,
                )
            )

        fig.update_layout(
            xaxis=dict(title=f"{cx.title()} (normalised)", range=[-0.05, 1.1]),
            yaxis=dict(title=f"{cy.title()} (normalised)", range=[-0.05, 1.1]),
            title=f"Pareto Frontier: {cx.title()} vs {cy.title()}",
            height=400,
            margin=dict(t=50, b=50),
        )
        return fig
