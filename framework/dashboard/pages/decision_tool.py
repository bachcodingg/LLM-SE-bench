"""
pages/decision_tool.py — Page 3: Interactive weight sliders + live matrix.

Users drag five sliders (auto-normalised to sum = 1) and the decision
matrix re-ranks in real time.  Includes a sensitivity sparkline and
constraint toggles.
"""

from __future__ import annotations

from typing import Any

_DASH_AVAILABLE = True
try:
    import dash  # type: ignore[import-untyped]
    import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    from dash import (  # type: ignore[import-untyped]
        Input,
        Output,
        State,
        callback_context,
        dcc,
        html,
    )
except ImportError:
    _DASH_AVAILABLE = False


_CRITERIA = ["correctness", "quality", "speed", "cost", "consistency"]
_DEFAULTS = {"correctness": 20, "quality": 20, "speed": 20, "cost": 20, "consistency": 20}
_PRESETS = {
    "equal": {"correctness": 20, "quality": 20, "speed": 20, "cost": 20, "consistency": 20},
    "devops": {"correctness": 25, "quality": 10, "speed": 35, "cost": 25, "consistency": 5},
    "audit": {"correctness": 30, "quality": 30, "speed": 5, "cost": 15, "consistency": 20},
    "budget": {"correctness": 30, "quality": 10, "speed": 15, "cost": 40, "consistency": 5},
}


def decision_tool_layout() -> Any:
    """Return the layout for the decision tool page."""
    if not _DASH_AVAILABLE:
        return None

    sliders = []
    for c in _CRITERIA:
        sliders.append(
            dbc.Row([
                dbc.Col(dbc.Label(c.title()), md=2),
                dbc.Col(
                    dcc.Slider(
                        id=f"weight-{c}",
                        min=0, max=100, step=1,
                        value=_DEFAULTS[c],
                        marks={0: "0", 50: "50", 100: "100"},
                        tooltip={"placement": "bottom"},
                    ),
                    md=8,
                ),
                dbc.Col(html.Span(id=f"weight-{c}-pct"), md=2),
            ], className="mb-2")
        )

    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Interactive Decision Tool", className="mb-3"),
                html.P(
                    "Adjust criterion weights with the sliders below.  "
                    "Weights are auto-normalised so they sum to 100%.  "
                    "The decision matrix updates in real time.",
                    className="text-muted",
                ),
            ]),
        ]),

        # Preset buttons
        dbc.Row([
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button("Equal", id="preset-equal", color="outline-primary", size="sm"),
                    dbc.Button("DevOps", id="preset-devops", color="outline-success", size="sm"),
                    dbc.Button("Audit", id="preset-audit", color="outline-warning", size="sm"),
                    dbc.Button("Budget", id="preset-budget", color="outline-danger", size="sm"),
                ]),
            ], className="mb-3"),
        ]),

        # Sliders
        dbc.Card(
            dbc.CardBody(sliders),
            className="mb-4",
        ),

        # Results
        dbc.Row([
            dbc.Col([
                html.H5("Live Rankings"),
                html.Div(id="live-rankings"),
            ], md=6),
            dbc.Col([
                html.H5("Score Breakdown"),
                dcc.Loading(dcc.Graph(id="live-bar")),
            ], md=6),
        ]),

        # Sensitivity
        dbc.Row([
            dbc.Col([
                html.H5("Sensitivity Analysis", className="mt-4"),
                html.P(
                    "How does the top pick change as each criterion weight "
                    "varies from 0% to 100%?",
                    className="text-muted small",
                ),
                dcc.Loading(dcc.Graph(id="sensitivity-chart")),
            ]),
        ]),
    ])


def register_decision_tool(app: Any) -> None:
    """Register decision tool callbacks on *app*."""
    if not _DASH_AVAILABLE:
        return

    # ── Preset buttons → slider values ────────────────────────────────
    for preset_name in _PRESETS:
        @app.callback(
            [Output(f"weight-{c}", "value") for c in _CRITERIA],
            [Input(f"preset-{preset_name}", "n_clicks")],
            prevent_initial_call=True,
        )
        def _apply_preset(n, _name=preset_name):  # noqa: B023
            vals = _PRESETS[_name]
            return [vals[c] for c in _CRITERIA]

    # ── Main update ───────────────────────────────────────────────────
    @app.callback(
        [
            Output("live-rankings", "children"),
            Output("live-bar", "figure"),
            Output("sensitivity-chart", "figure"),
        ] + [Output(f"weight-{c}-pct", "children") for c in _CRITERIA],
        [Input(f"weight-{c}", "value") for c in _CRITERIA],
        [State("summaries-store", "data")],
    )
    def update_tool(*args) -> tuple:
        slider_vals = list(args[:5])
        summaries_data = args[5]

        # Normalise
        total = sum(slider_vals) or 1
        weights = {
            c: slider_vals[i] / total
            for i, c in enumerate(_CRITERIA)
        }
        pct_labels = [f"{weights[c]:.0%}" for c in _CRITERIA]

        from framework.decision_matrix import CRITERIA, DecisionMatrixEngine

        try:
            from contracts import StatisticalSummary
        except ImportError:
            from datetime import datetime

            from pydantic import BaseModel
            from pydantic import Field as PField

            class StatisticalSummary(BaseModel):  # type: ignore[no-redef]
                metric_name: str = ""
                model_id: str = ""
                mean: float = 0.0
                std_dev: float = 0.0
                computed_at: datetime = PField(default_factory=datetime.utcnow)

        summaries = [StatisticalSummary(**d) for d in (summaries_data or [])]
        engine = DecisionMatrixEngine(summaries, weights=weights)
        matrices = engine.build()

        # Rankings table
        rank_rows = []
        for dm in matrices:
            cls = "table-success" if dm.rank == 1 else ""
            rank_rows.append(
                html.Tr([
                    html.Td(html.Strong(str(dm.rank))),
                    html.Td(dm.model_id),
                    html.Td(f"{dm.weighted_total:.4f}"),
                ], className=cls)
            )

        rankings_table = dbc.Table(
            [
                html.Thead(html.Tr([
                    html.Th("#"), html.Th("Model"), html.Th("Score"),
                ])),
                html.Tbody(rank_rows),
            ],
            bordered=True, hover=True, responsive=True, size="sm",
        )

        # Bar chart
        fig_bar = go.Figure()
        colors = ["#3182ce", "#e53e3e", "#38a169", "#d69e2e", "#805ad5"]
        for i, dm in enumerate(matrices):
            scores_map = {s.criterion: s for s in dm.scores}
            fig_bar.add_trace(
                go.Bar(
                    name=dm.model_id,
                    x=[c.title() for c in CRITERIA],
                    y=[
                        scores_map[c].normalised_value * scores_map[c].weight
                        if c in scores_map else 0
                        for c in CRITERIA
                    ],
                    marker_color=colors[i % len(colors)],
                )
            )
        fig_bar.update_layout(
            barmode="stack",
            yaxis=dict(title="Weighted Contribution"),
            height=300,
            margin=dict(t=20, b=40),
        )

        # Sensitivity
        fig_sens = go.Figure()
        for criterion in CRITERIA:
            sa = engine.sensitivity_analysis(criterion, steps=11)
            model_ids = list(sa[0]["totals"].keys()) if sa else []
            for mid in model_ids:
                fig_sens.add_trace(
                    go.Scatter(
                        x=[s["weight_value"] for s in sa],
                        y=[s["totals"].get(mid, 0) for s in sa],
                        mode="lines",
                        name=f"{mid} ({criterion})",
                        visible="legendonly" if criterion != "correctness" else True,
                    )
                )

        fig_sens.update_layout(
            xaxis=dict(title="Criterion Weight"),
            yaxis=dict(title="Weighted Total"),
            height=350,
            margin=dict(t=20, b=40),
        )

        return (rankings_table, fig_bar, fig_sens) + tuple(pct_labels)
