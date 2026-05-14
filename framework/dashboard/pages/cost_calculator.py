"""
pages/cost_calculator.py — Page 4: Project-size cost projection tool.

Users enter project parameters (number of problems, runs per problem,
average tokens) and the tool projects total cost per model based on
observed per-call costs from the statistical summaries.
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


def cost_calculator_layout() -> Any:
    """Return the layout for the cost calculator page."""
    if not _DASH_AVAILABLE:
        return None

    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Cost Calculator", className="mb-3"),
                html.P(
                    "Estimate total costs for your project based on "
                    "observed per-call costs.  Adjust the parameters below.",
                    className="text-muted",
                ),
            ]),
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("Project Parameters"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Number of problems"),
                            dbc.Input(
                                id="calc-problems", type="number",
                                value=100, min=1, max=10000, step=1,
                            ),
                        ], md=4),
                        dbc.Col([
                            dbc.Label("Runs per problem"),
                            dbc.Input(
                                id="calc-runs", type="number",
                                value=3, min=1, max=100, step=1,
                            ),
                        ], md=4),
                        dbc.Col([
                            dbc.Label("Models to evaluate"),
                            dbc.Input(
                                id="calc-models", type="number",
                                value=3, min=1, max=20, step=1,
                            ),
                        ], md=4),
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Avg. prompt tokens per call"),
                            dbc.Input(
                                id="calc-prompt-tokens", type="number",
                                value=800, min=10, max=100000, step=10,
                            ),
                        ], md=4),
                        dbc.Col([
                            dbc.Label("Avg. completion tokens per call"),
                            dbc.Input(
                                id="calc-completion-tokens", type="number",
                                value=1500, min=10, max=100000, step=10,
                            ),
                        ], md=4),
                        dbc.Col([
                            dbc.Label("Budget limit (USD)"),
                            dbc.Input(
                                id="calc-budget", type="number",
                                value=200.0, min=0, max=100000, step=1,
                            ),
                        ], md=4),
                    ]),
                ])),
            ]),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.H5("Cost Projections"),
                dcc.Loading(dcc.Graph(id="cost-bar-chart")),
            ], md=7),
            dbc.Col([
                html.H5("Summary"),
                html.Div(id="cost-summary"),
            ], md=5),
        ]),

        dbc.Row([
            dbc.Col([
                html.H5("Cost vs. Correctness Trade-off", className="mt-4"),
                dcc.Loading(dcc.Graph(id="cost-vs-accuracy")),
            ]),
        ]),
    ])


def register_cost_calculator(app: Any) -> None:
    """Register cost calculator callbacks on *app*."""
    if not _DASH_AVAILABLE:
        return

    # ── Published pricing (USD per 1K tokens, approximate Q1 2025) ───
    _PRICING = {
        "claude-3.5-sonnet": {"prompt": 0.003, "completion": 0.015},
        "gpt-4-turbo": {"prompt": 0.010, "completion": 0.030},
        "gemini-1.5-pro": {"prompt": 0.00125, "completion": 0.005},
    }
    _DEFAULT_PRICING = {"prompt": 0.005, "completion": 0.015}

    @app.callback(
        [
            Output("cost-bar-chart", "figure"),
            Output("cost-summary", "children"),
            Output("cost-vs-accuracy", "figure"),
        ],
        [
            Input("calc-problems", "value"),
            Input("calc-runs", "value"),
            Input("calc-models", "value"),
            Input("calc-prompt-tokens", "value"),
            Input("calc-completion-tokens", "value"),
            Input("calc-budget", "value"),
        ],
        [State("summaries-store", "data")],
    )
    def update_costs(
        n_problems, n_runs, n_models,
        avg_prompt, avg_completion, budget,
        summaries_data,
    ):
        from framework.decision_matrix import DecisionMatrixEngine, CRITERION_METRIC_MAP

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

        n_problems = n_problems or 100
        n_runs = n_runs or 3
        n_models_param = n_models or 3
        avg_prompt = avg_prompt or 800
        avg_completion = avg_completion or 1500
        budget = budget or 200.0

        total_calls_per_model = n_problems * n_runs

        summaries = [StatisticalSummary(**d) for d in (summaries_data or [])]

        # Extract model IDs and per-call costs
        model_ids = sorted({s.model_id for s in summaries})
        if not model_ids:
            model_ids = list(_PRICING.keys())[:n_models_param]

        # Get observed costs or estimate from pricing
        model_costs: dict[str, float] = {}
        for mid in model_ids:
            # Try to get from summaries
            cost_summary = None
            for s in summaries:
                if s.model_id == mid and s.metric_name == "cost_usd":
                    cost_summary = s
                    break

            if cost_summary and cost_summary.mean > 0:
                model_costs[mid] = cost_summary.mean
            else:
                pricing = _PRICING.get(mid, _DEFAULT_PRICING)
                per_call = (
                    avg_prompt / 1000 * pricing["prompt"]
                    + avg_completion / 1000 * pricing["completion"]
                )
                model_costs[mid] = per_call

        # Total projected costs
        projected: dict[str, float] = {
            mid: cost * total_calls_per_model
            for mid, cost in model_costs.items()
        }

        # Get pass rates for cost-vs-accuracy
        pass_rates: dict[str, float] = {}
        for s in summaries:
            if s.metric_name == "pass_rate":
                pass_rates[s.model_id] = s.mean

        # ── Bar chart ─────────────────────────────────────────────────
        fig_bar = go.Figure()
        colors = []
        for mid in model_ids:
            if projected.get(mid, 0) > budget:
                colors.append("#e53e3e")
            else:
                colors.append("#38a169")

        fig_bar.add_trace(
            go.Bar(
                x=model_ids,
                y=[projected.get(mid, 0) for mid in model_ids],
                marker_color=colors,
                text=[f"${projected.get(mid, 0):.2f}" for mid in model_ids],
                textposition="auto",
            )
        )
        fig_bar.add_hline(
            y=budget, line_dash="dash", line_color="red",
            annotation_text=f"Budget ${budget:.0f}",
        )
        fig_bar.update_layout(
            yaxis=dict(title="Projected Total Cost (USD)"),
            xaxis=dict(title="Model"),
            height=350,
            margin=dict(t=30, b=40),
        )

        # ── Summary table ─────────────────────────────────────────────
        summary_rows = []
        for mid in model_ids:
            per_call = model_costs.get(mid, 0)
            total = projected.get(mid, 0)
            within = total <= budget
            cls = "table-success" if within else "table-danger"
            summary_rows.append(
                html.Tr([
                    html.Td(mid),
                    html.Td(f"${per_call:.4f}"),
                    html.Td(f"${total:.2f}"),
                    html.Td(f"{total_calls_per_model}"),
                    html.Td("✓" if within else "✗"),
                ], className=cls)
            )

        summary_table = dbc.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Model"),
                    html.Th("$/call"),
                    html.Th("Total $"),
                    html.Th("Calls"),
                    html.Th("In budget"),
                ])),
                html.Tbody(summary_rows),
            ],
            bordered=True, hover=True, responsive=True, size="sm",
        )

        total_all = sum(projected.values())
        info = html.Div([
            summary_table,
            html.P(
                f"Total API calls: {total_calls_per_model * len(model_ids):,}  |  "
                f"Total cost (all models): ${total_all:.2f}",
                className="mt-2 text-muted small",
            ),
        ])

        # ── Cost vs accuracy scatter ──────────────────────────────────
        fig_scatter = go.Figure()
        scatter_colors = ["#3182ce", "#e53e3e", "#38a169", "#d69e2e", "#805ad5"]
        for i, mid in enumerate(model_ids):
            acc = pass_rates.get(mid, 0.5)
            cost = projected.get(mid, 0)
            fig_scatter.add_trace(
                go.Scatter(
                    x=[cost],
                    y=[acc],
                    mode="markers+text",
                    marker=dict(size=16, color=scatter_colors[i % len(scatter_colors)]),
                    text=[mid],
                    textposition="top center",
                    name=mid,
                )
            )
        fig_scatter.add_vline(
            x=budget, line_dash="dash", line_color="red",
            annotation_text="Budget",
        )
        fig_scatter.update_layout(
            xaxis=dict(title="Projected Cost (USD)"),
            yaxis=dict(title="Pass Rate", range=[0, 1.05]),
            height=350,
            margin=dict(t=30, b=40),
        )

        return fig_bar, info, fig_scatter
