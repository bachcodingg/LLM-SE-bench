"""
framework.dashboard.pages.refactoring — why a refactoring was or was not credited.

The God Class tasks have two bars to clear: the behaviour suite still passes,
and :mod:`quality.structural` credits the new structure. A refactoring that
changes nothing clears the first and not the second, which is the gap this
page exists to show.

Three views:

**By model.** Behaviour kept against structure credited, from the run
artifact. The distance between the two bars is the finding.

**Check a refactoring.** Paste code for a task and run it through the same
``evaluate_patch`` the MCP server exposes: tests in Docker, then the
structural score, then the reasons in plain sentences.

**Every evaluation.** Each recorded refactoring with its per-axis scores and
the reasons it was not credited.

Dash is imported lazily, as in the other pages.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from quality.structural import SCORE_WEIGHTS

logger = logging.getLogger(__name__)

__all__ = [
    "layout",
    "register_callbacks",
    "load_refactoring_rows",
    "summarise_by_model",
    "axes_figure",
]

DEFAULT_RUN = Path("results/2026-10-run/evaluations.csv")
DATASET = "godclass"
AXES = list(SCORE_WEIGHTS)
AXIS_LABELS = {
    "coupling_reduction": "Coupling",
    "cohesion_improvement": "Cohesion",
    "complexity_dispersal": "Dispersal",
    "size_reduction": "Size",
}


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def load_refactoring_rows(path: Path | str = DEFAULT_RUN) -> list[dict[str, Any]]:
    """God Class rows of a run artifact. Empty when the artifact is absent."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["dataset"] == DATASET]


def summarise_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per model: share of refactorings that kept behaviour, and that were credited."""
    summary: list[dict[str, Any]] = []
    for model in sorted({row["model_id"] for row in rows}):
        mine = [row for row in rows if row["model_id"] == model]
        kept = sum(
            1 for row in mine
            if _truthy(row["compile_success"])
            and row["tests_total"] not in ("", "0")
            and row["tests_passed"] == row["tests_total"]
        )
        credited = sum(1 for row in mine if _truthy(row.get("structural_credited")))
        summary.append({
            "model_id": model,
            "evaluations": len(mine),
            "behaviour_kept": kept / len(mine),
            "structure_credited": credited / len(mine),
        })
    return summary


def axes_figure(axes: dict[str, float], title: str = "") -> Any:
    """Horizontal bar per structural axis. Negative bars are worse than the original."""
    import plotly.graph_objects as go

    values = [float(axes.get(axis) or 0.0) for axis in AXES]
    figure = go.Figure(go.Bar(
        x=values,
        y=[AXIS_LABELS[axis] for axis in AXES],
        orientation="h",
        marker_color=["#38a169" if value > 0 else "#e53e3e" if value < 0 else "#a0aec0"
                      for value in values],
        text=[f"{value:+.2f}" for value in values],
        textposition="auto",
    ))
    figure.update_layout(
        title=title, height=230, margin={"l": 80, "r": 20, "t": 40 if title else 10, "b": 30},
        xaxis={"range": [-1, 1], "zeroline": True, "title": "change vs original (-1 to +1)"},
    )
    return figure


def _model_figure(summary: list[dict[str, Any]]) -> Any:
    import plotly.graph_objects as go

    models = [row["model_id"] for row in summary]
    figure = go.Figure([
        go.Bar(name="Behaviour kept (tests pass)", x=models,
               y=[row["behaviour_kept"] for row in summary], marker_color="#6b7f99"),
        go.Bar(name="Structure credited", x=models,
               y=[row["structure_credited"] for row in summary], marker_color="#c15f3c"),
    ])
    figure.update_layout(
        barmode="group", height=320, yaxis={"tickformat": ".0%", "range": [0, 1]},
        margin={"t": 30}, legend={"orientation": "h", "y": -0.2},
    )
    return figure


def _task_ids() -> list[str]:
    from bench.datasets.godclass import GodClassDataset

    return GodClassDataset(data_dir="data").list_problem_ids()


def _original(task_id: str) -> str:
    from bench.datasets.godclass import GodClassDataset

    return GodClassDataset(data_dir="data").get_original_code(task_id)


def layout(run_path: Path | str = DEFAULT_RUN) -> Any:
    import dash_bootstrap_components as dbc
    from dash import dash_table, dcc, html

    rows = load_refactoring_rows(run_path)
    summary = summarise_by_model(rows)
    tasks = _task_ids()

    by_model: Any
    if summary:
        by_model = dcc.Graph(figure=_model_figure(summary), id="refactor-by-model")
    else:
        by_model = dbc.Alert(
            f"No run artifact at {run_path}. Build one with scripts/build_run_artifact.py.",
            color="secondary",
        )

    table_rows = [
        {
            "model": row["model_id"],
            "task": row["problem_id"],
            "sample": row["sample_index"],
            "tests": f"{row['tests_passed']}/{row['tests_total']}",
            "credited": "yes" if _truthy(row.get("structural_credited")) else "no",
            "composite": row.get("structural_composite", ""),
            **{AXIS_LABELS[axis]: row.get(f"structural_{axis}", "") for axis in AXES},
            "why not credited": row.get("structural_reasons", ""),
        }
        for row in rows
    ]

    return html.Div([
        html.H3("Refactoring: behaviour kept is not structure improved"),
        html.P(
            "A refactoring that changes nothing passes every test. Each God Class "
            "result must also be credited by quality.structural: coupling and "
            "cohesion of the extracted classes, complexity dispersal, size, and "
            "anti-gaming checks, including that the API-preserving facade "
            "actually gave up the original's logic.",
            className="text-muted",
        ),
        html.H5("By model", className="mt-3"),
        by_model,
        html.H5("Check a refactoring", className="mt-4"),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(tasks, tasks[0] if tasks else None, id="refactor-task",
                             clearable=False),
                dcc.Textarea(
                    id="refactor-code", value=_original(tasks[0]) if tasks else "",
                    style={"width": "100%", "height": "360px", "fontFamily": "monospace",
                           "fontSize": "12px"},
                    className="mt-2",
                ),
                dbc.Button("Evaluate", id="refactor-run", color="primary", className="mt-2"),
                html.Small(
                    " Runs the task's JUnit suite in the Docker sandbox, then scores "
                    "the structure. Same code path as the MCP tool evaluate_patch.",
                    className="text-muted",
                ),
            ], md=6),
            dbc.Col(dcc.Loading(html.Div(id="refactor-result")), md=6),
        ]),
        html.H5("Every recorded refactoring", className="mt-4"),
        dash_table.DataTable(
            data=table_rows,
            columns=[{"name": key, "id": key} for key in (table_rows[0] if table_rows else {})],
            filter_action="native", sort_action="native", page_size=15,
            style_cell={"textAlign": "left", "fontSize": "12px", "whiteSpace": "normal",
                        "maxWidth": "420px"},
        ),
    ])


def render_result(result: Any) -> Any:
    """The verdict, the per-axis chart, and the reasons, for one evaluation."""
    import dash_bootstrap_components as dbc
    from dash import dcc, html

    if not result.ok:
        return dbc.Alert(result.error, color="danger")

    credited = bool(result.structural.get("credited"))
    colour = "success" if result.verdict == "pass" else "warning" if result.verdict == "partial" \
        else "danger"
    headline = {
        "pass": "Pass: behaviour kept and structure credited.",
        "partial": "Not credited: the tests pass, but the structure did not earn credit.",
        "fail": "Fail: the tests do not pass.",
        "error": "Error: the code did not compile.",
    }.get(result.verdict, result.verdict)

    children: list[Any] = [
        dbc.Alert([html.Strong(headline), html.Br(),
                   f"Tests {result.tests_passed}/{result.tests_total}"
                   f"{'' if result.executed else ' (NOT executed: Docker unavailable)'}"
                   f" · structurally credited: {'yes' if credited else 'no'}"],
                  color=colour),
    ]
    axes = result.structural.get("structure", {}).get("axes", {})
    if axes:
        composite = result.structural["structure"].get("composite", 0.0)
        children.append(dcc.Graph(figure=axes_figure(axes, f"Composite {composite:+.3f}")))
    if result.structural_reasons:
        children.append(html.H6("Why"))
        children.append(html.Ul([html.Li(reason) for reason in result.structural_reasons]))
    return html.Div(children)


def register_callbacks(app: Any) -> None:
    from dash import Input, Output, State

    @app.callback(Output("refactor-code", "value"), Input("refactor-task", "value"),
                  prevent_initial_call=True)
    def _load_original(task_id: str) -> str:
        return _original(task_id) if task_id else ""

    @app.callback(Output("refactor-result", "children"), Input("refactor-run", "n_clicks"),
                  State("refactor-task", "value"), State("refactor-code", "value"),
                  prevent_initial_call=True)
    def _evaluate(_: int, task_id: str, code: str) -> Any:
        from mcp_servers.tools.bench import evaluate_patch

        return render_result(evaluate_patch(task_id, code or ""))
