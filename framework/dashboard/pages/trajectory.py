"""
framework.dashboard.pages.trajectory — replay and side-by-side diff (M9).

The single most compelling view in the dashboard, and the one nobody
builds: two models on the same task, aligned step by step, with the moment
they diverged marked. A leaderboard tells you Claude scored higher. This
tells you Claude opened the right file at step 2 and Gemini was still
listing directories at step 9.

Three views:

**Replay.** Step through one episode: the thought, the tool call, the result,
the running cost, the test state after each step.

**Diff.** Two episodes side by side, aligned positionally, divergence
marked.

**Failure taxonomy.** The distribution over failure classes, drillable to
the episodes behind each one.

Dash is imported lazily throughout. The dashboard is optional; the analysis
underneath it is not, and importing this module must not require a web
framework.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["layout", "register_callbacks", "build_replay_rows", "build_diff_rows"]

#: Colour per failure class. Environment failures are grey on purpose: they
#: are the harness's fault, and colouring them like a model failure is how a
#: broken sandbox gets read as a weak model.
FAILURE_COLOURS: dict[str, str] = {
    "solved": "#38a169",
    "localisation_failure": "#dd6b20",
    "environment_failure": "#a0aec0",
    "syntactic_failure": "#d69e2e",
    "semantic_failure": "#3182ce",
    "regression": "#e53e3e",
    "budget_exhaustion": "#805ad5",
    "loop": "#b83280",
    "refusal_or_abandonment": "#718096",
    "tamper": "#c53030",
    "unknown": "#cbd5e0",
}


def load_trajectories(store_dir: Path | str = "results/trajectories") -> list[Any]:
    """Every stored trajectory, newest first. Empty when there are none."""
    from agent.trajectory import TrajectoryStore

    try:
        return sorted(
            TrajectoryStore(store_dir).load_all(),
            key=lambda trajectory: trajectory.started_at,
            reverse=True,
        )
    except Exception as exc:
        logger.warning("Could not load trajectories from %s: %s", store_dir, exc)
        return []


def build_replay_rows(trajectory: Any) -> list[dict[str, Any]]:
    """One row per step, with cost accumulated as the episode progresses.

    The running total is what makes a replay useful for cost work: it shows
    *where* the money went, and it is almost always one or two steps rather
    than all of them.
    """
    rows: list[dict[str, Any]] = []
    running_cost = 0.0
    running_tokens = 0

    for step in trajectory.steps:
        running_cost += step.cost_eur
        running_tokens += step.total_tokens
        rows.append({
            "step": step.step_index,
            "tool": step.tool_name or "(no tool call)",
            "arguments": _short(step.tool_args),
            "thought": step.thought_text[:200],
            "result": step.result_preview[:200],
            "error": "yes" if step.tool_error else "",
            "files": ", ".join(step.files_touched),
            "tests": (
                f"{step.tests_passing_after}/{step.tests_total_after}"
                if step.tests_passing_after is not None else ""
            ),
            "tokens": step.total_tokens,
            "cost_eur": round(step.cost_eur, 6),
            "cumulative_cost_eur": round(running_cost, 6),
            "cumulative_tokens": running_tokens,
        })
    return rows


def build_diff_rows(left: Any, right: Any) -> tuple[list[dict[str, Any]], int | None]:
    """Aligned rows for two episodes, plus the divergence step."""
    from agent.analysis import diff_trajectories

    diff = diff_trajectories(left, right)
    rows = [
        {
            "step": row["step"],
            "left": row["left"],
            "right": row["right"],
            "left_tests": row["left_tests"] if row["left_tests"] is not None else "",
            "right_tests": row["right_tests"] if row["right_tests"] is not None else "",
            "diverged": "" if row["same"] else "<<<",
        }
        for row in diff.aligned
    ]
    return rows, diff.divergence_step


def _short(value: Any, limit: int = 80) -> str:
    import json

    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[: limit - 3] + "..."


def failure_distribution(trajectories: list[Any]) -> dict[str, Any]:
    """Failure classes across every loaded episode."""
    from agent.analysis import aggregate_failures, classify_failure

    verdicts = [classify_failure(trajectory) for trajectory in trajectories]
    summary = aggregate_failures(verdicts)
    summary["episodes_by_class"] = {}
    for trajectory, verdict in zip(trajectories, verdicts):
        summary["episodes_by_class"].setdefault(
            verdict.failure_class.value, []
        ).append(trajectory.episode_id)
    return summary


def layout(store_dir: Path | str = "results/trajectories") -> Any:
    """The page layout. Imports Dash lazily."""
    from dash import dash_table, dcc, html

    trajectories = load_trajectories(store_dir)
    options = [
        {
            "label": (
                f"{t.task_id or t.episode_id} — {t.model_id} "
                f"[{t.scaffold}] {'solved' if t.solved else t.stop_condition}"
            ),
            "value": t.episode_id,
        }
        for t in trajectories
    ]

    if not trajectories:
        return html.Div([
            html.H2("Trajectories"),
            html.P(
                "No episodes found. Run one with "
                "`llm-se-bench agent run --task <id> --model claude "
                "--budget-eur 0.50`, or `--dry-run` to see the view with "
                "an episode that costs nothing.",
                style={"color": "#718096"},
            ),
        ], style={"padding": "2rem"})

    table_style = {
        "style_cell": {
            "textAlign": "left", "fontFamily": "monospace", "fontSize": "12px",
            "whiteSpace": "normal", "height": "auto", "padding": "6px",
        },
        "style_header": {"fontWeight": "bold", "backgroundColor": "#edf2f7"},
        "page_size": 25,
    }

    return html.Div([
        html.H2("Trajectories"),
        html.P(
            f"{len(trajectories)} episode(s). Replay one, or align two on the "
            f"same task to see where they diverged.",
            style={"color": "#718096"},
        ),

        html.H3("Replay", style={"marginTop": "1.5rem"}),
        dcc.Dropdown(
            id="replay-episode", options=options,
            value=options[0]["value"], clearable=False,
        ),
        html.Div(id="replay-summary", style={"margin": "1rem 0"}),
        dash_table.DataTable(
            id="replay-table",
            columns=[
                {"name": "#", "id": "step"},
                {"name": "Tool", "id": "tool"},
                {"name": "Arguments", "id": "arguments"},
                {"name": "Result", "id": "result"},
                {"name": "Files", "id": "files"},
                {"name": "Tests", "id": "tests"},
                {"name": "EUR so far", "id": "cumulative_cost_eur"},
            ],
            style_data_conditional=[{
                "if": {"filter_query": '{error} = "yes"'},
                "backgroundColor": "#fff5f5",
            }],
            **table_style,
        ),

        html.H3("Side-by-side", style={"marginTop": "2rem"}),
        html.Div([
            dcc.Dropdown(
                id="diff-left", options=options, value=options[0]["value"],
                clearable=False, style={"width": "48%", "display": "inline-block"},
            ),
            dcc.Dropdown(
                id="diff-right", options=options,
                value=options[min(1, len(options) - 1)]["value"],
                clearable=False,
                style={"width": "48%", "display": "inline-block", "marginLeft": "4%"},
            ),
        ]),
        html.Div(id="diff-summary", style={"margin": "1rem 0"}),
        dash_table.DataTable(
            id="diff-table",
            columns=[
                {"name": "#", "id": "step"},
                {"name": "Left", "id": "left"},
                {"name": "Tests", "id": "left_tests"},
                {"name": "Right", "id": "right"},
                {"name": "Tests", "id": "right_tests"},
                {"name": "", "id": "diverged"},
            ],
            style_data_conditional=[{
                "if": {"filter_query": '{diverged} = "<<<"'},
                "backgroundColor": "#fffaf0",
            }],
            **table_style,
        ),

        html.H3("Failure taxonomy", style={"marginTop": "2rem"}),
        dcc.Graph(id="failure-chart"),
        html.Div(id="failure-note", style={"color": "#718096", "fontSize": "13px"}),

        dcc.Store(id="trajectory-store-dir", data=str(store_dir)),
    ], style={"padding": "2rem", "maxWidth": "1400px"})


def register_callbacks(app: Any, store_dir: Path | str = "results/trajectories") -> None:
    """Wire the page's interactivity."""
    import plotly.graph_objects as go
    from dash import Input, Output, html

    @app.callback(
        Output("replay-table", "data"),
        Output("replay-summary", "children"),
        Input("replay-episode", "value"),
    )
    def _replay(episode_id: str):
        from agent.analysis import classify_failure, compute_process_metrics
        from agent.trajectory import TrajectoryStore

        trajectory = TrajectoryStore(store_dir).load(episode_id)
        if trajectory is None:
            return [], html.P(f"Episode {episode_id} could not be loaded.")

        metrics = compute_process_metrics(trajectory)
        verdict = classify_failure(trajectory)

        badges = [
            ("Outcome", "solved" if trajectory.solved else trajectory.stop_condition),
            ("Failure class", verdict.failure_class.value),
            ("Steps", str(trajectory.num_steps)),
            ("First edit at", str(metrics.steps_to_first_edit or "never")),
            ("Repeated calls", str(metrics.repeated_calls)),
            ("Cost", f"EUR {trajectory.total_cost_eur:.4f}"),
            ("Cache hit rate", f"{metrics.cache_hit_rate:.0%}"),
        ]
        children = [
            html.Span(
                [html.Strong(f"{label}: "), value],
                style={
                    "marginRight": "1.5rem", "padding": "4px 8px",
                    "backgroundColor": "#f7fafc", "borderRadius": "4px",
                },
            )
            for label, value in badges
        ]
        if not trajectory.tests_executed:
            children.append(html.Div(
                "Tests were never executed in this episode — the sandbox fell "
                "back to a structural check. These numbers are not a "
                "benchmark score.",
                style={"color": "#c05621", "marginTop": "0.75rem"},
            ))
        if trajectory.context.get("compactions"):
            children.append(html.Div(
                f"The conversation was compacted "
                f"{trajectory.context['compactions']} time(s): this episode "
                f"dropped detail, which is worth knowing before blaming the "
                f"model for forgetting something.",
                style={"color": "#718096", "marginTop": "0.5rem"},
            ))

        return build_replay_rows(trajectory), children

    @app.callback(
        Output("diff-table", "data"),
        Output("diff-summary", "children"),
        Input("diff-left", "value"),
        Input("diff-right", "value"),
    )
    def _diff(left_id: str, right_id: str):
        from agent.trajectory import TrajectoryStore

        store = TrajectoryStore(store_dir)
        left, right = store.load(left_id), store.load(right_id)
        if left is None or right is None:
            return [], html.P("One of the episodes could not be loaded.")

        rows, divergence = build_diff_rows(left, right)
        note: list[Any] = [
            html.Span(f"{left.model_id} ({left.scaffold}) vs "
                      f"{right.model_id} ({right.scaffold})"),
        ]
        if left.task_id != right.task_id:
            note.append(html.Div(
                f"Different tasks ({left.task_id} and {right.task_id}). The "
                f"alignment is positional, so this comparison is not "
                f"meaningful.",
                style={"color": "#c53030", "marginTop": "0.5rem"},
            ))
        elif divergence is None:
            note.append(html.Div(
                "The two episodes took identical steps throughout.",
                style={"color": "#718096", "marginTop": "0.5rem"},
            ))
        else:
            note.append(html.Div(
                f"Diverged at step {divergence}.",
                style={"color": "#c05621", "marginTop": "0.5rem"},
            ))
        return rows, note

    @app.callback(
        Output("failure-chart", "figure"),
        Output("failure-note", "children"),
        Input("replay-episode", "value"),
    )
    def _failures(_episode_id: str):
        trajectories = load_trajectories(store_dir)
        if not trajectories:
            return go.Figure(), ""

        summary = failure_distribution(trajectories)
        distribution = summary["distribution"]
        figure = go.Figure(go.Bar(
            x=list(distribution.values()),
            y=list(distribution),
            orientation="h",
            marker_color=[
                FAILURE_COLOURS.get(name, "#cbd5e0") for name in distribution
            ],
        ))
        figure.update_layout(
            height=320,
            margin={"l": 200, "r": 20, "t": 20, "b": 40},
            xaxis_title="episodes",
            plot_bgcolor="white",
        )
        note = (
            f"{summary['episodes']} episode(s). "
            f"{summary['rule_coverage']:.0%} classified by deterministic rule; "
            f"{summary['ambiguous']} ambiguous. Environment failures are grey: "
            f"they are the harness's fault, not the model's."
        )
        return figure, note
