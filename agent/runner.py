"""
agent.runner — run the agent loop against llm-se-bench tasks.

:mod:`agent.loop` knows nothing about the benchmark: it takes a workspace, a
test runner and a prompt.  This module is the join — it turns a task id into
those three things, runs the episode, and stores the trajectory.

The comparison it exists to support is the one the guide calls out as
publishable on its own: **agent mode versus single-shot, same tasks, with
cost.** :func:`compare_to_single_shot` produces exactly that table.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from agent.loop import AgentConfig, AgentLoop, make_sandbox_test_runner
from agent.termination import TerminationPolicy
from agent.trajectory import Trajectory, TrajectoryStore
from agent.workspace import Workspace
from mcp_servers.registry import get_registry

logger = logging.getLogger(__name__)

__all__ = [
    "build_episode",
    "run_episode",
    "run_episodes",
    "compare_to_single_shot",
    "EpisodeSetupError",
]


class EpisodeSetupError(ValueError):
    """A task could not be turned into a runnable episode."""


TASK_PROMPT = """\
{prompt}

The workspace contains the code and its test suite. The test file is
read-only: fix the code so the existing tests pass. Call list_dir to see
what is there.
"""


def _class_name_in(source: str) -> str:
    """Public class name declared in *source*, or '' when there is none."""
    import re

    match = re.search(r"public\s+(?:final\s+|abstract\s+)?class\s+(\w+)", source)
    return match.group(1) if match else ""


def build_episode(
    task_id: str,
    data_dir: str = "data",
) -> tuple[Workspace, str, str, str]:
    """Turn a task id into ``(workspace, task_prompt, test_class, dataset)``.

    The test file goes into the workspace's read-only set. That is the
    difference between measuring whether a model can fix code and measuring
    whether it can notice the tests are editable.

    Raises:
        EpisodeSetupError: when the task is unknown, or has no test suite to
            work against.
    """
    registry = get_registry(data_dir)
    location = registry.locate(task_id)
    if location is None:
        raise EpisodeSetupError(
            f"Unknown task {task_id!r}. {len(registry.task_ids())} tasks are available."
        )

    adapter = location.adapter
    # Called for its side effect: raises KeyError on an id the registry
    # knows but the adapter does not, which is a bug worth surfacing here
    # rather than three steps into the episode.
    adapter.get_problem(task_id)

    junit_code = ""
    getter = getattr(adapter, "get_junit_code", None)
    if callable(getter):
        try:
            junit_code = getter(task_id) or ""
        except Exception:
            junit_code = ""
    if not junit_code.strip():
        raise EpisodeSetupError(
            f"Task {task_id!r} has no JUnit suite, so there is nothing for the "
            f"agent to iterate against. Agent mode needs an executable target."
        )

    test_class = _class_name_in(junit_code) or f"{task_id}Test"
    files = {f"{test_class}.java": junit_code}

    # Starting code: the buggy class or the God Class. Code-generation
    # tasks start empty, which is legitimate — the agent creates the file.
    starting_source = ""
    for accessor in ("get_buggy_code", "get_original_code"):
        method = getattr(adapter, accessor, None)
        if not callable(method):
            continue
        try:
            starting_source = method(task_id) or ""
        except Exception:
            continue
        if starting_source:
            break

    if starting_source:
        source_class = _class_name_in(starting_source) or task_id
        files[f"{source_class}.java"] = starting_source

    workspace = Workspace(files=files, read_only={f"{test_class}.java"})
    prompt = TASK_PROMPT.format(prompt=adapter.format_prompt(task_id))
    return workspace, prompt, test_class, location.dataset_name


def run_episode(
    task_id: str,
    model_id: str,
    client: Any,
    policy: TerminationPolicy,
    data_dir: str = "data",
    scaffold: str = "react",
    store: TrajectoryStore | None = None,
    dry_run: bool = False,
    **config_kwargs: Any,
) -> Trajectory:
    """Run one episode and return — and optionally store — its trajectory.

    Parameters
    ----------
    client
        Anything with ``send_conversation``. An ``LLMClient`` in production.
    policy
        Ceilings for the episode. ``max_cost_eur`` is required.
    dry_run
        Force the sandbox's structural check instead of real execution. The
        agent is told, in the tool output, that nothing was executed.
    """
    workspace, prompt, test_class, dataset = build_episode(task_id, data_dir)

    loop = AgentLoop(
        client=client,
        workspace=workspace,
        test_runner=make_sandbox_test_runner(
            test_class_name=test_class, dry_run=dry_run
        ),
        config=AgentConfig(
            model_id=model_id,
            policy=policy,
            scaffold=scaffold,
            task_id=task_id,
            dataset=dataset,
            **config_kwargs,
        ),
    )
    trajectory = loop.run(prompt)
    if store is not None:
        store.save(trajectory)
    return trajectory


def run_episodes(
    task_ids: Iterable[str],
    model_id: str,
    client: Any,
    policy: TerminationPolicy,
    data_dir: str = "data",
    scaffold: str = "react",
    store: TrajectoryStore | None = None,
    dry_run: bool = False,
    total_budget_eur: float | None = None,
) -> list[Trajectory]:
    """Run several episodes, stopping if the run-level budget runs out.

    ``policy.max_cost_eur`` bounds a single episode. ``total_budget_eur``
    bounds the whole run: without it, N tasks can cost N times the per-episode
    ceiling, which is rarely what anyone intended when they set that ceiling.
    """
    trajectories: list[Trajectory] = []
    spent = 0.0

    for task_id in task_ids:
        if total_budget_eur is not None and spent >= total_budget_eur:
            logger.warning(
                "Run budget of €%.2f reached after %d episode(s); %s not attempted.",
                total_budget_eur, len(trajectories), task_id,
            )
            break

        remaining = (
            min(policy.max_cost_eur, total_budget_eur - spent)
            if total_budget_eur is not None
            else policy.max_cost_eur
        )
        if remaining <= 0:
            break

        episode_policy = TerminationPolicy(
            max_cost_eur=remaining,
            max_steps=policy.max_steps,
            max_tokens=policy.max_tokens,
            no_progress_steps=policy.no_progress_steps,
            wall_clock_seconds=policy.wall_clock_seconds,
        )
        try:
            trajectory = run_episode(
                task_id=task_id,
                model_id=model_id,
                client=client,
                policy=episode_policy,
                data_dir=data_dir,
                scaffold=scaffold,
                store=store,
                dry_run=dry_run,
            )
        except EpisodeSetupError as exc:
            # A task that cannot be set up is a harness limitation, not a
            # model failure, and must not be recorded as one.
            logger.warning("Skipping %s: %s", task_id, exc)
            continue

        trajectories.append(trajectory)
        spent += trajectory.total_cost_eur

    return trajectories


def compare_to_single_shot(
    trajectories: list[Trajectory],
    results_dir: Path | str = "results",
) -> dict[str, Any]:
    """Compare agent episodes against the single-shot run on the same tasks.

    The headline the comparison exists for is not "the agent is better" — it
    usually is, on a task it can iterate on — but **what the improvement
    cost**. An agent that solves 5% more tasks at 8× the price is worse for
    anyone with a budget, and a leaderboard reporting only resolve rate hides
    that entirely.

    Returns a dict with both axes and the ratio between them. Tasks with no
    single-shot result are excluded from the paired comparison and counted
    separately, rather than being treated as single-shot failures.
    """
    import json

    if not trajectories:
        return {"error": "no trajectories supplied"}

    model_id = trajectories[0].model_id
    agent_by_task = {t.task_id: t for t in trajectories}

    single_shot: dict[str, bool] = {}
    for jsonl in Path(results_dir).glob(f"*/{model_id}/results.jsonl"):
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            task = row.get("problem_id")
            if task in agent_by_task:
                # Several runs per task: solved once is solved.
                single_shot[task] = single_shot.get(task, False) or row.get("verdict") == "pass"

    paired = [task for task in agent_by_task if task in single_shot]
    unpaired = sorted(set(agent_by_task) - set(single_shot))

    agent_solved = sum(1 for task in paired if agent_by_task[task].solved)
    single_solved = sum(1 for task in paired if single_shot[task])
    agent_cost = sum(agent_by_task[task].total_cost_eur for task in paired)

    return {
        "model_id": model_id,
        "tasks_compared": len(paired),
        "tasks_without_single_shot_result": unpaired,
        "agent": {
            "solved": agent_solved,
            "resolve_rate": round(agent_solved / len(paired), 4) if paired else 0.0,
            "total_cost_eur": round(agent_cost, 4),
            "cost_per_task_eur": round(agent_cost / len(paired), 6) if paired else 0.0,
            "cost_per_solve_eur": round(agent_cost / agent_solved, 6) if agent_solved else None,
            "mean_steps": (
                round(sum(agent_by_task[t].num_steps for t in paired) / len(paired), 2)
                if paired else 0.0
            ),
        },
        "single_shot": {
            "solved": single_solved,
            "resolve_rate": round(single_solved / len(paired), 4) if paired else 0.0,
            "note": (
                "Single-shot cost is not repeated here: it is recorded in the "
                "cost database per call, not per task, and the published "
                "per-evaluation figures are in results/2026-05-run/summary.csv."
            ),
        },
        "delta": {
            "solved": agent_solved - single_solved,
            "resolve_rate": (
                round((agent_solved - single_solved) / len(paired), 4) if paired else 0.0
            ),
        },
        "caveat": (
            "Agent and single-shot results are not scored identically: the "
            "agent iterates against the same suite it is judged by, which "
            "single-shot does not get to do. Read this as 'what iteration "
            "buys, and what it costs', not as a like-for-like capability "
            "comparison."
        ),
    }
