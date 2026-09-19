"""
agent.scaffolds — interchangeable agent strategies behind one interface (M1).

Most papers compare models and publish conclusions about models. What they
actually compared was model-plus-scaffold, and the scaffold was theirs. If
your loop is weak, every model looks weak, and the conclusion you publish is
about your loop.

The only way out is to hold the model fixed and vary the scaffold. That
needs several scaffolds behind one interface, which is what this is:

``single_shot``
    The v1 baseline: one completion, no tools, no iteration. Here so that
    "agent mode is better" has something to be better *than*, measured the
    same way on the same tasks.
``react``
    The default. Think, call a tool, observe, repeat.
``plan_then_execute``
    Produce a plan first, then execute it. Trades a turn for direction.
``external``
    Shell out to an off-the-shelf harness. The scaffold confound is only
    really addressed by comparing against a loop you did not write, and
    this is where that plugs in.

Every scaffold records its name on the trajectory, so results can be split
by scaffold after the fact and the confound becomes measurable instead of
invisible.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from agent.loop import SYSTEM_PROMPT, AgentConfig, AgentLoop
from agent.termination import StopCondition
from agent.tools import ToolOutcome
from agent.trajectory import Trajectory, TrajectoryStep
from agent.workspace import Workspace
from llm_gateway.conversation import Conversation, Message

logger = logging.getLogger(__name__)

__all__ = [
    "Scaffold",
    "SingleShotScaffold",
    "ReActScaffold",
    "PlanThenExecuteScaffold",
    "ExternalHarnessScaffold",
    "SCAFFOLDS",
    "get_scaffold",
]


class Scaffold(ABC):
    """One strategy for turning a task into an attempt.

    Every scaffold takes the same inputs and returns the same
    :class:`~agent.trajectory.Trajectory`, so a comparison across scaffolds
    is a comparison of like with like.
    """

    #: Recorded on every trajectory this scaffold produces.
    name: str = "base"

    @abstractmethod
    def run(
        self,
        client: Any,
        workspace: Workspace,
        test_runner: Callable[[Workspace], ToolOutcome],
        config: AgentConfig,
        task_prompt: str,
    ) -> Trajectory:
        """Attempt the task and return the trajectory."""


class ReActScaffold(Scaffold):
    """Think, call a tool, observe, repeat. The default.

    A thin wrapper over :class:`~agent.loop.AgentLoop`, which is where the
    loop actually lives; this exists so ReAct sits behind the same interface
    as everything it is compared against.
    """

    name = "react"

    def run(self, client, workspace, test_runner, config, task_prompt) -> Trajectory:
        loop = AgentLoop(
            client=client,
            workspace=workspace,
            test_runner=test_runner,
            config=_with_scaffold(config, self.name),
        )
        return loop.run(task_prompt)


class PlanThenExecuteScaffold(Scaffold):
    """Ask for a plan, then execute with the plan in context.

    One extra call before any tool is used. The bet is that a model which
    has written down what it intends to do wanders less; the cost is a turn
    and the tokens to carry the plan through every subsequent turn.

    Whether the bet pays is the kind of question this class exists to
    answer, not to assume — so nothing here assumes it does.
    """

    name = "plan_then_execute"

    PLANNING_PROMPT = """\
Before touching anything, write a short plan.

You cannot call tools yet. Answer in at most six numbered lines:
which file you expect to change, what you expect to be wrong with it, and
how you will verify the fix.

If the task does not give you enough to plan with, say what you will look
at first and why. Do not guess at code you have not seen.
"""

    def run(self, client, workspace, test_runner, config, task_prompt) -> Trajectory:
        scaffold_config = _with_scaffold(config, self.name)

        plan_text = ""
        plan_cost = 0.0
        plan_tokens = 0
        try:
            planning = Conversation(
                system=self.PLANNING_PROMPT,
                messages=[Message.user(task_prompt)],
            )
            # No tools offered: the planning turn must not be able to act.
            turn = client.send_conversation(
                conversation=planning,
                tools=[],
                model_id=config.model_id,
                max_tokens=min(1024, config.max_tokens_per_call),
                temperature=config.temperature,
            )
            plan_text = turn.text.strip()
            plan_tokens = turn.usage.total
            from agent.loop import _default_cost_of

            plan_cost = _default_cost_of(turn)
        except Exception as exc:
            # A failed planning call must not lose the episode: fall through
            # and execute without a plan, with the failure recorded.
            logger.warning("Planning turn failed, executing without a plan: %s", exc)

        prompt = task_prompt
        if plan_text:
            prompt = (
                f"{task_prompt}\n\n"
                f"You already wrote this plan. Follow it, and say so if you "
                f"need to depart from it:\n\n{plan_text}"
            )

        loop = AgentLoop(
            client=client,
            workspace=workspace,
            test_runner=test_runner,
            config=scaffold_config,
        )
        # The planning call is real spend and must be inside the ceiling
        # from the first execution step, not added on afterwards.
        loop.governor.record(plan_cost)

        trajectory = loop.run(prompt)
        trajectory.total_cost_eur = round(trajectory.total_cost_eur, 6)
        trajectory.total_tokens += plan_tokens
        if plan_text:
            trajectory.steps.insert(0, TrajectoryStep(
                step_index=0,
                thought_text=plan_text,
                tool_name="",
                result_preview="(planning turn — no tool call)",
                cost_eur=round(plan_cost, 6),
            ))
        return trajectory


class SingleShotScaffold(Scaffold):
    """One completion, no tools. The v1 baseline.

    Not an agent. It is here so that every claim of the form "iterating
    helps" has a control measured on the same tasks, by the same scorer, in
    the same units — which is the only way the claim means anything.

    The model sees the task and the starting code and returns a whole file.
    The tests run once. There is no second chance, which is the point.
    """

    name = "single_shot"

    PROMPT = """\
Return the complete corrected Java file and nothing else.

Output one fenced code block containing the whole file. No explanation, no
diff, no partial content — what you return replaces the file entirely.
"""

    def run(self, client, workspace, test_runner, config, task_prompt) -> Trajectory:
        import time
        import uuid
        from datetime import datetime, timezone

        from agent.loop import _default_cost_of
        from llm_gateway.models import PromptRenderer

        started = time.monotonic()
        scaffold_config = _with_scaffold(config, self.name)

        trajectory = Trajectory(
            episode_id=f"ep-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}",
            task_id=config.task_id,
            dataset=config.dataset,
            model_id=config.model_id,
            scaffold=self.name,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        baseline = test_runner(workspace)
        trajectory.baseline_passed = baseline.tests_passed
        trajectory.baseline_total = baseline.tests_total
        trajectory.tests_executed = baseline.executed

        editable = [
            path for path in workspace.list_paths("*.java")
            if not workspace.is_read_only(path)
        ]
        context = "\n\n".join(
            f"// {path}\n{workspace.read(path)}"
            for path in workspace.list_paths()
        )

        try:
            turn = client.send_conversation(
                conversation=Conversation(
                    system=self.PROMPT,
                    messages=[Message.user(f"{task_prompt}\n\n{context}")],
                ),
                tools=[],
                model_id=config.model_id,
                max_tokens=scaffold_config.max_tokens_per_call,
                temperature=config.temperature,
            )
        except Exception as exc:
            trajectory.error = f"{type(exc).__name__}: {exc}"
            trajectory.stop_condition = StopCondition.ERROR.value
            trajectory.finished_at = datetime.now(timezone.utc).isoformat()
            trajectory.wall_clock_seconds = time.monotonic() - started
            return trajectory

        code = PromptRenderer.extract_code_blocks(turn.text) or turn.text
        target = editable[0] if editable else f"{config.task_id or 'Solution'}.java"

        outcome: ToolOutcome | None = None
        if code.strip():
            try:
                workspace.write(target, code, step_index=1)
                outcome = test_runner(workspace)
            except Exception as exc:
                trajectory.error = f"could not apply the completion: {exc}"

        trajectory.steps.append(TrajectoryStep(
            step_index=1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            thought_text=turn.text[:2000],
            tool_name="single_shot_completion",
            tool_args={"target": target},
            result_preview=(outcome.content[:400] if outcome else "no code extracted"),
            tool_error=bool(outcome and outcome.is_error),
            tokens_in=turn.usage.input_tokens,
            tokens_out=turn.usage.output_tokens,
            cache_read_tokens=turn.usage.cache_read_tokens,
            cost_eur=round(_default_cost_of(turn), 6),
            latency_ms=turn.latency_ms,
            files_touched=[target] if code.strip() else [],
            tests_passing_after=outcome.tests_passed if outcome else None,
            tests_total_after=outcome.tests_total if outcome else None,
        ))

        executed = outcome.executed if outcome else trajectory.tests_executed
        solved = bool(
            outcome
            and executed
            and outcome.tests_total
            and outcome.tests_passed == outcome.tests_total
        )
        regressed = bool(
            outcome
            and trajectory.baseline_passed is not None
            and outcome.tests_passed is not None
            and outcome.tests_passed < trajectory.baseline_passed
        )

        trajectory.solved = solved and not regressed
        trajectory.regressed = regressed
        trajectory.tests_executed = executed
        trajectory.stop_condition = (
            StopCondition.SUCCESS.value if trajectory.solved
            else StopCondition.ABANDONED.value
        )
        trajectory.final_passed = outcome.tests_passed if outcome else None
        trajectory.final_total = outcome.tests_total if outcome else None
        trajectory.total_cost_eur = trajectory.steps[0].cost_eur
        trajectory.total_tokens = turn.usage.total
        trajectory.files_changed = workspace.diff_summary()
        trajectory.final_workspace = workspace.snapshot()
        trajectory.finished_at = datetime.now(timezone.utc).isoformat()
        trajectory.wall_clock_seconds = time.monotonic() - started
        return trajectory


class ExternalHarnessScaffold(Scaffold):
    """Shell out to an off-the-shelf agent, then score the result here.

    The scaffold confound is not really answered by comparing two loops you
    wrote. This runs somebody else's — Claude Code, Aider, OpenHands — over
    the same workspace and scores the output with the same scorer, so
    "my loop is competitive" becomes a measurement.

    It records a trajectory with no steps: an external harness does not
    expose its trajectory, and inventing one would be worse than admitting
    the gap. Outcome, cost and files changed are real; process metrics are
    not available and are left empty rather than fabricated.

    Parameters
    ----------
    command
        Argv template. ``{workspace}`` and ``{prompt}`` are substituted.
    timeout_s
        Wall-clock ceiling. An external harness has no budget governor of
        ours, so this is the only ceiling that applies to it.
    """

    name = "external"

    def __init__(
        self,
        command: list[str],
        timeout_s: int = 900,
        cost_eur: float = 0.0,
    ) -> None:
        self.command = command
        self.timeout_s = timeout_s
        #: External harnesses report cost inconsistently or not at all. It
        #: is taken as a parameter rather than guessed, and defaults to 0.0
        #: — visibly wrong rather than plausibly wrong.
        self.cost_eur = cost_eur

    def run(self, client, workspace, test_runner, config, task_prompt) -> Trajectory:
        import time
        import uuid
        from datetime import datetime, timezone

        started = time.monotonic()
        trajectory = Trajectory(
            episode_id=f"ep-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}",
            task_id=config.task_id,
            dataset=config.dataset,
            model_id=config.model_id,
            scaffold=self.name,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        baseline = test_runner(workspace)
        trajectory.baseline_passed = baseline.tests_passed
        trajectory.baseline_total = baseline.tests_total

        staging = Path(tempfile.mkdtemp(prefix="external-agent-"))
        try:
            workspace.materialise(staging)
            argv = [
                part.format(workspace=str(staging), prompt=task_prompt)
                for part in self.command
            ]
            logger.info("External harness: %s", " ".join(argv[:3]))
            try:
                completed = subprocess.run(
                    argv,
                    cwd=staging,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
                trajectory.error = "" if completed.returncode == 0 else (
                    f"exit {completed.returncode}: {completed.stderr[:300]}"
                )
            except subprocess.TimeoutExpired:
                trajectory.error = f"external harness timed out after {self.timeout_s}s"
            except FileNotFoundError:
                trajectory.error = f"external harness not found: {argv[0]!r}"

            # Read back whatever it left behind.
            for path in sorted(staging.rglob("*.java")):
                relative = str(path.relative_to(staging)).replace("\\", "/")
                if workspace.is_read_only(relative):
                    continue
                try:
                    workspace.write(relative, path.read_text(encoding="utf-8"), step_index=1)
                except Exception as exc:
                    logger.debug("Could not read back %s: %s", relative, exc)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        outcome = test_runner(workspace)
        solved = bool(
            outcome.executed
            and outcome.tests_total
            and outcome.tests_passed == outcome.tests_total
        )
        regressed = bool(
            trajectory.baseline_passed is not None
            and outcome.tests_passed is not None
            and outcome.tests_passed < trajectory.baseline_passed
        )

        trajectory.solved = solved and not regressed
        trajectory.regressed = regressed
        trajectory.tests_executed = outcome.executed
        trajectory.stop_condition = (
            StopCondition.ERROR.value if trajectory.error
            else StopCondition.SUCCESS.value if trajectory.solved
            else StopCondition.ABANDONED.value
        )
        trajectory.final_passed = outcome.tests_passed
        trajectory.final_total = outcome.tests_total
        trajectory.total_cost_eur = self.cost_eur
        trajectory.files_changed = workspace.diff_summary()
        trajectory.final_workspace = workspace.snapshot()
        trajectory.finished_at = datetime.now(timezone.utc).isoformat()
        trajectory.wall_clock_seconds = time.monotonic() - started
        return trajectory


#: Scaffolds that need no constructor arguments.
SCAFFOLDS: dict[str, type[Scaffold]] = {
    ReActScaffold.name: ReActScaffold,
    PlanThenExecuteScaffold.name: PlanThenExecuteScaffold,
    SingleShotScaffold.name: SingleShotScaffold,
}


def get_scaffold(name: str) -> Scaffold:
    """Return a scaffold by name.

    Raises:
        KeyError: for an unknown name, listing the known ones. ``external``
            is excluded because it needs a command and must be constructed
            directly.
    """
    if name not in SCAFFOLDS:
        raise KeyError(
            f"unknown scaffold {name!r}; known: {sorted(SCAFFOLDS)}. "
            f"'external' takes a command and must be constructed directly."
        )
    return SCAFFOLDS[name]()


def _with_scaffold(config: AgentConfig, name: str) -> AgentConfig:
    """A copy of *config* tagged with the scaffold that is running."""
    from dataclasses import replace

    return replace(config, scaffold=name)


#: Re-exported so a caller can build a config without importing agent.loop.
DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT
