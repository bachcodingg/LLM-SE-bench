"""
agent.loop — the tool-calling loop.

::

    give the agent a failing task and a tool set
    while not terminated:
        ask the model               (budget checked BEFORE the call)
        run the tools it asked for
        feed the results back
    record everything

The loop is provider-agnostic: it speaks only the neutral types in
:mod:`llm_gateway.conversation`, so Anthropic's ``tool_use`` blocks,
OpenAI's JSON-string function arguments and Gemini's id-less function calls
all arrive here in the same shape.

Two behaviours are worth stating up front because they are easy to get
wrong and expensive when you do.

**The budget is checked before each call, not after.** Steps and tokens can
be counted afterwards; money cannot, because the call is what spends it.
:class:`~agent.termination.BudgetGovernor` refuses a step whose worst case
would cross the ceiling. There is no default ceiling — an unbounded loop
against a paid API is the single most expensive mistake available here.

**A malformed tool call is a policy decision, not a crash.** Models emit
invalid JSON. The policy is configurable, the rate is recorded, and the
default — tell the model and let it retry — costs one step rather than the
episode.
"""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from agent.context import CompactionPolicy, ContextManager
from agent.termination import (
    BudgetExceeded,
    BudgetGovernor,
    EpisodeState,
    StopCondition,
    TerminationPolicy,
    check_termination,
)
from agent.tools import TOOL_SCHEMAS, AgentToolset, ToolContext, ToolOutcome
from agent.trajectory import PREVIEW_CHARS, Trajectory, TrajectoryStep
from agent.workspace import Workspace
from llm_gateway.conversation import (
    AssistantTurn,
    Conversation,
    Message,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AgentLoop",
    "AgentConfig",
    "MalformedCallPolicy",
    "SYSTEM_PROMPT",
    "make_sandbox_test_runner",
]

#: What to do when a provider sends tool arguments that will not parse.
MalformedCallPolicy = Literal["reprompt", "fail_step", "abort"]

SYSTEM_PROMPT = """\
You are fixing a Java task inside a small workspace. You have tools; use \
them. Do not ask the user questions — there is no user, and a turn spent \
asking is a turn wasted.

How to work:

1. Call list_dir once to see what is there.
2. Read the code under test and the test file. The test file tells you \
exactly what is expected; read it before writing anything.
3. Make one coherent change with apply_patch. It is a whole-file write: \
send the complete new contents of the file, never a fragment or a diff.
4. Call run_tests. Read the failure output before editing again.
5. Repeat until every test passes.

Constraints that are enforced, not advisory:

- Test files are read-only. apply_patch will refuse them. Fix the code, \
not the test.
- You have a limited number of steps and a limited budget. run_tests is \
the expensive tool; do not call it after every keystroke.
- Repeating a call you have already made, with the same arguments, ends \
the episode as making no progress.

When every test passes, say so and stop calling tools.
"""


def make_sandbox_test_runner(
    test_class_name: str | None = None,
    timeout_s: int = 120,
    dry_run: bool = False,
) -> Callable[[Workspace], ToolOutcome]:
    """Build the ``run_tests`` implementation backed by the Docker sandbox.

    The workspace is materialised into a fresh temporary directory for each
    run and discarded after, so a run cannot leak state into the next one.

    Parameters
    ----------
    test_class_name
        JUnit class to run. Inferred from the first file containing
        ``@Test`` when omitted.
    timeout_s
        Ceiling for compilation and for the test run.
    dry_run
        Force the sandbox's structural check. The resulting counts are
        optimistic and the outcome says so, loudly, in the text the agent
        reads — an agent must not be able to "pass" by never compiling.
    """

    def run(workspace: Workspace) -> ToolOutcome:
        from bench.sandbox.docker_sandbox import DockerSandbox
        from mcp_servers.truncation import truncate_log

        sandbox = DockerSandbox(
            timeout_compile=timeout_s, timeout_test=timeout_s, dry_run=dry_run
        )
        executed = not dry_run and sandbox.is_docker_available()

        with tempfile.TemporaryDirectory(prefix="agent-tests-") as staging:
            workspace.materialise(Path(staging))
            result = sandbox.run_files(
                workspace.snapshot(),
                test_class_name=test_class_name,
                problem_id="agent-episode",
            )

        raw = "\n".join(
            part for part in (
                result.compile_stdout, result.compile_stderr,
                result.test_stdout, result.test_stderr, result.error_message,
            ) if part
        )
        log, truncation = truncate_log(raw, max_lines=80)

        if not result.compiled:
            header = "COMPILE FAILED — no test ran."
        elif result.tests_total == 0:
            header = "Compiled, but no @Test method was found. 0/0 is not a pass."
        else:
            header = (
                f"{result.tests_passed}/{result.tests_total} tests passed."
                if result.tests_passed < result.tests_total
                else f"All {result.tests_total} tests passed."
            )
        if not executed:
            header = (
                "EXECUTION SKIPPED — Docker was unavailable, so this is a "
                "structural check, not a test run. The counts below are "
                "optimistic and do not mean the code works.\n" + header
            )

        return ToolOutcome(
            content=f"{header}\n\n{log}" if log else header,
            is_error=not result.compiled,
            truncated=truncation.truncated,
            tests_passed=result.tests_passed,
            tests_total=result.tests_total,
            compiled=result.compiled,
            executed=executed,
        )

    return run


@dataclass
class AgentConfig:
    """How one episode is run."""

    model_id: str
    policy: TerminationPolicy
    system_prompt: str = SYSTEM_PROMPT
    max_tokens_per_call: int = 4096
    temperature: float = 0.0
    scaffold: str = "react"
    malformed_call_policy: MalformedCallPolicy = "reprompt"
    #: Worst-case euros for one call, used by the budget governor to refuse
    #: a step *before* making it. Zero disables the projection and leaves
    #: only the after-the-fact ceiling, which can overshoot by one call.
    worst_case_call_eur: float = 0.0
    dataset: str = ""
    task_id: str = ""
    #: Keep the conversation inside the context window. Without it a long
    #: episode eventually overflows and the provider errors at step 30,
    #: which is the harness's failure recorded as the model's.
    compaction: "CompactionPolicy | None" = None


@dataclass
class _StepRecord:
    """Scratch space for one step, before it becomes a TrajectoryStep."""

    calls: list[ToolCall] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)
    outcomes: list[ToolOutcome] = field(default_factory=list)


class AgentLoop:
    """Runs one episode: one task, one model, one workspace.

    Parameters
    ----------
    client
        Anything with ``send_conversation(conversation, tools, model_id,
        max_tokens, temperature) -> AssistantTurn``. In production an
        ``LLMClient``; in tests a scripted stub, which is how the loop is
        covered without an API key.
    workspace
        The files the agent works on. Test files belong in its read-only
        set.
    test_runner
        What ``run_tests`` calls. Usually from
        :func:`make_sandbox_test_runner`.
    config
        Model, ceilings, prompt, policies.
    cost_of
        Maps an ``AssistantTurn`` to euros. Defaults to the gateway's
        pricing table.
    """

    def __init__(
        self,
        client: Any,
        workspace: Workspace,
        test_runner: Callable[[Workspace], ToolOutcome],
        config: AgentConfig,
        cost_of: Callable[[AssistantTurn], float] | None = None,
    ) -> None:
        self.client = client
        self.workspace = workspace
        self.config = config
        self.cost_of = cost_of or _default_cost_of
        self.context = ToolContext(workspace=workspace, test_runner=test_runner)
        self.toolset = AgentToolset(self.context)
        self.governor = BudgetGovernor(
            ceiling_eur=config.policy.max_cost_eur,
            worst_case_call_eur=config.worst_case_call_eur,
        )
        # Named for what it manages, not "context": self.context is already
        # the ToolContext the tools act on.
        self.context_manager = ContextManager(
            model_id=config.model_id,
            policy=config.compaction or CompactionPolicy(),
        )

    # ── entry point ───────────────────────────────────────────────────

    def run(self, task_prompt: str) -> Trajectory:
        """Run the episode to termination and return its trajectory.

        Never raises for anything the model or a tool did: a provider
        failure becomes ``StopCondition.ERROR`` with the message recorded,
        because an exception escaping here loses the steps that led to it —
        which are the interesting part.
        """
        episode_id = f"ep-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
        started = time.monotonic()

        trajectory = Trajectory(
            episode_id=episode_id,
            task_id=self.config.task_id,
            dataset=self.config.dataset,
            model_id=self.config.model_id,
            scaffold=self.config.scaffold,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        state = EpisodeState()
        self._establish_baseline(state, trajectory)

        conversation = Conversation(
            system=self.config.system_prompt,
            messages=[Message.user(task_prompt)],
        )

        stop: StopCondition | None = None
        while stop is None:
            try:
                self.governor.check()
            except BudgetExceeded as exc:
                logger.info("Episode %s: %s", episode_id, exc)
                stop = StopCondition.MAX_COST
                trajectory.error = str(exc)
                break

            # Compact before sending, not after overflowing. The provider
            # reports an over-long conversation as an error, by which point
            # the episode is already lost.
            if self.context_manager.needs_compaction(conversation):
                self.context_manager.compact(conversation, step_index=state.steps)

            try:
                turn = self.client.send_conversation(
                    conversation=conversation,
                    tools=TOOL_SCHEMAS,
                    model_id=self.config.model_id,
                    max_tokens=self.config.max_tokens_per_call,
                    temperature=self.config.temperature,
                )
            except Exception as exc:
                logger.exception("Episode %s: provider call failed", episode_id)
                state.error = f"{type(exc).__name__}: {exc}"
                trajectory.error = state.error
                stop = StopCondition.ERROR
                break

            cost = self.cost_of(turn)
            self.governor.record(cost)
            state.cost_eur = self.governor.spent_eur
            state.total_tokens += turn.usage.total

            conversation.append_turn(turn)

            if not turn.wants_tools:
                # The model stopped asking for tools. Either it is done, or
                # it gave up; the test state decides which, not its prose.
                state.model_stopped_calling_tools = True
                self._record_text_only_step(trajectory, state, turn, cost)
                stop = check_termination(state, self.config.policy)
                if stop is None:
                    stop = StopCondition.ABANDONED
                break

            abort = self._execute_step(trajectory, state, conversation, turn, cost)
            if abort is not None:
                stop = abort
                break

            stop = check_termination(state, self.config.policy)

        self._finalise(trajectory, state, stop or StopCondition.ERROR, started)
        return trajectory

    # ── internals ─────────────────────────────────────────────────────

    def _establish_baseline(self, state: EpisodeState, trajectory: Trajectory) -> None:
        """Record the test state before the agent touches anything.

        Without this there is no way to distinguish "fixed the target" from
        "fixed the target and broke three other things", and the second must
        not score as success.
        """
        try:
            outcome = self.context.test_runner(self.workspace)
        except Exception as exc:
            logger.warning("Baseline test run failed: %s", exc)
            return
        state.baseline_passed = outcome.tests_passed
        state.baseline_total = outcome.tests_total
        state.tests_executed = outcome.executed
        trajectory.baseline_passed = outcome.tests_passed
        trajectory.baseline_total = outcome.tests_total
        trajectory.tests_executed = outcome.executed
        if not outcome.executed:
            logger.warning(
                "Episode %s: tests are not being executed (no Docker, or "
                "dry-run). This episode cannot be solved and its result is "
                "not a benchmark score.",
                trajectory.episode_id,
            )

    def _execute_step(
        self,
        trajectory: Trajectory,
        state: EpisodeState,
        conversation: Conversation,
        turn: AssistantTurn,
        cost: float,
    ) -> StopCondition | None:
        """Run one turn's tool calls. Returns a stop condition, or None.

        The cost of the call is attributed to the *first* step of the turn.
        A turn requesting three tools made one API call; spreading its cost
        across three steps would misreport every per-step cost figure.
        """
        results: list[ToolResult] = []
        edits_before = self.workspace.effective_edits()
        tests_before = (state.tests_passed, state.tests_total)

        for position, call in enumerate(turn.tool_calls):
            state.steps += 1
            self.context.step_index = state.steps

            if call.malformed:
                outcome = self._handle_malformed(call)
                if outcome is None:
                    trajectory.error = (
                        f"Provider sent unparseable tool arguments for "
                        f"{call.name!r} and the policy is 'abort'."
                    )
                    return StopCondition.ERROR
            else:
                outcome = self.toolset.dispatch(call.name, call.arguments)

            result = ToolResult(
                call_id=call.call_id,
                name=call.name,
                content=outcome.content,
                is_error=outcome.is_error,
            )
            results.append(result)

            if outcome.tests_total is not None:
                state.tests_passed = outcome.tests_passed
                state.tests_total = outcome.tests_total
                state.tests_executed = outcome.executed

            trajectory.steps.append(
                TrajectoryStep(
                    step_index=state.steps,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    # Only the first step of a turn carries its text and cost.
                    thought_text=turn.text if position == 0 else "",
                    tool_name=call.name,
                    tool_args=dict(call.arguments),
                    tool_result_hash=result.content_hash(),
                    result_preview=outcome.content[:PREVIEW_CHARS],
                    tool_error=outcome.is_error,
                    malformed_call=call.malformed,
                    tokens_in=turn.usage.input_tokens if position == 0 else 0,
                    tokens_out=turn.usage.output_tokens if position == 0 else 0,
                    cache_read_tokens=turn.usage.cache_read_tokens if position == 0 else 0,
                    cost_eur=cost if position == 0 else 0.0,
                    latency_ms=turn.latency_ms if position == 0 else 0.0,
                    files_touched=[
                        edit.path
                        for edit in self.workspace.edits
                        if edit.step_index == state.steps and edit.operation != "no-op"
                    ],
                    tests_passing_after=outcome.tests_passed,
                    tests_total_after=outcome.tests_total,
                )
            )

            if state.steps >= self.config.policy.max_steps:
                break

        conversation.append_results(results)

        # Progress means a file actually changed, or the test result moved.
        # Reading the same file again is not progress, and neither is
        # rewriting a file with identical content.
        made_progress = (
            self.workspace.effective_edits() > edits_before
            or (state.tests_passed, state.tests_total) != tests_before
        )
        state.record_progress(made_progress)
        state.effective_edits = self.workspace.effective_edits()
        return None

    def _handle_malformed(self, call: ToolCall) -> ToolOutcome | None:
        """Apply the malformed-call policy. None means abort the episode."""
        if self.config.malformed_call_policy == "abort":
            return None
        if self.config.malformed_call_policy == "fail_step":
            return ToolOutcome(
                content=(
                    f"The arguments for {call.name} were not valid JSON and "
                    f"the step was discarded."
                ),
                is_error=True,
            )
        return ToolOutcome(
            content=(
                f"The arguments you sent for {call.name} were not valid JSON:\n"
                f"{call.raw_arguments[:500]}\n\n"
                f"Send the arguments again as a JSON object matching the "
                f"tool's schema."
            ),
            is_error=True,
        )

    def _record_text_only_step(
        self,
        trajectory: Trajectory,
        state: EpisodeState,
        turn: AssistantTurn,
        cost: float,
    ) -> None:
        """Record a turn that made no tool call, so its cost is not lost."""
        state.steps += 1
        trajectory.steps.append(
            TrajectoryStep(
                step_index=state.steps,
                timestamp=datetime.now(timezone.utc).isoformat(),
                thought_text=turn.text,
                tool_name="",
                tool_result_hash="",
                result_preview="",
                tokens_in=turn.usage.input_tokens,
                tokens_out=turn.usage.output_tokens,
                cache_read_tokens=turn.usage.cache_read_tokens,
                cost_eur=cost,
                latency_ms=turn.latency_ms,
                tests_passing_after=state.tests_passed,
                tests_total_after=state.tests_total,
            )
        )

    def _finalise(
        self,
        trajectory: Trajectory,
        state: EpisodeState,
        stop: StopCondition,
        started: float,
    ) -> None:
        """Fill in the trajectory's outcome fields."""
        trajectory.finished_at = datetime.now(timezone.utc).isoformat()
        trajectory.stop_condition = stop.value
        trajectory.solved = stop.is_success
        trajectory.regressed = state.regressed
        trajectory.total_cost_eur = round(self.governor.spent_eur, 6)
        trajectory.total_tokens = state.total_tokens
        trajectory.wall_clock_seconds = time.monotonic() - started
        trajectory.final_passed = state.tests_passed
        trajectory.final_total = state.tests_total
        trajectory.files_changed = self.workspace.diff_summary()
        trajectory.edit_churn = self.workspace.churn()
        trajectory.final_workspace = self.workspace.snapshot()
        trajectory.context = self.context_manager.stats()

        logger.info(
            "Episode %s ended: %s after %d step(s), €%.4f, solved=%s",
            trajectory.episode_id, stop.value, trajectory.num_steps,
            trajectory.total_cost_eur, trajectory.solved,
        )


def _default_cost_of(turn: AssistantTurn) -> float:
    """Euros for *turn*, from the gateway's pricing table.

    Cache reads are charged at the full input rate, which **overstates**
    the cost: every provider discounts cached input. Overstating is the
    right direction for a budget check — an agent that stops slightly early
    is better than one that overruns — but it means a reported agent cost is
    an upper bound, not a measurement, until per-tier cache pricing exists.
    """
    from llm_gateway.config import GatewayConfig
    from settings import usd_to_eur

    pricing = GatewayConfig().get_pricing(turn.model_id)
    usd = (
        (turn.usage.input_tokens + turn.usage.cache_read_tokens + turn.usage.cache_write_tokens)
        * pricing.cost_per_prompt_token
        + turn.usage.output_tokens * pricing.cost_per_completion_token
    )
    return usd_to_eur(usd)
