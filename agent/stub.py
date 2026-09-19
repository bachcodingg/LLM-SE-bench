"""
agent.stub — a scripted client for dry runs and plumbing checks.

Exercises the whole loop — tool schemas, dispatch, the workspace, the
trajectory table, termination — without an API key, a network or a cent
spent.  It is what ``llm-se-bench agent run --dry-run`` uses and what CI
runs on every push.

**It does not try to solve the task, and it must not.** It walks a fixed
plan: look around, read what it finds, run the tests, stop. A stub that
solved tasks would make a green dry run look like a capability result, and
someone would eventually quote it as one.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_gateway.conversation import (
    AssistantTurn,
    Conversation,
    TokenUsage,
    ToolCall,
    ToolSchema,
)

logger = logging.getLogger(__name__)

__all__ = ["StubAgentClient"]


class StubAgentClient:
    """A deterministic client that explores the workspace and then stops.

    Its plan, in order:

    1. ``list_dir`` — see what is there.
    2. ``read_file`` on each file the listing named, one per turn.
    3. ``run_tests`` — confirm the sandbox path works end to end.
    4. Stop calling tools, which the loop records as abandonment.

    Abandonment is the honest outcome: the stub genuinely did not solve
    anything.

    Parameters
    ----------
    max_reads
        Ceiling on step 2, so a large workspace does not produce a hundred
        pointless turns.
    """

    def __init__(self, max_reads: int = 4) -> None:
        self.max_reads = max_reads
        self.calls = 0
        self._reset()

    def _reset(self) -> None:
        """Start a fresh episode's plan. Does not clear the call counter."""
        self._to_read: list[str] = []
        self._plan_started = False
        self._ran_tests = False

    def send_conversation(
        self,
        conversation: Conversation,
        tools: list[ToolSchema],
        model_id: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AssistantTurn:
        """Return the next scripted turn. Makes no network call."""
        self.calls += 1

        # One stub instance is reused across every episode of a run, so it
        # has to notice when a new one begins. A conversation holding only
        # the task message is a fresh episode; without this reset, episode
        # two starts wherever episode one left off and runs a single step.
        if len(conversation.messages) <= 1:
            self._reset()

        self._observe(conversation)

        if not self._plan_started:
            self._plan_started = True
            return self._call("list_dir", {}, "Dry run: listing the workspace.", model_id)

        if self._to_read:
            path = self._to_read.pop(0)
            return self._call(
                "read_file", {"path": path}, f"Dry run: reading {path}.", model_id
            )

        if not self._ran_tests:
            self._ran_tests = True
            return self._call("run_tests", {}, "Dry run: running the suite.", model_id)

        return AssistantTurn(
            text=(
                "Dry run complete. This stub explores the workspace and stops; "
                "it does not attempt a fix, so the episode ends unsolved by "
                "design. Nothing here is a capability result."
            ),
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=0, output_tokens=0),
            model_id=model_id,
        )

    # ── internals ─────────────────────────────────────────────────────

    def _observe(self, conversation: Conversation) -> None:
        """Pick the files to read out of the most recent list_dir result."""
        for message in reversed(conversation.messages):
            if not message.tool_results:
                continue
            for result in message.tool_results:
                if result.name == "list_dir" and not result.is_error:
                    self._to_read = self._paths_in(result.content)[: self.max_reads]
                    return
            return

    @staticmethod
    def _paths_in(listing: str) -> list[str]:
        """Extract file paths from a ``list_dir`` result.

        Entries are indented and shaped ``  <path>  (<n> lines)``. The
        header — ``2 file(s):`` — also contains a bracket, so indentation
        and a file extension are both required; matching on the bracket
        alone picks the header up as a filename.
        """
        paths: list[str] = []
        for line in listing.splitlines():
            if not line.startswith(" ") or "(" not in line:
                continue
            candidate = line.split("(")[0].strip()
            if candidate and "." in candidate and not candidate.endswith(":"):
                paths.append(candidate)
        return paths

    @staticmethod
    def _call(name: str, arguments: dict[str, Any], text: str, model_id: str) -> AssistantTurn:
        return AssistantTurn(
            text=text,
            tool_calls=[
                ToolCall(call_id=f"stub-{name}-{len(arguments)}", name=name, arguments=arguments)
            ],
            stop_reason="tool_use",
            # Zero tokens: a dry run must not consume the token ceiling, and
            # must not look like it spent anything.
            usage=TokenUsage(input_tokens=0, output_tokens=0),
            model_id=model_id,
        )
