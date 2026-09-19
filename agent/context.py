"""
agent.context — context window management and prompt-cache control (M1).

A long episode outgrows the context window. When it does, the provider
returns an error and the episode dies at step 30 having done nothing wrong —
which is the harness's failure, recorded as the model's.

Two mechanisms:

**Compaction.** When the conversation approaches the window, replace the
oldest exchanges with a summary and keep the recent ones verbatim. What gets
summarised, and what survives intact, is a policy decision with real
consequences, so it is explicit and configurable rather than buried.

**Cache control.** Every provider that offers prompt caching prices a cache
read far below a fresh input token, and an agent conversation is mostly a
repeat of itself. Marking a stable prefix as cacheable is the single largest
cost lever in a long episode. The catch: a cache breakpoint invalidates
everything after it on any change, so the breakpoint goes *before* the part
that grows, not after.

Token counts here are estimates. Getting an exact count needs a provider
round trip, which defeats the purpose of checking before sending. The
estimate is deliberately conservative — it overestimates — so compaction
fires slightly early rather than slightly late.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from llm_gateway.conversation import Conversation, Message

logger = logging.getLogger(__name__)

__all__ = [
    "CompactionPolicy",
    "ContextManager",
    "estimate_tokens",
    "CONTEXT_WINDOWS",
]

#: Usable context per model, in tokens. Conservative: the real windows are
#: larger, and leaving headroom for the response is not optional.
CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-7": 180_000,
    "claude-sonnet-4-6": 180_000,
    "claude-haiku-4-5-20251001": 180_000,
    "gpt-4o": 110_000,
    "gpt-4-turbo": 110_000,
    "gemini-2.5-pro": 900_000,
    "gemini-2.5-flash": 900_000,
    "gemini-2.0-flash": 900_000,
}

#: Fallback for a model with no entry. Small on purpose: compacting a
#: conversation that did not need it costs a summary, while failing to
#: compact one that did costs the episode.
DEFAULT_CONTEXT_WINDOW = 100_000

#: Characters per token. English prose runs about 4; source code, which is
#: what fills these conversations, runs denser. 3.3 overestimates the token
#: count slightly, which is the safe direction.
_CHARS_PER_TOKEN = 3.3


def estimate_tokens(text: str) -> int:
    """Approximate token count for *text*.

    An estimate, not a measurement: an exact count needs a provider call,
    and the point of this is to decide *before* making one. Biased high.
    """
    return int(len(text) / _CHARS_PER_TOKEN) + 1


@dataclass
class CompactionPolicy:
    """When to compact, and what survives.

    Attributes
    ----------
    trigger_fraction
        Compact when the conversation exceeds this share of the window.
        0.7 leaves room for the response plus the tool results the next
        turn will add.
    keep_recent_exchanges
        Recent messages kept verbatim. The last few turns are where the
        agent's working state lives; summarising them is how an agent
        forgets what it just tried and repeats it.
    keep_first_message
        Keep the original task verbatim, always. It is small, it is the
        objective, and an agent that loses it starts solving a summary of
        its task.
    max_tool_result_chars
        Tool results older than the recent window are clipped to this
        before being summarised. A build log from step 4 is the largest
        thing in the conversation and the least useful at step 25.
    """

    trigger_fraction: float = 0.7
    keep_recent_exchanges: int = 6
    keep_first_message: bool = True
    max_tool_result_chars: int = 400

    def __post_init__(self) -> None:
        if not 0.1 <= self.trigger_fraction <= 0.95:
            raise ValueError("trigger_fraction must be between 0.1 and 0.95")
        if self.keep_recent_exchanges < 2:
            raise ValueError(
                "keep_recent_exchanges must be at least 2: the last assistant "
                "turn and its tool results have to survive together, or the "
                "conversation is left with a tool call whose result is gone."
            )


@dataclass
class CompactionEvent:
    """A record of one compaction, for the trajectory."""

    step_index: int
    messages_before: int
    messages_after: int
    tokens_before: int
    tokens_after: int
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "messages_before": self.messages_before,
            "messages_after": self.messages_after,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_before - self.tokens_after,
        }


class ContextManager:
    """Keeps a conversation inside the window, and marks it for caching.

    Parameters
    ----------
    model_id
        Used to look up the window size.
    policy
        When and how to compact.
    reserve_tokens
        Held back for the model's response. Counting the window as fully
        available is how a conversation that "fits" still overflows.
    """

    def __init__(
        self,
        model_id: str,
        policy: CompactionPolicy | None = None,
        reserve_tokens: int = 8_000,
    ) -> None:
        self.model_id = model_id
        self.policy = policy or CompactionPolicy()
        self.window = CONTEXT_WINDOWS.get(model_id, DEFAULT_CONTEXT_WINDOW)
        self.reserve_tokens = reserve_tokens
        self.events: list[CompactionEvent] = []

    @property
    def usable_tokens(self) -> int:
        return max(1, self.window - self.reserve_tokens)

    def measure(self, conversation: Conversation) -> int:
        """Estimated tokens the conversation currently occupies."""
        total = estimate_tokens(conversation.system)
        for message in conversation.messages:
            total += estimate_tokens(message.text)
            for call in message.tool_calls:
                total += estimate_tokens(str(call.arguments)) + 10
            for result in message.tool_results:
                total += estimate_tokens(result.content) + 10
        return total

    def utilisation(self, conversation: Conversation) -> float:
        """Share of the usable window in use, 0.0 upward. Can exceed 1.0."""
        return self.measure(conversation) / self.usable_tokens

    def needs_compaction(self, conversation: Conversation) -> bool:
        return self.utilisation(conversation) >= self.policy.trigger_fraction

    def compact(self, conversation: Conversation, step_index: int = 0) -> Conversation:
        """Replace old exchanges with a summary, in place. Returns it.

        The summary is mechanical — what was read, what was written, what
        the tests said — not model-generated. A summarisation call costs
        money and latency at exactly the moment the episode is already
        expensive, and the facts worth keeping are facts the harness
        already has.
        """
        before_tokens = self.measure(conversation)
        before_count = len(conversation.messages)

        keep = self.policy.keep_recent_exchanges
        if before_count <= keep + 1:
            return conversation

        head = conversation.messages[:1] if self.policy.keep_first_message else []
        tail = conversation.messages[-keep:]
        middle = conversation.messages[len(head):-keep]
        if not middle:
            return conversation

        summary = self._summarise(middle)
        conversation.messages = [
            *head,
            Message.user(summary),
            *tail,
        ]

        event = CompactionEvent(
            step_index=step_index,
            messages_before=before_count,
            messages_after=len(conversation.messages),
            tokens_before=before_tokens,
            tokens_after=self.measure(conversation),
            summary=summary,
        )
        self.events.append(event)
        logger.info(
            "Compacted context at step %d: %d -> %d messages, ~%d -> ~%d tokens",
            step_index, event.messages_before, event.messages_after,
            event.tokens_before, event.tokens_after,
        )
        return conversation

    def _summarise(self, messages: list[Message]) -> str:
        """A factual digest of the messages being dropped.

        Deliberately lists *what was already tried*. The failure mode
        compaction introduces is an agent that forgets and repeats itself,
        so the summary is written to prevent exactly that.
        """
        read: list[str] = []
        written: list[str] = []
        searched: list[str] = []
        test_lines: list[str] = []
        errors = 0

        for message in messages:
            for call in message.tool_calls:
                path = str(call.arguments.get("path", ""))
                if call.name == "read_file" and path:
                    read.append(path)
                elif call.name == "apply_patch" and path:
                    written.append(path)
                elif call.name == "grep":
                    searched.append(str(call.arguments.get("pattern", "")))
            for result in message.tool_results:
                if result.is_error:
                    errors += 1
                if result.name == "run_tests":
                    first = result.content.strip().splitlines()
                    if first:
                        test_lines.append(first[0][: self.policy.max_tool_result_chars])

        parts = [
            "[Earlier turns were removed to stay inside the context window. "
            "What happened in them:]",
        ]
        if read:
            parts.append(f"- Read: {', '.join(dict.fromkeys(read))}")
        if searched:
            parts.append(f"- Searched for: {', '.join(dict.fromkeys(searched))}")
        if written:
            parts.append(f"- Edited: {', '.join(dict.fromkeys(written))}")
        if test_lines:
            parts.append("- Test runs, oldest first:")
            parts.extend(f"    {line}" for line in test_lines[-5:])
        if errors:
            parts.append(f"- {errors} tool call(s) returned an error.")
        parts.append(
            "[Do not repeat work listed above. If you need a file's current "
            "contents, read it again — it may have changed since.]"
        )
        return "\n".join(parts)

    # ── prompt caching ────────────────────────────────────────────────

    def cache_breakpoint(self, conversation: Conversation) -> int:
        """Index of the last message that is worth marking cacheable.

        Everything up to and including this index is stable across turns:
        the task, the early exploration, the parts that will not change
        again. Everything after it is the growing tail.

        Returns -1 when the prefix is too small to be worth a breakpoint.
        Providers charge a premium to *write* a cache entry, so caching a
        prefix that will be used twice loses money.
        """
        stable = max(0, len(conversation.messages) - self.policy.keep_recent_exchanges)
        if stable < 2:
            return -1

        prefix_tokens = estimate_tokens(conversation.system) + sum(
            estimate_tokens(message.text)
            + sum(estimate_tokens(r.content) for r in message.tool_results)
            for message in conversation.messages[:stable]
        )
        # Below roughly a thousand tokens the cache write costs more than
        # the reads will save.
        return stable - 1 if prefix_tokens >= 1024 else -1

    def stats(self) -> dict[str, Any]:
        """What compaction did over the episode, for the trajectory."""
        return {
            "model_id": self.model_id,
            "window": self.window,
            "usable_tokens": self.usable_tokens,
            "compactions": len(self.events),
            "tokens_saved": sum(
                event.tokens_before - event.tokens_after for event in self.events
            ),
            "events": [event.to_dict() for event in self.events],
        }
