"""
llm_gateway.conversation — provider-neutral multi-turn, tool-calling types.

``Prompt``/``LLMResponse`` in :mod:`contracts` model a single completion:
one system message, one user message, one block of text back.  An agent
needs something else — a growing conversation, a tool schema, a request to
call a tool, and the result fed back in.

The three providers disagree about all of it:

=================  ====================  ======================  ==========================
Concept            Anthropic             OpenAI                  Google
=================  ====================  ======================  ==========================
Tool declaration   ``tools[].input_schema``  ``tools[].function.parameters``  ``function_declarations[].parameters``
Call request       ``tool_use`` block    ``message.tool_calls[]`` ``function_call`` part
Call arguments     dict                  **JSON string**         dict
Result message     ``user`` + ``tool_result``  ``role: "tool"``   ``function_response`` part
Result linkage     ``tool_use_id``       ``tool_call_id``        function *name*
System prompt      top-level ``system``  a message               ``system_instruction``
=================  ====================  ======================  ==========================

Every one of those differences is a place to get it wrong silently. So the
loop speaks only the types below and each adapter translates once, in one
place, with a test for the round trip.

Nothing here talks to a network.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "ToolSchema",
    "ToolCall",
    "ToolResult",
    "TokenUsage",
    "AssistantTurn",
    "Message",
    "Conversation",
    "StopReason",
]

#: Why the model stopped. ``tool_use`` means it wants a tool run and the
#: loop must continue; ``end_turn`` means it is done; ``max_tokens`` means
#: it was cut off mid-thought and the turn may be unusable.
StopReason = Literal["end_turn", "tool_use", "max_tokens", "stop_sequence", "error"]


class ToolSchema(BaseModel):
    """A tool offered to the model.

    ``parameters`` is a JSON Schema object. All three providers accept the
    ``{"type": "object", "properties": {...}, "required": [...]}`` subset,
    which is what the adapters assume; anything more exotic (``oneOf``,
    ``$ref``) is a portability risk and is not used here.
    """

    name: str = Field(description="Tool name. Must match ^[a-zA-Z0-9_-]{1,64}$.")
    description: str = Field(
        description="What the tool does, written for a model: preconditions, "
                    "units, failure modes."
    )
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema for the arguments object.",
    )


class ToolCall(BaseModel):
    """A model's request to run one tool.

    ``call_id`` is the provider's own identifier, carried through unchanged
    so the result can be linked back. Gemini has no per-call id, so its
    adapter synthesises one; see :mod:`llm_gateway.adapters.gemini`.
    """

    call_id: str = Field(description="Provider-assigned id linking call to result.")
    name: str = Field(description="Tool to run.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed arguments. Always a dict here, even for OpenAI, "
                    "which sends a JSON string on the wire.",
    )
    malformed: bool = Field(
        default=False,
        description="True when the provider's arguments could not be parsed "
                    "as JSON. `arguments` is then empty and `raw_arguments` "
                    "holds what arrived. Models do emit invalid JSON; the "
                    "loop decides the policy rather than crashing.",
    )
    raw_arguments: str = Field(
        default="", description="The unparsed argument text, kept when malformed."
    )


class ToolResult(BaseModel):
    """The outcome of running a tool, to be fed back to the model."""

    call_id: str = Field(description="The ToolCall.call_id this answers.")
    name: str = Field(default="", description="Tool that ran. Gemini links by name.")
    content: str = Field(description="What the model sees. Already truncated.")
    is_error: bool = Field(
        default=False,
        description="True when the tool failed. Providers render errors "
                    "differently but every one of them benefits from the "
                    "model being told explicitly.",
    )

    def content_hash(self) -> str:
        """SHA-256 of the content, for the trajectory table.

        Tool output is often large and often repeated. The trajectory stores
        the hash so a loop that calls the same tool with the same arguments
        five times is visible as five identical hashes without storing the
        payload five times.
        """
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


class TokenUsage(BaseModel):
    """Token counts for one API call.

    Cache reads and cache writes are priced differently from base input
    tokens by every provider that offers them. Folding them into
    ``input_tokens`` makes a cost comparison wrong in a way that looks
    right, so they are kept separate.
    """

    input_tokens: int = Field(default=0, description="Uncached input tokens.")
    output_tokens: int = Field(default=0, description="Generated tokens.")
    cache_read_tokens: int = Field(
        default=0, description="Input tokens served from the provider's prompt cache."
    )
    cache_write_tokens: int = Field(
        default=0, description="Input tokens written into the provider's prompt cache."
    )

    @property
    def total(self) -> int:
        """Every token the call touched, cached or not."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


class AssistantTurn(BaseModel):
    """One assistant response: some text, and zero or more tool calls."""

    text: str = Field(
        default="",
        description="Prose the model emitted alongside its tool calls. Often "
                    "its reasoning, which is what makes a trajectory readable.",
    )
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: StopReason = Field(default="end_turn")
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(default=0.0)
    model_id: str = Field(default="")
    raw: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific fields worth keeping for debugging. "
                    "Never read by the loop.",
    )

    @property
    def wants_tools(self) -> bool:
        """True when the loop must run tools and continue."""
        return bool(self.tool_calls)


class Message(BaseModel):
    """One turn in the conversation, in neutral form.

    A ``user`` message carries either ``text`` (the task, or a nudge) or
    ``tool_results`` (the answers to the previous assistant turn's calls),
    never both — every provider models those as distinct message shapes.
    """

    role: Literal["user", "assistant"] = Field(description="Who is speaking.")
    text: str = Field(default="")
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Assistant messages only."
    )
    tool_results: list[ToolResult] = Field(
        default_factory=list, description="User messages only."
    )

    @classmethod
    def user(cls, text: str) -> Message:
        """A plain user message."""
        return cls(role="user", text=text)

    @classmethod
    def from_turn(cls, turn: AssistantTurn) -> Message:
        """The assistant message corresponding to *turn*."""
        return cls(role="assistant", text=turn.text, tool_calls=list(turn.tool_calls))

    @classmethod
    def tool_output(cls, results: list[ToolResult]) -> Message:
        """The user message carrying tool results back to the model."""
        return cls(role="user", tool_results=list(results))


class Conversation(BaseModel):
    """System prompt plus an ordered list of messages.

    The system prompt is held separately rather than as a first message,
    because two of the three providers put it outside the message list and
    the third needs it converted. Keeping it separate means the conversion
    happens once, in the adapter.
    """

    system: str = Field(default="")
    messages: list[Message] = Field(default_factory=list)

    def append(self, message: Message) -> Conversation:
        """Append *message* and return self, for chaining."""
        self.messages.append(message)
        return self

    def append_turn(self, turn: AssistantTurn) -> Conversation:
        """Append the assistant message for *turn*."""
        return self.append(Message.from_turn(turn))

    def append_results(self, results: list[ToolResult]) -> Conversation:
        """Append the user message carrying *results*."""
        return self.append(Message.tool_output(results))

    def transcript(self) -> str:
        """A readable rendering, for logs and trajectory dumps."""
        lines: list[str] = []
        if self.system:
            lines.append(f"[system] {self.system}")
        for message in self.messages:
            if message.tool_results:
                for result in message.tool_results:
                    marker = "error" if result.is_error else "ok"
                    lines.append(f"[tool:{result.name} {marker}] {result.content}")
            elif message.tool_calls:
                if message.text:
                    lines.append(f"[assistant] {message.text}")
                for call in message.tool_calls:
                    lines.append(
                        f"[call:{call.name}] {json.dumps(call.arguments, default=str)}"
                    )
            else:
                lines.append(f"[{message.role}] {message.text}")
        return "\n".join(lines)
