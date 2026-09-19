"""
llm_gateway.adapters.gemini — Google Gemini function calling.

Shape of the wire format, and the two places it differs enough to matter:

* Tools are ``[{"function_declarations": [{name, description, parameters}]}]``
  — one wrapper object holding every declaration, not one per tool.
* Roles are ``user`` and ``model``, not ``user`` and ``assistant``.
* A call arrives as a ``function_call`` part with ``name`` and ``args``.
  **There is no call id.** Gemini links a result to a call by function
  *name*. The adapter therefore synthesises deterministic ids on decode and
  discards them on encode, linking by name instead. The consequence is real
  and is stated here rather than discovered later: two concurrent calls to
  the same function in one turn cannot be told apart on the way back. The
  agent loop runs tool calls sequentially, which sidesteps it.
* Results go back as ``function_response`` parts with ``name`` and a
  ``response`` dict.
* The system prompt is ``system_instruction`` on the model object, not part
  of the contents.
* ``usage_metadata`` reports ``cached_content_token_count`` separately.
"""

from __future__ import annotations

from typing import Any

from llm_gateway.adapters import ProviderAdapter
from llm_gateway.conversation import (
    AssistantTurn,
    Conversation,
    StopReason,
    TokenUsage,
    ToolCall,
    ToolSchema,
)

__all__ = ["GeminiAdapter", "synthetic_call_id"]

_STOP_REASONS: dict[str, StopReason] = {
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "error",
    "RECITATION": "error",
    "OTHER": "error",
}


def synthetic_call_id(function_name: str, index: int) -> str:
    """Deterministic stand-in for the call id Gemini does not send.

    Deterministic rather than a UUID so a recorded trajectory replays to
    byte-identical ids.
    """
    return f"gemini-{function_name}-{index}"


class GeminiAdapter(ProviderAdapter):
    """Neutral types to and from the Gemini generate_content API."""

    provider_name = "gemini"

    def encode_tools(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        if not tools:
            return []
        return [{
            "function_declarations": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in tools
            ]
        }]

    def encode_conversation(self, conversation: Conversation) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for message in conversation.messages:
            if message.tool_results:
                contents.append({
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": result.name,
                                "response": (
                                    {"error": result.content}
                                    if result.is_error
                                    else {"result": result.content}
                                ),
                            }
                        }
                        for result in message.tool_results
                    ],
                })
            elif message.tool_calls:
                parts: list[dict[str, Any]] = []
                if message.text:
                    parts.append({"text": message.text})
                # call_id is dropped: Gemini links results by function name.
                parts.extend(
                    {"function_call": {"name": call.name, "args": call.arguments}}
                    for call in message.tool_calls
                )
                contents.append({"role": "model", "parts": parts})
            else:
                contents.append({
                    "role": "model" if message.role == "assistant" else "user",
                    "parts": [{"text": message.text}],
                })
        return contents

    def encode_request(
        self,
        conversation: Conversation,
        tools: list[ToolSchema],
        model_id: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model_id,
            "contents": self.encode_conversation(conversation),
            "generation_config": {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        }
        # Not a request field: the caller puts it on the GenerativeModel.
        if conversation.system:
            request["system_instruction"] = conversation.system
        if tools:
            request["tools"] = self.encode_tools(tools)
        return request

    def decode_response(self, response: Any, model_id: str) -> AssistantTurn:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return AssistantTurn(
                stop_reason="error",
                model_id=model_id,
                raw={"error": "response contained no candidates"},
            )

        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for index, part in enumerate(parts):
            function_call = getattr(part, "function_call", None)
            if function_call is not None and getattr(function_call, "name", ""):
                name = function_call.name
                arguments = getattr(function_call, "args", None) or {}
                calls.append(
                    ToolCall(
                        call_id=synthetic_call_id(name, index),
                        name=name,
                        arguments=dict(arguments),
                    )
                )
                continue
            text = getattr(part, "text", "")
            if text:
                text_parts.append(text)

        meta = getattr(response, "usage_metadata", None)
        cached = getattr(meta, "cached_content_token_count", 0) or 0 if meta else 0
        prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0 if meta else 0
        usage = TokenUsage(
            input_tokens=max(0, prompt_tokens - cached),
            output_tokens=(getattr(meta, "candidates_token_count", 0) or 0) if meta else 0,
            cache_read_tokens=cached,
        )

        raw_finish = getattr(candidate, "finish_reason", None)
        finish_name = getattr(raw_finish, "name", None) or str(raw_finish or "STOP")
        stop_reason: StopReason = _STOP_REASONS.get(finish_name, "end_turn")
        # Gemini reports STOP even when it emitted a function call, so the
        # presence of calls is what decides whether the loop continues.
        if calls:
            stop_reason = "tool_use"

        return AssistantTurn(
            text="".join(text_parts),
            tool_calls=calls,
            stop_reason=stop_reason,
            usage=usage,
            model_id=model_id,
            raw={"finish_reason": finish_name},
        )
