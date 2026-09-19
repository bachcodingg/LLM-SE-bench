"""
llm_gateway.adapters.openai — OpenAI Chat Completions tool calling.

Shape of the wire format:

* Tools are ``{"type": "function", "function": {name, description, parameters}}``.
* A call arrives in ``message.tool_calls[]`` with ``id``, ``function.name``
  and ``function.arguments`` — **a JSON string**, not a dict, and a string
  the model generated, so it can be invalid JSON. That is the one real
  hazard in this adapter and it is handled explicitly rather than allowed
  to raise mid-episode.
* Results go back as separate messages with ``role: "tool"`` and
  ``tool_call_id`` — one message per result, unlike Anthropic's single
  user message holding several blocks.
* The system prompt is the first message in the list.
* Cached input tokens appear under
  ``usage.prompt_tokens_details.cached_tokens`` and are *included in*
  ``prompt_tokens``, so they must be subtracted out rather than added.
"""

from __future__ import annotations

import json
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

__all__ = ["OpenAIAdapter"]

_STOP_REASONS: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "error",
}


class OpenAIAdapter(ProviderAdapter):
    """Neutral types to and from the OpenAI Chat Completions API."""

    provider_name = "gpt4"

    def encode_tools(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def encode_conversation(self, conversation: Conversation) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if conversation.system:
            messages.append({"role": "system", "content": conversation.system})

        for message in conversation.messages:
            if message.tool_results:
                # One message per result: OpenAI has no multi-result message.
                messages.extend(
                    {
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": (
                            f"ERROR: {result.content}" if result.is_error else result.content
                        ),
                    }
                    for result in message.tool_results
                )
            elif message.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": message.text or None,
                    "tool_calls": [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                # Back to a JSON string: round-tripping the
                                # model's own call through the type system
                                # must not change its shape.
                                "arguments": (
                                    call.raw_arguments
                                    if call.malformed
                                    else json.dumps(call.arguments, default=str)
                                ),
                            },
                        }
                        for call in message.tool_calls
                    ],
                })
            else:
                messages.append({"role": message.role, "content": message.text})
        return messages

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
            "messages": self.encode_conversation(conversation),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            request["tools"] = self.encode_tools(tools)
            request["tool_choice"] = "auto"
        return request

    def decode_response(self, response: Any, model_id: str) -> AssistantTurn:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return AssistantTurn(
                stop_reason="error",
                model_id=model_id,
                raw={"error": "response contained no choices"},
            )

        choice = choices[0]
        message = getattr(choice, "message", None)
        text = (getattr(message, "content", None) or "") if message else ""

        calls: list[ToolCall] = []
        for raw_call in getattr(message, "tool_calls", None) or []:
            function = getattr(raw_call, "function", None)
            name = getattr(function, "name", "") if function else ""
            raw_arguments = (getattr(function, "arguments", "") if function else "") or ""
            try:
                parsed = json.loads(raw_arguments) if raw_arguments else {}
                malformed = not isinstance(parsed, dict)
            except json.JSONDecodeError:
                parsed, malformed = {}, True

            calls.append(
                ToolCall(
                    call_id=getattr(raw_call, "id", ""),
                    name=name,
                    arguments=parsed if not malformed else {},
                    malformed=malformed,
                    raw_arguments=raw_arguments if malformed else "",
                )
            )

        usage_obj = getattr(response, "usage", None)
        prompt_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
        details = getattr(usage_obj, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0 if details else 0
        usage = TokenUsage(
            # cached_tokens is a subset of prompt_tokens, not an addition.
            input_tokens=max(0, prompt_tokens - cached),
            output_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
            cache_read_tokens=cached,
        )

        raw_stop = getattr(choice, "finish_reason", None) or "stop"
        return AssistantTurn(
            text=text,
            tool_calls=calls,
            stop_reason=_STOP_REASONS.get(raw_stop, "end_turn"),
            usage=usage,
            model_id=model_id,
            raw={
                "api_model": getattr(response, "model", ""),
                "api_id": getattr(response, "id", ""),
                "finish_reason": raw_stop,
            },
        )
