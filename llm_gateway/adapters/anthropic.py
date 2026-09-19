"""
llm_gateway.adapters.anthropic — Anthropic Messages API tool calling.

Shape of the wire format:

* Tools are ``{"name", "description", "input_schema"}``.
* A call arrives as a ``tool_use`` content block with ``id``, ``name`` and
  ``input`` (already a dict — Anthropic does not stringify arguments).
* Results go back as a ``user`` message whose content is a list of
  ``tool_result`` blocks keyed by ``tool_use_id``.
* The system prompt is a top-level ``system`` parameter.
* ``usage`` reports ``cache_read_input_tokens`` and
  ``cache_creation_input_tokens`` separately from ``input_tokens``, and
  they are priced differently.
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

__all__ = ["AnthropicAdapter"]

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
}


class AnthropicAdapter(ProviderAdapter):
    """Neutral types to and from the Anthropic Messages API."""

    provider_name = "claude"

    def encode_tools(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    def encode_conversation(self, conversation: Conversation) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for message in conversation.messages:
            if message.tool_results:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": result.call_id,
                            "content": result.content,
                            "is_error": result.is_error,
                        }
                        for result in message.tool_results
                    ],
                })
            elif message.tool_calls:
                content: list[dict[str, Any]] = []
                if message.text:
                    content.append({"type": "text", "text": message.text})
                content.extend(
                    {
                        "type": "tool_use",
                        "id": call.call_id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                    for call in message.tool_calls
                )
                messages.append({"role": "assistant", "content": content})
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
        if conversation.system:
            request["system"] = conversation.system
        if tools:
            request["tools"] = self.encode_tools(tools)
        return request

    def decode_response(self, response: Any, model_id: str) -> AssistantTurn:
        text_parts: list[str] = []
        calls: list[ToolCall] = []

        for block in getattr(response, "content", None) or []:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
            elif block_type == "tool_use":
                calls.append(
                    ToolCall(
                        call_id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        # Anthropic sends a dict, so there is no parse step
                        # and no malformed-JSON case to handle here.
                        arguments=dict(getattr(block, "input", None) or {}),
                    )
                )

        usage_obj = getattr(response, "usage", None)
        usage = TokenUsage(
            input_tokens=getattr(usage_obj, "input_tokens", 0) or 0,
            output_tokens=getattr(usage_obj, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage_obj, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage_obj, "cache_creation_input_tokens", 0) or 0,
        )

        raw_stop = getattr(response, "stop_reason", None) or "end_turn"
        return AssistantTurn(
            text="".join(text_parts),
            tool_calls=calls,
            stop_reason=_STOP_REASONS.get(raw_stop, "end_turn"),
            usage=usage,
            model_id=model_id,
            raw={"api_model": getattr(response, "model", ""), "stop_reason": raw_stop},
        )
