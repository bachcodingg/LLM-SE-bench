"""
Anthropic Claude API client.

Wraps the ``anthropic`` SDK to conform to the ``LLMClient`` interface.
Handles message formatting, streaming disabled (for benchmarking determinism),
and provider-specific error mapping.

Usage::

    client = ClaudeClient(config)
    response = client.send_prompt(prompt)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from contracts import LLMResponse, Prompt
from llm_gateway.adapters.anthropic import AnthropicAdapter
from llm_gateway.clients.base import LLMClient, _new_id
from llm_gateway.conversation import AssistantTurn, Conversation, ToolSchema

logger = logging.getLogger(__name__)


class ClaudeClient(LLMClient):
    """Concrete LLM client for Anthropic Claude models.

    Uses the ``anthropic.Anthropic`` SDK client under the hood.  The
    ``_call_api`` method translates a ``Prompt`` into the Messages API format
    and returns an ``LLMResponse``.

    Attributes:
        provider_name: Always ``"claude"``.
    """

    provider_name: str = "claude"
    adapter = AnthropicAdapter()

    def _get_sdk_client(self) -> Any:
        """Lazily construct and return an ``anthropic.Anthropic`` instance.

        Returns:
            A configured Anthropic SDK client.
        """
        if not hasattr(self, "_sdk_client"):
            import anthropic

            prov_cfg = self.config.providers.get("claude")
            if prov_cfg is None:
                raise ValueError("No 'claude' provider in GatewayConfig.providers")
            api_key = prov_cfg.resolve_api_key()
            kwargs: dict[str, Any] = {"api_key": api_key}
            if prov_cfg.base_url:
                kwargs["base_url"] = prov_cfg.base_url
            kwargs["timeout"] = prov_cfg.timeout_seconds
            kwargs["max_retries"] = prov_cfg.max_retries
            self._sdk_client = anthropic.Anthropic(**kwargs)
        return self._sdk_client

    def _call_api(self, prompt: Prompt) -> LLMResponse:
        """Send *prompt* to the Claude Messages API.

        Args:
            prompt: A ``Prompt`` with ``system_message``, ``user_message``,
                ``model_id``, ``temperature``, and ``max_tokens`` populated.

        Returns:
            An ``LLMResponse`` with token counts, finish reason, and raw text.

        Raises:
            anthropic.APIError: On any API-level failure.
        """
        client = self._get_sdk_client()

        messages = [{"role": "user", "content": prompt.user_message}]

        kwargs: dict[str, Any] = {
            "model": prompt.model_id,
            "messages": messages,
            "max_tokens": prompt.max_tokens,
            "temperature": prompt.temperature,
        }
        if prompt.system_message:
            kwargs["system"] = prompt.system_message

        logger.debug("Claude API call: model=%s tokens=%d", prompt.model_id, prompt.max_tokens)
        api_response = client.messages.create(**kwargs)

        raw_text = ""
        for block in api_response.content:
            if hasattr(block, "text"):
                raw_text += block.text

        return LLMResponse(
            response_id=_new_id("resp"),
            prompt_id=prompt.prompt_id,
            model_id=prompt.model_id,
            raw_text=raw_text,
            finish_reason=api_response.stop_reason or "stop",
            prompt_tokens=api_response.usage.input_tokens,
            completion_tokens=api_response.usage.output_tokens,
            latency_ms=0.0,  # filled in by base class
            created_at=datetime.utcnow(),
            metadata={
                "provider": "claude",
                "api_model": api_response.model,
            },
        )

    def _call_api_with_tools(
        self,
        conversation: Conversation,
        tools: list[ToolSchema],
        model_id: str,
        max_tokens: int,
        temperature: float,
    ) -> AssistantTurn:
        """Send a tool-enabled conversation to the Messages API."""
        client = self._get_sdk_client()
        request = self.adapter.encode_request(
            conversation=conversation,
            tools=tools,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        logger.debug(
            "Claude tool call: model=%s messages=%d tools=%d",
            model_id, len(conversation.messages), len(tools),
        )
        return self.adapter.decode_response(client.messages.create(**request), model_id)
