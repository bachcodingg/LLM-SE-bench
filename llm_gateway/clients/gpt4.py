"""
OpenAI GPT-4 API client.

Wraps the ``openai`` SDK (v1+) to conform to the ``LLMClient`` interface.
Supports both ``gpt-4-turbo`` and ``gpt-4o`` model variants via the Chat
Completions endpoint.

Usage::

    client = GPT4Client(config)
    response = client.send_prompt(prompt)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from contracts import LLMResponse, Prompt
from llm_gateway.adapters.openai import OpenAIAdapter
from llm_gateway.clients.base import LLMClient, _new_id
from llm_gateway.conversation import AssistantTurn, Conversation, ToolSchema

logger = logging.getLogger(__name__)


class GPT4Client(LLMClient):
    """Concrete LLM client for OpenAI GPT-4 models.

    Uses the ``openai.OpenAI`` SDK client under the hood.  The ``_call_api``
    method translates a ``Prompt`` into the Chat Completions format and
    returns an ``LLMResponse``.

    Attributes:
        provider_name: Always ``"gpt4"``.
    """

    provider_name: str = "gpt4"
    adapter = OpenAIAdapter()

    def _get_sdk_client(self) -> Any:
        """Lazily construct and return an ``openai.OpenAI`` instance.

        Returns:
            A configured OpenAI SDK client.
        """
        if not hasattr(self, "_sdk_client"):
            import openai

            prov_cfg = self.config.providers.get("gpt4")
            if prov_cfg is None:
                raise ValueError("No 'gpt4' provider in GatewayConfig.providers")
            api_key = prov_cfg.resolve_api_key()
            kwargs: dict[str, Any] = {"api_key": api_key}
            if prov_cfg.base_url:
                kwargs["base_url"] = prov_cfg.base_url
            kwargs["timeout"] = prov_cfg.timeout_seconds
            kwargs["max_retries"] = prov_cfg.max_retries
            self._sdk_client = openai.OpenAI(**kwargs)
        return self._sdk_client

    def _call_api(self, prompt: Prompt) -> LLMResponse:
        """Send *prompt* to the OpenAI Chat Completions API.

        Args:
            prompt: A ``Prompt`` with ``system_message``, ``user_message``,
                ``model_id``, ``temperature``, and ``max_tokens`` populated.

        Returns:
            An ``LLMResponse`` with token counts, finish reason, and raw text.

        Raises:
            openai.APIError: On any API-level failure.
        """
        client = self._get_sdk_client()

        messages: list[dict[str, str]] = []
        if prompt.system_message:
            messages.append({"role": "system", "content": prompt.system_message})
        messages.append({"role": "user", "content": prompt.user_message})

        logger.debug("GPT-4 API call: model=%s tokens=%d", prompt.model_id, prompt.max_tokens)
        api_response = client.chat.completions.create(
            model=prompt.model_id,
            messages=messages,
            temperature=prompt.temperature,
            max_tokens=prompt.max_tokens,
        )

        choice = api_response.choices[0]
        raw_text = choice.message.content or ""

        usage = api_response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return LLMResponse(
            response_id=_new_id("resp"),
            prompt_id=prompt.prompt_id,
            model_id=prompt.model_id,
            raw_text=raw_text,
            finish_reason=choice.finish_reason or "stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=0.0,
            created_at=datetime.utcnow(),
            metadata={
                "provider": "gpt4",
                "api_model": api_response.model,
                "api_id": api_response.id,
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
        """Send a tool-enabled conversation to the Chat Completions API."""
        client = self._get_sdk_client()
        request = self.adapter.encode_request(
            conversation=conversation,
            tools=tools,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        logger.debug(
            "GPT-4 tool call: model=%s messages=%d tools=%d",
            model_id, len(conversation.messages), len(tools),
        )
        return self.adapter.decode_response(
            client.chat.completions.create(**request), model_id
        )
