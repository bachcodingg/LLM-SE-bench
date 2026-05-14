"""
Google Gemini API client.

Wraps the ``google-generativeai`` SDK to conform to the ``LLMClient``
interface.  Uses ``GenerativeModel.generate_content`` for single-turn
completions.

Usage::

    client = GeminiClient(config)
    response = client.send_prompt(prompt)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from contracts import LLMResponse, Prompt
from llm_gateway.clients.base import LLMClient, _new_id

logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    """Concrete LLM client for Google Gemini models.

    Uses the ``google.generativeai`` SDK under the hood.  The ``_call_api``
    method translates a ``Prompt`` into ``generate_content`` parameters and
    returns an ``LLMResponse``.

    Attributes:
        provider_name: Always ``"gemini"``.
    """

    provider_name: str = "gemini"

    def _get_sdk_client(self) -> Any:
        """Lazily configure the SDK and return a ``GenerativeModel``.

        The Gemini SDK is configured globally via ``genai.configure``,
        then a model-specific ``GenerativeModel`` is returned.

        Returns:
            A ``google.generativeai.GenerativeModel`` instance.
        """
        if not hasattr(self, "_sdk_client"):
            import google.generativeai as genai

            prov_cfg = self.config.providers.get("gemini")
            if prov_cfg is None:
                raise ValueError("No 'gemini' provider in GatewayConfig.providers")
            api_key = prov_cfg.resolve_api_key()
            genai.configure(api_key=api_key)
            model_name = prov_cfg.default_model or "gemini-1.5-pro"
            self._sdk_client = genai.GenerativeModel(model_name)
            self._genai = genai
        return self._sdk_client

    def _call_api(self, prompt: Prompt) -> LLMResponse:
        """Send *prompt* to the Gemini ``generate_content`` API.

        Args:
            prompt: A ``Prompt`` with ``system_message``, ``user_message``,
                ``model_id``, ``temperature``, and ``max_tokens`` populated.

        Returns:
            An ``LLMResponse`` with token counts, finish reason, and raw text.

        Raises:
            google.api_core.exceptions.GoogleAPIError: On any API failure.
        """
        import google.generativeai as genai

        prov_cfg = self.config.providers.get("gemini")
        if prov_cfg is None:
            raise ValueError("No 'gemini' provider in GatewayConfig.providers")

        # Ensure API key is configured globally before creating a model.
        self._get_sdk_client()

        # Build the model (use prompt.model_id if it differs from default).
        model_name = prompt.model_id
        model = genai.GenerativeModel(
            model_name,
            system_instruction=prompt.system_message if prompt.system_message else None,
        )

        generation_config = genai.types.GenerationConfig(
            temperature=prompt.temperature,
            max_output_tokens=prompt.max_tokens,
        )

        logger.debug("Gemini API call: model=%s tokens=%d", model_name, prompt.max_tokens)
        api_response = model.generate_content(
            prompt.user_message,
            generation_config=generation_config,
        )

        raw_text = api_response.text or ""

        # Token counts via usage_metadata (may be None on some SDK versions).
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(api_response, "usage_metadata") and api_response.usage_metadata:
            meta = api_response.usage_metadata
            prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0
            completion_tokens = getattr(meta, "candidates_token_count", 0) or 0

        # Finish reason mapping.
        finish_reason = "stop"
        if api_response.candidates:
            candidate = api_response.candidates[0]
            fr = getattr(candidate, "finish_reason", None)
            if fr is not None:
                finish_reason = str(fr.name).lower() if hasattr(fr, "name") else str(fr)

        return LLMResponse(
            response_id=_new_id("resp"),
            prompt_id=prompt.prompt_id,
            model_id=prompt.model_id,
            raw_text=raw_text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=0.0,
            created_at=datetime.utcnow(),
            metadata={
                "provider": "gemini",
                "api_model": model_name,
            },
        )
