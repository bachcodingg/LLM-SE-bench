"""
llm_gateway.adapters — translation between neutral types and provider wire formats.

One adapter per provider, each implementing :class:`ProviderAdapter`:

``encode_*``
    Neutral types out to the provider's request shape.
``decode_response``
    The provider's response back to a neutral :class:`AssistantTurn`.

Adapters are pure: no network, no SDK client, no state.  That is what makes
the round trip testable without an API key, and it is where the value is —
every one of the differences these classes paper over is a chance to get
tool calling silently wrong.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llm_gateway.conversation import AssistantTurn, Conversation, ToolSchema

__all__ = ["ProviderAdapter", "get_adapter", "ADAPTERS"]


class ProviderAdapter(ABC):
    """Translates between the neutral conversation types and one provider."""

    #: Provider key, matching ``GatewayConfig.providers``.
    provider_name: str = "base"

    @abstractmethod
    def encode_tools(self, tools: list[ToolSchema]) -> Any:
        """Neutral tool schemas into the provider's declaration format."""

    @abstractmethod
    def encode_conversation(self, conversation: Conversation) -> Any:
        """Neutral conversation into the provider's message list.

        Returns whatever that provider's SDK wants; the system prompt is
        handled by :meth:`encode_request`, not here, because two providers
        put it outside the message list.
        """

    @abstractmethod
    def encode_request(
        self,
        conversation: Conversation,
        tools: list[ToolSchema],
        model_id: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """The complete keyword arguments for one SDK call."""

    @abstractmethod
    def decode_response(self, response: Any, model_id: str) -> AssistantTurn:
        """The provider's response object into a neutral :class:`AssistantTurn`."""


def _anthropic() -> type[ProviderAdapter]:
    from llm_gateway.adapters.anthropic import AnthropicAdapter

    return AnthropicAdapter


def _openai() -> type[ProviderAdapter]:
    from llm_gateway.adapters.openai import OpenAIAdapter

    return OpenAIAdapter


def _gemini() -> type[ProviderAdapter]:
    from llm_gateway.adapters.gemini import GeminiAdapter

    return GeminiAdapter


#: ``{provider key: () -> adapter class}``, matching the keys used by
#: ``GatewayConfig.provider_for_model``.
ADAPTERS = {
    "claude": _anthropic,
    "gpt4": _openai,
    "gemini": _gemini,
}


def get_adapter(provider: str) -> ProviderAdapter:
    """Return an adapter instance for *provider*.

    Raises:
        KeyError: for an unknown provider, naming the known ones.
    """
    if provider not in ADAPTERS:
        raise KeyError(
            f"no tool-calling adapter for provider {provider!r}; "
            f"known: {sorted(ADAPTERS)}"
        )
    return ADAPTERS[provider]()()
