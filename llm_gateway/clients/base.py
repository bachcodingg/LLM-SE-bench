"""
Abstract base class for LLM provider clients.

Every concrete provider (Claude, GPT-4, Gemini) inherits from ``LLMClient``
and implements ``_call_api``.  The base class owns the cross-cutting plumbing:
rate limiting, caching, cost tracking, audit logging, and code extraction.

``LLMClientFactory`` maps a ``model_id`` string to the right concrete client.
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from contracts import CostRecord, LLMResponse, Prompt
from llm_gateway.models import PromptRenderer

if TYPE_CHECKING:
    from llm_gateway.audit import AuditLogger
    from llm_gateway.cache import ResponseCache
    from llm_gateway.config import GatewayConfig
    from llm_gateway.conversation import AssistantTurn, Conversation, ToolSchema
    from llm_gateway.cost_tracker import CostTracker
    from llm_gateway.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


def _new_id(prefix: str = "resp") -> str:
    """Generate a short unique id.

    Args:
        prefix: String prepended to the UUID segment.

    Returns:
        E.g. ``"resp-a3f8b2c1"``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class LLMClient(ABC):
    """Abstract base class for LLM provider clients.

    Subclasses must implement ``_call_api`` which performs the raw HTTP call
    and returns a partially-populated ``LLMResponse``.

    The public ``send_prompt`` method orchestrates:
    1. Rate-limit gating
    2. Cache lookup
    3. Raw API call (if cache miss)
    4. Code extraction from the response
    5. Cost recording
    6. Audit logging
    7. Cache storage

    Attributes:
        provider_name: Short provider label (``"claude"``, ``"gpt4"``, ``"gemini"``).
        config:        The gateway-wide configuration.
        cache:         Optional response cache.
        cost_tracker:  Optional cost tracker.
        rate_limiter:  Optional rate limiter.
        audit_logger:  Optional audit logger.
    """

    provider_name: str = "base"

    def __init__(
        self,
        config: GatewayConfig,
        cache: ResponseCache | None = None,
        cost_tracker: CostTracker | None = None,
        rate_limiter: RateLimiter | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        """Initialise the client with shared infrastructure.

        Args:
            config:       Gateway configuration.
            cache:        Response cache (SQLite-backed).
            cost_tracker: Cost accounting tracker.
            rate_limiter: Token-bucket rate limiter.
            audit_logger: JSONL audit logger.
        """
        self.config = config
        self.cache = cache
        self.cost_tracker = cost_tracker
        self.rate_limiter = rate_limiter
        self.audit_logger = audit_logger

    # ── abstract hook ──────────────────────────────────────────────────

    @abstractmethod
    def _call_api(self, prompt: Prompt) -> LLMResponse:
        """Perform the actual API call.

        Implementations must populate at minimum: ``response_id``,
        ``prompt_id``, ``model_id``, ``raw_text``, ``finish_reason``,
        ``prompt_tokens``, ``completion_tokens``, and ``latency_ms``.

        Args:
            prompt: The fully-formed ``Prompt`` to send.

        Returns:
            A ``LLMResponse`` with provider-specific fields filled in.

        Raises:
            Exception: On any API-level error.
        """
        ...

    # ── public entry point ─────────────────────────────────────────────

    def send_prompt(self, prompt: Prompt) -> LLMResponse:
        """Send a prompt through the full gateway pipeline.

        Handles rate limiting → cache lookup → API call → code extraction →
        cost tracking → audit → cache store.

        Args:
            prompt: A ``Prompt`` object (typically built by ``PromptRenderer``).

        Returns:
            A complete ``LLMResponse`` with ``extracted_code`` populated.
        """
        # 1 — Rate-limit gate
        if self.rate_limiter:
            self.rate_limiter.acquire(self.provider_name)

        # 2 — Cache lookup
        prompt_hash = PromptRenderer.hash_prompt(
            prompt.system_message, prompt.user_message
        )
        if self.cache:
            cached = self.cache.get(prompt_hash, prompt.model_id)
            if cached is not None:
                logger.info(
                    "Cache hit for prompt_hash=%s model=%s",
                    prompt_hash[:12],
                    prompt.model_id,
                )
                cached.metadata["cache_hit"] = True
                if self.audit_logger:
                    self.audit_logger.log_event(
                        event_type="cache_hit",
                        prompt=prompt,
                        response=cached,
                        prompt_hash=prompt_hash,
                    )
                return cached

        # 3 — API call
        t0 = time.perf_counter()
        response = self._call_api(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        response.latency_ms = elapsed_ms
        response.metadata["cache_hit"] = False
        response.metadata["prompt_hash"] = prompt_hash

        # 4 — Code extraction
        response.extracted_code = PromptRenderer.extract_code_blocks(response.raw_text)

        # 5 — Cost recording
        cost_record: CostRecord | None = None
        if self.cost_tracker:
            cost_record = self.cost_tracker.record_cost(response)

        # 6 — Audit logging
        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="api_call",
                prompt=prompt,
                response=response,
                prompt_hash=prompt_hash,
                cost_usd=cost_record.total_cost_usd if cost_record else 0.0,
            )

        # 7 — Cache store
        if self.cache:
            self.cache.put(prompt_hash, prompt.model_id, response)

        return response

    # ── multi-turn, tool-calling entry point ───────────────────────────

    def _call_api_with_tools(
        self,
        conversation: Conversation,
        tools: list[ToolSchema],
        model_id: str,
        max_tokens: int,
        temperature: float,
    ) -> AssistantTurn:
        """Perform one tool-calling API call.

        Subclasses that support tool use override this.  The default
        raises, so a provider without an implementation fails loudly at the
        first agent step rather than silently degrading to text-only
        completion — which would look like a very stupid model.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement tool calling; "
            f"it can only be used for single-shot evaluation."
        )

    def send_conversation(
        self,
        conversation: Conversation,
        tools: list[ToolSchema],
        model_id: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AssistantTurn:
        """Send a multi-turn, tool-enabled conversation.

        The single-shot sibling of this method is ``send_prompt``.  The
        cross-cutting plumbing is the same except for one deliberate
        omission: **responses are not cached.**  A cache key covering a whole
        conversation almost never hits, and a key covering less than the whole
        conversation would return an answer to a different question. Prompt
        caching — the provider-side kind that actually pays off here — is
        reported through ``AssistantTurn.usage.cache_read_tokens``.

        Args:
            conversation: The conversation so far, system prompt included.
            tools:        Tools to offer. May be empty.
            model_id:     Model to call.
            max_tokens:   Output-token ceiling for this call.
            temperature:  Sampling temperature.

        Returns:
            A neutral ``AssistantTurn``, whatever the provider.
        """
        if self.rate_limiter:
            self.rate_limiter.acquire(self.provider_name)

        t0 = time.perf_counter()
        turn = self._call_api_with_tools(
            conversation=conversation,
            tools=tools,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        turn.latency_ms = (time.perf_counter() - t0) * 1000.0

        cost_record = self._record_turn_cost(turn, model_id)

        if self.audit_logger:
            # Audit wants the contract types, so the turn is projected onto
            # them. The projection is lossy — tool calls do not survive it —
            # and the trajectory table is what preserves the detail.
            self.audit_logger.log_event(
                event_type="agent_turn",
                prompt=Prompt(
                    prompt_id=_new_id("pmt"),
                    problem_id="agent",
                    model_id=model_id,
                    system_message=conversation.system,
                    user_message=f"<{len(conversation.messages)} messages>",
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                response=self._turn_as_response(turn, model_id),
                prompt_hash="",
                cost_usd=cost_record.total_cost_usd if cost_record else 0.0,
            )

        return turn

    def _record_turn_cost(self, turn: AssistantTurn, model_id: str) -> CostRecord | None:
        """Write a CostRecord for *turn*, if a tracker is attached.

        Cache reads are charged at the input rate here, which **understates**
        the saving: every provider discounts cached input, most heavily.
        Correct per-tier cache pricing needs a price table this project does
        not yet have, so the figure is deliberately conservative — an agent
        run will not look cheaper than it was.
        """
        if not self.cost_tracker:
            return None
        return self.cost_tracker.record_cost(self._turn_as_response(turn, model_id))

    @staticmethod
    def _turn_as_response(turn: AssistantTurn, model_id: str) -> LLMResponse:
        """Project an AssistantTurn onto the LLMResponse contract."""
        return LLMResponse(
            response_id=_new_id("resp"),
            prompt_id=_new_id("pmt"),
            model_id=model_id,
            raw_text=turn.text,
            finish_reason=turn.stop_reason,
            prompt_tokens=turn.usage.input_tokens + turn.usage.cache_read_tokens,
            completion_tokens=turn.usage.output_tokens,
            latency_ms=turn.latency_ms,
            metadata={
                "agent_turn": True,
                "tool_calls": [call.name for call in turn.tool_calls],
                "cache_read_tokens": turn.usage.cache_read_tokens,
                "cache_write_tokens": turn.usage.cache_write_tokens,
            },
        )


class LLMClientFactory:
    """Factory that maps model-id strings to concrete ``LLMClient`` instances.

    Usage::

        factory = LLMClientFactory(config)
        client = factory.get_client("claude-3-5-sonnet-20241022")
        response = client.send_prompt(prompt)

    Attributes:
        config:       Gateway configuration.
        cache:        Shared response cache.
        cost_tracker: Shared cost tracker.
        rate_limiter: Shared rate limiter.
        audit_logger: Shared audit logger.
    """

    def __init__(
        self,
        config: GatewayConfig,
        cache: ResponseCache | None = None,
        cost_tracker: CostTracker | None = None,
        rate_limiter: RateLimiter | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        """Initialise the factory with shared infrastructure.

        Args:
            config:       Gateway configuration.
            cache:        Response cache instance.
            cost_tracker: Cost tracker instance.
            rate_limiter: Rate limiter instance.
            audit_logger: Audit logger instance.
        """
        self.config = config
        self.cache = cache
        self.cost_tracker = cost_tracker
        self.rate_limiter = rate_limiter
        self.audit_logger = audit_logger
        self._clients: dict[str, LLMClient] = {}

    def get_client(self, model_id: str) -> LLMClient:
        """Return a client for the given *model_id*, creating one if needed.

        Caches client instances so repeated calls with the same *model_id*
        return the same object.

        Args:
            model_id: Model identifier (e.g. ``"gpt-4o"``).

        Returns:
            A concrete ``LLMClient`` subclass.

        Raises:
            ValueError: If the model id cannot be mapped to a provider.
        """
        if model_id in self._clients:
            return self._clients[model_id]

        provider = self.config.provider_for_model(model_id)

        # Lazy imports to avoid circular dependencies.
        from llm_gateway.clients.claude import ClaudeClient
        from llm_gateway.clients.gemini import GeminiClient
        from llm_gateway.clients.gpt4 import GPT4Client

        cls_map: dict[str, type[LLMClient]] = {
            "claude": ClaudeClient,
            "gpt4": GPT4Client,
            "gemini": GeminiClient,
        }

        client_cls = cls_map[provider]
        client = client_cls(
            config=self.config,
            cache=self.cache,
            cost_tracker=self.cost_tracker,
            rate_limiter=self.rate_limiter,
            audit_logger=self.audit_logger,
        )
        self._clients[model_id] = client
        return client
