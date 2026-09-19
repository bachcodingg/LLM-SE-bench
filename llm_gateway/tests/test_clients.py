"""
Tests for LLM client implementations and the client factory.

All provider SDK calls are mocked — no real API requests are made.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from contracts import LLMResponse, Prompt
from llm_gateway.audit import AuditLogger
from llm_gateway.cache import ResponseCache
from llm_gateway.clients.base import LLMClientFactory, _new_id
from llm_gateway.clients.claude import ClaudeClient
from llm_gateway.clients.gemini import GeminiClient
from llm_gateway.clients.gpt4 import GPT4Client
from llm_gateway.config import GatewayConfig
from llm_gateway.cost_tracker import CostTracker
from llm_gateway.rate_limiter import RateLimiter

# ── helpers ────────────────────────────────────────────────────────────

def _make_prompt(model_id: str = "gpt-4o") -> Prompt:
    """Build a simple Prompt for tests."""
    return Prompt(
        prompt_id="pmt-unit-001",
        problem_id="prob-001",
        model_id=model_id,
        system_message="You are a tester.",
        user_message="Write a unit test.",
        temperature=0.0,
        max_tokens=1024,
    )


def _make_response(prompt_id: str = "pmt-unit-001", model_id: str = "gpt-4o") -> LLMResponse:
    """Build a simple LLMResponse for tests."""
    return LLMResponse(
        response_id="resp-unit-001",
        prompt_id=prompt_id,
        model_id=model_id,
        raw_text="```java\nSystem.out.println(1);\n```",
        finish_reason="stop",
        prompt_tokens=100,
        completion_tokens=20,
        latency_ms=500.0,
        metadata={"provider": "mock"},
    )


# ── _new_id ────────────────────────────────────────────────────────────

class TestNewId:
    """Tests for the _new_id utility."""

    def test_prefix(self) -> None:
        """Generated id starts with the given prefix."""
        result = _new_id("foo")
        assert result.startswith("foo-")

    def test_uniqueness(self) -> None:
        """Successive calls produce distinct ids."""
        ids = {_new_id() for _ in range(100)}
        assert len(ids) == 100


# ── ClaudeClient ───────────────────────────────────────────────────────

class TestClaudeClient:
    """Tests for ClaudeClient._call_api with mocked anthropic SDK."""

    def _make_client(self, gateway_config: GatewayConfig) -> ClaudeClient:
        """Instantiate a ClaudeClient without shared infra."""
        return ClaudeClient(config=gateway_config)

    @patch("llm_gateway.clients.claude.ClaudeClient._get_sdk_client")
    def test_call_api_success(self, mock_sdk: MagicMock, gateway_config: GatewayConfig) -> None:
        """Successful Claude API call returns a valid LLMResponse."""
        # Build mock SDK response.
        mock_block = MagicMock()
        mock_block.text = "```java\nint x = 1;\n```"

        mock_usage = MagicMock()
        mock_usage.input_tokens = 50
        mock_usage.output_tokens = 15

        mock_api_resp = MagicMock()
        mock_api_resp.content = [mock_block]
        mock_api_resp.usage = mock_usage
        mock_api_resp.stop_reason = "end_turn"
        mock_api_resp.model = "claude-3-5-sonnet-20241022"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_api_resp
        mock_sdk.return_value = mock_client

        client = self._make_client(gateway_config)
        prompt = _make_prompt(model_id="claude-3-5-sonnet-20241022")
        resp = client._call_api(prompt)

        assert isinstance(resp, LLMResponse)
        assert resp.prompt_id == prompt.prompt_id
        assert resp.model_id == prompt.model_id
        assert "int x = 1;" in resp.raw_text
        assert resp.prompt_tokens == 50
        assert resp.completion_tokens == 15
        assert resp.finish_reason == "end_turn"
        assert resp.metadata["provider"] == "claude"

    @patch("llm_gateway.clients.claude.ClaudeClient._get_sdk_client")
    def test_call_api_empty_system(self, mock_sdk: MagicMock, gateway_config: GatewayConfig) -> None:
        """System message omitted when empty."""
        mock_block = MagicMock()
        mock_block.text = "output"
        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5
        mock_api_resp = MagicMock()
        mock_api_resp.content = [mock_block]
        mock_api_resp.usage = mock_usage
        mock_api_resp.stop_reason = "stop"
        mock_api_resp.model = "claude-3-5-sonnet-20241022"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_api_resp
        mock_sdk.return_value = mock_client

        client = self._make_client(gateway_config)
        prompt = _make_prompt(model_id="claude-3-5-sonnet-20241022")
        prompt.system_message = ""
        client._call_api(prompt)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "system" not in call_kwargs


# ── GPT4Client ─────────────────────────────────────────────────────────

class TestGPT4Client:
    """Tests for GPT4Client._call_api with mocked openai SDK."""

    def _make_client(self, gateway_config: GatewayConfig) -> GPT4Client:
        """Instantiate a GPT4Client without shared infra."""
        return GPT4Client(config=gateway_config)

    @patch("llm_gateway.clients.gpt4.GPT4Client._get_sdk_client")
    def test_call_api_success(self, mock_sdk: MagicMock, gateway_config: GatewayConfig) -> None:
        """Successful GPT-4 API call returns a valid LLMResponse."""
        mock_message = MagicMock()
        mock_message.content = "```java\nreturn 42;\n```"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 80
        mock_usage.completion_tokens = 25

        mock_api_resp = MagicMock()
        mock_api_resp.choices = [mock_choice]
        mock_api_resp.usage = mock_usage
        mock_api_resp.model = "gpt-4o"
        mock_api_resp.id = "chatcmpl-test"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_api_resp
        mock_sdk.return_value = mock_client

        client = self._make_client(gateway_config)
        prompt = _make_prompt(model_id="gpt-4o")
        resp = client._call_api(prompt)

        assert isinstance(resp, LLMResponse)
        assert resp.prompt_tokens == 80
        assert resp.completion_tokens == 25
        assert resp.finish_reason == "stop"
        assert "return 42;" in resp.raw_text
        assert resp.metadata["provider"] == "gpt4"

    @patch("llm_gateway.clients.gpt4.GPT4Client._get_sdk_client")
    def test_call_api_no_usage(self, mock_sdk: MagicMock, gateway_config: GatewayConfig) -> None:
        """Handles None usage gracefully."""
        mock_message = MagicMock()
        mock_message.content = "code"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_api_resp = MagicMock()
        mock_api_resp.choices = [mock_choice]
        mock_api_resp.usage = None
        mock_api_resp.model = "gpt-4o"
        mock_api_resp.id = "x"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_api_resp
        mock_sdk.return_value = mock_client

        client = self._make_client(gateway_config)
        resp = client._call_api(_make_prompt())
        assert resp.prompt_tokens == 0
        assert resp.completion_tokens == 0


# ── GeminiClient ───────────────────────────────────────────────────────

class TestGeminiClient:
    """Tests for GeminiClient._call_api with mocked google SDK."""

    def _make_client(self, gateway_config: GatewayConfig) -> GeminiClient:
        """Instantiate a GeminiClient without shared infra."""
        return GeminiClient(config=gateway_config)

    def test_call_api_success(self, gateway_config: GatewayConfig) -> None:
        """Successful Gemini API call returns a valid LLMResponse."""
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 60
        mock_usage.candidates_token_count = 30

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = MagicMock()
        mock_candidate.finish_reason.name = "STOP"

        mock_api_resp = MagicMock()
        mock_api_resp.text = '```java\nString s = "hi";\n```'
        mock_api_resp.usage_metadata = mock_usage
        mock_api_resp.candidates = [mock_candidate]

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_api_resp

        # Build a fake google.generativeai module
        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.types.GenerationConfig = MagicMock()

        import sys
        fake_google = types.ModuleType("google")
        fake_google.generativeai = mock_genai  # type: ignore[attr-defined]
        fake_google.api_core = MagicMock()  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {
            "google": fake_google,
            "google.generativeai": mock_genai,
            "google.generativeai.types": mock_genai.types,
            "google.api_core": fake_google.api_core,
            "google.api_core.exceptions": MagicMock(),
        }):
            client = self._make_client(gateway_config)
            prompt = _make_prompt(model_id="gemini-1.5-pro")
            resp = client._call_api(prompt)

        assert isinstance(resp, LLMResponse)
        assert resp.prompt_tokens == 60
        assert resp.completion_tokens == 30
        assert resp.metadata["provider"] == "gemini"


# ── LLMClient.send_prompt pipeline ────────────────────────────────────

class TestSendPromptPipeline:
    """Integration-level tests for the full send_prompt pipeline."""

    def _make_concrete_client(
        self, gateway_config: GatewayConfig, tmp_path, with_infra: bool = True
    ) -> GPT4Client:
        """Build a GPT4Client with optional cache / tracker / audit."""
        cache = ResponseCache(db_path=tmp_path / "test.db") if with_infra else None
        tracker = CostTracker(gateway_config) if with_infra else None
        limiter = RateLimiter(gateway_config) if with_infra else None
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl") if with_infra else None
        return GPT4Client(
            config=gateway_config,
            cache=cache,
            cost_tracker=tracker,
            rate_limiter=limiter,
            audit_logger=audit,
        )

    @patch("llm_gateway.clients.gpt4.GPT4Client._call_api")
    def test_cache_miss_then_hit(
        self, mock_api: MagicMock, gateway_config: GatewayConfig, tmp_path
    ) -> None:
        """First call hits API; second call returns cached response."""
        mock_api.return_value = _make_response()

        client = self._make_concrete_client(gateway_config, tmp_path)
        prompt = _make_prompt()

        # First call — cache miss.
        resp1 = client.send_prompt(prompt)
        assert resp1.metadata.get("cache_hit") is False
        assert mock_api.call_count == 1

        # Second call — cache hit.
        resp2 = client.send_prompt(prompt)
        assert resp2.metadata.get("cache_hit") is True
        assert mock_api.call_count == 1  # no additional API call

    @patch("llm_gateway.clients.gpt4.GPT4Client._call_api")
    def test_code_extraction(
        self, mock_api: MagicMock, gateway_config: GatewayConfig, tmp_path
    ) -> None:
        """Extracted code is populated from raw_text."""
        mock_api.return_value = _make_response()
        client = self._make_concrete_client(gateway_config, tmp_path)
        resp = client.send_prompt(_make_prompt())
        assert "System.out.println(1);" in resp.extracted_code

    @patch("llm_gateway.clients.gpt4.GPT4Client._call_api")
    def test_cost_tracked(
        self, mock_api: MagicMock, gateway_config: GatewayConfig, tmp_path
    ) -> None:
        """Cost tracker records the call cost."""
        mock_api.return_value = _make_response()
        client = self._make_concrete_client(gateway_config, tmp_path)
        client.send_prompt(_make_prompt())

        assert client.cost_tracker is not None
        assert client.cost_tracker.request_count() == 1
        assert client.cost_tracker.total_cost_usd() > 0

    @patch("llm_gateway.clients.gpt4.GPT4Client._call_api")
    def test_audit_logged(
        self, mock_api: MagicMock, gateway_config: GatewayConfig, tmp_path
    ) -> None:
        """Audit logger records the event."""
        mock_api.return_value = _make_response()
        client = self._make_concrete_client(gateway_config, tmp_path)
        client.send_prompt(_make_prompt())

        assert client.audit_logger is not None
        assert client.audit_logger.count() == 1

    @patch("llm_gateway.clients.gpt4.GPT4Client._call_api")
    def test_no_infra(
        self, mock_api: MagicMock, gateway_config: GatewayConfig, tmp_path
    ) -> None:
        """Pipeline works without cache / tracker / limiter / audit."""
        mock_api.return_value = _make_response()
        client = self._make_concrete_client(gateway_config, tmp_path, with_infra=False)
        resp = client.send_prompt(_make_prompt())
        assert isinstance(resp, LLMResponse)


# ── LLMClientFactory ──────────────────────────────────────────────────

class TestLLMClientFactory:
    """Tests for the client factory."""

    def test_get_claude_client(self, gateway_config: GatewayConfig) -> None:
        """Factory returns a ClaudeClient for claude model ids."""
        factory = LLMClientFactory(config=gateway_config)
        client = factory.get_client("claude-3-5-sonnet-20241022")
        assert isinstance(client, ClaudeClient)

    def test_get_gpt4_client(self, gateway_config: GatewayConfig) -> None:
        """Factory returns a GPT4Client for gpt model ids."""
        factory = LLMClientFactory(config=gateway_config)
        client = factory.get_client("gpt-4o")
        assert isinstance(client, GPT4Client)

    def test_get_gemini_client(self, gateway_config: GatewayConfig) -> None:
        """Factory returns a GeminiClient for gemini model ids."""
        factory = LLMClientFactory(config=gateway_config)
        client = factory.get_client("gemini-1.5-pro")
        assert isinstance(client, GeminiClient)

    def test_unknown_model_raises(self, gateway_config: GatewayConfig) -> None:
        """Unknown model id raises ValueError."""
        factory = LLMClientFactory(config=gateway_config)
        with pytest.raises(ValueError, match="Cannot infer provider"):
            factory.get_client("unknown-model-xyz")

    def test_caches_client_instances(self, gateway_config: GatewayConfig) -> None:
        """Same model id returns the same client instance."""
        factory = LLMClientFactory(config=gateway_config)
        c1 = factory.get_client("gpt-4o")
        c2 = factory.get_client("gpt-4o")
        assert c1 is c2
