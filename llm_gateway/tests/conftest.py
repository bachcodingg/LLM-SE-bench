"""
Shared pytest fixtures for llm_gateway tests.

Provides pre-built ``GatewayConfig``, ``Prompt``, and ``LLMResponse`` objects
so that individual test modules can focus on behaviour rather than boilerplate.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from contracts import LLMResponse, Prompt
from llm_gateway.config import GatewayConfig, ProviderConfig


@pytest.fixture()
def gateway_config() -> GatewayConfig:
    """Return a ``GatewayConfig`` with dummy API keys baked in.

    No real API calls should be made in unit tests — all provider calls
    are mocked.
    """
    return GatewayConfig(
        providers={
            "claude": ProviderConfig(api_key="sk-test-claude", default_model="claude-3-5-sonnet-20241022"),
            "gpt4": ProviderConfig(api_key="sk-test-openai", default_model="gpt-4o"),
            "gemini": ProviderConfig(api_key="sk-test-gemini", default_model="gemini-1.5-pro"),
        },
    )


@pytest.fixture()
def sample_prompt() -> Prompt:
    """Return a minimal ``Prompt`` for testing."""
    return Prompt(
        prompt_id="pmt-test-001",
        problem_id="prob-fizzbuzz",
        model_id="gpt-4o",
        system_message="You are a helpful assistant.",
        user_message="Implement FizzBuzz in Java.",
        temperature=0.0,
        max_tokens=2048,
        created_at=datetime(2026, 1, 1),
        metadata={"template_name": "codegen_zero_shot.j2"},
    )


@pytest.fixture()
def sample_response() -> LLMResponse:
    """Return a minimal ``LLMResponse`` for testing."""
    return LLMResponse(
        response_id="resp-test-001",
        prompt_id="pmt-test-001",
        model_id="gpt-4o",
        raw_text='Here is the solution:\n```java\npublic class FizzBuzz {\n    public static String fizzBuzz(int n) {\n        if (n % 15 == 0) return "FizzBuzz";\n        if (n % 3 == 0) return "Fizz";\n        if (n % 5 == 0) return "Buzz";\n        return String.valueOf(n);\n    }\n}\n```\n',
        extracted_code="",
        finish_reason="stop",
        prompt_tokens=150,
        completion_tokens=80,
        latency_ms=1234.5,
        created_at=datetime(2026, 1, 1),
        metadata={"provider": "gpt4", "cache_hit": False},
    )


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Return the pytest-provided temporary directory."""
    return tmp_path
