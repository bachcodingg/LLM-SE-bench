"""
Tests for GatewayConfig and its sub-components.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from llm_gateway.config import (
    GatewayConfig,
    ProviderConfig,
)


class TestProviderConfig:
    """Tests for ProviderConfig."""

    def test_resolve_api_key_direct(self) -> None:
        """Directly set api_key is returned."""
        pc = ProviderConfig(api_key="sk-direct")
        assert pc.resolve_api_key() == "sk-direct"

    def test_resolve_api_key_env(self) -> None:
        """API key from environment variable."""
        pc = ProviderConfig(env_var="TEST_LLM_KEY")
        with patch.dict(os.environ, {"TEST_LLM_KEY": "sk-from-env"}):
            assert pc.resolve_api_key() == "sk-from-env"

    def test_resolve_api_key_missing(self) -> None:
        """Missing key raises ValueError."""
        pc = ProviderConfig(env_var="NONEXISTENT_KEY_12345")
        with pytest.raises(ValueError, match="No API key"):
            pc.resolve_api_key()

    def test_direct_key_takes_precedence(self) -> None:
        """Direct api_key overrides env_var."""
        pc = ProviderConfig(api_key="sk-direct", env_var="TEST_LLM_KEY")
        with patch.dict(os.environ, {"TEST_LLM_KEY": "sk-env"}):
            assert pc.resolve_api_key() == "sk-direct"


class TestGatewayConfig:
    """Tests for the top-level config."""

    def test_defaults(self) -> None:
        """Default config has three providers and pricing."""
        cfg = GatewayConfig()
        assert "claude" in cfg.providers
        assert "gpt4" in cfg.providers
        assert "gemini" in cfg.providers
        assert len(cfg.pricing) >= 3

    def test_get_pricing_known(self) -> None:
        """Known model returns correct pricing."""
        cfg = GatewayConfig()
        p = cfg.get_pricing("gpt-4o")
        assert p.cost_per_prompt_token > 0

    def test_get_pricing_unknown(self) -> None:
        """Unknown model returns zero-cost tier."""
        cfg = GatewayConfig()
        p = cfg.get_pricing("fantasy-model-v99")
        assert p.cost_per_prompt_token == 0.0
        assert p.cost_per_completion_token == 0.0

    def test_provider_for_model_claude(self) -> None:
        """Claude model ids map to 'claude'."""
        cfg = GatewayConfig()
        assert cfg.provider_for_model("claude-3-5-sonnet-20241022") == "claude"

    def test_provider_for_model_gpt(self) -> None:
        """GPT model ids map to 'gpt4'."""
        cfg = GatewayConfig()
        assert cfg.provider_for_model("gpt-4o") == "gpt4"

    def test_provider_for_model_gemini(self) -> None:
        """Gemini model ids map to 'gemini'."""
        cfg = GatewayConfig()
        assert cfg.provider_for_model("gemini-1.5-pro") == "gemini"

    def test_provider_for_model_unknown(self) -> None:
        """Unknown model id raises ValueError."""
        cfg = GatewayConfig()
        with pytest.raises(ValueError, match="Cannot infer provider"):
            cfg.provider_for_model("llama-3-70b")

    def test_from_yaml(self, tmp_path: Path) -> None:
        """from_yaml loads overrides on top of defaults."""
        yaml_data = {
            "cache_db_path": "/tmp/custom.db",
            "providers": {
                "claude": {"api_key": "sk-yaml-key"},
            },
            "rate_limits": {
                "claude": {"requests_per_minute": 30},
            },
        }
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as fh:
            yaml.dump(yaml_data, fh)

        cfg = GatewayConfig.from_yaml(cfg_path)
        assert cfg.cache_db_path == "/tmp/custom.db"
        assert cfg.providers["claude"].api_key == "sk-yaml-key"
        assert cfg.rate_limits["claude"].requests_per_minute == 30
        # Other defaults still present.
        assert "gpt4" in cfg.providers

    def test_from_yaml_missing_file(self) -> None:
        """Missing YAML file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            GatewayConfig.from_yaml("/nonexistent/path.yaml")

    def test_from_yaml_empty_file(self, tmp_path: Path) -> None:
        """Empty YAML file produces defaults."""
        cfg_path = tmp_path / "empty.yaml"
        cfg_path.write_text("")
        cfg = GatewayConfig.from_yaml(cfg_path)
        assert "claude" in cfg.providers

    def test_from_yaml_new_provider(self, tmp_path: Path) -> None:
        """YAML can add a new provider not in defaults."""
        yaml_data = {
            "providers": {
                "custom": {"api_key": "sk-custom", "default_model": "custom-v1"},
            },
        }
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as fh:
            yaml.dump(yaml_data, fh)

        cfg = GatewayConfig.from_yaml(cfg_path)
        assert "custom" in cfg.providers
        assert cfg.providers["custom"].api_key == "sk-custom"

    def test_from_yaml_custom_pricing(self, tmp_path: Path) -> None:
        """YAML pricing section replaces defaults entirely."""
        yaml_data = {
            "pricing": [
                {"model_id": "test-model", "cost_per_prompt_token": 0.001, "cost_per_completion_token": 0.002},
            ],
        }
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as fh:
            yaml.dump(yaml_data, fh)

        cfg = GatewayConfig.from_yaml(cfg_path)
        assert len(cfg.pricing) == 1
        assert cfg.pricing[0].model_id == "test-model"
