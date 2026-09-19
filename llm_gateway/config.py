"""
Gateway configuration management.

Loads settings from a YAML file (or sensible defaults) and exposes them as a
typed ``GatewayConfig`` object.  Provider-specific sections hold API keys,
endpoints, and model aliases.  Cross-cutting settings (cache path, audit log
path, rate-limit windows) live at the top level.

Typical usage::

    cfg = GatewayConfig.from_yaml("config.yaml")
    # or use built-in defaults
    cfg = GatewayConfig()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ── provider-level config ──────────────────────────────────────────────

@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider (Claude / GPT-4 / Gemini).

    Attributes:
        api_key:         API key (falls back to ``env_var`` if empty).
        env_var:         Name of the environment variable holding the API key.
        default_model:   Default model string sent to the provider.
        base_url:        Override for the provider's base URL (useful for proxies).
        max_retries:     Maximum retries on transient errors.
        timeout_seconds: Per-request timeout.
        extra:           Any additional provider-specific key-value pairs.
    """

    api_key: str = ""
    env_var: str = ""
    default_model: str = ""
    base_url: str = ""
    max_retries: int = 3
    timeout_seconds: float = 120.0
    extra: dict[str, Any] = field(default_factory=dict)

    def resolve_api_key(self) -> str:
        """Return the API key, reading from the environment if necessary.

        Raises:
            ValueError: If no key can be resolved.
        """
        if self.api_key:
            return self.api_key
        if self.env_var:
            key = os.environ.get(self.env_var, "")
            if key:
                return key
        raise ValueError(
            f"No API key configured (env_var={self.env_var!r}).  "
            "Set the key directly or export the environment variable."
        )


# ── pricing config ─────────────────────────────────────────────────────

@dataclass
class PricingTier:
    """Token pricing for a single model.

    Attributes:
        model_id:                Model identifier string.
        cost_per_prompt_token:   USD per input token.
        cost_per_completion_token: USD per output token.
    """

    model_id: str = ""
    cost_per_prompt_token: float = 0.0
    cost_per_completion_token: float = 0.0


# ── rate-limit config ──────────────────────────────────────────────────

@dataclass
class RateLimitConfig:
    """Token-bucket parameters for a provider.

    Attributes:
        requests_per_minute: Sustained request cap.
        tokens_per_minute:   Sustained token cap (prompt + completion).
        burst_multiplier:    Allowed burst above the sustained rate (e.g. 1.5).
        retry_base_seconds:  Initial back-off wait on 429.
        retry_max_seconds:   Ceiling for exponential back-off.
    """

    requests_per_minute: int = 60
    tokens_per_minute: int = 100_000
    burst_multiplier: float = 1.5
    retry_base_seconds: float = 1.0
    retry_max_seconds: float = 60.0


# ── top-level gateway config ──────────────────────────────────────────

# Default pricing as of early-2026 estimates (USD)
_DEFAULT_PRICING: list[dict[str, Any]] = [
    # Claude 4.x
    {"model_id": "claude-opus-4-7", "cost_per_prompt_token": 15e-6, "cost_per_completion_token": 75e-6},
    {"model_id": "claude-sonnet-4-6", "cost_per_prompt_token": 3e-6, "cost_per_completion_token": 15e-6},
    {"model_id": "claude-haiku-4-5-20251001", "cost_per_prompt_token": 0.8e-6, "cost_per_completion_token": 4e-6},
    # Claude 3.x (legacy)
    {"model_id": "claude-3-5-sonnet-20241022", "cost_per_prompt_token": 3e-6, "cost_per_completion_token": 15e-6},
    # GPT-4
    {"model_id": "gpt-4o", "cost_per_prompt_token": 2.5e-6, "cost_per_completion_token": 10e-6},
    {"model_id": "gpt-4-turbo", "cost_per_prompt_token": 10e-6, "cost_per_completion_token": 30e-6},
    # Gemini 2.x
    {"model_id": "gemini-2.5-pro", "cost_per_prompt_token": 1.25e-6, "cost_per_completion_token": 10e-6},
    {"model_id": "gemini-2.5-flash", "cost_per_prompt_token": 0.15e-6, "cost_per_completion_token": 0.6e-6},
    {"model_id": "gemini-2.0-flash", "cost_per_prompt_token": 0.1e-6, "cost_per_completion_token": 0.4e-6},
]


@dataclass
class GatewayConfig:
    """Top-level configuration for the LLM Gateway.

    Holds provider configs, pricing tiers, rate limits, cache / audit paths,
    and template directories.

    Attributes:
        providers:       Mapping of provider name → ``ProviderConfig``.
        pricing:         List of ``PricingTier`` objects for cost tracking.
        rate_limits:     Mapping of provider name → ``RateLimitConfig``.
        cache_db_path:   Path to the SQLite cache database.
        audit_log_path:  Path to the JSONL audit log.
        template_dir:    Directory containing Jinja2 prompt templates.
        log_level:       Python logging level string.
    """

    providers: dict[str, ProviderConfig] = field(default_factory=lambda: {
        "claude": ProviderConfig(
            env_var="ANTHROPIC_API_KEY",
            default_model="claude-sonnet-4-6",
        ),
        "gpt4": ProviderConfig(
            env_var="OPENAI_API_KEY",
            default_model="gpt-4o",
        ),
        "gemini": ProviderConfig(
            env_var="GOOGLE_API_KEY",
            default_model="gemini-2.5-flash",
        ),
    })
    pricing: list[PricingTier] = field(default_factory=lambda: [
        PricingTier(**p) for p in _DEFAULT_PRICING
    ])
    rate_limits: dict[str, RateLimitConfig] = field(default_factory=lambda: {
        "claude": RateLimitConfig(requests_per_minute=50, tokens_per_minute=80_000),
        "gpt4": RateLimitConfig(requests_per_minute=60, tokens_per_minute=150_000),
        "gemini": RateLimitConfig(requests_per_minute=60, tokens_per_minute=120_000),
    })
    cache_db_path: str = "llm_cache.db"
    audit_log_path: str = "audit_log.jsonl"
    template_dir: str = str(Path(__file__).resolve().parent / "templates")
    log_level: str = "INFO"

    # ── serialisation helpers ──────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> GatewayConfig:
        """Load configuration from a YAML file.

        Missing keys fall back to defaults.  Provider and pricing sections are
        merged on top of the built-in defaults so that a minimal config file
        need only specify overrides.

        Args:
            path: Filesystem path to the YAML config.

        Returns:
            A fully-populated ``GatewayConfig`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        cfg = cls()

        # -- providers
        for name, prov_dict in raw.get("providers", {}).items():
            if name in cfg.providers:
                for k, v in prov_dict.items():
                    if hasattr(cfg.providers[name], k):
                        setattr(cfg.providers[name], k, v)
            else:
                cfg.providers[name] = ProviderConfig(**prov_dict)

        # -- pricing
        if "pricing" in raw:
            cfg.pricing = [PricingTier(**p) for p in raw["pricing"]]

        # -- rate limits
        for name, rl_dict in raw.get("rate_limits", {}).items():
            if name in cfg.rate_limits:
                for k, v in rl_dict.items():
                    if hasattr(cfg.rate_limits[name], k):
                        setattr(cfg.rate_limits[name], k, v)
            else:
                cfg.rate_limits[name] = RateLimitConfig(**rl_dict)

        # -- scalars
        for scalar in ("cache_db_path", "audit_log_path", "template_dir", "log_level"):
            if scalar in raw:
                setattr(cfg, scalar, raw[scalar])

        return cfg

    def get_pricing(self, model_id: str) -> PricingTier:
        """Look up pricing for *model_id*.

        Returns a zero-cost ``PricingTier`` if the model is unknown.

        Args:
            model_id: The model identifier to look up.

        Returns:
            The matching ``PricingTier``, or a zero-cost default.
        """
        for tier in self.pricing:
            if tier.model_id == model_id:
                return tier
        return PricingTier(model_id=model_id)

    def provider_for_model(self, model_id: str) -> str:
        """Infer the provider name from a *model_id* string.

        Uses simple prefix heuristics.

        Args:
            model_id: e.g. ``"claude-3-5-sonnet-20241022"``

        Returns:
            Provider key such as ``"claude"``, ``"gpt4"``, or ``"gemini"``.

        Raises:
            ValueError: If the model id cannot be mapped to any provider.
        """
        mid = model_id.lower()
        if "claude" in mid:
            return "claude"
        if "gpt" in mid:
            return "gpt4"
        if "gemini" in mid:
            return "gemini"
        raise ValueError(f"Cannot infer provider for model_id={model_id!r}")
