"""
llm_gateway — Unified LLM API abstraction for llm-se-bench.

Provides a single interface across Anthropic Claude, OpenAI GPT-4, and Google
Gemini.  Every other component in llm-se-bench interacts with LLMs exclusively
through this gateway.

Public surface:
    GatewayConfig     – YAML-driven configuration
    LLMClientFactory  – build a client by model-id string
    ClaudeClient      – Anthropic Claude provider
    GPT4Client        – OpenAI GPT-4 provider
    GeminiClient      – Google Gemini provider
    ResponseCache     – SQLite content-addressed response cache
    CostTracker       – per-request / cumulative cost accounting
    RateLimiter       – token-bucket rate limiter with retry back-off
    AuditLogger       – append-only JSONL audit trail
    PromptRenderer    – Jinja2-based prompt template engine
"""

__version__ = "0.1.0"

from llm_gateway.audit import AuditLogger
from llm_gateway.cache import ResponseCache
from llm_gateway.clients.base import LLMClient, LLMClientFactory
from llm_gateway.clients.claude import ClaudeClient
from llm_gateway.clients.gemini import GeminiClient
from llm_gateway.clients.gpt4 import GPT4Client
from llm_gateway.config import GatewayConfig
from llm_gateway.cost_tracker import CostTracker
from llm_gateway.models import PromptRenderer
from llm_gateway.rate_limiter import RateLimiter

__all__ = [
    "GatewayConfig",
    "LLMClient",
    "LLMClientFactory",
    "ClaudeClient",
    "GPT4Client",
    "GeminiClient",
    "ResponseCache",
    "CostTracker",
    "RateLimiter",
    "AuditLogger",
    "PromptRenderer",
]
