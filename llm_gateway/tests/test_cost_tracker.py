"""
Tests for the CostTracker module.
"""

from __future__ import annotations

from contracts import LLMResponse
from llm_gateway.config import GatewayConfig
from llm_gateway.cost_tracker import CostTracker


def _resp(
    model_id: str = "gpt-4o",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> LLMResponse:
    """Build a minimal LLMResponse for cost testing."""
    return LLMResponse(
        response_id="resp-ct-001",
        prompt_id="pmt-001",
        model_id=model_id,
        raw_text="test",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


class TestCostTracker:
    """Tests for CostTracker cost computation and aggregation."""

    def test_record_cost_basic(self, gateway_config: GatewayConfig) -> None:
        """record_cost produces a CostRecord with correct values."""
        tracker = CostTracker(gateway_config)
        record = tracker.record_cost(_resp(model_id="gpt-4o", prompt_tokens=1000, completion_tokens=500))
        # gpt-4o pricing: 2.5e-6 prompt, 10e-6 completion
        expected = 1000 * 2.5e-6 + 500 * 10e-6
        assert abs(record.total_cost_usd - expected) < 1e-9

    def test_total_cost(self, gateway_config: GatewayConfig) -> None:
        """total_cost_usd sums across multiple calls."""
        tracker = CostTracker(gateway_config)
        tracker.record_cost(_resp())
        tracker.record_cost(_resp())
        assert tracker.total_cost_usd() > 0
        assert tracker.request_count() == 2

    def test_cost_by_model(self, gateway_config: GatewayConfig) -> None:
        """cost_by_model groups costs correctly."""
        tracker = CostTracker(gateway_config)
        tracker.record_cost(_resp(model_id="gpt-4o"))
        tracker.record_cost(_resp(model_id="claude-3-5-sonnet-20241022"))
        by_model = tracker.cost_by_model()
        assert "gpt-4o" in by_model
        assert "claude-3-5-sonnet-20241022" in by_model

    def test_total_tokens(self, gateway_config: GatewayConfig) -> None:
        """total_tokens sums prompt and completion tokens."""
        tracker = CostTracker(gateway_config)
        tracker.record_cost(_resp(prompt_tokens=100, completion_tokens=50))
        t = tracker.total_tokens()
        assert t["prompt_tokens"] == 100
        assert t["completion_tokens"] == 50
        assert t["total_tokens"] == 150

    def test_summary(self, gateway_config: GatewayConfig) -> None:
        """summary returns a complete dict."""
        tracker = CostTracker(gateway_config)
        tracker.record_cost(_resp())
        s = tracker.summary()
        assert "total_cost_usd" in s
        assert "total_requests" in s
        assert "tokens" in s
        assert "cost_by_model" in s

    def test_unknown_model_zero_cost(self, gateway_config: GatewayConfig) -> None:
        """Unknown model produces zero-cost record."""
        tracker = CostTracker(gateway_config)
        record = tracker.record_cost(_resp(model_id="unknown-model"))
        assert record.total_cost_usd == 0.0

    def test_reset(self, gateway_config: GatewayConfig) -> None:
        """reset clears the ledger."""
        tracker = CostTracker(gateway_config)
        tracker.record_cost(_resp())
        tracker.reset()
        assert tracker.request_count() == 0
        assert tracker.total_cost_usd() == 0.0
