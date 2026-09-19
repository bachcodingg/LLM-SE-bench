"""
Tests for llm_gateway.budget (M7).

The one that matters: cache tiers must be priced at their own rates.
Charging a cache read at the full input rate is the error that makes an
agent run look an order of magnitude more expensive than it is, and it is
invisible because the total still looks plausible.
"""

from __future__ import annotations

import pytest

from llm_gateway.budget import (
    CACHE_PRICING,
    BudgetSimulator,
    Configuration,
    normalise_by_cost,
    pareto_frontier,
    power_two_proportions,
    price_call,
    required_n_two_proportions,
)

INPUT_RATE = 3e-6
OUTPUT_RATE = 15e-6


class TestPricing:
    def test_basic_call(self):
        breakdown = price_call(
            input_tokens=1000, output_tokens=500,
            cost_per_input_token=INPUT_RATE, cost_per_output_token=OUTPUT_RATE,
        )
        assert breakdown["input_usd"] == pytest.approx(0.003)
        assert breakdown["output_usd"] == pytest.approx(0.0075)
        assert breakdown["total_usd"] == pytest.approx(0.0105)

    def test_cache_reads_are_discounted(self):
        breakdown = price_call(
            input_tokens=0, output_tokens=0, cache_read_tokens=1000,
            cost_per_input_token=INPUT_RATE, provider="claude",
        )
        expected = 1000 * INPUT_RATE * CACHE_PRICING["claude"].read
        assert breakdown["cache_read_usd"] == pytest.approx(expected)

    def test_cache_writes_cost_a_premium(self):
        """Caching a prefix used twice loses money; that is why the premium exists."""
        breakdown = price_call(
            input_tokens=0, output_tokens=0, cache_write_tokens=1000,
            cost_per_input_token=INPUT_RATE, provider="claude",
        )
        assert breakdown["cache_write_usd"] > 1000 * INPUT_RATE

    def test_the_saving_from_caching_is_reported(self):
        breakdown = price_call(
            input_tokens=100, output_tokens=50, cache_read_tokens=10_000,
            cost_per_input_token=INPUT_RATE, cost_per_output_token=OUTPUT_RATE,
            provider="claude",
        )
        assert breakdown["cache_saving_usd"] > 0

    def test_an_unknown_provider_falls_back_conservatively(self):
        """Overstating cost is the safe direction for a ceiling."""
        breakdown = price_call(
            input_tokens=0, output_tokens=0, cache_read_tokens=1000,
            cost_per_input_token=INPUT_RATE, provider="unknown-provider",
        )
        assert breakdown["cache_read_usd"] == pytest.approx(1000 * INPUT_RATE)
        assert breakdown["pricing_source"] == "conservative-default"

    def test_batch_applies_a_discount(self):
        normal = price_call(
            input_tokens=1000, output_tokens=1000,
            cost_per_input_token=INPUT_RATE, cost_per_output_token=OUTPUT_RATE,
            provider="claude",
        )
        batched = price_call(
            input_tokens=1000, output_tokens=1000,
            cost_per_input_token=INPUT_RATE, cost_per_output_token=OUTPUT_RATE,
            provider="claude", batch=True,
        )
        assert batched["total_usd"] < normal["total_usd"]
        assert batched["batch_discount_applied"] is True

    def test_mispricing_cache_reads_inflates_the_total(self):
        """The bug this module exists to fix, demonstrated."""
        correct = price_call(
            input_tokens=200, output_tokens=100, cache_read_tokens=50_000,
            cost_per_input_token=INPUT_RATE, cost_per_output_token=OUTPUT_RATE,
            provider="claude",
        )["total_usd"]
        naive = (200 + 50_000) * INPUT_RATE + 100 * OUTPUT_RATE
        assert naive > correct * 2


class TestCostNormalisation:
    def test_sorts_by_euros_per_solve_not_resolve_rate(self):
        rows = normalise_by_cost([
            {"label": "expensive", "solved": 50, "attempted": 100, "cost_eur": 10.0},
            {"label": "cheap", "solved": 45, "attempted": 100, "cost_eur": 1.0},
        ])
        assert rows[0].label == "cheap"
        assert rows[0].resolve_rate < rows[1].resolve_rate

    def test_eur_per_solve(self):
        row = normalise_by_cost([
            {"label": "a", "solved": 10, "attempted": 20, "cost_eur": 5.0}
        ])[0]
        assert row.eur_per_solve == pytest.approx(0.5)

    def test_unmeasured_cost_is_none_not_zero(self):
        """Zero would sort an unmeasured run ahead of every measured one."""
        row = normalise_by_cost([
            {"label": "a", "solved": 5, "attempted": 10, "cost_eur": 0.0}
        ])[0]
        assert row.solves_per_eur is None

    def test_unmeasured_sorts_last(self):
        rows = normalise_by_cost([
            {"label": "unmeasured", "solved": 10, "attempted": 10, "cost_eur": 0.0},
            {"label": "measured", "solved": 5, "attempted": 10, "cost_eur": 1.0},
        ])
        assert rows[-1].label == "unmeasured"

    def test_no_solves_gives_no_cost_per_solve(self):
        row = normalise_by_cost([
            {"label": "a", "solved": 0, "attempted": 10, "cost_eur": 5.0}
        ])[0]
        assert row.eur_per_solve is None

    def test_per_token_and_per_hour(self):
        row = normalise_by_cost([{
            "label": "a", "solved": 10, "attempted": 10,
            "cost_eur": 1.0, "tokens": 2_000_000, "wall_clock_seconds": 7200,
        }])[0]
        assert row.solves_per_million_tokens == pytest.approx(5.0)
        assert row.solves_per_hour == pytest.approx(5.0)


class TestBudgetSimulator:
    def test_affordability(self):
        simulator = BudgetSimulator(budget_eur=10.0)
        rows = simulator.affordable([
            Configuration("cheap", cost_per_episode_eur=0.01, tasks=100),
            Configuration("dear", cost_per_episode_eur=1.0, tasks=100),
        ])
        by_label = {row["label"]: row for row in rows}
        assert by_label["cheap"]["affordable"] is True
        assert by_label["dear"]["affordable"] is False

    def test_reports_power_alongside_affordability(self):
        """Affording a run is not the same as affording one that can see anything."""
        rows = BudgetSimulator(budget_eur=100.0).affordable([
            Configuration("tiny", cost_per_episode_eur=0.01, tasks=10),
            Configuration("large", cost_per_episode_eur=0.01, tasks=2000),
        ])
        by_label = {row["label"]: row for row in rows}
        assert by_label["large"]["power_for_5pp"] > by_label["tiny"]["power_for_5pp"]

    def test_allocation_trade_off(self):
        allocation = BudgetSimulator(budget_eur=10.0).allocate(
            cost_per_episode_eur=0.05, tasks=100
        )
        assert allocation["affordable_episodes"] == 200
        by_repeats = {option["repeats"]: option for option in allocation["options"]}
        assert by_repeats[1]["tasks_covered"] >= by_repeats[3]["tasks_covered"]
        assert by_repeats[3]["detects_flakiness"] is True
        assert by_repeats[1]["detects_flakiness"] is False

    def test_rejects_a_non_positive_budget(self):
        with pytest.raises(ValueError):
            BudgetSimulator(budget_eur=0.0)


class TestPower:
    def test_power_rises_with_sample_size(self):
        small = power_two_proportions(n_per_group=20, p1=0.5, effect=0.05)
        large = power_two_proportions(n_per_group=2000, p1=0.5, effect=0.05)
        assert large > small

    def test_power_rises_with_effect_size(self):
        small = power_two_proportions(n_per_group=200, p1=0.5, effect=0.02)
        large = power_two_proportions(n_per_group=200, p1=0.5, effect=0.20)
        assert large > small

    def test_required_n_is_large_for_small_effects(self):
        """The number nobody budgets for, which is why it is worth computing."""
        assert required_n_two_proportions(p1=0.5, effect=0.05) > 1000

    def test_power_is_bounded(self):
        assert 0.0 <= power_two_proportions(n_per_group=2, p1=0.5, effect=0.5) <= 1.0

    def test_tiny_sample_has_no_power(self):
        assert power_two_proportions(n_per_group=1, p1=0.5, effect=0.05) == 0.0


class TestParetoFrontier:
    def test_dominated_points_are_identified(self):
        result = pareto_frontier([
            {"label": "best", "quality": 0.9, "cost": 1.0, "latency": 1.0},
            {"label": "dominated", "quality": 0.5, "cost": 2.0, "latency": 2.0},
        ])
        assert [row["label"] for row in result["frontier"]] == ["best"]
        assert result["dominated"][0]["label"] == "dominated"
        assert result["dominated"][0]["dominated_by"] == ["best"]

    def test_a_real_trade_off_stays_on_the_frontier(self):
        result = pareto_frontier([
            {"label": "accurate", "quality": 0.9, "cost": 10.0, "latency": 5.0},
            {"label": "cheap", "quality": 0.7, "cost": 1.0, "latency": 1.0},
        ])
        assert len(result["frontier"]) == 2

    def test_identical_points_do_not_dominate_each_other(self):
        result = pareto_frontier([
            {"label": "a", "quality": 0.8, "cost": 1.0, "latency": 1.0},
            {"label": "b", "quality": 0.8, "cost": 1.0, "latency": 1.0},
        ])
        assert len(result["frontier"]) == 2

    def test_frontier_is_sorted_by_the_primary_axis(self):
        result = pareto_frontier([
            {"label": "low", "quality": 0.6, "cost": 1.0, "latency": 1.0},
            {"label": "high", "quality": 0.9, "cost": 5.0, "latency": 5.0},
        ])
        assert result["frontier"][0]["label"] == "high"

    def test_empty_input(self):
        assert pareto_frontier([])["frontier"] == []
