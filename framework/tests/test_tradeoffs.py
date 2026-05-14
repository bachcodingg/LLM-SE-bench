"""
test_tradeoffs.py — Tests for Pareto frontier and trade-off analysis.

Covers:
- Dominance relation
- Pareto frontier identification
- Pairwise trade-offs and MRS computation
- 2-D frontier
- TradeoffAnalyzer high-level API
"""

from __future__ import annotations

import pytest

from framework.decision_matrix import CRITERIA
from framework.tradeoffs import (
    ParetoPoint,
    ParetoResult,
    TradeoffAnalyzer,
    TradeoffPair,
    _dominates,
    compute_2d_frontier,
    compute_pareto_frontier,
    compute_pairwise_tradeoffs,
)


# =====================================================================
# Dominance
# =====================================================================


class TestDominance:
    def test_strictly_better(self):
        a = {"correctness": 0.9, "quality": 0.8, "speed": 0.7,
             "cost": 0.6, "consistency": 0.5}
        b = {"correctness": 0.8, "quality": 0.7, "speed": 0.6,
             "cost": 0.5, "consistency": 0.4}
        assert _dominates(a, b)
        assert not _dominates(b, a)

    def test_equal(self):
        a = {c: 0.5 for c in CRITERIA}
        b = {c: 0.5 for c in CRITERIA}
        assert not _dominates(a, b)
        assert not _dominates(b, a)

    def test_partial_better(self):
        a = {"correctness": 0.9, "quality": 0.5, "speed": 0.7,
             "cost": 0.6, "consistency": 0.5}
        b = {"correctness": 0.8, "quality": 0.6, "speed": 0.6,
             "cost": 0.5, "consistency": 0.5}
        # a is better on correctness+speed+cost, worse on quality
        assert not _dominates(a, b)
        assert not _dominates(b, a)

    def test_at_least_as_good_plus_one_strictly(self):
        a = {"correctness": 0.9, "quality": 0.5, "speed": 0.5,
             "cost": 0.5, "consistency": 0.5}
        b = {"correctness": 0.8, "quality": 0.5, "speed": 0.5,
             "cost": 0.5, "consistency": 0.5}
        assert _dominates(a, b)


# =====================================================================
# Pareto frontier
# =====================================================================


class TestParetoFrontier:
    def test_simple_frontier(self):
        # C is NOT dominated: A has quality=0 < C's 0.3, B has correctness=0 < C's 0.3
        # All three are Pareto-optimal in 5-D
        normalised = {
            "A": {"correctness": 1.0, "quality": 0.0, "speed": 0.5,
                  "cost": 0.5, "consistency": 0.5},
            "B": {"correctness": 0.0, "quality": 1.0, "speed": 0.5,
                  "cost": 0.5, "consistency": 0.5},
            "C": {"correctness": 0.3, "quality": 0.3, "speed": 0.3,
                  "cost": 0.3, "consistency": 0.3},
        }
        points = compute_pareto_frontier(normalised)
        frontier = [p for p in points if p.is_pareto]
        # All three on frontier (none dominates another)
        assert len(frontier) == 3

    def test_dominated_point(self):
        # D is strictly dominated by A (A ≥ D on every criterion, > on some)
        normalised = {
            "A": {"correctness": 0.9, "quality": 0.8, "speed": 0.7,
                  "cost": 0.6, "consistency": 0.5},
            "D": {"correctness": 0.5, "quality": 0.4, "speed": 0.3,
                  "cost": 0.2, "consistency": 0.1},
        }
        points = compute_pareto_frontier(normalised)
        frontier = [p for p in points if p.is_pareto]
        dominated = [p for p in points if not p.is_pareto]
        assert len(frontier) == 1
        assert frontier[0].model_id == "A"
        assert len(dominated) == 1
        assert dominated[0].model_id == "D"

    def test_all_on_frontier(self):
        normalised = {
            "A": {"correctness": 1.0, "quality": 0.0, "speed": 0.5,
                  "cost": 0.5, "consistency": 0.5},
            "B": {"correctness": 0.0, "quality": 1.0, "speed": 0.5,
                  "cost": 0.5, "consistency": 0.5},
        }
        points = compute_pareto_frontier(normalised)
        assert all(p.is_pareto for p in points)

    def test_single_model(self):
        normalised = {"only": {c: 0.5 for c in CRITERIA}}
        points = compute_pareto_frontier(normalised)
        assert len(points) == 1
        assert points[0].is_pareto

    def test_with_synthetic_data(self, synthetic_summaries):
        analyzer = TradeoffAnalyzer(synthetic_summaries)
        result = analyzer.analyze()
        # At least one model on frontier
        assert len(result.frontier) >= 1
        # All models accounted for
        assert len(result.points) == 3


# =====================================================================
# Pairwise trade-offs
# =====================================================================


class TestPairwiseTradeoffs:
    def test_basic(self):
        normalised = {
            "A": {"correctness": 0.9, "cost": 0.3},
            "B": {"correctness": 0.6, "cost": 0.8},
        }
        pairs = compute_pairwise_tradeoffs(
            normalised, [("correctness", "cost")]
        )
        assert len(pairs) == 1
        p = pairs[0]
        assert p.model_a == "A"
        assert p.model_b == "B"
        assert p.delta_x > 0  # A better on correctness
        assert p.delta_y < 0  # A worse on cost

    def test_mrs_computed(self):
        normalised = {
            "A": {"correctness": 0.8, "cost": 0.4},
            "B": {"correctness": 0.5, "cost": 0.7},
        }
        pairs = compute_pairwise_tradeoffs(
            normalised, [("correctness", "cost")]
        )
        assert pairs[0].marginal_rate is not None

    def test_zero_delta_x(self):
        normalised = {
            "A": {"correctness": 0.5, "cost": 0.4},
            "B": {"correctness": 0.5, "cost": 0.7},
        }
        pairs = compute_pairwise_tradeoffs(
            normalised, [("correctness", "cost")]
        )
        assert pairs[0].marginal_rate is None

    def test_default_pairs_all_combinations(self):
        normalised = {
            "A": {c: 0.5 for c in CRITERIA},
            "B": {c: 0.6 for c in CRITERIA},
        }
        pairs = compute_pairwise_tradeoffs(normalised)
        # C(5,2) = 10 pairs × 1 model pair = 10
        assert len(pairs) == 10

    def test_tradeoff_summary(self):
        t = TradeoffPair(
            model_a="A", model_b="B",
            criterion_x="correctness", criterion_y="cost",
            delta_x=0.3, delta_y=-0.4, marginal_rate=1.333,
        )
        assert "Switching" in t.summary
        assert "MRS" in t.summary


# =====================================================================
# 2-D frontier
# =====================================================================


class TestFrontier2D:
    def test_2d(self):
        normalised = {
            "A": {"correctness": 0.9, "cost": 0.3},
            "B": {"correctness": 0.5, "cost": 0.8},
            "C": {"correctness": 0.4, "cost": 0.4},
        }
        points = compute_2d_frontier(normalised, "correctness", "cost")
        frontier = [p for p in points if p.is_pareto]
        # A dominates C in 2-D? A: (0.9, 0.3) vs C: (0.4, 0.4)
        # A > C on correctness, but A < C on cost. So neither dominates.
        # B: (0.5, 0.8): B > C on both → C dominated by B
        assert {p.model_id for p in frontier} == {"A", "B"}

    def test_2d_from_analyzer(self, synthetic_summaries):
        analyzer = TradeoffAnalyzer(synthetic_summaries)
        points = analyzer.frontier_2d("correctness", "cost")
        assert len(points) == 3


# =====================================================================
# TradeoffAnalyzer
# =====================================================================


class TestTradeoffAnalyzer:
    def test_analyze(self, synthetic_summaries):
        analyzer = TradeoffAnalyzer(synthetic_summaries)
        result = analyzer.analyze()
        assert isinstance(result, ParetoResult)
        assert len(result.points) == 3
        assert len(result.frontier) >= 1
        assert len(result.tradeoffs) > 0

    def test_tradeoff_table(self, synthetic_summaries):
        analyzer = TradeoffAnalyzer(synthetic_summaries)
        table = analyzer.tradeoff_table()
        assert len(table) > 0
        assert "model_a" in table[0]
        assert "delta_x" in table[0]
        assert "summary" in table[0]

    def test_to_dict(self, synthetic_summaries):
        analyzer = TradeoffAnalyzer(synthetic_summaries)
        d = analyzer.to_dict()
        assert "frontier" in d
        assert "dominated" in d
        assert "tradeoffs" in d

    def test_normalised_scores_property(self, synthetic_summaries):
        analyzer = TradeoffAnalyzer(synthetic_summaries)
        analyzer.analyze()
        ns = analyzer.normalised_scores
        assert len(ns) == 3

    def test_specific_criteria_pairs(self, synthetic_summaries):
        analyzer = TradeoffAnalyzer(synthetic_summaries)
        result = analyzer.analyze(
            criteria_pairs=[("correctness", "cost")]
        )
        # 3 model pairs × 1 criterion pair = 3 tradeoffs
        assert len(result.tradeoffs) == 3
