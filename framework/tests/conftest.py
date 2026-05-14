"""
conftest.py — Shared pytest fixtures for framework tests.

Provides synthetic :class:`StatisticalSummary` data for three LLM models
(claude-3.5-sonnet, gpt-4-turbo, gemini-1.5-pro) across all five
metrics expected by the decision engine:

    pass_rate, maintainability_index, latency_ms, cost_usd,
    consistency_score

Values are chosen so that models have clear trade-offs:

*   **Claude** — highest correctness & quality, mid cost, mid speed
*   **GPT-4**  — mid correctness, mid quality, slowest, most expensive
*   **Gemini** — lowest correctness, lower quality, fastest, cheapest
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

# Make framework importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from contracts import StatisticalSummary
except ImportError:
    from pydantic import BaseModel, Field as PField

    class StatisticalSummary(BaseModel):  # type: ignore[no-redef]
        metric_name: str = ""
        model_id: str = ""
        n: int = 0
        mean: float = 0.0
        std_dev: float = 0.0
        median: float = 0.0
        min_val: float = 0.0
        max_val: float = 0.0
        ci_lower_95: float = 0.0
        ci_upper_95: float = 0.0
        computed_at: datetime = PField(default_factory=datetime.utcnow)


# ── Synthetic data constants ──────────────────────────────────────────

MODEL_IDS = ["claude-3.5-sonnet", "gpt-4-turbo", "gemini-1.5-pro"]

# (metric_name, model_id) → (mean, std_dev, median, min_val, max_val, ci_lo, ci_hi)
_RAW: dict[tuple[str, str], tuple[float, ...]] = {
    # pass_rate — Claude best, Gemini worst
    ("pass_rate", "claude-3.5-sonnet"):      (0.82, 0.05, 0.83, 0.70, 0.92, 0.80, 0.84),
    ("pass_rate", "gpt-4-turbo"):            (0.75, 0.08, 0.76, 0.58, 0.88, 0.72, 0.78),
    ("pass_rate", "gemini-1.5-pro"):         (0.68, 0.10, 0.69, 0.45, 0.85, 0.64, 0.72),
    # maintainability_index — Claude best, Gemini mid
    ("maintainability_index", "claude-3.5-sonnet"): (72.0, 8.0, 73.0, 50.0, 90.0, 69.0, 75.0),
    ("maintainability_index", "gpt-4-turbo"):       (65.0, 10.0, 66.0, 40.0, 85.0, 61.0, 69.0),
    ("maintainability_index", "gemini-1.5-pro"):    (60.0, 12.0, 61.0, 30.0, 80.0, 55.0, 65.0),
    # latency_ms — Gemini fastest, GPT-4 slowest
    ("latency_ms", "claude-3.5-sonnet"):     (3500.0, 800.0, 3400.0, 1500.0, 6000.0, 3200.0, 3800.0),
    ("latency_ms", "gpt-4-turbo"):           (5200.0, 1200.0, 5100.0, 2500.0, 9000.0, 4700.0, 5700.0),
    ("latency_ms", "gemini-1.5-pro"):        (2200.0, 600.0, 2100.0, 1000.0, 4000.0, 1900.0, 2500.0),
    # cost_usd — Gemini cheapest, GPT-4 most expensive
    ("cost_usd", "claude-3.5-sonnet"):       (0.035, 0.010, 0.033, 0.015, 0.060, 0.031, 0.039),
    ("cost_usd", "gpt-4-turbo"):             (0.065, 0.020, 0.062, 0.030, 0.120, 0.057, 0.073),
    ("cost_usd", "gemini-1.5-pro"):          (0.012, 0.004, 0.011, 0.005, 0.025, 0.010, 0.014),
    # consistency_score — Claude most consistent
    ("consistency_score", "claude-3.5-sonnet"): (0.90, 0.04, 0.91, 0.80, 0.96, 0.88, 0.92),
    ("consistency_score", "gpt-4-turbo"):       (0.78, 0.08, 0.79, 0.55, 0.92, 0.74, 0.82),
    ("consistency_score", "gemini-1.5-pro"):    (0.72, 0.10, 0.73, 0.50, 0.88, 0.68, 0.76),
}


def build_synthetic_summaries() -> list[StatisticalSummary]:
    """Build a list of 15 synthetic StatisticalSummary objects."""
    summaries: list[StatisticalSummary] = []
    for (metric, model), vals in _RAW.items():
        mean, std, med, mn, mx, ci_lo, ci_hi = vals
        summaries.append(
            StatisticalSummary(
                metric_name=metric,
                model_id=model,
                n=30,
                mean=mean,
                std_dev=std,
                median=med,
                min_val=mn,
                max_val=mx,
                ci_lower_95=ci_lo,
                ci_upper_95=ci_hi,
            )
        )
    return summaries


def summaries_to_json_dicts(
    summaries: list[StatisticalSummary],
) -> list[dict[str, Any]]:
    """Serialise summaries to JSON-friendly dicts."""
    out = []
    for s in summaries:
        d = s.model_dump() if hasattr(s, "model_dump") else s.dict()
        # Ensure datetime is string
        if "computed_at" in d and isinstance(d["computed_at"], datetime):
            d["computed_at"] = d["computed_at"].isoformat()
        out.append(d)
    return out


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_summaries() -> list[StatisticalSummary]:
    """Return 15 synthetic summaries for 3 models × 5 metrics."""
    return build_synthetic_summaries()


@pytest.fixture
def summary_dicts(synthetic_summaries) -> list[dict]:
    """Summaries as plain dicts (for JSON round-trip tests)."""
    return summaries_to_json_dicts(synthetic_summaries)


@pytest.fixture
def summary_json_path(
    synthetic_summaries, tmp_path: Path
) -> Path:
    """Write summaries to a temp JSON file and return the path."""
    data = summaries_to_json_dicts(synthetic_summaries)
    p = tmp_path / "statistical_summary.json"
    p.write_text(json.dumps(data, indent=2, default=str))
    return p


@pytest.fixture
def profiles_dir() -> Path:
    """Return the path to the built-in profiles directory."""
    return Path(__file__).resolve().parent.parent / "profiles"
