"""
stats/tests/conftest.py — Synthetic data fixtures for all C4 tests.

Generates realistic benchmark data:
    * 3 models: claude-3.5-sonnet, gpt-4-turbo, gemini-1.5-pro
    * 4 datasets: humaneval, mbpp, defects4j, godclass
    * 3 runs per (model, dataset, problem)
    * ~50 problems per dataset → 50 × 3 × 3 = 450 rows per dataset
"""

from __future__ import annotations

import json
import sqlite3
import uuid

import numpy as np
import pandas as pd
import pytest

# ── constants ────────────────────────────────────────────────────────

MODELS = ["claude-3.5-sonnet", "gpt-4-turbo", "gemini-1.5-pro"]
DATASETS = ["humaneval", "mbpp", "defects4j", "godclass"]
N_PROBLEMS = {"humaneval": 50, "mbpp": 50, "defects4j": 30, "godclass": 10}
N_RUNS = 3
DIFFICULTIES = ["easy", "medium", "hard"]

# Model-specific performance characteristics (mean pass rate, quality)
MODEL_PROFILES = {
    "claude-3.5-sonnet": {
        "pass_rate": 0.72,
        "quality_mean": 68,
        "cost_per_prompt_token": 3e-6,
        "cost_per_completion_token": 15e-6,
        "latency_mean": 3500,
    },
    "gpt-4-turbo": {
        "pass_rate": 0.68,
        "quality_mean": 65,
        "cost_per_prompt_token": 10e-6,
        "cost_per_completion_token": 30e-6,
        "latency_mean": 4200,
    },
    "gemini-1.5-pro": {
        "pass_rate": 0.60,
        "quality_mean": 62,
        "cost_per_prompt_token": 3.5e-6,
        "cost_per_completion_token": 10.5e-6,
        "latency_mean": 2800,
    },
}

DIFFICULTY_MODIFIERS = {"easy": 0.15, "medium": 0.0, "hard": -0.20}


# ── synthetic data generation ────────────────────────────────────────

def _generate_evaluation_results(rng: np.random.Generator) -> pd.DataFrame:
    """Generate synthetic EvaluationResult records."""
    rows = []
    for dataset in DATASETS:
        n_probs = N_PROBLEMS[dataset]
        for prob_idx in range(n_probs):
            problem_id = f"{dataset}-{prob_idx:04d}"
            difficulty = rng.choice(DIFFICULTIES, p=[0.3, 0.5, 0.2])
            for model in MODELS:
                profile = MODEL_PROFILES[model]
                base_pr = profile["pass_rate"] + DIFFICULTY_MODIFIERS[difficulty]
                for run in range(N_RUNS):
                    eval_id = str(uuid.uuid4())[:12]
                    response_id = f"resp-{eval_id}"

                    # Score and pass/fail
                    score = float(np.clip(
                        rng.normal(base_pr, 0.15), 0, 1
                    ))
                    is_pass = rng.random() < base_pr
                    tests_total = rng.integers(5, 20)
                    tests_passed = int(tests_total * score)
                    compile_success = rng.random() < 0.92

                    if not compile_success:
                        is_pass = False
                        tests_passed = 0
                        score = 0.0
                        verdict = "error"
                    elif is_pass:
                        verdict = "pass"
                    elif score > 0.3:
                        verdict = "partial"
                    else:
                        verdict = "fail"

                    latency = max(500, rng.normal(profile["latency_mean"], 800))
                    prompt_tokens = int(rng.normal(1200, 300))
                    completion_tokens = int(rng.normal(800, 200))

                    rows.append({
                        "evaluation_id": eval_id,
                        "response_id": response_id,
                        "problem_id": problem_id,
                        "model_id": model,
                        "dataset": dataset,
                        "difficulty": difficulty,
                        "suite_id": f"suite-{problem_id}",
                        "verdict": verdict,
                        "tests_total": int(tests_total),
                        "tests_passed": int(tests_passed),
                        "weighted_score": round(score, 4),
                        "compile_success": compile_success,
                        "is_pass": is_pass,
                        "pass_rate": round(score, 4),
                        "latency_ms": round(latency, 1),
                        "prompt_tokens": max(100, prompt_tokens),
                        "completion_tokens": max(50, completion_tokens),
                    })
    return pd.DataFrame(rows)


def _generate_quality_metrics(
    results_df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Generate synthetic QualityMetrics records."""
    rows = []
    for _, r in results_df.iterrows():
        model = r["model_id"]
        profile = MODEL_PROFILES[model]
        q_mean = profile["quality_mean"]

        rows.append({
            "metrics_id": f"qm-{r['evaluation_id']}",
            "response_id": r["response_id"],
            "problem_id": r["problem_id"],
            "model_id": model,
            "dataset": r["dataset"],
            "lines_of_code": max(5, int(rng.normal(45, 20))),
            "cyclomatic_complexity": max(1, round(rng.normal(8, 4), 1)),
            "maintainability_index": round(np.clip(rng.normal(q_mean, 12), 10, 100), 1),
            "halstead_volume": max(50, round(rng.normal(500, 200), 1)),
            "lint_warnings": max(0, int(rng.normal(3, 2))),
            "lint_errors": max(0, int(rng.poisson(0.5))),
            "type_coverage_pct": round(np.clip(rng.normal(60, 20), 0, 100), 1),
            "docstring_coverage_pct": round(np.clip(rng.normal(40, 25), 0, 100), 1),
        })
    return pd.DataFrame(rows)


def _generate_cost_records(
    results_df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Generate synthetic CostRecord entries."""
    rows = []
    for _, r in results_df.iterrows():
        model = r["model_id"]
        profile = MODEL_PROFILES[model]
        pt = int(r.get("prompt_tokens", 1000))
        ct = int(r.get("completion_tokens", 500))
        total = pt * profile["cost_per_prompt_token"] + ct * profile["cost_per_completion_token"]

        rows.append({
            "record_id": f"cost-{r['evaluation_id']}",
            "response_id": r["response_id"],
            "model_id": model,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "cost_per_prompt_token": profile["cost_per_prompt_token"],
            "cost_per_completion_token": profile["cost_per_completion_token"],
            "total_cost_usd": round(total, 6),
            "created_at": "2026-04-01T12:00:00",
        })
    return pd.DataFrame(rows)


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def rng():
    """Seeded RNG for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def results_df(rng):
    """Session-scoped synthetic evaluation results."""
    return _generate_evaluation_results(rng)


@pytest.fixture(scope="session")
def quality_df(results_df, rng):
    """Session-scoped synthetic quality metrics."""
    return _generate_quality_metrics(results_df, rng)


@pytest.fixture(scope="session")
def cost_df(results_df, rng):
    """Session-scoped synthetic cost records."""
    return _generate_cost_records(results_df, rng)


@pytest.fixture(scope="session")
def merged_df(results_df, quality_df):
    """Results merged with quality metrics."""
    from stats._loader import merge_results_quality
    return merge_results_quality(results_df, quality_df)


@pytest.fixture()
def results_dir(results_df, tmp_path):
    """Write synthetic results to JSONL files on disk."""
    for (dataset, model), grp in results_df.groupby(["dataset", "model_id"]):
        d = tmp_path / "results" / dataset / model
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "results.jsonl", "w") as f:
            for _, row in grp.iterrows():
                f.write(json.dumps(row.to_dict(), default=str) + "\n")
    return tmp_path / "results"


@pytest.fixture()
def quality_dir(quality_df, tmp_path):
    """Write synthetic quality metrics to CSV files on disk."""
    for (dataset, model), grp in quality_df.groupby(["dataset", "model_id"]):
        d = tmp_path / "quality" / dataset / model
        d.mkdir(parents=True, exist_ok=True)
        grp.to_csv(d / "metrics.csv", index=False)
    return tmp_path / "quality"


@pytest.fixture()
def cost_db(cost_df, tmp_path):
    """Write synthetic cost data to a SQLite database."""
    db_path = tmp_path / "llm_cache.db"
    conn = sqlite3.connect(str(db_path))
    cost_df.to_sql("cost_records", conn, if_exists="replace", index=False)
    conn.close()
    return db_path
