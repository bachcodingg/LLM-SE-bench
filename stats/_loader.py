"""
stats/_loader.py — Data-loading utilities for the Statistical Engine.

Reads:
    - results/{dataset}/{model}/results.jsonl  → EvaluationResult records
    - quality/{dataset}/{model}/metrics.csv    → QualityMetrics records
    - llm_cache.db                             → CostRecord data via SQL
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── JSONL loader (C2 evaluation results) ──────────────────────────────

def load_evaluation_results(base_dir: str | Path) -> pd.DataFrame:
    """Load all ``results.jsonl`` files under *base_dir* into a single DataFrame.

    Expected directory layout::

        base_dir/
          {dataset}/
            {model}/
              results.jsonl   ← one JSON object per line

    Each JSON line must be an ``EvaluationResult`` dict (see contracts.py).

    Returns
    -------
    pd.DataFrame
        Columns include *dataset*, *model_id*, *problem_id*, *verdict*,
        *weighted_score*, *tests_passed*, *tests_total*, *compile_success*,
        *evaluated_at*, plus any extra keys present in the JSONL.
    """
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Results directory not found: {base}")

    rows: list[dict[str, Any]] = []
    for jsonl_path in sorted(base.rglob("results.jsonl")):
        parts = jsonl_path.relative_to(base).parts
        if len(parts) < 3:
            logger.warning("Skipping %s — unexpected path depth", jsonl_path)
            continue
        dataset, model = parts[0], parts[1]
        with open(jsonl_path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "%s:%d — invalid JSON, skipping", jsonl_path, lineno
                    )
                    continue
                record["dataset"] = dataset
                record["model_id"] = record.get("model_id", model)
                rows.append(record)

    if not rows:
        logger.warning("No evaluation results found under %s", base)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Derive convenience columns
    if "tests_total" in df.columns and "tests_passed" in df.columns:
        df["pass_rate"] = np.where(
            df["tests_total"] > 0,
            df["tests_passed"] / df["tests_total"],
            0.0,
        )
    if "verdict" in df.columns:
        df["is_pass"] = df["verdict"].str.lower() == "pass"
    logger.info(
        "Loaded %d evaluation results from %s (%d datasets, %d models)",
        len(df),
        base,
        df["dataset"].nunique(),
        df["model_id"].nunique(),
    )
    return df


# ── CSV loader (C3 quality metrics) ──────────────────────────────────

def load_quality_metrics(base_dir: str | Path) -> pd.DataFrame:
    """Load all ``metrics.csv`` files under *base_dir* into a single DataFrame.

    Expected layout::

        base_dir/
          {dataset}/
            {model}/
              metrics.csv

    Returns
    -------
    pd.DataFrame
        Columns include *dataset*, *model_id*, plus all CSV columns
        (e.g. *cyclomatic_complexity*, *maintainability_index*, …).
    """
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Quality directory not found: {base}")

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(base.rglob("metrics.csv")):
        parts = csv_path.relative_to(base).parts
        if len(parts) < 3:
            logger.warning("Skipping %s — unexpected path depth", csv_path)
            continue
        dataset, model = parts[0], parts[1]
        try:
            chunk = pd.read_csv(csv_path)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", csv_path, exc)
            continue
        chunk["dataset"] = dataset
        chunk["model_id"] = chunk.get("model_id", model)
        frames.append(chunk)

    if not frames:
        logger.warning("No quality metrics found under %s", base)
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded %d quality-metric rows from %s", len(df), base
    )
    return df


# ── SQLite loader (C1 cost data) ─────────────────────────────────────

_COST_QUERY = """
SELECT
    record_id,
    response_id,
    model_id,
    prompt_tokens,
    completion_tokens,
    cost_per_prompt_token,
    cost_per_completion_token,
    total_cost_usd,
    created_at
FROM cost_records
"""


def load_cost_data(db_path: str | Path) -> pd.DataFrame:
    """Read the ``cost_records`` table from the C1 SQLite cache.

    Parameters
    ----------
    db_path : str | Path
        Path to ``llm_cache.db``.

    Returns
    -------
    pd.DataFrame
        One row per API call with token counts, per-token prices, and total
        cost in USD.
    """
    db = Path(db_path)
    if not db.exists():
        raise FileNotFoundError(f"Cost database not found: {db}")

    conn = sqlite3.connect(str(db))
    try:
        df = pd.read_sql_query(_COST_QUERY, conn)
    finally:
        conn.close()

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    logger.info("Loaded %d cost records from %s", len(df), db)
    return df


# ── Merge helper ─────────────────────────────────────────────────────

def merge_results_quality(
    results_df: pd.DataFrame,
    quality_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join evaluation results with quality metrics on *response_id*.

    Falls back to joining on *(problem_id, model_id)* if *response_id* is not
    present in both frames.
    """
    if results_df.empty or quality_df.empty:
        return results_df

    if "response_id" in results_df.columns and "response_id" in quality_df.columns:
        merged = results_df.merge(
            quality_df,
            on="response_id",
            how="left",
            suffixes=("", "_qual"),
        )
    elif {"problem_id", "model_id"}.issubset(results_df.columns) and \
         {"problem_id", "model_id"}.issubset(quality_df.columns):
        merged = results_df.merge(
            quality_df,
            on=["problem_id", "model_id"],
            how="left",
            suffixes=("", "_qual"),
        )
    else:
        logger.warning("Cannot merge: no common join keys found")
        return results_df

    logger.info(
        "Merged results (%d rows) + quality (%d rows) → %d rows",
        len(results_df), len(quality_df), len(merged),
    )
    return merged
