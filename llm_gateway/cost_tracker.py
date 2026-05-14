"""
Cost tracker for the LLM Gateway.

Computes and records monetary cost for every LLM call using published
per-token pricing from ``GatewayConfig``.  Maintains an in-memory ledger
of ``CostRecord`` objects and exposes cumulative / per-model summaries.

Usage::

    tracker = CostTracker(config)
    record = tracker.record_cost(response)
    print(tracker.total_cost_usd())
    print(tracker.cost_by_model())
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from contracts import CostRecord, LLMResponse
from llm_gateway.config import GatewayConfig

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cost_records (
    record_id           TEXT PRIMARY KEY,
    response_id         TEXT,
    model_id            TEXT,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    cost_per_prompt_token       REAL,
    cost_per_completion_token   REAL,
    total_cost_usd      REAL,
    created_at          TEXT
);
"""

logger = logging.getLogger(__name__)


def _new_id(prefix: str = "cost") -> str:
    """Generate a short unique id.

    Args:
        prefix: String prepended to the UUID segment.

    Returns:
        E.g. ``"cost-ab12cd34"``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class CostTracker:
    """Tracks per-request and cumulative costs across LLM calls.

    Pricing data comes from the ``GatewayConfig.pricing`` list.  Each call to
    ``record_cost`` creates a ``CostRecord``, appends it to the internal
    ledger, and returns it to the caller.

    Attributes:
        config:  The gateway configuration supplying pricing tiers.
        records: Ordered list of all ``CostRecord`` objects created so far.
    """

    def __init__(self, config: GatewayConfig, db_path: str | Path | None = None) -> None:
        """Initialise the tracker.

        Args:
            config:  Gateway configuration containing pricing tiers.
            db_path: Optional path to SQLite file for persistent cost records.
        """
        self.config = config
        self.records: list[CostRecord] = []
        self.db_path = str(db_path) if db_path else None
        if self.db_path:
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(_SCHEMA)

    def record_cost(self, response: LLMResponse) -> CostRecord:
        """Create a ``CostRecord`` for a single response.

        Looks up per-token pricing from config, computes total cost, and
        appends to the internal ledger.

        Args:
            response: The ``LLMResponse`` whose cost to compute.

        Returns:
            The newly created ``CostRecord``.
        """
        pricing = self.config.get_pricing(response.model_id)

        record = CostRecord(
            record_id=_new_id("cost"),
            response_id=response.response_id,
            model_id=response.model_id,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cost_per_prompt_token=pricing.cost_per_prompt_token,
            cost_per_completion_token=pricing.cost_per_completion_token,
            created_at=datetime.utcnow(),
        )
        record.recalculate()

        self.records.append(record)
        if self.db_path:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cost_records VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        record.record_id, record.response_id, record.model_id,
                        record.prompt_tokens, record.completion_tokens,
                        record.cost_per_prompt_token, record.cost_per_completion_token,
                        record.total_cost_usd,
                        record.created_at.isoformat() if record.created_at else None,
                    ),
                )
        logger.debug(
            "Cost recorded: model=%s tokens=%d+%d cost=$%.6f",
            response.model_id,
            response.prompt_tokens,
            response.completion_tokens,
            record.total_cost_usd,
        )
        return record

    # ── aggregate queries ──────────────────────────────────────────────

    def total_cost_usd(self) -> float:
        """Return the cumulative cost across all recorded calls.

        Returns:
            Total cost in USD.
        """
        return sum(r.total_cost_usd for r in self.records)

    def total_tokens(self) -> dict[str, int]:
        """Return aggregate token counts.

        Returns:
            Dict with ``prompt_tokens``, ``completion_tokens``,
            and ``total_tokens`` keys.
        """
        pt = sum(r.prompt_tokens for r in self.records)
        ct = sum(r.completion_tokens for r in self.records)
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}

    def cost_by_model(self) -> dict[str, float]:
        """Return cumulative cost grouped by model.

        Returns:
            ``{model_id: total_cost_usd}`` mapping.
        """
        by_model: dict[str, float] = defaultdict(float)
        for r in self.records:
            by_model[r.model_id] += r.total_cost_usd
        return dict(by_model)

    def request_count(self) -> int:
        """Return the total number of recorded requests.

        Returns:
            Integer request count.
        """
        return len(self.records)

    def summary(self) -> dict[str, Any]:
        """Return a full cost summary.

        Returns:
            Dict containing ``total_cost_usd``, ``total_requests``,
            ``tokens``, and ``cost_by_model``.
        """
        return {
            "total_cost_usd": self.total_cost_usd(),
            "total_requests": self.request_count(),
            "tokens": self.total_tokens(),
            "cost_by_model": self.cost_by_model(),
        }

    def reset(self) -> None:
        """Clear the internal ledger.

        Useful between benchmark runs.
        """
        self.records.clear()
