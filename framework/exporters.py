"""
exporters.py — JSON and CSV export for decision matrices and recommendations.

Writes structured output to ``reports/decision_matrix.{json,csv}`` (or
custom paths).  The JSON exporter produces a single document containing
the full ranked matrix plus metadata.  The CSV exporter produces a flat
table suitable for spreadsheet import or CI/CD quality-gate parsing.

Usage
-----
>>> engine = DecisionMatrixEngine(summaries, profile_name="devops")
>>> matrices = engine.build()
>>> JSONExporter(engine).export("reports/decision_matrix.json")
>>> CSVExporter(engine).export("reports/decision_matrix.csv")
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from framework.decision_matrix import CRITERIA, DecisionMatrixEngine

try:
    from contracts import DecisionMatrix, Recommendation
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)


# ======================================================================
# JSONExporter
# ======================================================================

class JSONExporter:
    """Export decision matrices and recommendations to JSON.

    Parameters
    ----------
    engine : DecisionMatrixEngine
        A built engine (i.e. ``.build()`` has been called).
    recommendations : list[dict] | None
        Optional recommendation dicts to include.
    """

    def __init__(
        self,
        engine: DecisionMatrixEngine,
        recommendations: list[dict[str, Any]] | None = None,
    ) -> None:
        self._engine = engine
        self._recommendations = recommendations or []

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Build the full JSON-serialisable document."""
        return {
            "meta": {
                "generated_at": datetime.utcnow().isoformat(),
                "profile": self._engine.profile_name,
                "weights": self._engine.weights,
                "model_count": len(self._engine.matrices),
            },
            "decision_matrix": self._engine.to_dict(),
            "recommendations": self._recommendations,
        }

    # ------------------------------------------------------------------
    def to_json(self, indent: int = 2) -> str:
        """Return the JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    # ------------------------------------------------------------------
    def export(self, path: str | Path) -> Path:
        """Write JSON to *path*, creating parent directories."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())
        logger.info("JSON exported to %s", p)
        return p


# ======================================================================
# CSVExporter
# ======================================================================

class CSVExporter:
    """Export decision matrices to a flat CSV table.

    Columns
    -------
    rank, model_id, weighted_total, profile,
    <criterion>_raw, <criterion>_norm, <criterion>_weight  (× 5 criteria)
    """

    def __init__(self, engine: DecisionMatrixEngine) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    @property
    def fieldnames(self) -> list[str]:
        base = ["rank", "model_id", "weighted_total", "profile"]
        for c in CRITERIA:
            base.extend([f"{c}_raw", f"{c}_norm", f"{c}_weight"])
        return base

    # ------------------------------------------------------------------
    def to_rows(self) -> list[dict[str, Any]]:
        """Return list of row dicts."""
        rows: list[dict[str, Any]] = []
        for dm_dict in self._engine.to_dict():
            row: dict[str, Any] = {
                "rank": dm_dict["rank"],
                "model_id": dm_dict["model_id"],
                "weighted_total": dm_dict["weighted_total"],
                "profile": dm_dict["profile"],
            }
            scores_by_criterion = {
                s["criterion"]: s for s in dm_dict["scores"]
            }
            for c in CRITERIA:
                s = scores_by_criterion.get(c, {})
                row[f"{c}_raw"] = s.get("raw_value", 0.0)
                row[f"{c}_norm"] = s.get("normalised_value", 0.0)
                row[f"{c}_weight"] = s.get("weight", 0.0)
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    def to_csv_string(self) -> str:
        """Return the CSV as a string."""
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=self.fieldnames)
        writer.writeheader()
        writer.writerows(self.to_rows())
        return buf.getvalue()

    # ------------------------------------------------------------------
    def export(self, path: str | Path) -> Path:
        """Write CSV to *path*, creating parent directories."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_csv_string())
        logger.info("CSV exported to %s", p)
        return p


# ======================================================================
# Multi-profile exporter
# ======================================================================

class MultiProfileExporter:
    """Export decision matrices for all available profiles.

    Produces one JSON and one CSV per profile, plus a combined summary.
    """

    def __init__(
        self,
        summaries: Sequence[Any],
        output_dir: str | Path = "reports",
        profiles_dir: str | Path | None = None,
    ) -> None:
        from framework.decision_matrix import ProfileLoader

        self._summaries = list(summaries)
        self._output_dir = Path(output_dir)
        self._loader = ProfileLoader(profiles_dir)

    # ------------------------------------------------------------------
    def export_all(self) -> dict[str, Path]:
        """Export matrices for every profile.  Returns ``{name: json_path}``."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        for name in self._loader.list_profiles():
            engine = DecisionMatrixEngine(
                self._summaries, profile_name=name
            )
            engine.build()

            json_path = self._output_dir / f"decision_matrix_{name}.json"
            csv_path = self._output_dir / f"decision_matrix_{name}.csv"

            JSONExporter(engine).export(json_path)
            CSVExporter(engine).export(csv_path)
            paths[name] = json_path

        # Combined summary
        combined = self._build_combined()
        combined_path = self._output_dir / "decision_matrix.json"
        combined_path.write_text(
            json.dumps(combined, indent=2, default=str)
        )
        paths["combined"] = combined_path
        return paths

    # ------------------------------------------------------------------
    def _build_combined(self) -> dict[str, Any]:
        profiles: dict[str, Any] = {}
        for name in self._loader.list_profiles():
            engine = DecisionMatrixEngine(
                self._summaries, profile_name=name
            )
            engine.build()
            profiles[name] = {
                "weights": engine.weights,
                "rankings": [
                    {
                        "rank": m.rank,
                        "model_id": m.model_id,
                        "weighted_total": round(m.weighted_total, 6),
                    }
                    for m in engine.matrices
                ],
            }
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "profiles": profiles,
        }
