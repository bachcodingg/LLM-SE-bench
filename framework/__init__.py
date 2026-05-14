"""
framework — Component 5: Decision Framework for llm-se-bench.

Practitioner decision matrix, constraint-based model recommendation,
Pareto frontier analysis, interactive dashboard, and report generation.

Public surface
--------------
DecisionMatrixEngine   Weighted multi-criteria decision analysis (MCDA).
ModelRecommender       Constraint-based model recommendation.
TradeoffAnalyzer       Pareto frontier computation and trade-off analysis.
ProfileLoader          Load YAML weighting profiles (DevOps / Audit / Budget).
CLIReporter            Terminal summary generator.
PDFReporter            WeasyPrint-based PDF report generator.
JSONExporter / CSVExporter   Structured export of decision matrices.

Dashboard (framework.dashboard)
    Launch via ``python -m framework.dashboard.app`` or ``launch_dashboard()``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from framework.decision_matrix import DecisionMatrixEngine, ProfileLoader
from framework.recommender import ModelRecommender
from framework.tradeoffs import TradeoffAnalyzer

__all__ = [
    "DecisionMatrixEngine",
    "ModelRecommender",
    "ProfileLoader",
    "TradeoffAnalyzer",
]
