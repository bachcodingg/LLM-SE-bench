"""
stats/ — Component 4: Statistical Engine for llm-se-bench.

Provides hypothesis testing, effect-size computation, cost modelling,
confidence-interval estimation, consistency analysis, correlation analysis,
publication-quality visualisation (25+ figure types), and LaTeX table
generation.

Public API
----------
    load_evaluation_results(base_dir) -> pd.DataFrame
    load_quality_metrics(base_dir) -> pd.DataFrame
    load_cost_data(db_path) -> pd.DataFrame
    run_full_analysis(results_dir, quality_dir, db_path, output_dir)
"""

from __future__ import annotations

__version__ = "0.1.0"

from stats.descriptive import DescriptiveAnalyzer
from stats.hypothesis import HypothesisEngine
from stats.effect_size import EffectSizeCalculator
from stats.confidence import BootstrapCI
from stats.cost_model import CostModel
from stats.consistency import ConsistencyAnalyzer
from stats.correlation import CorrelationAnalyzer
from stats.visualisation import VisualisationEngine
from stats.report_data import ReportDataGenerator
from stats._loader import (
    load_evaluation_results,
    load_quality_metrics,
    load_cost_data,
)
from stats._pipeline import run_full_analysis

__all__ = [
    "DescriptiveAnalyzer",
    "HypothesisEngine",
    "EffectSizeCalculator",
    "BootstrapCI",
    "CostModel",
    "ConsistencyAnalyzer",
    "CorrelationAnalyzer",
    "VisualisationEngine",
    "ReportDataGenerator",
    "load_evaluation_results",
    "load_quality_metrics",
    "load_cost_data",
    "run_full_analysis",
]
