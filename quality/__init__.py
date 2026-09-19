"""
quality — Component 3 of llm-se-bench: Code Quality Analyzer.

Analyses LLM-generated Java code across multiple quality dimensions:
    - CK design metrics (WMC, DIT, NOC, CBO, RFC, LCOM) via javalang AST
    - Cyclomatic & cognitive complexity
    - Readability heuristics (naming, comments, structure)
    - Refactoring evaluation (before/after God Class metrics delta)
    - Expert rating framework (rubric + CSV collection)
    - Inter-rater reliability (Cohen's kappa)
    - Diff/patch analysis (location accuracy, minimality)

Input:  Java source files at results/{dataset}/{model}/code/*.java
Output: quality/{dataset}/{model}/metrics.csv  (QualityMetrics)
        quality/{dataset}/{model}/ck_metrics.csv  (CKMetrics per class)
"""

__version__ = "0.1.0"

from .ck_metrics import (
    CKMetricsExtractor,
    ClassMetrics,
    count_logical_loc,
    extract_from_directory,
    extract_from_file,
)
from .complexity import (
    ComplexityAnalyzer,
    FileComplexity,
    MethodComplexity,
)
from .complexity import (
    analyse_file as analyse_complexity,
)
from .diff_analysis import (
    DiffStats,
    LocationAccuracy,
    PatchAnalysis,
    analyse_location_accuracy,
    analyse_patch,
    compute_diff,
)
from .expert_rating import (
    AutomatedRubricScorer,
    Rating,
    create_blank_rating_csv,
    get_rubric_text,
    read_ratings_csv,
    write_ratings_csv,
)
from .irr import (
    KappaResult,
    cohens_kappa,
    fleiss_kappa,
    interpret_kappa,
    percent_agreement,
    weighted_kappa,
)
from .readability import (
    ReadabilityReport,
    ReadabilityScorer,
)
from .readability import (
    score_file as score_readability,
)
from .refactoring import (
    GOD_CLASS_THRESHOLDS,
    MetricDelta,
    RefactoringEvaluator,
    RefactoringResult,
    is_god_class,
)
from .reports import (
    FileQualityReport,
    QualityReportGenerator,
)

__all__ = [
    # CK metrics
    "CKMetricsExtractor", "ClassMetrics", "count_logical_loc",
    "extract_from_file", "extract_from_directory",
    # Complexity
    "ComplexityAnalyzer", "FileComplexity", "MethodComplexity",
    "analyse_complexity",
    # Readability
    "ReadabilityScorer", "ReadabilityReport", "score_readability",
    # Refactoring
    "RefactoringEvaluator", "RefactoringResult", "MetricDelta",
    "is_god_class", "GOD_CLASS_THRESHOLDS",
    # Expert rating
    "AutomatedRubricScorer", "Rating",
    "read_ratings_csv", "write_ratings_csv",
    "create_blank_rating_csv", "get_rubric_text",
    # IRR
    "cohens_kappa", "weighted_kappa", "fleiss_kappa",
    "percent_agreement", "interpret_kappa", "KappaResult",
    # Diff analysis
    "compute_diff", "analyse_patch", "analyse_location_accuracy",
    "DiffStats", "PatchAnalysis", "LocationAccuracy",
    # Reports
    "QualityReportGenerator", "FileQualityReport",
]
