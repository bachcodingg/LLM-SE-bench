"""
reports.py — Per-file quality report generation and CSV output.

Orchestrates the full quality analysis pipeline for a single Java file or a
batch of files, producing:
    - QualityMetrics CSV (matching contracts.QualityMetrics fields)
    - CKMetrics CSV (matching contracts.CKMetrics fields)
    - Per-file JSON reports with detailed breakdowns

Output directory structure:
    quality/{dataset}/{model}/metrics.csv       — QualityMetrics
    quality/{dataset}/{model}/ck_metrics.csv     — CKMetrics per class
    quality/{dataset}/{model}/reports/           — per-file JSON
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .ck_metrics import CKMetricsExtractor, ClassMetrics, count_logical_loc
from .complexity import ComplexityAnalyzer, FileComplexity
from .readability import ReadabilityReport, ReadabilityScorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QualityMetrics CSV columns (from contracts.py)
# ---------------------------------------------------------------------------

QUALITY_METRICS_HEADER = [
    "metrics_id", "response_id", "problem_id",
    "lines_of_code", "cyclomatic_complexity", "maintainability_index",
    "halstead_volume", "lint_warnings", "lint_errors",
    "type_coverage_pct", "docstring_coverage_pct",
    "raw_lint_output", "computed_at",
]

CK_METRICS_HEADER = [
    "metrics_id", "response_id", "class_name",
    "wmc", "dit", "noc", "cbo", "rfc", "lcom",
    "computed_at",
]


# ---------------------------------------------------------------------------
# Halstead volume approximation
# ---------------------------------------------------------------------------

# Java operators
_OPERATORS = frozenset({
    "+", "-", "*", "/", "%", "++", "--",
    "==", "!=", ">", "<", ">=", "<=",
    "&&", "||", "!",
    "&", "|", "^", "~", "<<", ">>", ">>>",
    "=", "+=", "-=", "*=", "/=", "%=",
    "&=", "|=", "^=", "<<=", ">>=", ">>>=",
    "?", ":", "->",
    ".", ",", ";", "(", ")", "[", "]", "{", "}",
    "new", "instanceof", "return", "throw", "throws",
    "if", "else", "for", "while", "do", "switch", "case",
    "break", "continue", "try", "catch", "finally",
    "class", "interface", "extends", "implements",
    "public", "private", "protected", "static", "final",
    "abstract", "synchronized", "volatile", "transient",
    "void", "import", "package",
})

_TOKEN_RE = re.compile(
    r'""".*?"""|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|'
    r'//[^\n]*|/\*.*?\*/|'
    r'[a-zA-Z_]\w*|'
    r'\d+\.?\d*[fFdDlL]?|'
    r'>>>?=?|<<=?|>>=?|&&|\|\||[+\-*/%&|^~!=<>]=?|'
    r'[(){}\[\];,.:?@]|->',
    re.DOTALL,
)


def _estimate_halstead(source: str) -> float:
    """
    Approximate Halstead volume from Java source.

    Volume = N * log2(n)
    where N = total operators + operands, n = distinct operators + operands.

    This is an approximation — a true Halstead computation requires a full
    tokeniser distinguishing operators from operands precisely.
    """
    tokens = _TOKEN_RE.findall(source)
    if not tokens:
        return 0.0

    operators = set()
    operands = set()
    total_operators = 0
    total_operands = 0

    for tok in tokens:
        if tok.startswith("//") or tok.startswith("/*"):
            continue
        if tok.startswith('"') or tok.startswith("'"):
            operands.add(tok)
            total_operands += 1
        elif tok in _OPERATORS or tok in {
            "public", "private", "protected", "static", "final",
            "abstract", "void", "class", "interface", "new",
            "return", "if", "else", "for", "while", "do", "switch",
            "case", "break", "continue", "try", "catch", "finally",
            "throw", "throws", "extends", "implements", "instanceof",
            "import", "package", "synchronized", "volatile", "transient",
        }:
            operators.add(tok)
            total_operators += 1
        elif tok[0].isdigit():
            operands.add(tok)
            total_operands += 1
        elif tok[0].isalpha() or tok[0] == "_":
            operands.add(tok)
            total_operands += 1
        else:
            operators.add(tok)
            total_operators += 1

    n1 = len(operators)
    n2 = len(operands)
    big_n1 = total_operators
    big_n2 = total_operands
    n = n1 + n2  # vocabulary
    big_n = big_n1 + big_n2  # length

    if n <= 1:
        return 0.0

    volume = big_n * math.log2(n)
    return volume


# ---------------------------------------------------------------------------
# Maintainability index
# ---------------------------------------------------------------------------

def _maintainability_index(
    halstead_volume: float,
    cyclomatic_complexity: float,
    loc: int,
) -> float:
    """
    Compute the Maintainability Index (MI).

    MI = 171 - 5.2 * ln(V) - 0.23 * CC - 16.2 * ln(LOC)

    Clamped to 0–100 range.  Higher = more maintainable.
    Reference: Oman & Hagemeister (1992).
    """
    if halstead_volume <= 0 or loc <= 0:
        return 100.0

    mi = (
        171.0
        - 5.2 * math.log(halstead_volume)
        - 0.23 * cyclomatic_complexity
        - 16.2 * math.log(loc)
    )
    return max(0.0, min(100.0, mi))


# ---------------------------------------------------------------------------
# Type coverage estimation
# ---------------------------------------------------------------------------

def _estimate_type_coverage(source: str) -> float:
    """
    Estimate type annotation coverage for Java.

    Java is statically typed, so almost all declarations have types.
    We estimate by checking for `var` usage (Java 10+) which uses
    type inference.  A fully typed Java file has ~100% coverage.
    """
    lines = source.splitlines()
    var_count = 0
    decl_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("*"):
            continue
        # Count 'var' declarations
        if re.search(r'\bvar\s+\w+', stripped):
            var_count += 1
            decl_count += 1
        # Count explicit type declarations (simplified)
        elif re.search(r'\b(int|long|double|float|boolean|char|byte|short|String|List|Map|Set|void)\b.*\b\w+\s*[=;(]', stripped):
            decl_count += 1

    if decl_count == 0:
        return 100.0  # no declarations to check

    typed = decl_count - var_count
    return min(100.0, (typed / decl_count) * 100.0)


# ---------------------------------------------------------------------------
# Per-file quality report
# ---------------------------------------------------------------------------

@dataclass
class FileQualityReport:
    """Complete quality analysis for a single Java file."""
    filename: str
    response_id: str = ""
    problem_id: str = ""
    ck_metrics: list[ClassMetrics] = field(default_factory=list)
    complexity: Optional[FileComplexity] = None
    readability: Optional[ReadabilityReport] = None
    loc: int = 0
    halstead_volume: float = 0.0
    maintainability_index: float = 0.0
    type_coverage_pct: float = 100.0
    docstring_coverage_pct: float = 0.0

    def to_quality_metrics_row(self) -> list:
        """Produce a CSV row matching contracts.QualityMetrics."""
        avg_cc = 0.0
        if self.complexity and self.complexity.methods:
            avg_cc = self.complexity.avg_cyclomatic

        return [
            f"qm-{uuid.uuid4().hex[:8]}",
            self.response_id,
            self.problem_id,
            self.loc,
            round(avg_cc, 2),
            round(self.maintainability_index, 2),
            round(self.halstead_volume, 2),
            0,  # lint_warnings (not computed here)
            0,  # lint_errors
            round(self.type_coverage_pct, 1),
            round(self.docstring_coverage_pct, 1),
            "",  # raw_lint_output
            datetime.utcnow().isoformat(),
        ]

    def to_ck_metrics_rows(self) -> list[list]:
        """Produce CSV rows matching contracts.CKMetrics (one per class)."""
        rows = []
        for ck in self.ck_metrics:
            rows.append([
                f"ck-{uuid.uuid4().hex[:8]}",
                self.response_id,
                ck.class_name,
                ck.wmc, ck.dit, ck.noc, ck.cbo, ck.rfc, ck.lcom,
                datetime.utcnow().isoformat(),
            ])
        return rows

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "response_id": self.response_id,
            "problem_id": self.problem_id,
            "loc": self.loc,
            "halstead_volume": round(self.halstead_volume, 2),
            "maintainability_index": round(self.maintainability_index, 2),
            "type_coverage_pct": round(self.type_coverage_pct, 1),
            "docstring_coverage_pct": round(self.docstring_coverage_pct, 1),
            "ck_metrics": [m.to_dict() for m in self.ck_metrics],
            "complexity": self.complexity.to_dict() if self.complexity else {},
            "readability": self.readability.to_dict() if self.readability else {},
        }


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

class QualityReportGenerator:
    """
    Orchestrates quality analysis for Java source files and produces
    CSV outputs matching the contracts.py data models.
    """

    def __init__(self):
        self._ck_extractor = CKMetricsExtractor()
        self._complexity_analyzer = ComplexityAnalyzer()
        self._readability_scorer = ReadabilityScorer()

    def analyse_file(
        self,
        source: str,
        filename: str = "<unknown>",
        response_id: str = "",
        problem_id: str = "",
    ) -> FileQualityReport:
        """
        Run the full quality analysis pipeline on a single Java source.

        Args:
            source: Java source code.
            filename: file label.
            response_id: identifier of the LLM response.
            problem_id: identifier of the problem.

        Returns:
            FileQualityReport with all metrics populated.
        """
        report = FileQualityReport(
            filename=filename,
            response_id=response_id,
            problem_id=problem_id,
        )

        # LOC
        report.loc = count_logical_loc(source)

        # CK metrics
        report.ck_metrics = self._ck_extractor.extract(source, filename)

        # Complexity
        report.complexity = self._complexity_analyzer.analyse(source, filename)

        # Readability
        report.readability = self._readability_scorer.score(source, filename)

        # Halstead volume
        report.halstead_volume = _estimate_halstead(source)

        # Maintainability index
        avg_cc = report.complexity.avg_cyclomatic if report.complexity else 0
        report.maintainability_index = _maintainability_index(
            report.halstead_volume, avg_cc, max(1, report.loc))

        # Type coverage
        report.type_coverage_pct = _estimate_type_coverage(source)

        # Documentation coverage
        if report.readability and "documentation" in report.readability.details:
            doc_info = report.readability.details["documentation"]
            total = doc_info.get("total_targets", 0)
            documented = doc_info.get("documented", 0)
            report.docstring_coverage_pct = (
                (documented / total * 100) if total > 0 else 0.0
            )

        return report

    def analyse_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        dataset: str = "unknown",
        model: str = "unknown",
    ) -> list[FileQualityReport]:
        """
        Analyse all .java files in input_dir and write CSV outputs.

        Args:
            input_dir: directory containing .java files (from C2).
            output_dir: root output directory for quality results.
            dataset: dataset label (e.g. 'humaneval').
            model: model label (e.g. 'gpt-4o').

        Returns:
            List of FileQualityReport objects.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) / dataset / model
        output_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)

        reports: list[FileQualityReport] = []
        java_files = sorted(input_dir.glob("*.java"))

        if not java_files:
            logger.warning("No .java files found in %s", input_dir)
            return reports

        for java_file in java_files:
            source = java_file.read_text(encoding="utf-8", errors="replace")

            # Derive IDs from filename
            stem = java_file.stem
            response_id = f"resp-{stem}"
            problem_id = f"prob-{stem}"

            report = self.analyse_file(
                source, filename=java_file.name,
                response_id=response_id, problem_id=problem_id,
            )
            reports.append(report)

            # Write per-file JSON report
            json_path = reports_dir / f"{stem}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

        # Write QualityMetrics CSV
        metrics_csv = output_dir / "metrics.csv"
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(QUALITY_METRICS_HEADER)
            for r in reports:
                writer.writerow(r.to_quality_metrics_row())

        logger.info("Wrote %d rows to %s", len(reports), metrics_csv)

        # Write CKMetrics CSV
        ck_csv = output_dir / "ck_metrics.csv"
        with open(ck_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CK_METRICS_HEADER)
            for r in reports:
                for row in r.to_ck_metrics_rows():
                    writer.writerow(row)

        logger.info("Wrote CK metrics to %s", ck_csv)

        return reports

    def analyse_results_tree(
        self,
        results_root: str | Path,
        output_root: str | Path,
    ) -> dict[str, list[FileQualityReport]]:
        """
        Analyse the full results/ directory tree produced by C2.

        Expected structure:
            results/{dataset}/{model}/code/*.java

        Produces:
            quality/{dataset}/{model}/metrics.csv
            quality/{dataset}/{model}/ck_metrics.csv
            quality/{dataset}/{model}/reports/*.json

        Returns:
            Dict mapping '{dataset}/{model}' → list of reports.
        """
        results_root = Path(results_root)
        output_root = Path(output_root)
        all_reports: dict[str, list[FileQualityReport]] = {}

        if not results_root.is_dir():
            logger.error("Results root not found: %s", results_root)
            return all_reports

        for dataset_dir in sorted(results_root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name

            for model_dir in sorted(dataset_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                model = model_dir.name

                code_dir = model_dir / "code"
                if not code_dir.is_dir():
                    # Try the model dir itself
                    code_dir = model_dir

                key = f"{dataset}/{model}"
                reports = self.analyse_directory(
                    code_dir, output_root, dataset, model)
                all_reports[key] = reports
                logger.info("Analysed %d files for %s", len(reports), key)

        return all_reports
