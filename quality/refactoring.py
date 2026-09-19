"""
refactoring.py — Before/after CK-metrics comparison for God Class refactoring.

Evaluates LLM-generated refactoring recommendations by:
    1. Computing CK metrics on the original God Class.
    2. Computing CK metrics on each proposed decomposition class.
    3. Measuring deltas (improvements / regressions) across all six CK dimensions.
    4. Validating structural coherence: do the decomposed classes collectively
       cover the original class's methods and fields?
    5. Scoring the refactoring on a 0–100 scale.

The God Class thresholds follow the Lanza & Marinescu heuristics:
    WMC  > 47   →  God Class indicator
    LCOM > 10   →  poor cohesion
    CBO  > 14   →  high coupling
    RFC  > 50   →  overly complex response set
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .ck_metrics import CKMetricsExtractor, ClassMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds (Lanza & Marinescu, "Object-Oriented Metrics in Practice", 2006)
# ---------------------------------------------------------------------------

GOD_CLASS_THRESHOLDS = {
    "wmc": 47,
    "lcom": 10,
    "cbo": 14,
    "rfc": 50,
}

# Ideal direction for each metric (lower is better for all CK metrics)
_LOWER_IS_BETTER = {"wmc", "cbo", "rfc", "lcom", "dit"}
_NEUTRAL = {"noc", "loc"}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MetricDelta:
    """Change in a single CK metric from before to after refactoring."""
    metric_name: str
    before: float
    after: float  # aggregate (max or sum, depending on metric)
    delta: float = 0.0
    improved: bool = False
    pct_change: float = 0.0

    def __post_init__(self):
        self.delta = self.after - self.before
        if self.before != 0:
            self.pct_change = (self.delta / abs(self.before)) * 100
        self.improved = (
            self.delta < 0 if self.metric_name in _LOWER_IS_BETTER
            else self.delta >= 0
        )


@dataclass
class RefactoringResult:
    """Complete evaluation of a single refactoring attempt."""
    original_file: str
    original_class: str = ""
    decomposed_files: list[str] = field(default_factory=list)
    decomposed_classes: list[str] = field(default_factory=list)
    original_metrics: Optional[ClassMetrics] = None
    decomposed_metrics: list[ClassMetrics] = field(default_factory=list)
    deltas: list[MetricDelta] = field(default_factory=list)
    method_coverage: float = 0.0
    field_coverage: float = 0.0
    overall_score: float = 0.0
    is_god_class: bool = False
    god_class_resolved: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "original_file": self.original_file,
            "original_class": self.original_class,
            "num_decomposed_classes": len(self.decomposed_classes),
            "decomposed_classes": self.decomposed_classes,
            "original_wmc": self.original_metrics.wmc if self.original_metrics else 0,
            "original_lcom": self.original_metrics.lcom if self.original_metrics else 0,
            "original_cbo": self.original_metrics.cbo if self.original_metrics else 0,
            "method_coverage": round(self.method_coverage, 2),
            "field_coverage": round(self.field_coverage, 2),
            "overall_score": round(self.overall_score, 1),
            "is_god_class": self.is_god_class,
            "god_class_resolved": self.god_class_resolved,
            "warnings": self.warnings,
            "deltas": {d.metric_name: {
                "before": d.before,
                "after": d.after,
                "delta": round(d.delta, 2),
                "pct_change": round(d.pct_change, 1),
                "improved": d.improved,
            } for d in self.deltas},
        }


# ---------------------------------------------------------------------------
# God Class detection
# ---------------------------------------------------------------------------

def is_god_class(metrics: ClassMetrics) -> bool:
    """
    Check whether the given class exceeds God Class thresholds.

    A class is flagged as God Class if it exceeds at least 2 of 4 thresholds.
    """
    violations = 0
    if metrics.wmc > GOD_CLASS_THRESHOLDS["wmc"]:
        violations += 1
    if metrics.lcom > GOD_CLASS_THRESHOLDS["lcom"]:
        violations += 1
    if metrics.cbo > GOD_CLASS_THRESHOLDS["cbo"]:
        violations += 1
    if metrics.rfc > GOD_CLASS_THRESHOLDS["rfc"]:
        violations += 1
    return violations >= 2


# ---------------------------------------------------------------------------
# Refactoring evaluator
# ---------------------------------------------------------------------------

class RefactoringEvaluator:
    """
    Evaluates God Class refactoring by comparing before/after CK metrics.

    Usage::

        evaluator = RefactoringEvaluator()
        result = evaluator.evaluate(
            original_source="...",
            decomposed_sources=["...", "..."],
        )
        print(result.overall_score)
    """

    def __init__(self):
        self._extractor = CKMetricsExtractor()

    def evaluate(
        self,
        original_source: str,
        decomposed_sources: list[str],
        original_filename: str = "Original.java",
        decomposed_filenames: Optional[list[str]] = None,
    ) -> RefactoringResult:
        """
        Compare original class metrics against decomposed class metrics.

        Args:
            original_source: Java source of the original God Class.
            decomposed_sources: list of Java sources for decomposed classes.
            original_filename: label for the original file.
            decomposed_filenames: labels for decomposed files.

        Returns:
            RefactoringResult with deltas and overall score.
        """
        if decomposed_filenames is None:
            decomposed_filenames = [
                f"Decomposed_{i}.java" for i in range(len(decomposed_sources))
            ]

        result = RefactoringResult(
            original_file=original_filename,
            decomposed_files=decomposed_filenames,
        )

        # Parse original
        orig_classes = self._extractor.extract(original_source, original_filename)
        if not orig_classes:
            result.warnings.append("Could not parse original source")
            return result

        orig = orig_classes[0]  # primary class
        result.original_class = orig.class_name
        result.original_metrics = orig
        result.is_god_class = is_god_class(orig)

        # Parse decomposed classes
        all_decomposed: list[ClassMetrics] = []
        for src, fname in zip(decomposed_sources, decomposed_filenames):
            classes = self._extractor.extract(src, fname)
            all_decomposed.extend(classes)

        if not all_decomposed:
            result.warnings.append("No classes found in decomposed sources")
            result.overall_score = 0.0
            return result

        result.decomposed_metrics = all_decomposed
        result.decomposed_classes = [c.class_name for c in all_decomposed]

        # Behaviour preservation.  Without this, a "decomposition" that
        # deletes most of the class scores well on every CK metric, because
        # deleting code is the easiest way to reduce complexity.
        result.method_coverage, result.field_coverage = self._compute_coverage(
            original_source, decomposed_sources
        )
        if result.method_coverage < 1.0:
            result.warnings.append(
                f"Only {result.method_coverage:.0%} of the original's methods "
                f"appear in the decomposed sources"
            )

        # Compute deltas
        result.deltas = self._compute_deltas(orig, all_decomposed)

        # Check if God Class is resolved
        result.god_class_resolved = not any(
            is_god_class(c) for c in all_decomposed
        )

        # Compute overall score
        result.overall_score = self._compute_score(result)

        return result

    def evaluate_from_files(
        self,
        original_path: str | Path,
        decomposed_paths: list[str | Path],
    ) -> RefactoringResult:
        """Convenience method that reads files from disk."""
        original_path = Path(original_path)
        original_source = original_path.read_text(encoding="utf-8", errors="replace")

        decomposed_sources = []
        decomposed_filenames = []
        for p in decomposed_paths:
            p = Path(p)
            decomposed_sources.append(p.read_text(encoding="utf-8", errors="replace"))
            decomposed_filenames.append(p.name)

        return self.evaluate(
            original_source, decomposed_sources,
            original_filename=original_path.name,
            decomposed_filenames=decomposed_filenames,
        )

    @staticmethod
    def _member_names(source: str) -> tuple[set[str], set[str]]:
        """Return (method names, field names) declared anywhere in *source*.

        Names, not signatures: a decomposition is free to change a method's
        visibility or move it between classes, and both should still count
        as preserved.
        """
        import javalang

        methods: set[str] = set()
        fields: set[str] = set()
        try:
            tree = javalang.parse.parse(source)
        except Exception:
            return methods, fields

        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            methods.add(node.name)
        for _, node in tree.filter(javalang.tree.FieldDeclaration):
            for declarator in node.declarators:
                fields.add(declarator.name)
        return methods, fields

    def _compute_coverage(
        self, original_source: str, decomposed_sources: list[str],
    ) -> tuple[float, float]:
        """Fraction of the original's methods and fields that survive.

        Returns ``(method_coverage, field_coverage)``, each in [0, 1]. A
        class with no methods (or no fields) reports 1.0 for that axis:
        nothing was there to lose.
        """
        orig_methods, orig_fields = self._member_names(original_source)

        after_methods: set[str] = set()
        after_fields: set[str] = set()
        for source in decomposed_sources:
            methods, fields = self._member_names(source)
            after_methods |= methods
            after_fields |= fields

        method_coverage = (
            len(orig_methods & after_methods) / len(orig_methods)
            if orig_methods else 1.0
        )
        field_coverage = (
            len(orig_fields & after_fields) / len(orig_fields)
            if orig_fields else 1.0
        )
        return method_coverage, field_coverage

    def _compute_deltas(
        self, original: ClassMetrics, decomposed: list[ClassMetrics],
    ) -> list[MetricDelta]:
        """Compute metric deltas between original and decomposed classes."""
        deltas = []

        # For WMC, CBO, RFC, LCOM: compare original against max of decomposed
        # (the worst decomposed class should still be better than the original)
        for metric in ("wmc", "cbo", "rfc", "lcom"):
            before = getattr(original, metric)
            # Use max: the worst case among decomposed classes
            after = max(getattr(c, metric) for c in decomposed)
            deltas.append(MetricDelta(metric_name=metric, before=before, after=after))

        # DIT: max across decomposed
        deltas.append(MetricDelta(
            metric_name="dit",
            before=original.dit,
            after=max(c.dit for c in decomposed),
        ))

        # LOC: sum of decomposed (total code should not balloon)
        deltas.append(MetricDelta(
            metric_name="loc",
            before=original.loc,
            after=sum(c.loc for c in decomposed),
        ))

        # Number of classes introduced
        deltas.append(MetricDelta(
            metric_name="num_classes",
            before=1,
            after=len(decomposed),
        ))

        return deltas

    def _compute_score(self, result: RefactoringResult) -> float:
        """
        Compute an overall refactoring quality score (0–100).

        Scoring rubric:
            - Metric improvements (WMC, LCOM, CBO, RFC down): 40 pts
            - God Class resolved: 20 pts
            - Reasonable decomposition size (2–6 classes): 15 pts
            - LOC not excessively increased (< 50%): 15 pts
            - No critical warnings: 10 pts
        """
        score = 0.0

        # Metric improvement (40 pts total, 10 per key metric)
        key_metrics = {"wmc", "lcom", "cbo", "rfc"}
        for delta in result.deltas:
            if delta.metric_name in key_metrics:
                if delta.improved:
                    if delta.pct_change <= -30:
                        score += 10  # significant improvement
                    elif delta.pct_change <= -10:
                        score += 7
                    else:
                        score += 4  # marginal improvement
                elif delta.delta == 0:
                    score += 2  # neutral

        # God Class resolved (20 pts)
        if result.god_class_resolved:
            score += 20
        elif not result.is_god_class:
            score += 10  # wasn't a god class to begin with

        # Decomposition size (15 pts)
        n_classes = len(result.decomposed_classes)
        if 2 <= n_classes <= 6:
            score += 15
        elif n_classes == 1:
            score += 0  # no decomposition
        elif n_classes <= 10:
            score += 8
        else:
            score += 3
            result.warnings.append(
                f"Excessive decomposition: {n_classes} classes")

        # LOC growth (15 pts)
        loc_delta = next(
            (d for d in result.deltas if d.metric_name == "loc"), None)
        if loc_delta:
            if loc_delta.pct_change <= 10:
                score += 15
            elif loc_delta.pct_change <= 30:
                score += 10
            elif loc_delta.pct_change <= 50:
                score += 5
            else:
                result.warnings.append(
                    f"LOC increased by {loc_delta.pct_change:.0f}%")

        # No critical warnings (10 pts)
        if not result.warnings:
            score += 10
        else:
            score += max(0, 10 - len(result.warnings) * 2)

        return min(100.0, score)
