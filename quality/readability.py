"""
readability.py — Automated readability scoring for Java source code.

Heuristic dimensions (each 0–100, then averaged into a composite):
    1. Naming quality:     camelCase compliance, identifier length, meaningfulness
    2. Comment density:    ratio of comment lines to code lines, Javadoc coverage
    3. Method structure:   method length, parameter count, return-point count
    4. Formatting:         consistent indentation, line length, blank-line spacing
    5. Documentation:      class-level and public-method Javadoc presence

Scores are intentionally heuristic — they approximate what a human reviewer
would flag, not a formal readability metric.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import javalang
from javalang.tree import (
    ClassDeclaration,
    ConstructorDeclaration,
    FieldDeclaration,
    InterfaceDeclaration,
    MethodDeclaration,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ReadabilityReport:
    """Readability assessment for a Java source file."""
    filename: str
    naming_score: float = 0.0
    comment_score: float = 0.0
    structure_score: float = 0.0
    formatting_score: float = 0.0
    documentation_score: float = 0.0
    composite_score: float = 0.0
    details: dict = field(default_factory=dict)

    def compute_composite(self) -> float:
        """Weighted average of sub-scores."""
        weights = {
            "naming": 0.25,
            "comment": 0.15,
            "structure": 0.25,
            "formatting": 0.15,
            "documentation": 0.20,
        }
        self.composite_score = (
            self.naming_score * weights["naming"]
            + self.comment_score * weights["comment"]
            + self.structure_score * weights["structure"]
            + self.formatting_score * weights["formatting"]
            + self.documentation_score * weights["documentation"]
        )
        return self.composite_score

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "naming_score": round(self.naming_score, 1),
            "comment_score": round(self.comment_score, 1),
            "structure_score": round(self.structure_score, 1),
            "formatting_score": round(self.formatting_score, 1),
            "documentation_score": round(self.documentation_score, 1),
            "composite_score": round(self.composite_score, 1),
        }


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_CAMEL_CASE = re.compile(r"^[a-z][a-zA-Z0-9]*$")
_PASCAL_CASE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
_UPPER_SNAKE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SINGLE_CHAR = re.compile(r"^[a-zA-Z]$")
_JAVADOC_OPEN = re.compile(r"^\s*/\*\*")
_BLOCK_COMMENT_OPEN = re.compile(r"^\s*/\*")
_BLOCK_COMMENT_CLOSE = re.compile(r"\*/\s*$")
_LINE_COMMENT = re.compile(r"^\s*//")
_BLANK_LINE = re.compile(r"^\s*$")

# Common meaningless identifiers (beyond single chars)
_MEANINGLESS = frozenset({
    "tmp", "temp", "foo", "bar", "baz", "x", "y", "z",
    "a", "b", "c", "d", "e", "aa", "bb", "cc", "dd",
    "val", "var", "data", "obj", "thing", "stuff",
    "ret", "res", "ans",
})

# Allowed single-char names in specific contexts
_LOOP_VARS = frozenset({"i", "j", "k", "n", "m"})


# ---------------------------------------------------------------------------
# Naming quality
# ---------------------------------------------------------------------------

def _score_naming(tree, source: str) -> tuple[float, dict]:
    """
    Evaluate identifier naming conventions.

    Checks:
    - Class names: PascalCase
    - Method / field names: camelCase
    - Constants (static final): UPPER_SNAKE_CASE
    - Identifier length: 2–30 chars preferred
    - Avoidance of meaningless names
    """
    scores = []
    details = {"violations": [], "total_identifiers": 0, "compliant": 0}

    for _, node in tree.filter(ClassDeclaration):
        details["total_identifiers"] += 1
        if _PASCAL_CASE.match(node.name):
            scores.append(100)
            details["compliant"] += 1
        else:
            scores.append(20)
            details["violations"].append(f"Class '{node.name}' not PascalCase")

    for _, node in tree.filter(InterfaceDeclaration):
        details["total_identifiers"] += 1
        if _PASCAL_CASE.match(node.name):
            scores.append(100)
            details["compliant"] += 1
        else:
            scores.append(20)
            details["violations"].append(f"Interface '{node.name}' not PascalCase")

    for _, node in tree.filter(MethodDeclaration):
        details["total_identifiers"] += 1
        name = node.name
        if _CAMEL_CASE.match(name):
            scores.append(100)
            details["compliant"] += 1
        else:
            scores.append(30)
            details["violations"].append(f"Method '{name}' not camelCase")

        # Length check
        if len(name) < 2:
            scores.append(20)
            details["violations"].append(f"Method '{name}' too short")
        elif len(name) > 30:
            scores.append(50)
            details["violations"].append(f"Method '{name}' very long ({len(name)} chars)")
        else:
            scores.append(100)

        # Meaningfulness
        if name.lower() in _MEANINGLESS:
            scores.append(10)
            details["violations"].append(f"Method '{name}' is a meaningless name")
        else:
            scores.append(100)

    for _, node in tree.filter(FieldDeclaration):
        is_constant = ("static" in (node.modifiers or set())
                       and "final" in (node.modifiers or set()))
        for decl in node.declarators:
            details["total_identifiers"] += 1
            name = decl.name
            if is_constant:
                if _UPPER_SNAKE.match(name):
                    scores.append(100)
                    details["compliant"] += 1
                else:
                    scores.append(40)
                    details["violations"].append(
                        f"Constant '{name}' not UPPER_SNAKE_CASE")
            else:
                if _CAMEL_CASE.match(name):
                    scores.append(100)
                    details["compliant"] += 1
                else:
                    scores.append(40)
                    details["violations"].append(f"Field '{name}' not camelCase")

            if name.lower() in _MEANINGLESS and name.lower() not in _LOOP_VARS:
                scores.append(20)
            else:
                scores.append(100)

    return (sum(scores) / len(scores) if scores else 50.0), details


# ---------------------------------------------------------------------------
# Comment density
# ---------------------------------------------------------------------------

def _score_comments(source: str) -> tuple[float, dict]:
    """
    Measure comment density relative to code volume.

    Target: 10–30% comment lines is ideal.
    Below 5% → poor; above 50% → over-commented.
    """
    lines = source.splitlines()
    total = len(lines)
    if total == 0:
        return 0.0, {"comment_lines": 0, "code_lines": 0, "ratio": 0.0}

    comment_lines = 0
    code_lines = 0
    blank_lines = 0
    in_block = False

    for line in lines:
        stripped = line.strip()
        if in_block:
            comment_lines += 1
            if "*/" in stripped:
                in_block = False
            continue

        if _BLANK_LINE.match(line):
            blank_lines += 1
        elif _JAVADOC_OPEN.match(line) or _BLOCK_COMMENT_OPEN.match(line):
            comment_lines += 1
            if "*/" not in stripped:
                in_block = True
        elif _LINE_COMMENT.match(line):
            comment_lines += 1
        else:
            code_lines += 1

    effective = code_lines + comment_lines
    if effective == 0:
        return 0.0, {"comment_lines": 0, "code_lines": 0, "ratio": 0.0}

    ratio = comment_lines / effective

    # Score based on ideal range
    if 0.10 <= ratio <= 0.30:
        score = 100.0
    elif 0.05 <= ratio < 0.10:
        score = 60.0 + (ratio - 0.05) / 0.05 * 40
    elif 0.30 < ratio <= 0.50:
        score = 100.0 - (ratio - 0.30) / 0.20 * 40
    elif ratio < 0.05:
        score = max(10.0, ratio / 0.05 * 60)
    else:
        score = max(20.0, 60 - (ratio - 0.50) * 100)

    details = {
        "comment_lines": comment_lines,
        "code_lines": code_lines,
        "blank_lines": blank_lines,
        "ratio": round(ratio, 3),
    }
    return min(100.0, score), details


# ---------------------------------------------------------------------------
# Structure quality
# ---------------------------------------------------------------------------

def _score_structure(tree) -> tuple[float, dict]:
    """
    Evaluate method structure heuristics:
    - Method length: < 20 lines ideal, > 50 poor
    - Parameter count: ≤ 4 ideal, > 7 poor
    - Single return point preferred
    """
    scores = []
    details = {
        "long_methods": [],
        "high_param_methods": [],
        "avg_method_length": 0.0,
    }
    method_lengths = []

    for _, class_node in tree.filter(ClassDeclaration):
        for member in (class_node.body or []):
            if not isinstance(member, (MethodDeclaration, ConstructorDeclaration)):
                continue

            method_name = member.name
            param_count = len(member.parameters or [])

            # Estimate line count from positions
            line_count = _estimate_method_length(member)
            method_lengths.append(line_count)

            # Length scoring
            if line_count <= 20:
                scores.append(100)
            elif line_count <= 40:
                scores.append(70)
            elif line_count <= 60:
                scores.append(40)
                details["long_methods"].append(
                    f"{method_name} ({line_count} lines)")
            else:
                scores.append(10)
                details["long_methods"].append(
                    f"{method_name} ({line_count} lines)")

            # Parameter scoring
            if param_count <= 3:
                scores.append(100)
            elif param_count <= 5:
                scores.append(70)
            elif param_count <= 7:
                scores.append(40)
                details["high_param_methods"].append(
                    f"{method_name} ({param_count} params)")
            else:
                scores.append(10)
                details["high_param_methods"].append(
                    f"{method_name} ({param_count} params)")

    if method_lengths:
        details["avg_method_length"] = round(
            sum(method_lengths) / len(method_lengths), 1)

    return (sum(scores) / len(scores) if scores else 50.0), details


def _estimate_method_length(method_node) -> int:
    """Estimate method line count from AST positions."""
    positions = []

    def _collect_positions(node):
        if node is None:
            return
        pos = getattr(node, "position", None)
        if pos is not None:
            positions.append(pos.line if hasattr(pos, "line") else pos[0])
        if isinstance(node, (list, tuple)):
            for child in node:
                _collect_positions(child)
        elif hasattr(node, "children"):
            for child in node.children:
                _collect_positions(child)

    _collect_positions(method_node)
    if len(positions) < 2:
        return 1
    return max(positions) - min(positions) + 1


# ---------------------------------------------------------------------------
# Formatting quality
# ---------------------------------------------------------------------------

def _score_formatting(source: str) -> tuple[float, dict]:
    """
    Check formatting consistency:
    - Line length: < 120 chars preferred
    - Consistent indentation (spaces vs tabs, no mixing)
    - Blank-line separation between methods
    """
    lines = source.splitlines()
    if not lines:
        return 50.0, {}

    penalties = 0
    total_checks = 0
    long_lines = 0
    mixed_indent = 0

    uses_tabs = any(line.startswith("\t") for line in lines if line.strip())
    uses_spaces = any(line.startswith("    ") for line in lines if line.strip())

    for line in lines:
        total_checks += 1

        # Line length
        if len(line) > 120:
            long_lines += 1
            penalties += 1
        elif len(line) > 100:
            penalties += 0.5

        # Mixed indentation
        if line.strip():
            has_tab = "\t" in line[:len(line) - len(line.lstrip())]
            has_space = "    " in line[:len(line) - len(line.lstrip())]
            if has_tab and has_space:
                mixed_indent += 1
                penalties += 1

    details = {
        "long_lines_over_120": long_lines,
        "mixed_indentation_lines": mixed_indent,
        "uses_tabs": uses_tabs,
        "uses_spaces": uses_spaces,
    }

    if total_checks == 0:
        return 50.0, details

    violation_rate = penalties / total_checks
    score = max(0.0, 100.0 - violation_rate * 200)
    return min(100.0, score), details


# ---------------------------------------------------------------------------
# Documentation coverage
# ---------------------------------------------------------------------------

def _score_documentation(tree, source: str) -> tuple[float, dict]:
    """
    Evaluate Javadoc coverage:
    - Class-level Javadoc present?
    - Public method Javadoc present?
    """
    lines = source.splitlines()
    total_targets = 0
    documented = 0
    details = {"undocumented": []}

    # Helper: check if a Javadoc comment precedes a given line
    javadoc_ranges = _find_javadoc_ranges(lines)

    for _, node in tree.filter(ClassDeclaration):
        total_targets += 1
        pos = getattr(node, "position", None)
        if pos and _has_preceding_javadoc(pos, javadoc_ranges):
            documented += 1
        else:
            details["undocumented"].append(f"class {node.name}")

    for _, node in tree.filter(MethodDeclaration):
        modifiers = node.modifiers or set()
        if "public" in modifiers:
            total_targets += 1
            pos = getattr(node, "position", None)
            if pos and _has_preceding_javadoc(pos, javadoc_ranges):
                documented += 1
            else:
                details["undocumented"].append(f"method {node.name}")

    details["total_targets"] = total_targets
    details["documented"] = documented

    if total_targets == 0:
        return 50.0, details

    coverage = documented / total_targets
    score = coverage * 100.0
    return score, details


def _find_javadoc_ranges(lines: list[str]) -> list[tuple[int, int]]:
    """Find (start_line, end_line) of all Javadoc comment blocks (1-indexed)."""
    ranges = []
    i = 0
    while i < len(lines):
        if _JAVADOC_OPEN.match(lines[i]):
            start = i + 1
            while i < len(lines) and "*/" not in lines[i]:
                i += 1
            ranges.append((start, i + 1))
        i += 1
    return ranges


def _has_preceding_javadoc(position, javadoc_ranges: list[tuple[int, int]]) -> bool:
    """Check if there's a Javadoc block ending just before the given position."""
    target_line = position.line if hasattr(position, "line") else position[0]
    for _, end in javadoc_ranges:
        # Javadoc should end 1–3 lines before the declaration
        if 0 < target_line - end <= 3:
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ReadabilityScorer:
    """Scores Java source readability across five heuristic dimensions."""

    def score(self, source: str, filename: str = "<unknown>") -> ReadabilityReport:
        """
        Analyse source and return a ReadabilityReport with sub-scores.

        Args:
            source: Java source code string.
            filename: label for the report.

        Returns:
            ReadabilityReport with composite and per-dimension scores.
        """
        report = ReadabilityReport(filename=filename)

        try:
            tree = javalang.parse.parse(source)
        except Exception as exc:
            logger.warning("Parse failure in %s: %s", filename, exc)
            report.composite_score = 0.0
            return report

        report.naming_score, naming_det = _score_naming(tree, source)
        report.comment_score, comment_det = _score_comments(source)
        report.structure_score, struct_det = _score_structure(tree)
        report.formatting_score, format_det = _score_formatting(source)
        report.documentation_score, doc_det = _score_documentation(tree, source)

        report.details = {
            "naming": naming_det,
            "comments": comment_det,
            "structure": struct_det,
            "formatting": format_det,
            "documentation": doc_det,
        }

        report.compute_composite()
        return report


def score_file(filepath: str | Path) -> ReadabilityReport:
    """Convenience: read a .java file and return its readability report."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error("File not found: %s", filepath)
        return ReadabilityReport(filename=str(filepath))
    source = filepath.read_text(encoding="utf-8", errors="replace")
    return ReadabilityScorer().score(source, filename=str(filepath))
