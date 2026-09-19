"""
complexity.py — Cyclomatic and cognitive complexity analysis for Java source.

Cyclomatic complexity (McCabe, 1976):
    CC = 1 + number of decision points (if, for, while, case, catch, &&, ||).

Cognitive complexity (SonarSource, 2017):
    Increments for breaks in linear flow, with nesting penalties.
    Structural: if/else/for/while/do/catch/switch/ternary → +1
    Nesting:    each structural keyword inside a nesting context → +nesting_level
    Hybrid:     &&/|| sequences broken by other operators → +1

Reference: G. Ann Campbell, "Cognitive Complexity: An Overview and Evaluation",
           SonarSource SA, 2017.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import javalang
from javalang.tree import (
    ClassDeclaration,
    ConstructorDeclaration,
    MethodDeclaration,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MethodComplexity:
    """Complexity results for a single method."""
    class_name: str
    method_name: str
    cyclomatic: int = 1
    cognitive: int = 0
    line_count: int = 0
    max_nesting_depth: int = 0
    parameter_count: int = 0

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "method_name": self.method_name,
            "cyclomatic": self.cyclomatic,
            "cognitive": self.cognitive,
            "line_count": self.line_count,
            "max_nesting_depth": self.max_nesting_depth,
            "parameter_count": self.parameter_count,
        }


@dataclass
class FileComplexity:
    """Aggregated complexity for an entire Java source file."""
    filename: str
    methods: list[MethodComplexity] = field(default_factory=list)
    total_cyclomatic: int = 0
    avg_cyclomatic: float = 0.0
    max_cyclomatic: int = 0
    total_cognitive: int = 0
    avg_cognitive: float = 0.0
    max_cognitive: int = 0
    max_nesting: int = 0

    def summarise(self) -> None:
        """Recompute aggregate fields from methods list."""
        if not self.methods:
            return
        cycs = [m.cyclomatic for m in self.methods]
        cogs = [m.cognitive for m in self.methods]
        nestings = [m.max_nesting_depth for m in self.methods]

        self.total_cyclomatic = sum(cycs)
        self.avg_cyclomatic = sum(cycs) / len(cycs)
        self.max_cyclomatic = max(cycs)
        self.total_cognitive = sum(cogs)
        self.avg_cognitive = sum(cogs) / len(cogs)
        self.max_cognitive = max(cogs)
        self.max_nesting = max(nestings) if nestings else 0

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "num_methods": len(self.methods),
            "total_cyclomatic": self.total_cyclomatic,
            "avg_cyclomatic": round(self.avg_cyclomatic, 2),
            "max_cyclomatic": self.max_cyclomatic,
            "total_cognitive": self.total_cognitive,
            "avg_cognitive": round(self.avg_cognitive, 2),
            "max_cognitive": self.max_cognitive,
            "max_nesting": self.max_nesting,
        }


# ---------------------------------------------------------------------------
# Cyclomatic complexity
# ---------------------------------------------------------------------------

def _walk_tree(node):
    """Yield all nodes in a javalang subtree."""
    if node is None:
        return
    yield node
    if isinstance(node, (list, tuple)):
        for child in node:
            yield from _walk_tree(child)
    elif hasattr(node, "children"):
        for child in node.children:
            yield from _walk_tree(child)


def _cyclomatic_complexity(method_body) -> int:
    """Count decision points in a method body."""
    if method_body is None:
        return 1
    cc = 1
    for node in _walk_tree(method_body):
        name = type(node).__name__
        if name in ("IfStatement", "ForStatement", "WhileStatement",
                     "DoStatement", "CatchClause"):
            cc += 1
        elif name == "SwitchStatementCase":
            if getattr(node, "case", None) is not None:
                cc += 1
        elif name == "TernaryExpression":
            cc += 1
        elif name == "BinaryOperation":
            if getattr(node, "operator", "") in ("&&", "||"):
                cc += 1
    return cc


# ---------------------------------------------------------------------------
# Cognitive complexity
# ---------------------------------------------------------------------------

# Node types that are structural increments AND add to nesting
_NESTING_STRUCTURAL = {
    "IfStatement", "ForStatement", "WhileStatement",
    "DoStatement", "SwitchStatement", "CatchClause",
}

# Additional structural increments (no nesting change)
_STRUCTURAL_ONLY = {"TernaryExpression"}


def _cognitive_complexity(method_body) -> tuple[int, int]:
    """
    Compute cognitive complexity and max nesting depth for a method body.

    Returns:
        (cognitive_score, max_nesting_depth)
    """
    if method_body is None:
        return 0, 0

    score = 0
    max_depth = 0

    def _walk(nodes, nesting: int) -> None:
        nonlocal score, max_depth

        if nodes is None:
            return
        if not isinstance(nodes, (list, tuple)):
            nodes = [nodes]

        prev_logical_op = None

        for node in nodes:
            if node is None:
                continue
            name = type(node).__name__

            if name in _NESTING_STRUCTURAL:
                score += 1 + nesting  # structural + nesting penalty
                max_depth = max(max_depth, nesting + 1)

                # Process the body at increased nesting
                if name == "IfStatement":
                    _walk(getattr(node, "then_statement", None), nesting + 1)
                    else_stmt = getattr(node, "else_statement", None)
                    if else_stmt is not None:
                        if type(else_stmt).__name__ == "IfStatement":
                            # else-if: +1 structural, no nesting increase
                            score += 1
                            _walk(getattr(else_stmt, "then_statement", None),
                                  nesting + 1)
                            _walk(getattr(else_stmt, "else_statement", None),
                                  nesting)
                        else:
                            score += 1  # else branch
                            _walk(else_stmt, nesting + 1)
                    continue

                elif name in ("ForStatement", "WhileStatement", "DoStatement"):
                    _walk(getattr(node, "body", None), nesting + 1)
                    continue

                elif name == "SwitchStatement":
                    for case in (getattr(node, "cases", None) or []):
                        _walk(getattr(case, "statements", None), nesting + 1)
                    continue

                elif name == "CatchClause":
                    _walk(getattr(node, "block", None), nesting + 1)
                    continue

            elif name == "TernaryExpression":
                score += 1 + nesting
                max_depth = max(max_depth, nesting + 1)

            elif name == "BinaryOperation":
                op = getattr(node, "operator", "")
                if op in ("&&", "||"):
                    if op != prev_logical_op:
                        score += 1
                    prev_logical_op = op
                else:
                    prev_logical_op = None

            elif name == "LambdaExpression":
                _walk(getattr(node, "body", None), nesting + 1)
                continue

            elif name == "TryStatement":
                _walk(getattr(node, "block", None), nesting)
                for catch in (getattr(node, "catches", None) or []):
                    _walk([catch], nesting)
                _walk(getattr(node, "finally_block", None), nesting)
                continue

            # Recurse into children
            if hasattr(node, "children"):
                for child in node.children:
                    if child is not None and not isinstance(child, str):
                        if isinstance(child, (list, tuple)):
                            _walk(child, nesting)
                        elif hasattr(child, "children"):
                            _walk([child], nesting)

    _walk(method_body, 0)
    return score, max_depth


# ---------------------------------------------------------------------------
# Nesting depth measurement
# ---------------------------------------------------------------------------

def _max_nesting_depth(method_body) -> int:
    """Compute the maximum nesting depth of control structures."""
    if method_body is None:
        return 0

    max_depth = 0

    def _walk_depth(nodes, depth: int) -> None:
        nonlocal max_depth
        if nodes is None:
            return
        if not isinstance(nodes, (list, tuple)):
            nodes = [nodes]

        for node in nodes:
            if node is None:
                continue
            name = type(node).__name__
            if name in _NESTING_STRUCTURAL:
                new_depth = depth + 1
                max_depth = max(max_depth, new_depth)
                if hasattr(node, "children"):
                    for child in node.children:
                        if child is not None:
                            if isinstance(child, (list, tuple)):
                                _walk_depth(child, new_depth)
                            elif hasattr(child, "children"):
                                _walk_depth([child], new_depth)
            elif hasattr(node, "children"):
                for child in node.children:
                    if child is not None:
                        if isinstance(child, (list, tuple)):
                            _walk_depth(child, depth)
                        elif hasattr(child, "children"):
                            _walk_depth([child], depth)

    _walk_depth(method_body, 0)
    return max_depth


# ---------------------------------------------------------------------------
# Method line-count estimation
# ---------------------------------------------------------------------------

def _estimate_method_lines(method_node) -> int:
    """Estimate method line count from AST position attributes."""
    if method_node.body is None:
        return 1

    positions = []
    for node in _walk_tree(method_node.body):
        pos = getattr(node, "position", None)
        if pos is not None:
            positions.append(pos.line if hasattr(pos, "line") else pos[0])

    if len(positions) < 2:
        return 1
    return max(positions) - min(positions) + 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ComplexityAnalyzer:
    """Analyses cyclomatic and cognitive complexity of Java source files."""

    def analyse(self, source: str, filename: str = "<unknown>") -> FileComplexity:
        """
        Parse Java source and compute per-method complexity metrics.

        Args:
            source: Java source code string.
            filename: label for the result.

        Returns:
            FileComplexity with per-method breakdowns.
        """
        result = FileComplexity(filename=filename)

        try:
            tree = javalang.parse.parse(source)
        except Exception as exc:
            logger.warning("Parse failure in %s: %s", filename, exc)
            return result

        for _, class_node in tree.filter(ClassDeclaration):
            self._process_class(class_node, result)

        result.summarise()
        return result

    def _process_class(self, class_node: ClassDeclaration,
                       result: FileComplexity) -> None:
        """Extract complexity for every method in a class."""
        class_name = class_node.name

        for member in (class_node.body or []):
            if isinstance(member, (MethodDeclaration, ConstructorDeclaration)):
                mc = MethodComplexity(
                    class_name=class_name,
                    method_name=member.name,
                )
                mc.cyclomatic = _cyclomatic_complexity(member.body)
                cog, nest = _cognitive_complexity(member.body)
                mc.cognitive = cog
                mc.max_nesting_depth = nest
                mc.line_count = _estimate_method_lines(member)
                mc.parameter_count = len(member.parameters or [])
                result.methods.append(mc)


def analyse_file(filepath: str | Path) -> FileComplexity:
    """Convenience: read a .java file and return its complexity profile."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error("File not found: %s", filepath)
        return FileComplexity(filename=str(filepath))

    source = filepath.read_text(encoding="utf-8", errors="replace")
    analyzer = ComplexityAnalyzer()
    return analyzer.analyse(source, filename=str(filepath))
