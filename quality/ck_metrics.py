"""
ck_metrics.py — Chidamber–Kemerer OO design metrics via javalang AST.

Computes per-class:
    LOC   – Lines of Code (logical, excluding blanks/comments)
    WMC   – Weighted Methods per Class (sum of cyclomatic complexities)
    DIT   – Depth of Inheritance Tree
    NOC   – Number of Children (requires multi-file context)
    CBO   – Coupling Between Objects
    RFC   – Response For a Class
    LCOM  – Lack of Cohesion of Methods (Henderson-Sellers variant)

Reference: Chidamber & Kemerer, "A Metrics Suite for Object Oriented Design",
           IEEE TSE 20(6), 1994.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import javalang
from javalang.tree import (
    ClassCreator,
    ClassDeclaration,
    ConstructorDeclaration,
    FieldDeclaration,
    InterfaceDeclaration,
    MemberReference,
    MethodDeclaration,
    MethodInvocation,
    ReferenceType,
    SuperMethodInvocation,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MethodInfo:
    """Internal representation of a parsed method for metrics computation."""
    name: str
    complexity: int = 1
    accessed_fields: set = field(default_factory=set)
    called_methods: set = field(default_factory=set)
    parameter_types: list = field(default_factory=list)
    return_type: str = ""
    line_count: int = 0


@dataclass
class ClassMetrics:
    """Computed CK metrics for a single Java class."""
    class_name: str
    package_name: str = ""
    loc: int = 0
    wmc: int = 0
    dit: int = 0
    noc: int = 0
    cbo: int = 0
    rfc: int = 0
    lcom: int = 0
    num_methods: int = 0
    num_fields: int = 0
    parent_class: str = ""

    def to_dict(self) -> dict:
        """Serialise to a flat dictionary for CSV output."""
        return {
            "class_name": self.class_name,
            "package_name": self.package_name,
            "loc": self.loc,
            "wmc": self.wmc,
            "dit": self.dit,
            "noc": self.noc,
            "cbo": self.cbo,
            "rfc": self.rfc,
            "lcom": self.lcom,
            "num_methods": self.num_methods,
            "num_fields": self.num_fields,
            "parent_class": self.parent_class,
        }


# ---------------------------------------------------------------------------
# Java standard-library types to exclude from CBO
# ---------------------------------------------------------------------------

_JAVA_STD_TYPES: frozenset[str] = frozenset({
    "String", "Object", "Integer", "Long", "Double", "Float", "Boolean",
    "Byte", "Short", "Character", "Void", "Number",
    "List", "ArrayList", "LinkedList",
    "Map", "HashMap", "TreeMap", "LinkedHashMap", "ConcurrentHashMap",
    "Set", "HashSet", "TreeSet", "LinkedHashSet",
    "Queue", "Deque", "ArrayDeque", "PriorityQueue",
    "Collection", "Collections", "Arrays", "Iterator", "Iterable",
    "Optional", "Stream", "Collectors",
    "StringBuilder", "StringBuffer", "Math", "System",
    "Exception", "RuntimeException", "Error", "Throwable",
    "IOException", "FileNotFoundException",
    "NullPointerException", "IllegalArgumentException",
    "IllegalStateException", "UnsupportedOperationException",
    "IndexOutOfBoundsException", "ClassCastException",
    "Comparable", "Comparator", "Serializable", "Cloneable",
    "Runnable", "Callable", "Thread", "Future",
    "Date", "Calendar", "SimpleDateFormat",
    "File", "Path", "InputStream", "OutputStream",
    "Reader", "Writer", "BufferedReader", "BufferedWriter",
    "PrintWriter", "PrintStream", "FileReader", "FileWriter",
    "URL", "HttpURLConnection",
    "Override", "Deprecated", "SuppressWarnings", "FunctionalInterface",
})


# ---------------------------------------------------------------------------
# AST walker utilities
# ---------------------------------------------------------------------------

def _walk_tree(node):
    """Recursively yield all nodes in a javalang AST subtree."""
    if node is None:
        return
    yield node
    if isinstance(node, (list, tuple)):
        for child in node:
            yield from _walk_tree(child)
    elif hasattr(node, "children"):
        for child in node.children:
            yield from _walk_tree(child)


def _count_complexity(method_node) -> int:
    """
    Compute cyclomatic complexity for a single method/constructor body.

    Counts decision points: if, for, while, do, case, catch, &&, ||,
    ternary (?:), plus the base path of 1.
    """
    complexity = 1  # base path

    if method_node.body is None:
        return complexity

    for node in _walk_tree(method_node.body):
        node_type = type(node).__name__
        if node_type in ("IfStatement", "ForStatement", "WhileStatement",
                         "DoStatement", "CatchClause"):
            complexity += 1
        elif node_type == "SwitchStatementCase":
            # default case doesn't add a decision
            if getattr(node, "case", None) is not None:
                complexity += 1
        elif node_type == "TernaryExpression":
            complexity += 1
        elif node_type == "BinaryOperation":
            op = getattr(node, "operator", "")
            if op in ("&&", "||"):
                complexity += 1

    return complexity


def _extract_field_names(class_node) -> set[str]:
    """Return the set of field names declared in a class."""
    names: set[str] = set()
    for field_decl in (class_node.fields or []):
        if isinstance(field_decl, FieldDeclaration):
            for declarator in field_decl.declarators:
                names.add(declarator.name)
    return names


def _extract_accessed_fields(method_node, class_fields: set[str]) -> set[str]:
    """Find which class fields a method body reads or writes."""
    accessed: set[str] = set()
    if method_node.body is None:
        return accessed

    for node in _walk_tree(method_node.body):
        if isinstance(node, MemberReference):
            name = getattr(node, "member", "")
            qualifier = getattr(node, "qualifier", "")
            # this.field or just field (no qualifier or qualifier == 'this')
            if name in class_fields and (not qualifier or qualifier == "this"):
                accessed.add(name)

    return accessed


def _extract_called_methods(method_node) -> set[str]:
    """Return the set of method names invoked within a method body."""
    called: set[str] = set()
    if method_node.body is None:
        return called

    for node in _walk_tree(method_node.body):
        if isinstance(node, MethodInvocation):
            called.add(node.member)
        elif isinstance(node, SuperMethodInvocation):
            called.add(f"super.{node.member}")

    return called


def _extract_referenced_types(node) -> set[str]:
    """
    Collect all non-primitive type names referenced in an AST subtree.
    Returns all reference types found; caller is responsible for filtering
    stdlib types based on configuration.
    """
    types: set[str] = set()
    for child in _walk_tree(node):
        if isinstance(child, ReferenceType):
            name = child.name
            if name:
                types.add(name)
        elif isinstance(child, ClassCreator):
            ctype = getattr(child, "type", None)
            if ctype:
                name = ctype.name if hasattr(ctype, "name") else str(ctype)
                if name:
                    types.add(name)
    return types


# ---------------------------------------------------------------------------
# LOC counter
# ---------------------------------------------------------------------------

_LINE_COMMENT_RE = re.compile(r"^\s*//")
_BLANK_RE = re.compile(r"^\s*$")


def count_logical_loc(source: str) -> int:
    """
    Count logical lines of code: total lines minus blank lines,
    single-line comments, and lines that are purely block-comment
    delimiters.  Approximation, but consistent and fast.
    """
    lines = source.splitlines()
    loc = 0
    in_block = False
    for line in lines:
        stripped = line.strip()
        if in_block:
            if "*/" in stripped:
                in_block = False
            continue
        if stripped.startswith("/*"):
            if "*/" not in stripped or stripped.endswith("*/"):
                in_block = "*/" not in stripped
            continue
        if _BLANK_RE.match(line):
            continue
        if _LINE_COMMENT_RE.match(line):
            continue
        loc += 1
    return loc


# ---------------------------------------------------------------------------
# DIT computation helpers
# ---------------------------------------------------------------------------

def _resolve_dit(class_name: str, parent_map: dict[str, str],
                 cache: dict[str, int], depth: int = 0) -> int:
    """Recursively resolve depth of inheritance tree."""
    if depth > 50:  # guard against cycles
        return depth
    if class_name in cache:
        return cache[class_name]
    parent = parent_map.get(class_name)
    if parent is None or parent == "Object":
        cache[class_name] = 0
        return 0
    dit = 1 + _resolve_dit(parent, parent_map, cache, depth + 1)
    cache[class_name] = dit
    return dit


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class CKMetricsExtractor:
    """
    Extracts Chidamber–Kemerer metrics from Java source code using the
    javalang AST parser.

    Usage::

        extractor = CKMetricsExtractor()
        metrics = extractor.extract(source_code)
        # metrics is a list of ClassMetrics, one per class in the file
    """

    def __init__(self, *, include_stdlib_coupling: bool = False):
        """
        Args:
            include_stdlib_coupling: if True, count java.* types in CBO.
                Default False excludes them for a cleaner project-coupling view.
        """
        self.include_stdlib_coupling = include_stdlib_coupling

    def extract(self, source: str, filename: str = "<unknown>") -> list[ClassMetrics]:
        """
        Parse *source* and return CK metrics for every class found.

        Args:
            source: Java source code as a string.
            filename: used for error messages only.

        Returns:
            List of ClassMetrics, one per class/interface.  Empty list if
            the file cannot be parsed.
        """
        try:
            tree = javalang.parse.parse(source)
        except javalang.parser.JavaSyntaxError as exc:
            logger.warning("Syntax error in %s: %s", filename, exc)
            return []
        except Exception as exc:
            logger.warning("Parse failure in %s: %s", filename, exc)
            return []

        package_name = tree.package.name if tree.package else ""
        file_loc = count_logical_loc(source)

        # First pass: build parent map and collect class nodes
        parent_map: dict[str, str] = {}
        class_nodes: list[tuple[str, ClassDeclaration | InterfaceDeclaration]] = []
        children_count: dict[str, int] = {}

        for _, node in tree.filter(ClassDeclaration):
            name = node.name
            parent = None
            if node.extends:
                parent = node.extends.name if hasattr(node.extends, "name") else str(node.extends)
            parent_map[name] = parent
            class_nodes.append((name, node))
            # Track children
            if parent and parent != "Object":
                children_count[parent] = children_count.get(parent, 0) + 1

        for _, node in tree.filter(InterfaceDeclaration):
            name = node.name
            parent_map[name] = None
            class_nodes.append((name, node))

        # Resolve DITs
        dit_cache: dict[str, int] = {}
        for cname in parent_map:
            _resolve_dit(cname, parent_map, dit_cache)

        # Second pass: compute per-class metrics
        results: list[ClassMetrics] = []
        for class_name, class_node in class_nodes:
            metrics = self._analyse_class(
                class_name, class_node, package_name, dit_cache,
                children_count, file_loc, len(class_nodes),
            )
            results.append(metrics)

        return results

    def _analyse_class(
        self,
        class_name: str,
        class_node,
        package_name: str,
        dit_cache: dict[str, int],
        children_count: dict[str, int],
        file_loc: int,
        num_classes: int,
    ) -> ClassMetrics:
        """Compute all CK metrics for one class node."""

        # ---- Fields ----
        class_fields = _extract_field_names(class_node)

        # ---- Methods (including constructors) ----
        methods: list[MethodInfo] = []

        method_nodes = []
        for member in (class_node.body or []):
            if isinstance(member, (MethodDeclaration, ConstructorDeclaration)):
                method_nodes.append(member)

        for m_node in method_nodes:
            mi = MethodInfo(name=m_node.name)
            mi.complexity = _count_complexity(m_node)
            mi.accessed_fields = _extract_accessed_fields(m_node, class_fields)
            mi.called_methods = _extract_called_methods(m_node)

            # Parameter types
            for param in (m_node.parameters or []):
                ptype = param.type
                if hasattr(ptype, "name"):
                    mi.parameter_types.append(ptype.name)

            # Return type
            if isinstance(m_node, MethodDeclaration) and m_node.return_type:
                rt = m_node.return_type
                mi.return_type = rt.name if hasattr(rt, "name") else str(rt)

            methods.append(mi)

        # ---- WMC: sum of cyclomatic complexities ----
        wmc = sum(m.complexity for m in methods)

        # ---- DIT ----
        dit = dit_cache.get(class_name, 0)

        # ---- NOC ----
        noc = children_count.get(class_name, 0)

        # ---- CBO: count of distinct external types referenced ----
        referenced = _extract_referenced_types(class_node)
        # Remove self-reference
        referenced.discard(class_name)
        if not self.include_stdlib_coupling:
            referenced -= _JAVA_STD_TYPES
        # Remove primitive-like names that aren't real types
        referenced -= {"java", "util", "io", "net", "lang", "text"}
        cbo = len(referenced)

        # ---- RFC: number of distinct methods in the response set ----
        # Response set = own methods + methods directly called by own methods
        own_method_names = {m.name for m in methods}
        called_from_all = set()
        for m in methods:
            called_from_all |= m.called_methods
        rfc = len(own_method_names | called_from_all)

        # ---- LCOM (Henderson-Sellers) ----
        lcom = self._compute_lcom(methods, class_fields)

        # ---- LOC: approximate per-class if multiple classes in file ----
        if num_classes > 1:
            loc = max(1, file_loc // num_classes)
        else:
            loc = file_loc

        # ---- Parent class ----
        parent_class = ""
        if isinstance(class_node, ClassDeclaration) and class_node.extends:
            parent_class = (class_node.extends.name
                            if hasattr(class_node.extends, "name")
                            else str(class_node.extends))

        return ClassMetrics(
            class_name=class_name,
            package_name=package_name,
            loc=loc,
            wmc=wmc,
            dit=dit,
            noc=noc,
            cbo=cbo,
            rfc=rfc,
            lcom=lcom,
            num_methods=len(methods),
            num_fields=len(class_fields),
            parent_class=parent_class,
        )

    @staticmethod
    def _compute_lcom(methods: list[MethodInfo], fields: set[str]) -> int:
        """
        LCOM using the Henderson-Sellers definition:

            LCOM* = (1/|F|) * Σ_f (|M| - μ(f)) / (|M| - 1)

        where μ(f) = number of methods accessing field f,
              |M| = total methods, |F| = total fields.

        Simplification used here (LCOM1-style count for robustness):
        Count pairs of methods sharing no fields vs pairs sharing ≥1 field.
        LCOM = max(0, P - Q) where P = non-sharing pairs, Q = sharing pairs.

        Falls back to 0 when fewer than 2 methods or 0 fields.
        """
        if len(methods) < 2 or len(fields) == 0:
            return 0

        sharing = 0
        not_sharing = 0
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                if methods[i].accessed_fields & methods[j].accessed_fields:
                    sharing += 1
                else:
                    not_sharing += 1

        return max(0, not_sharing - sharing)


# ---------------------------------------------------------------------------
# Convenience: analyse a file on disk
# ---------------------------------------------------------------------------

def extract_from_file(filepath: str | Path,
                      include_stdlib_coupling: bool = False) -> list[ClassMetrics]:
    """
    Read a .java file and return CK metrics for all classes in it.

    Args:
        filepath: path to the .java source file.
        include_stdlib_coupling: whether to count java.* types in CBO.

    Returns:
        List of ClassMetrics.  Empty if the file cannot be read or parsed.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error("File not found: %s", filepath)
        return []

    source = filepath.read_text(encoding="utf-8", errors="replace")
    extractor = CKMetricsExtractor(include_stdlib_coupling=include_stdlib_coupling)
    return extractor.extract(source, filename=str(filepath))


def extract_from_directory(directory: str | Path,
                           include_stdlib_coupling: bool = False,
                           recursive: bool = True) -> dict[str, list[ClassMetrics]]:
    """
    Scan a directory for .java files and extract CK metrics from each.

    Args:
        directory: root directory to scan.
        include_stdlib_coupling: whether to count java.* types in CBO.
        recursive: if True, scan subdirectories.

    Returns:
        Dict mapping relative file paths to their ClassMetrics lists.
    """
    directory = Path(directory)
    if not directory.is_dir():
        logger.error("Not a directory: %s", directory)
        return {}

    pattern = "**/*.java" if recursive else "*.java"
    results: dict[str, list[ClassMetrics]] = {}
    extractor = CKMetricsExtractor(include_stdlib_coupling=include_stdlib_coupling)

    for java_file in sorted(directory.glob(pattern)):
        try:
            source = java_file.read_text(encoding="utf-8", errors="replace")
            metrics = extractor.extract(source, filename=str(java_file))
            rel_path = str(java_file.relative_to(directory))
            results[rel_path] = metrics
        except Exception as exc:
            logger.warning("Error processing %s: %s", java_file, exc)

    return results
