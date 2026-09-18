"""
mcp_servers.tools.godclass — God Class detection, planning and scoring.

Wraps C3's :mod:`quality.refactoring` and adds a decomposition planner.

**The planner is static analysis, not a model call.** It clusters members by
what they touch and returns a plan; it is deterministic for a given input,
costs nothing, and writes no code. That is a deliberate choice: a tool an
agent calls speculatively must not make a paid API call, and a planner that
asks a model which methods belong together is not evidence of anything —
the agent could have asked the model itself.

Three strategies:

``field_clusters`` (default)
    Partition methods by the fields they access, using connected components
    over the method-field bipartite graph. This is LCOM4's own definition of
    cohesion turned into a decomposition: each component is, by that
    measure, an independent responsibility.

``responsibility``
    Cluster by shared method-name prefix (``loadX``/``saveX``/``validateX``).
    Cheaper and cruder than field clustering, but it works on classes whose
    methods are static and touch no fields, where ``field_clusters`` finds
    nothing to work with.

``layered``
    Split accessors from behaviour: trivial getters and setters into one
    class, everything else into another. Useful for a data class that has
    accreted logic; useless for anything else.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from mcp_servers.models import (
    ClassMetricsModel,
    DetectGodClassesResult,
    ExtractedClassPlan,
    GodClassFinding,
    MetricDeltaModel,
    ProposeDecompositionResult,
    ScoreDecompositionResult,
    SourceFile,
)
from mcp_servers.tools.jvm import (
    ToolInputError,
    _read_directory,
    _severity_for,
    _to_metrics_model,
    _validate_sources,
)

logger = logging.getLogger(__name__)

__all__ = [
    "detect_god_classes",
    "propose_decomposition",
    "score_decomposition",
    "STRATEGIES",
]

#: Strategies ``propose_decomposition`` accepts.
STRATEGIES = ("field_clusters", "responsibility", "layered")

#: Thresholds from quality/refactoring.py. A class is a God Class when it
#: breaches at least 2 of these 4.
_GOD_CLASS_THRESHOLDS = {"wmc": 47, "lcom": 1, "cbo": 14, "rfc": 50}

_ACCESSOR = re.compile(r"^(get|set|is|has)[A-Z_]")
_VERB_PREFIX = re.compile(r"^(?P<verb>[a-z]+)(?P<rest>[A-Z].*)?$")


# ──────────────────────────────────────────────────────────────────────
# detect_god_classes
# ──────────────────────────────────────────────────────────────────────

def _violations_for(metrics: ClassMetricsModel) -> dict[str, float]:
    """Which God Class thresholds *metrics* breaches, and by what ratio."""
    return {
        name: getattr(metrics, name) / threshold
        for name, threshold in _GOD_CLASS_THRESHOLDS.items()
        if getattr(metrics, name) > threshold
    }


def detect_god_classes(
    project_path: str = "",
    source_files: list[SourceFile] | None = None,
) -> DetectGodClassesResult:
    """Find God Classes in Java sources. Parses only — nothing is executed.

    Supply either *project_path* or *source_files*, not both.
    """
    if bool(project_path) == bool(source_files):
        return DetectGodClassesResult(
            ok=False,
            error="Supply exactly one of project_path or source_files.",
        )

    try:
        if project_path:
            files, _ = _read_directory(project_path)
        else:
            files = _validate_sources(source_files or [])
    except ToolInputError as exc:
        return DetectGodClassesResult(ok=False, error=str(exc))

    from quality.ck_metrics import CKMetricsExtractor

    extractor = CKMetricsExtractor()
    findings: list[GodClassFinding] = []
    failures: list[str] = []
    analysed = 0

    for name, content in files.items():
        try:
            extracted = extractor.extract(content, name)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        for item in extracted:
            analysed += 1
            metrics = _to_metrics_model(item)
            violations = _violations_for(metrics)
            if len(violations) < 2:
                continue
            findings.append(
                GodClassFinding(
                    class_name=metrics.class_name,
                    file=name,
                    severity=_severity_for(max(violations.values())),  # type: ignore[arg-type]
                    violations=sorted(violations),
                    metrics=metrics,
                )
            )

    order = {"extreme": 0, "severe": 1, "moderate": 2}
    findings.sort(key=lambda f: (order[f.severity], -len(f.violations), f.class_name))

    return DetectGodClassesResult(
        ok=True,
        god_classes=findings,
        classes_analysed=analysed,
        files_analysed=len(files),
        parse_failures=failures,
    )


# ──────────────────────────────────────────────────────────────────────
# propose_decomposition
# ──────────────────────────────────────────────────────────────────────

def _parse_class(source: str, filename: str):
    """Return (javalang class node, field names) for the first class declared."""
    import javalang

    tree = javalang.parse.parse(source)
    for _, node in tree.filter(javalang.tree.ClassDeclaration):
        fields = {
            declarator.name
            for member in node.body
            if isinstance(member, javalang.tree.FieldDeclaration)
            for declarator in member.declarators
        }
        return node, fields
    return None, set()


def _method_field_map(
    class_node, field_names: set[str]
) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Analyse every method in the class.

    Returns ``({method: fields it references}, {method: cyclomatic
    complexity})``. Both are keyed by method *name*, so an overloaded method
    collapses into one entry — acceptable, because overloads of the same
    name virtually always belong in the same extracted class.
    """
    import javalang

    from quality.ck_metrics import _count_complexity, _extract_accessed_fields

    fields_used: dict[str, set[str]] = {}
    complexity: dict[str, int] = {}
    for member in class_node.body:
        if not isinstance(member, javalang.tree.MethodDeclaration):
            continue
        try:
            accessed = _extract_accessed_fields(member, field_names)
        except Exception:
            accessed = set()
        fields_used[member.name] = fields_used.get(member.name, set()) | accessed
        try:
            complexity[member.name] = complexity.get(member.name, 0) + _count_complexity(member)
        except Exception:
            complexity[member.name] = complexity.get(member.name, 0) + 1
    return fields_used, complexity


def _connected_components(
    method_fields: dict[str, set[str]]
) -> list[tuple[list[str], set[str]]]:
    """Group methods that share at least one field, transitively.

    This is LCOM4: the components are the class's independent
    responsibilities. Methods touching no field form their own component
    each, which is why ``propose_decomposition`` falls back to another
    strategy when that dominates.
    """
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for method, fields in method_fields.items():
        find(f"m:{method}")
        for field in fields:
            union(f"m:{method}", f"f:{field}")

    groups: dict[str, tuple[list[str], set[str]]] = {}
    for method, fields in method_fields.items():
        root = find(f"m:{method}")
        methods, field_set = groups.setdefault(root, ([], set()))
        methods.append(method)
        field_set.update(fields)

    return sorted(groups.values(), key=lambda g: (-len(g[0]), g[0][0] if g[0] else ""))


def _name_for(methods: list[str], fields: set[str], class_name: str, index: int) -> str:
    """Derive a plausible class name from what the cluster holds."""
    if fields:
        stem = sorted(fields)[0].lstrip("_")
        stem = re.sub(r"(List|Map|Set|Array|Count|Cache|s)$", "", stem) or stem
        return f"{stem[:1].upper()}{stem[1:]}Manager"
    if methods:
        match = _VERB_PREFIX.match(methods[0])
        if match and match.group("rest"):
            return f"{match.group('rest')}Service"
        return f"{methods[0][:1].upper()}{methods[0][1:]}Service"
    return f"{class_name}Part{index + 1}"


def _cluster_by_field(method_fields, class_name) -> list[ExtractedClassPlan]:
    plans: list[ExtractedClassPlan] = []
    for index, (methods, fields) in enumerate(_connected_components(method_fields)):
        plans.append(
            ExtractedClassPlan(
                proposed_name=_name_for(methods, fields, class_name, index),
                methods=sorted(methods),
                fields=sorted(fields),
                rationale=(
                    f"{len(methods)} method(s) form a connected component over "
                    f"{len(fields)} field(s); no method here touches a field "
                    f"used by another cluster."
                ),
                estimated_wmc=sum(_METHOD_COMPLEXITY.get(m, 1) for m in methods),
            )
        )
    return plans


def _cluster_by_responsibility(method_fields, class_name) -> list[ExtractedClassPlan]:
    buckets: dict[str, list[str]] = {}
    for method in method_fields:
        match = _VERB_PREFIX.match(method)
        verb = match.group("verb") if match else "misc"
        buckets.setdefault(verb, []).append(method)

    plans: list[ExtractedClassPlan] = []
    for index, (verb, methods) in enumerate(
        sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ):
        fields: set[str] = set()
        for method in methods:
            fields |= method_fields[method]
        plans.append(
            ExtractedClassPlan(
                proposed_name=f"{verb[:1].upper()}{verb[1:]}{class_name}",
                methods=sorted(methods),
                fields=sorted(fields),
                rationale=f"{len(methods)} method(s) share the '{verb}' verb prefix.",
                estimated_wmc=sum(_METHOD_COMPLEXITY.get(m, 1) for m in methods),
            )
        )
    return plans


def _cluster_layered(method_fields, class_name) -> list[ExtractedClassPlan]:
    accessors = sorted(m for m in method_fields if _ACCESSOR.match(m))
    behaviour = sorted(m for m in method_fields if not _ACCESSOR.match(m))

    plans: list[ExtractedClassPlan] = []
    if accessors:
        fields: set[str] = set()
        for method in accessors:
            fields |= method_fields[method]
        plans.append(
            ExtractedClassPlan(
                proposed_name=f"{class_name}Data",
                methods=accessors,
                fields=sorted(fields),
                rationale=f"{len(accessors)} accessor(s): state, with no behaviour.",
                estimated_wmc=sum(_METHOD_COMPLEXITY.get(m, 1) for m in accessors),
            )
        )
    if behaviour:
        fields = set()
        for method in behaviour:
            fields |= method_fields[method]
        plans.append(
            ExtractedClassPlan(
                proposed_name=f"{class_name}Service",
                methods=behaviour,
                fields=sorted(fields),
                rationale=f"{len(behaviour)} non-accessor method(s): the behaviour.",
                estimated_wmc=sum(_METHOD_COMPLEXITY.get(m, 1) for m in behaviour),
            )
        )
    return plans


def propose_decomposition(
    source: str = "",
    class_path: str = "",
    strategy: str = "field_clusters",
) -> ProposeDecompositionResult:
    """Plan a decomposition of one God Class by static analysis.

    Returns which members to extract and why. Writes no code, makes no API
    call, and is deterministic for a given input.
    """
    if strategy not in STRATEGIES:
        return ProposeDecompositionResult(
            ok=False,
            error=f"Unknown strategy {strategy!r}; expected one of {list(STRATEGIES)}.",
        )
    if bool(source) == bool(class_path):
        return ProposeDecompositionResult(
            ok=False, error="Supply exactly one of source or class_path."
        )

    filename = "Input.java"
    if class_path:
        path = Path(class_path).expanduser()
        if not path.is_file():
            return ProposeDecompositionResult(
                ok=False, error=f"{class_path!r} is not a readable file."
            )
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return ProposeDecompositionResult(
                ok=False, error=f"Could not read {class_path!r}: {exc}"
            )
        filename = path.name

    from quality.ck_metrics import CKMetricsExtractor

    try:
        class_node, field_names = _parse_class(source, filename)
    except Exception as exc:
        return ProposeDecompositionResult(
            ok=False, error=f"Could not parse Java source: {type(exc).__name__}: {exc}"
        )
    if class_node is None:
        return ProposeDecompositionResult(
            ok=False, error="No class declaration found in the source."
        )

    class_name = class_node.name
    _METHOD_COMPLEXITY.clear()
    method_fields = _method_field_map(class_node, field_names)

    warnings: list[str] = []
    if not method_fields:
        return ProposeDecompositionResult(
            ok=False,
            error=f"{class_name} declares no methods; there is nothing to decompose.",
        )

    effective = strategy
    if strategy == "field_clusters" and not any(method_fields.values()):
        effective = "responsibility"
        warnings.append(
            "No method in this class accesses a field, so field clustering "
            "has nothing to work with; fell back to the 'responsibility' "
            "strategy. This is normal for a class of static helpers."
        )

    builders = {
        "field_clusters": _cluster_by_field,
        "responsibility": _cluster_by_responsibility,
        "layered": _cluster_layered,
    }
    plans = builders[effective](method_fields, class_name)

    metrics_before = None
    is_god_class = False
    try:
        extracted = CKMetricsExtractor().extract(source, filename)
        if extracted:
            metrics_before = _to_metrics_model(extracted[0])
            is_god_class = len(_violations_for(metrics_before)) >= 2
    except Exception as exc:
        warnings.append(f"Could not compute CK metrics: {exc}")

    if not is_god_class:
        warnings.append(
            f"{class_name} breaches fewer than 2 of the 4 God Class "
            f"thresholds, so decomposing it may not be warranted."
        )
    if len(plans) == 1:
        warnings.append(
            "The analysis found a single cohesive cluster: this class does "
            "not split cleanly along the axis this strategy measures. Try "
            "another strategy before splitting it anyway."
        )

    unassigned = set(method_fields) - {m for plan in plans for m in plan.methods}
    if unassigned:
        warnings.append(f"Methods left unassigned: {sorted(unassigned)}")

    rationale = {
        "field_clusters": (
            "Methods were partitioned into connected components over the "
            "fields they access — the same relation LCOM4 measures, so each "
            "component is an independent responsibility by that definition."
        ),
        "responsibility": (
            "Methods were grouped by shared verb prefix. This reads intent "
            "from naming, so it is only as good as the naming."
        ),
        "layered": (
            "Trivial accessors were separated from behaviour, on the "
            "assumption this is a data class that accreted logic."
        ),
    }[effective]

    return ProposeDecompositionResult(
        ok=True,
        class_name=class_name,
        strategy=effective,
        is_god_class=is_god_class,
        extracted_classes=plans,
        rationale=rationale,
        warnings=warnings,
        metrics_before=metrics_before,
    )


# ──────────────────────────────────────────────────────────────────────
# score_decomposition
# ──────────────────────────────────────────────────────────────────────

def score_decomposition(
    before: str,
    after: list[str],
) -> ScoreDecompositionResult:
    """Score a decomposition on structural improvement and preservation.

    Test results are not part of this score. A refactoring that changes
    nothing passes every test, so behaviour preservation is necessary and
    not sufficient; run the suite separately and require both.
    """
    if not before or not before.strip():
        return ScoreDecompositionResult(ok=False, error="`before` is empty.")
    if not after:
        return ScoreDecompositionResult(
            ok=False, error="`after` is empty; supply at least one decomposed class."
        )
    if any(not source.strip() for source in after):
        return ScoreDecompositionResult(
            ok=False,
            error="One of the `after` sources is empty. An empty extracted "
                  "class is the classic way to game a structural metric, so "
                  "it is rejected rather than scored.",
        )

    from quality.refactoring import RefactoringEvaluator

    try:
        result = RefactoringEvaluator().evaluate(
            original_source=before,
            decomposed_sources=list(after),
        )
    except Exception as exc:
        return ScoreDecompositionResult(
            ok=False, error=f"Evaluation failed: {type(exc).__name__}: {exc}"
        )

    deltas = [
        MetricDeltaModel(
            metric=delta.metric_name,
            before=delta.before,
            after=delta.after,
            delta=round(delta.delta, 3),
            pct_change=round(delta.pct_change, 2),
            improved=delta.improved,
        )
        for delta in result.deltas
    ]
    by_metric = {delta.metric: delta for delta in deltas}

    warnings = list(result.warnings)
    if result.method_coverage < 1.0:
        warnings.append(
            f"method_coverage is {result.method_coverage:.2f}: "
            f"{(1 - result.method_coverage) * 100:.0f}% of the original's "
            f"methods are absent after. Behaviour was dropped, not moved."
        )
    if result.field_coverage < 1.0:
        warnings.append(
            f"field_coverage is {result.field_coverage:.2f}: some of the "
            f"original's state is absent after."
        )

    return ScoreDecompositionResult(
        ok=True,
        original_class=result.original_class,
        decomposed_classes=list(result.decomposed_classes),
        overall_score=round(result.overall_score, 2),
        is_god_class=result.is_god_class,
        god_class_resolved=result.god_class_resolved,
        loc_delta=by_metric["loc"].delta if "loc" in by_metric else 0.0,
        coupling_delta=by_metric["cbo"].delta if "cbo" in by_metric else 0.0,
        method_coverage=round(result.method_coverage, 4),
        field_coverage=round(result.field_coverage, 4),
        deltas=deltas,
        warnings=warnings,
    )
