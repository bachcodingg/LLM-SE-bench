"""
quality.structural — behaviour preservation and structural scoring (M5).

Almost every benchmark scores pass/fail on tests. For refactoring that is
the wrong objective function, and wrong in a specific, exploitable way: **a
refactoring that changes nothing passes every test.** Behaviour preservation
is necessary and nowhere near sufficient.

So a refactoring is scored on three independent axes and they are reported
separately:

1. **Behaviour preserved** — the suite still passes, the public API is
   unchanged, and pre/post versions agree on generated inputs.
2. **Structure improved** — CK metric deltas, with published weights.
3. **Not gamed** — extracted classes have real behaviour, nothing was
   deleted rather than moved, complexity was dispersed rather than hidden,
   and the facade that keeps the public API delegates instead of keeping
   the logic.

Coupling and cohesion are measured on the *extracted* classes. The facade
is coupled to every class it delegates to by design, so including it would
score every API-preserving decomposition as worse than the original.

The composite is reported *alongside* the per-axis deltas, never instead of
them. A single number hides disagreement between axes, and disagreement
between axes is usually the interesting part: a decomposition that halves
coupling while doubling total complexity is a real finding, and a composite
that averages it away destroys it.

Weights are module constants, not magic numbers buried in a function, so a
paper can state them and a reader can change them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from typing import Any

from quality.ck_metrics import CKMetricsExtractor, ClassMetrics

logger = logging.getLogger(__name__)

__all__ = [
    "SCORE_WEIGHTS",
    "MethodSignature",
    "ApiDiff",
    "StructuralScore",
    "extract_public_api",
    "diff_public_api",
    "score_structure",
    "explain_score",
    "GamingGuard",
]

#: Published weights for the composite structural score. They sum to 1.0.
#: Coupling is weighted heaviest because it is the axis a decomposition is
#: usually *for*; size is weighted lightest and is penalised when it falls
#: through deletion rather than movement.
SCORE_WEIGHTS: dict[str, float] = {
    "coupling_reduction": 0.30,     # CBO
    "cohesion_improvement": 0.30,   # LCOM
    "complexity_dispersal": 0.25,   # WMC spread across extracted classes
    "size_reduction": 0.15,         # LOC, penalised if it is just deletion
}

#: A class smaller than this, with no branching, is a shell rather than a
#: responsibility. Splitting into shells improves every CK metric at once.
MIN_MEANINGFUL_WMC = 2
MIN_MEANINGFUL_METHODS = 1

#: Share of the original's branching the API-preserving facade may keep.
#: A facade that delegates has one path per method; one that kept more than
#: this did not move the logic it claims to have extracted.
FACADE_MAX_RETAINED_BRANCHING = 0.5

_MODIFIER_WORDS = frozenset({
    "public", "protected", "private", "static", "final", "abstract",
    "synchronized", "native", "strictfp", "default",
})

#: Everything before a method name, then the name and its parameter list.
#: Modifiers and the return type are *not* split here: a return type can
#: contain generics, and a character class permissive enough for those is
#: permissive enough to swallow the modifiers too. The prefix is captured
#: whole and split token-wise afterwards, which is unambiguous.
#:
#: Generics containing spaces (``Map<String, Integer>``) are not matched.
#: They are rare in a return position and the cost of missing one is a
#: method absent from the diff, which is visible, rather than a wrong
#: signature, which is not.
_METHOD = re.compile(
    r"(?P<prefix>(?:[\w<>\[\],.$]+\s+)+)"
    r"(?P<name>\w+)\s*\((?P<params>[^)]*)\)\s*(?:throws\s+[\w,\s.]+)?\s*[{;]"
)


# ──────────────────────────────────────────────────────────────────────
# Public API diffing
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MethodSignature:
    """One public method, by the parts a caller depends on."""

    name: str
    return_type: str
    parameter_types: tuple[str, ...]
    is_static: bool = False

    def __str__(self) -> str:
        static = "static " if self.is_static else ""
        return f"{static}{self.return_type} {self.name}({', '.join(self.parameter_types)})"


@dataclass
class ApiDiff:
    """What changed in the public surface across a refactoring.

    A refactoring that removes a public method has changed behaviour for
    every caller outside the test suite — which the test suite, by
    construction, cannot tell you.
    """

    removed: list[MethodSignature] = field(default_factory=list)
    added: list[MethodSignature] = field(default_factory=list)
    retained: list[MethodSignature] = field(default_factory=list)

    @property
    def breaking(self) -> bool:
        """True when something callers depended on is gone."""
        return bool(self.removed)

    @property
    def preserved_fraction(self) -> float:
        """Share of the original public API that survived, 0.0-1.0."""
        original = len(self.retained) + len(self.removed)
        return len(self.retained) / original if original else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "breaking": self.breaking,
            "preserved_fraction": round(self.preserved_fraction, 4),
            "removed": [str(s) for s in self.removed],
            "added": [str(s) for s in self.added],
            "retained_count": len(self.retained),
        }


def _normalise_type(raw: str) -> str:
    """Strip whitespace and package qualification from a type name.

    ``java.util.List<String>`` and ``List<String>`` are the same dependency
    from a caller's point of view, and a refactoring that changes only the
    import should not read as an API break.
    """
    cleaned = re.sub(r"\s+", "", raw.strip())
    if not cleaned:
        return ""
    base, _, generics = cleaned.partition("<")
    base = base.rsplit(".", 1)[-1]
    return f"{base}<{generics}" if generics else base


def extract_public_api(source: str) -> set[MethodSignature]:
    """Public and protected method signatures declared in *source*.

    Regex rather than a full parse: this runs over sources that may not
    compile, which is precisely when a parser is least useful. Constructors
    and control-flow keywords are excluded explicitly, because they match
    the same shape.
    """
    keywords = {"if", "for", "while", "switch", "catch", "return", "new", "synchronized"}
    signatures: set[MethodSignature] = set()

    class_names = set(re.findall(r"\b(?:class|interface|enum)\s+(\w+)", source))

    for match in _METHOD.finditer(source):
        tokens = match.group("prefix").split()
        modifiers = {token for token in tokens if token in _MODIFIER_WORDS}
        remainder = [token for token in tokens if token not in _MODIFIER_WORDS]

        if "private" in modifiers:
            continue
        # Package-private methods are not part of the published surface.
        if not modifiers & {"public", "protected"}:
            continue
        # The return type is the last non-modifier token. More than one
        # means the prefix ran into the previous statement; skip rather
        # than guess.
        if not remainder:
            continue

        name = match.group("name")
        if name in keywords or name in class_names:
            continue

        return_type = _normalise_type(remainder[-1])
        if not return_type or return_type in keywords:
            continue

        parameters = tuple(
            _normalise_type(part.strip().rsplit(" ", 1)[0])
            for part in match.group("params").split(",")
            if part.strip()
        )
        signatures.add(MethodSignature(
            name=name,
            return_type=return_type,
            parameter_types=parameters,
            is_static="static" in modifiers,
        ))
    return signatures


def diff_public_api(before: str, after_sources: list[str]) -> ApiDiff:
    """Compare the original's public API against the decomposed classes'.

    The decomposed sources are pooled: a method that moved from ``Order``
    to ``OrderManager`` is retained, not removed, because the behaviour
    still exists somewhere in the result.
    """
    original = extract_public_api(before)
    resulting: set[MethodSignature] = set()
    for source in after_sources:
        resulting |= extract_public_api(source)

    return ApiDiff(
        removed=sorted(original - resulting, key=str),
        added=sorted(resulting - original, key=str),
        retained=sorted(original & resulting, key=str),
    )


# ──────────────────────────────────────────────────────────────────────
# Anti-gaming
# ──────────────────────────────────────────────────────────────────────

@dataclass
class GamingGuard:
    """Checks that a decomposition moved behaviour rather than losing it.

    The attack these exist to stop: split a God Class into eight classes
    with no method bodies. Every CK metric improves — WMC falls, LCOM
    falls, CBO falls — and the naive composite score goes up.
    """

    shell_classes: list[str] = field(default_factory=list)
    empty_classes: list[str] = field(default_factory=list)
    behaviour_lost: bool = False
    api_broken: bool = False
    size_collapsed: bool = False
    facade_retains_logic: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def gamed(self) -> bool:
        """True when the result should not be credited as an improvement."""
        return bool(
            self.empty_classes or self.behaviour_lost
            or self.api_broken or self.size_collapsed
            or self.facade_retains_logic
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gamed": self.gamed,
            "shell_classes": self.shell_classes,
            "empty_classes": self.empty_classes,
            "behaviour_lost": self.behaviour_lost,
            "api_broken": self.api_broken,
            "size_collapsed": self.size_collapsed,
            "facade_retains_logic": self.facade_retains_logic,
            "notes": self.notes,
        }


def _branching(metrics: ClassMetrics) -> int:
    """Decision points beyond one path per method: WMC minus method count."""
    return max(0, metrics.wmc - metrics.num_methods)


def _check_gaming(
    before_metrics: ClassMetrics,
    after_metrics: list[ClassMetrics],
    api: ApiDiff,
    size_collapse_threshold: float = 0.5,
    original_branching: int = 0,
    facade: ClassMetrics | None = None,
) -> GamingGuard:
    """Apply every anti-gaming check."""
    guard = GamingGuard()

    if facade is not None and original_branching > 0:
        kept = _branching(facade) / original_branching
        if kept > FACADE_MAX_RETAINED_BRANCHING:
            guard.facade_retains_logic = True
            guard.notes.append(
                f"{facade.class_name} still holds {kept:.0%} of the original's "
                f"branching. Classes were added around the logic, not "
                f"extracted from it."
            )

    for metrics in after_metrics:
        if metrics.num_methods == 0:
            guard.empty_classes.append(metrics.class_name)
        elif (
            metrics.wmc < MIN_MEANINGFUL_WMC
            and metrics.num_methods <= MIN_MEANINGFUL_METHODS
        ):
            guard.shell_classes.append(metrics.class_name)

    if guard.empty_classes:
        guard.notes.append(
            f"{len(guard.empty_classes)} extracted class(es) declare no "
            f"methods. Splitting into empty shells improves every CK metric "
            f"without improving anything."
        )
    if guard.shell_classes:
        guard.notes.append(
            f"{len(guard.shell_classes)} extracted class(es) hold a single "
            f"trivial method. Not conclusive on its own, but a decomposition "
            f"made mostly of these has dispersed names, not responsibilities."
        )

    if api.breaking:
        guard.api_broken = True
        guard.notes.append(
            f"{len(api.removed)} public method(s) no longer exist: "
            f"{', '.join(str(s) for s in api.removed[:5])}. The test suite "
            f"cannot see this; every other caller can."
        )

    total_after = sum(m.loc for m in after_metrics)
    if before_metrics.loc and total_after < before_metrics.loc * size_collapse_threshold:
        guard.size_collapsed = True
        guard.notes.append(
            f"Total size fell from {before_metrics.loc} to {total_after} "
            f"logical lines, below {size_collapse_threshold:.0%} of the "
            f"original. Code was deleted, not moved."
        )

    total_wmc_after = sum(m.wmc for m in after_metrics)
    if before_metrics.wmc and total_wmc_after < before_metrics.wmc * size_collapse_threshold:
        guard.behaviour_lost = True
        guard.notes.append(
            f"Total complexity fell from {before_metrics.wmc} to "
            f"{total_wmc_after}. Decomposition redistributes complexity; it "
            f"does not remove this much of it."
        )

    return guard


# ──────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────

@dataclass
class StructuralScore:
    """The three axes, kept separate, plus a composite of the third."""

    # Axis 1 — behaviour
    tests_pass: bool | None = None
    api: ApiDiff = field(default_factory=ApiDiff)
    differential_agreement: float | None = None

    # Axis 2 — structure
    coupling_reduction: float = 0.0
    cohesion_improvement: float = 0.0
    complexity_dispersal: float = 0.0
    size_reduction: float = 0.0
    composite: float = 0.0

    # Axis 3 — gaming
    guard: GamingGuard = field(default_factory=GamingGuard)

    # Raw numbers behind the normalised axes.
    before: dict[str, float] = field(default_factory=dict)
    after: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def behaviour_preserved(self) -> bool | None:
        """True only when an *executable* check ran and every check passed.

        The API diff is necessary and not sufficient, which is this
        module's whole thesis applied to itself: an unchanged public surface
        proves nothing about what happens behind it. So a passing verdict
        requires the suite or a differential run, and ``None`` — unknown —
        is returned when neither happened.

        ``None`` must never be read as "preserved". An unrun check is not a
        passed check.
        """
        executable = [
            check for check in (
                self.tests_pass,
                None if self.differential_agreement is None
                else self.differential_agreement >= 1.0,
            )
            if check is not None
        ]
        if not executable:
            return None
        return all(executable) and not self.api.breaking

    @property
    def credited(self) -> bool:
        """True when the improvement should count.

        Requires behaviour preserved, no gaming, and a positive composite.
        All three: a refactoring that improves structure by breaking the API
        is not a refactoring.
        """
        return bool(
            self.behaviour_preserved
            and not self.guard.gamed
            and self.composite > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "credited": self.credited,
            "behaviour": {
                "preserved": self.behaviour_preserved,
                "tests_pass": self.tests_pass,
                "api": self.api.to_dict(),
                "differential_agreement": self.differential_agreement,
            },
            "structure": {
                "composite": round(self.composite, 4),
                "weights": SCORE_WEIGHTS,
                "axes": {
                    "coupling_reduction": round(self.coupling_reduction, 4),
                    "cohesion_improvement": round(self.cohesion_improvement, 4),
                    "complexity_dispersal": round(self.complexity_dispersal, 4),
                    "size_reduction": round(self.size_reduction, 4),
                },
                "before": self.before,
                "after": self.after,
            },
            "gaming": self.guard.to_dict(),
            "warnings": self.warnings,
        }


def explain_score(payload: dict[str, Any]) -> list[str]:
    """Plain-sentence reasons a refactoring was not credited.

    Takes :meth:`StructuralScore.to_dict` output, so it also works on scores
    read back from a results file. Empty when the score was credited. Each
    reason starts with the axis it comes from.
    """
    if not payload or payload.get("credited"):
        return []
    behaviour = payload.get("behaviour", {})
    gaming = payload.get("gaming", {})
    composite = payload.get("structure", {}).get("composite", 0.0)

    reasons: list[str] = []
    if behaviour.get("tests_pass") is False:
        reasons.append("Behaviour: the refactored code fails the original's test suite.")
    api = behaviour.get("api", {})
    if api.get("breaking"):
        removed = ", ".join(api.get("removed", [])[:4])
        reasons.append(f"API: public methods callers depend on are gone ({removed}).")
    notes = gaming.get("notes", [])
    reasons.extend(f"Gaming: {note}" for note in notes)
    if not gaming.get("gamed") and composite <= 0:
        reasons.append(
            f"Structure: the composite is {composite:+.3f}; coupling, cohesion, "
            f"dispersal and size did not improve on balance."
        )
    reasons.extend(
        f"Note: {warning}" for warning in payload.get("warnings", []) if warning not in notes
    )
    return reasons


def _dispersal(classes: list[ClassMetrics]) -> float:
    """How close the most complex class is to an even share of total WMC.

    1.0 is perfectly even; 0.0 for a single class or no complexity.
    """
    total = sum(m.wmc for m in classes)
    worst = max((m.wmc for m in classes), default=0)
    if len(classes) <= 1 or total <= 0 or worst <= 0:
        return 0.0
    return max(0.0, min(1.0, (total / len(classes)) / worst))


def _normalised_reduction(before: float, after: float) -> float:
    """Fractional improvement in a lower-is-better metric, clamped to [-1, 1].

    Returns 0.0 when there was nothing to improve: a class with CBO 0 cannot
    have its coupling reduced, and crediting it for that would reward
    refactoring code that did not need it.
    """
    if before <= 0:
        return 0.0
    return max(-1.0, min(1.0, (before - after) / before))


def score_structure(
    before_source: str,
    after_sources: list[str],
    tests_pass: bool | None = None,
    differential_agreement: float | None = None,
) -> StructuralScore:
    """Score a refactoring on behaviour, structure and gaming.

    Parameters
    ----------
    before_source
        The original class.
    after_sources
        The decomposed classes.
    tests_pass
        Whether the suite still passes. ``None`` means it was not run, which
        is recorded as unknown rather than assumed.
    differential_agreement
        Fraction of generated inputs on which pre and post versions produced
        the same output, from :mod:`quality.difftest`. ``None`` when not run.
    """
    score = StructuralScore(
        tests_pass=tests_pass,
        differential_agreement=differential_agreement,
    )

    extractor = CKMetricsExtractor()
    try:
        before_classes = extractor.extract(before_source, "Before.java")
    except Exception as exc:
        score.warnings.append(f"Could not parse the original: {exc}")
        return score
    if not before_classes:
        score.warnings.append("The original source declared no class.")
        return score

    before_metrics = before_classes[0]
    after_metrics: list[ClassMetrics] = []
    for index, source in enumerate(after_sources):
        try:
            after_metrics.extend(extractor.extract(source, f"After_{index}.java"))
        except Exception as exc:
            score.warnings.append(f"Could not parse decomposed source {index}: {exc}")
    if not after_metrics:
        score.warnings.append("No class was found in the decomposed sources.")
        return score

    score.api = diff_public_api(before_source, after_sources)

    # A decomposition that keeps the public API keeps the original class as
    # a facade, and a facade delegating to n extracted classes is coupled to
    # all n by design. Judging coupling and cohesion on the facade therefore
    # scores every API-preserving refactoring as negative. So the extracted
    # classes are judged on coupling and cohesion, and the facade on whether
    # it gave up the logic (see the guard). Types already in the original,
    # such as nested DTOs, are neither.
    original_names = {m.class_name for m in before_classes}
    facade = next(
        (m for m in after_metrics if m.class_name == before_metrics.class_name), None
    )
    extracted = [m for m in after_metrics if m.class_name not in original_names]
    pieces = extracted or after_metrics

    max_cbo = max(m.cbo for m in pieces)
    max_lcom = max(m.lcom for m in pieces)
    max_wmc = max(m.wmc for m in after_metrics)
    total_loc = sum(m.loc for m in after_metrics)
    total_wmc = sum(m.wmc for m in after_metrics)

    # The original may declare nested helper types (a DTO, say). They are
    # classes in the "after" count too, so size and dispersal are measured
    # against every original class: an unchanged file must score zero.
    before_total_loc = sum(m.loc for m in before_classes)
    before_total_wmc = sum(m.wmc for m in before_classes)

    score.before = {
        "cbo": float(before_metrics.cbo), "lcom": float(before_metrics.lcom),
        "wmc": float(before_metrics.wmc), "loc": float(before_total_loc),
        "rfc": float(before_metrics.rfc), "classes": float(len(before_classes)),
        "branching": float(_branching(before_metrics)),
    }
    score.after = {
        "cbo_max": float(max_cbo), "lcom_max": float(max_lcom),
        "wmc_max": float(max_wmc), "wmc_total": float(total_wmc),
        "loc_total": float(total_loc),
        "rfc_max": float(max(m.rfc for m in after_metrics)),
        "classes": float(len(after_metrics)),
        "extracted_classes": float(len(extracted)),
    }
    if facade is not None:
        score.after["facade_cbo"] = float(facade.cbo)
        score.after["facade_branching"] = float(_branching(facade))

    # Coupling and cohesion: compare against the *worst* extracted class.
    # The claim a decomposition makes is that every piece is better than the
    # whole was, and the worst piece is what tests that claim.
    score.coupling_reduction = _normalised_reduction(before_metrics.cbo, max_cbo)
    score.cohesion_improvement = _normalised_reduction(before_metrics.lcom, max_lcom)

    # Complexity dispersal: did WMC spread out, or just move? Perfect
    # dispersal over n classes puts total/n in each. This measures how close
    # the worst class is to that ideal, which is why it rewards balance and
    # not merely "fewer methods in one file". Only the gain over the
    # original's own spread is credited.
    score.complexity_dispersal = max(
        0.0, _dispersal(after_metrics) - _dispersal(before_classes)
    )
    if len(after_metrics) == 1:
        score.warnings.append(
            "A single resulting class: nothing was dispersed."
        )

    # Size: credit a modest reduction, but a collapse is deletion, and the
    # guard below will catch it. Growth is expected and lightly penalised —
    # extracting classes costs boilerplate.
    raw_size = _normalised_reduction(before_total_loc, total_loc)
    score.size_reduction = raw_size if -0.25 <= raw_size <= 0.35 else max(0.0, raw_size * 0.25)

    score.guard = _check_gaming(
        replace(before_metrics, loc=before_total_loc, wmc=before_total_wmc),
        after_metrics,
        score.api,
        original_branching=_branching(before_metrics),
        facade=facade,
    )

    score.composite = round(sum(
        SCORE_WEIGHTS[axis] * getattr(score, axis)
        for axis in SCORE_WEIGHTS
    ), 4)

    # Gaming zeroes the composite rather than reducing it. A partial credit
    # for a gamed result is an invitation to game it partially.
    if score.guard.gamed:
        score.warnings.extend(score.guard.notes)
        score.composite = 0.0

    return score
