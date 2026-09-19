"""
diff_analysis.py — Patch size and location-accuracy analysis for LLM-generated
code changes (bug fixes and refactoring).

Compares LLM-generated patches against reference patches to measure:
    - Patch size:  lines added / removed / changed
    - Location accuracy:  did the LLM modify the correct file regions?
    - Precision & recall of changed lines vs. reference
    - Minimality:  ratio of necessary changes to total changes
    - Collateral damage:  unrelated lines modified

Uses unified-diff format for comparison.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DiffStats:
    """Statistics about a single diff/patch."""
    filename: str = ""
    lines_added: int = 0
    lines_removed: int = 0
    lines_modified: int = 0
    hunks: int = 0
    total_lines_original: int = 0
    total_lines_modified: int = 0
    change_ratio: float = 0.0  # (added + removed) / total_original

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "lines_modified": self.lines_modified,
            "hunks": self.hunks,
            "total_lines_original": self.total_lines_original,
            "total_lines_modified": self.total_lines_modified,
            "change_ratio": round(self.change_ratio, 4),
        }


@dataclass
class LocationAccuracy:
    """How well the LLM targeted the correct lines."""
    # Lines changed by LLM
    llm_changed_lines: set = field(default_factory=set)
    # Lines changed in reference patch
    reference_changed_lines: set = field(default_factory=set)
    # Overlap
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    # Lines changed by LLM outside the reference region
    collateral_lines: int = 0

    def to_dict(self) -> dict:
        return {
            "llm_changed_count": len(self.llm_changed_lines),
            "reference_changed_count": len(self.reference_changed_lines),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "collateral_lines": self.collateral_lines,
        }


@dataclass
class PatchAnalysis:
    """Complete analysis comparing an LLM patch against a reference."""
    llm_diff_stats: DiffStats = field(default_factory=DiffStats)
    reference_diff_stats: DiffStats = field(default_factory=DiffStats)
    location_accuracy: LocationAccuracy = field(default_factory=LocationAccuracy)
    minimality_score: float = 0.0  # 0-1: how concise is the LLM patch?
    size_ratio: float = 0.0  # LLM patch size / reference patch size

    def to_dict(self) -> dict:
        return {
            "llm_diff": self.llm_diff_stats.to_dict(),
            "reference_diff": self.reference_diff_stats.to_dict(),
            "location_accuracy": self.location_accuracy.to_dict(),
            "minimality_score": round(self.minimality_score, 4),
            "size_ratio": round(self.size_ratio, 4),
        }


# ---------------------------------------------------------------------------
# Diff computation
# ---------------------------------------------------------------------------

def compute_diff(original: str, modified: str,
                 filename: str = "") -> tuple[list[str], DiffStats]:
    """
    Compute unified diff between original and modified source.

    Args:
        original: original source code.
        modified: modified source code.
        filename: label for the diff header.

    Returns:
        (diff_lines, DiffStats)
    """
    orig_lines = original.splitlines(keepends=True)
    mod_lines = modified.splitlines(keepends=True)

    diff_lines = list(difflib.unified_diff(
        orig_lines, mod_lines,
        fromfile=f"a/{filename}", tofile=f"b/{filename}",
        lineterm="",
    ))

    stats = DiffStats(
        filename=filename,
        total_lines_original=len(orig_lines),
        total_lines_modified=len(mod_lines),
    )

    in_hunk = False
    for line in diff_lines:
        if line.startswith("@@"):
            stats.hunks += 1
            in_hunk = True
        elif in_hunk:
            if line.startswith("+") and not line.startswith("+++"):
                stats.lines_added += 1
            elif line.startswith("-") and not line.startswith("---"):
                stats.lines_removed += 1

    stats.lines_modified = min(stats.lines_added, stats.lines_removed)
    if stats.total_lines_original > 0:
        stats.change_ratio = (
            (stats.lines_added + stats.lines_removed) /
            stats.total_lines_original
        )

    return diff_lines, stats


def get_changed_line_numbers(original: str, modified: str) -> set[int]:
    """
    Return 1-indexed line numbers from the *original* that were changed
    or removed in the modified version.
    """
    orig_lines = original.splitlines()
    mod_lines = modified.splitlines()

    changed: set[int] = set()
    matcher = difflib.SequenceMatcher(None, orig_lines, mod_lines)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            for i in range(i1, i2):
                changed.add(i + 1)  # 1-indexed

    return changed


# ---------------------------------------------------------------------------
# Location accuracy analysis
# ---------------------------------------------------------------------------

def analyse_location_accuracy(
    original: str,
    llm_modified: str,
    reference_modified: str,
) -> LocationAccuracy:
    """
    Compare which lines the LLM changed vs. which lines the reference
    patch changed.

    Args:
        original: the original (buggy) source.
        llm_modified: the LLM's proposed fix.
        reference_modified: the known-correct fix.

    Returns:
        LocationAccuracy with precision, recall, and F1 of changed lines.
    """
    llm_changed = get_changed_line_numbers(original, llm_modified)
    ref_changed = get_changed_line_numbers(original, reference_modified)

    result = LocationAccuracy(
        llm_changed_lines=llm_changed,
        reference_changed_lines=ref_changed,
    )

    tp = len(llm_changed & ref_changed)
    fp = len(llm_changed - ref_changed)
    fn = len(ref_changed - llm_changed)

    result.true_positives = tp
    result.false_positives = fp
    result.false_negatives = fn
    result.collateral_lines = fp

    if tp + fp > 0:
        result.precision = tp / (tp + fp)
    if tp + fn > 0:
        result.recall = tp / (tp + fn)
    if result.precision + result.recall > 0:
        result.f1_score = (
            2 * result.precision * result.recall /
            (result.precision + result.recall)
        )

    return result


# ---------------------------------------------------------------------------
# Full patch analysis
# ---------------------------------------------------------------------------

def analyse_patch(
    original: str,
    llm_modified: str,
    reference_modified: str,
    filename: str = "",
) -> PatchAnalysis:
    """
    Comprehensive patch analysis comparing LLM output against reference.

    Args:
        original: original source code.
        llm_modified: LLM-generated modification.
        reference_modified: known-correct modification.
        filename: file label.

    Returns:
        PatchAnalysis with diff stats, location accuracy, and minimality.
    """
    _, llm_stats = compute_diff(original, llm_modified, filename)
    _, ref_stats = compute_diff(original, reference_modified, filename)
    loc_accuracy = analyse_location_accuracy(original, llm_modified,
                                              reference_modified)

    # Minimality: ratio of reference changes to LLM changes
    # If the LLM made fewer changes but they're all correct, that's ideal.
    llm_total = llm_stats.lines_added + llm_stats.lines_removed
    ref_total = ref_stats.lines_added + ref_stats.lines_removed

    if llm_total > 0:
        minimality = min(1.0, ref_total / llm_total)
    else:
        minimality = 0.0 if ref_total > 0 else 1.0

    size_ratio = llm_total / ref_total if ref_total > 0 else float("inf")

    return PatchAnalysis(
        llm_diff_stats=llm_stats,
        reference_diff_stats=ref_stats,
        location_accuracy=loc_accuracy,
        minimality_score=minimality,
        size_ratio=size_ratio if size_ratio != float("inf") else 0.0,
    )


def analyse_patch_from_files(
    original_path: str | Path,
    llm_path: str | Path,
    reference_path: str | Path,
) -> PatchAnalysis:
    """Convenience: read files from disk and run patch analysis."""
    original_path = Path(original_path)
    llm_path = Path(llm_path)
    reference_path = Path(reference_path)

    original = original_path.read_text(encoding="utf-8", errors="replace")
    llm_modified = llm_path.read_text(encoding="utf-8", errors="replace")
    reference = reference_path.read_text(encoding="utf-8", errors="replace")

    return analyse_patch(
        original, llm_modified, reference,
        filename=original_path.name,
    )
