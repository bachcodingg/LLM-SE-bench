"""
Tests for the refactoring dashboard page's data layer.

The page's claim is the gap between "behaviour kept" and "structure
credited", so the summary must count the two independently: a row whose
tests pass and whose structure is not credited counts towards one and not
the other.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from framework.dashboard.pages.refactoring import (
    AXES,
    axes_figure,
    load_refactoring_rows,
    summarise_by_model,
)

FIELDS = [
    "model_id", "dataset", "problem_id", "sample_index", "verdict", "tests_passed",
    "tests_total", "compile_success", "structural_credited", "structural_composite",
    "structural_reasons",
]


def _row(model: str, passed: str, total: str, credited: str, dataset: str = "godclass") -> dict:
    return {
        "model_id": model, "dataset": dataset, "problem_id": "GC_001", "sample_index": "1",
        "verdict": "pass" if credited == "True" else "partial", "tests_passed": passed,
        "tests_total": total, "compile_success": "True", "structural_credited": credited,
        "structural_composite": "0.1", "structural_reasons": "",
    }


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    path = tmp_path / "evaluations.csv"
    rows = [
        _row("a", "3", "3", "True"),
        _row("a", "3", "3", "False"),   # kept behaviour, not credited
        _row("a", "1", "3", "False"),   # broke behaviour
        _row("b", "3", "3", "False"),
        _row("b", "3", "3", "True", dataset="humaneval-java"),  # not a refactoring
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_only_refactoring_rows_are_loaded(artifact):
    assert {row["dataset"] for row in load_refactoring_rows(artifact)} == {"godclass"}


def test_a_missing_artifact_is_empty_not_an_error(tmp_path):
    assert load_refactoring_rows(tmp_path / "absent.csv") == []


def test_kept_and_credited_are_counted_independently(artifact):
    summary = {row["model_id"]: row for row in summarise_by_model(load_refactoring_rows(artifact))}
    assert summary["a"]["behaviour_kept"] == pytest.approx(2 / 3)
    assert summary["a"]["structure_credited"] == pytest.approx(1 / 3)
    assert summary["b"]["behaviour_kept"] == 1.0
    assert summary["b"]["structure_credited"] == 0.0


def test_axes_figure_has_one_bar_per_axis():
    pytest.importorskip("plotly")
    figure = axes_figure({"coupling_reduction": -0.3, "cohesion_improvement": 0.9})
    assert len(figure.data[0].x) == len(AXES)
    assert figure.data[0].x[0] == pytest.approx(-0.3)
