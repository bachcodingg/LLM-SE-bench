"""
Tests for the maintenance scripts (M10 and the status check).

These are the scripts that decide whether the build fails and whether the
project is honest about its own state. A regression gate that cannot itself
regress-test is not much of a gate.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    """Import a script from scripts/ by path.

    They live outside the package tree — they are tools, not library code —
    so a normal import will not find them.
    """
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_scripts_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


regression_gate = _load("regression_gate")
project_status = _load("project_status")


def _write_summary(directory: Path, rows: list[dict[str, object]]) -> Path:
    """Write a minimal summary.csv of the shape the gate reads."""
    directory.mkdir(parents=True, exist_ok=True)
    header = "model_id,dataset,pass_rate\n"
    body = "".join(
        f"{row['model_id']},{row.get('dataset', 'ALL')},{row['pass_rate']}\n"
        for row in rows
    )
    (directory / "summary.csv").write_text(header + body, encoding="utf-8")
    return directory


class TestRegressionGateResolveRates:
    def test_identical_runs_pass(self, tmp_path):
        rows = [{"model_id": "claude", "pass_rate": 1.0}]
        baseline = _write_summary(tmp_path / "base", rows)
        candidate = _write_summary(tmp_path / "cand", rows)

        result = regression_gate.compare_resolve_rates(baseline, candidate)
        assert result["passed"] is True
        assert result["comparisons"][0]["status"] == "within_tolerance"

    def test_a_small_drop_is_tolerated(self):
        """A gate that fails on noise gets turned off, which is worse."""
        assert regression_gate.DEFAULT_TOLERANCE >= 0.03

    def test_a_drop_beyond_tolerance_fails(self, tmp_path):
        baseline = _write_summary(tmp_path / "base", [
            {"model_id": "claude", "pass_rate": 1.0}
        ])
        candidate = _write_summary(tmp_path / "cand", [
            {"model_id": "claude", "pass_rate": 0.80}
        ])
        result = regression_gate.compare_resolve_rates(baseline, candidate)
        assert result["passed"] is False
        assert "fell" in result["failures"][0]

    def test_a_drop_within_tolerance_passes(self, tmp_path):
        baseline = _write_summary(tmp_path / "base", [
            {"model_id": "claude", "pass_rate": 1.0}
        ])
        candidate = _write_summary(tmp_path / "cand", [
            {"model_id": "claude", "pass_rate": 0.97}
        ])
        assert regression_gate.compare_resolve_rates(baseline, candidate)["passed"] is True

    def test_a_large_drop_is_a_hard_regression(self, tmp_path):
        baseline = _write_summary(tmp_path / "base", [
            {"model_id": "claude", "pass_rate": 1.0}
        ])
        candidate = _write_summary(tmp_path / "cand", [
            {"model_id": "claude", "pass_rate": 0.2}
        ])
        result = regression_gate.compare_resolve_rates(baseline, candidate)
        assert result["comparisons"][0]["status"] == "hard_regression"
        assert "beyond any plausible noise" in result["failures"][0]

    def test_an_improvement_passes_and_is_labelled(self, tmp_path):
        baseline = _write_summary(tmp_path / "base", [
            {"model_id": "claude", "pass_rate": 0.5}
        ])
        candidate = _write_summary(tmp_path / "cand", [
            {"model_id": "claude", "pass_rate": 0.9}
        ])
        result = regression_gate.compare_resolve_rates(baseline, candidate)
        assert result["passed"] is True
        assert result["comparisons"][0]["status"] == "improvement"

    def test_a_missing_model_fails(self, tmp_path):
        """A model that silently stopped being evaluated is a regression."""
        baseline = _write_summary(tmp_path / "base", [
            {"model_id": "claude", "pass_rate": 1.0},
            {"model_id": "gpt-4o", "pass_rate": 1.0},
        ])
        candidate = _write_summary(tmp_path / "cand", [
            {"model_id": "claude", "pass_rate": 1.0}
        ])
        result = regression_gate.compare_resolve_rates(baseline, candidate)
        assert result["passed"] is False
        assert "absent from the run" in result["failures"][0]

    def test_a_new_model_does_not_fail_the_gate(self, tmp_path):
        baseline = _write_summary(tmp_path / "base", [
            {"model_id": "claude", "pass_rate": 1.0}
        ])
        candidate = _write_summary(tmp_path / "cand", [
            {"model_id": "claude", "pass_rate": 1.0},
            {"model_id": "new-model", "pass_rate": 0.4},
        ])
        result = regression_gate.compare_resolve_rates(baseline, candidate)
        assert result["passed"] is True
        assert any(row["status"] == "new" for row in result["comparisons"])

    def test_non_all_rows_are_ignored(self, tmp_path):
        directory = tmp_path / "base"
        directory.mkdir()
        (directory / "summary.csv").write_text(
            "model_id,dataset,pass_rate\n"
            "claude,ALL,1.0\n"
            "claude,defects4j,0.1\n",
            encoding="utf-8",
        )
        assert regression_gate.load_summary(directory)["claude"]["pass_rate"] == "1.0"

    def test_a_missing_summary_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            regression_gate.load_summary(tmp_path)

    def test_a_summary_with_no_all_rows_raises(self, tmp_path):
        directory = tmp_path / "base"
        directory.mkdir()
        (directory / "summary.csv").write_text(
            "model_id,dataset,pass_rate\nclaude,defects4j,1.0\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="no ALL rows"):
            regression_gate.load_summary(directory)


class TestPromptDrift:
    def test_matching_hashes_pass(self, tmp_path):
        directory = tmp_path / "run"
        directory.mkdir()
        (directory / "manifest.json").write_text(
            json.dumps({
                "run_name": "test",
                "prompt_template_sha256": regression_gate.prompt_fingerprints(),
            }),
            encoding="utf-8",
        )
        result = regression_gate.check_prompt_drift(directory)
        assert result["passed"] is True
        assert "remain valid" in result["advice"]

    def test_a_changed_template_is_detected(self, tmp_path):
        """A silent prompt edit invalidates every published comparison."""
        fingerprints = dict(regression_gate.prompt_fingerprints())
        name = next(iter(fingerprints))
        fingerprints[name] = "0" * 64

        directory = tmp_path / "run"
        directory.mkdir()
        (directory / "manifest.json").write_text(
            json.dumps({"prompt_template_sha256": fingerprints}), encoding="utf-8"
        )
        result = regression_gate.check_prompt_drift(directory)
        assert result["passed"] is False
        assert name in result["changed"]
        assert "makes stale" in result["advice"]

    def test_a_removed_template_fails(self, tmp_path):
        fingerprints = dict(regression_gate.prompt_fingerprints())
        fingerprints["gone.j2"] = "0" * 64

        directory = tmp_path / "run"
        directory.mkdir()
        (directory / "manifest.json").write_text(
            json.dumps({"prompt_template_sha256": fingerprints}), encoding="utf-8"
        )
        result = regression_gate.check_prompt_drift(directory)
        assert result["passed"] is False
        assert "gone.j2" in result["removed"]

    def test_a_missing_manifest_is_reported(self, tmp_path):
        result = regression_gate.check_prompt_drift(tmp_path)
        assert result["passed"] is False
        assert "no manifest.json" in result["reason"]

    def test_the_committed_baseline_matches_the_current_templates(self):
        """The repository's own published numbers are still valid."""
        result = regression_gate.check_prompt_drift(REPO_ROOT / "results" / "2026-05-run")
        assert result["passed"] is True, result

    def test_fingerprints_cover_every_template(self):
        templates = list((REPO_ROOT / "llm_gateway" / "templates").glob("*.j2"))
        assert len(regression_gate.prompt_fingerprints()) == len(templates)


class TestGateRendering:
    def test_markdown_leads_with_the_verdict(self, tmp_path):
        baseline = _write_summary(tmp_path / "base", [
            {"model_id": "claude", "pass_rate": 1.0}
        ])
        candidate = _write_summary(tmp_path / "cand", [
            {"model_id": "claude", "pass_rate": 0.5}
        ])
        resolve = regression_gate.compare_resolve_rates(baseline, candidate)
        markdown = regression_gate.render_markdown(resolve, None)
        assert "Resolve rate: FAILED" in markdown
        assert "| Model |" in markdown


class TestProjectStatus:
    def test_every_module_is_assessed(self):
        statuses = project_status.module_statuses()
        assert len(statuses) == len(project_status.MODULES)
        assert all(status.source_exists for status in statuses)

    def test_states_are_from_the_known_set(self):
        allowed = {"missing", "untested", "library", "not run", "working"}
        assert {status.state for status in project_status.module_statuses()} <= allowed

    def test_a_module_with_no_artifact_is_a_library_not_a_failure(self):
        """Lumping pure functions in with 'never run' dilutes the signal."""
        statuses = {s.name: s for s in project_status.module_statuses()}
        assert statuses["M6 tamper detection"].state == "library"

    def test_the_task_factory_reports_as_never_run(self):
        """The honest answer until it has actually mined something."""
        statuses = {s.name: s for s in project_status.module_statuses()}
        assert statuses["M3 task factory"].state == "not run"

    def test_test_counts_are_positive_for_tested_modules(self):
        statuses = {s.name: s for s in project_status.module_statuses()}
        assert statuses["M1 agent harness"].test_count > 100

    def test_the_suite_total_does_not_double_count_shared_directories(self):
        """agent/tests is evidence for both M1 and M4; summing overstates."""
        per_module = sum(s.test_count for s in project_status.module_statuses())
        assert project_status.suite_test_count() < per_module

    def test_no_placeholders_remain(self):
        assert project_status.find_placeholders() == {}

    def test_publication_checks_have_the_expected_shape(self):
        checks = project_status.publication_checks()
        names = {check["check"] for check in checks}
        assert names == {"published", "placeholders", "run_artifact", "demo_gif"}
        assert all("detail" in check for check in checks)

    def test_the_gif_check_is_not_blocking(self):
        gif = next(
            c for c in project_status.publication_checks() if c["check"] == "demo_gif"
        )
        assert gif.get("blocking") is False

    def test_nothing_blocks_publication(self):
        """If this fails, read the report: something needs doing before a push."""
        assert project_status.main(["--json"]) == 0

    def test_renders_without_crashing(self):
        text = project_status.render(
            project_status.module_statuses(), project_status.publication_checks()
        )
        assert "PUBLICATION" in text
        assert "MODULE" in text
