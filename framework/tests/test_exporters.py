"""
test_exporters.py — Tests for JSON/CSV exporters, CLI report, and PDF report.

Covers:
- JSONExporter (to_dict, to_json, export)
- CSVExporter (fieldnames, to_rows, to_csv_string, export)
- MultiProfileExporter
- CLIReporter (render, sections)
- PDFReporter (render_html, export fallback)
"""

from __future__ import annotations

import csv
import json
from io import StringIO

from framework.decision_matrix import DecisionMatrixEngine
from framework.exporters import (
    CSVExporter,
    JSONExporter,
    MultiProfileExporter,
)

# =====================================================================
# JSONExporter
# =====================================================================


class TestJSONExporter:
    def test_to_dict(self, synthetic_summaries):
        engine = DecisionMatrixEngine(synthetic_summaries)
        engine.build()
        exp = JSONExporter(engine)
        d = exp.to_dict()
        assert "meta" in d
        assert "decision_matrix" in d
        assert d["meta"]["model_count"] == 3

    def test_to_json_valid(self, synthetic_summaries):
        engine = DecisionMatrixEngine(synthetic_summaries)
        engine.build()
        exp = JSONExporter(engine)
        j = exp.to_json()
        parsed = json.loads(j)
        assert len(parsed["decision_matrix"]) == 3

    def test_export(self, synthetic_summaries, tmp_path):
        engine = DecisionMatrixEngine(synthetic_summaries)
        engine.build()
        exp = JSONExporter(engine)
        p = exp.export(tmp_path / "test.json")
        assert p.exists()
        data = json.loads(p.read_text())
        assert len(data["decision_matrix"]) == 3

    def test_with_recommendations(self, synthetic_summaries):
        engine = DecisionMatrixEngine(synthetic_summaries)
        engine.build()
        recs = [{"model": "claude", "confidence": 0.9}]
        exp = JSONExporter(engine, recommendations=recs)
        d = exp.to_dict()
        assert d["recommendations"] == recs

    def test_profile_in_meta(self, synthetic_summaries, profiles_dir):
        engine = DecisionMatrixEngine(
            synthetic_summaries,
            profile_name="devops",
            profiles_dir=str(profiles_dir),
        )
        engine.build()
        exp = JSONExporter(engine)
        d = exp.to_dict()
        assert d["meta"]["profile"] == "devops"


# =====================================================================
# CSVExporter
# =====================================================================


class TestCSVExporter:
    def test_fieldnames(self, synthetic_summaries):
        engine = DecisionMatrixEngine(synthetic_summaries)
        engine.build()
        exp = CSVExporter(engine)
        fn = exp.fieldnames
        assert "rank" in fn
        assert "model_id" in fn
        assert "correctness_raw" in fn
        assert "cost_norm" in fn

    def test_to_rows(self, synthetic_summaries):
        engine = DecisionMatrixEngine(synthetic_summaries)
        engine.build()
        exp = CSVExporter(engine)
        rows = exp.to_rows()
        assert len(rows) == 3
        assert rows[0]["rank"] == 1

    def test_to_csv_string(self, synthetic_summaries):
        engine = DecisionMatrixEngine(synthetic_summaries)
        engine.build()
        exp = CSVExporter(engine)
        csv_str = exp.to_csv_string()
        reader = csv.DictReader(StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 3
        assert "model_id" in rows[0]

    def test_export(self, synthetic_summaries, tmp_path):
        engine = DecisionMatrixEngine(synthetic_summaries)
        engine.build()
        exp = CSVExporter(engine)
        p = exp.export(tmp_path / "test.csv")
        assert p.exists()
        assert p.read_text().startswith("rank,model_id")

    def test_round_trip(self, synthetic_summaries, tmp_path):
        """Write CSV, read it back, verify values."""
        engine = DecisionMatrixEngine(synthetic_summaries)
        engine.build()
        p = CSVExporter(engine).export(tmp_path / "rt.csv")
        with open(p) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert float(rows[0]["weighted_total"]) > 0


# =====================================================================
# MultiProfileExporter
# =====================================================================


class TestMultiProfileExporter:
    def test_export_all(self, synthetic_summaries, tmp_path, profiles_dir):
        exp = MultiProfileExporter(
            synthetic_summaries,
            output_dir=tmp_path / "reports",
            profiles_dir=str(profiles_dir),
        )
        paths = exp.export_all()
        assert "devops" in paths
        assert "audit" in paths
        assert "budget" in paths
        assert "combined" in paths
        assert all(p.exists() for p in paths.values())

    def test_combined_has_all_profiles(
        self, synthetic_summaries, tmp_path, profiles_dir
    ):
        exp = MultiProfileExporter(
            synthetic_summaries,
            output_dir=tmp_path / "reports",
            profiles_dir=str(profiles_dir),
        )
        paths = exp.export_all()
        combined = json.loads(paths["combined"].read_text())
        assert "devops" in combined["profiles"]
        assert "audit" in combined["profiles"]
        assert "budget" in combined["profiles"]


# =====================================================================
# CLIReporter
# =====================================================================


class TestCLIReporter:
    def test_render(self, synthetic_summaries, profiles_dir):
        from framework.cli_report import CLIReporter

        reporter = CLIReporter(
            synthetic_summaries,
            profile_name="devops",
            profiles_dir=str(profiles_dir),
            color=False,
        )
        text = reporter.render()
        assert "Decision Report" in text
        assert "DevOps" in text or "devops" in text
        assert "Recommended" in text
        assert "Pareto" in text

    def test_contains_all_models(self, synthetic_summaries, profiles_dir):
        from framework.cli_report import CLIReporter

        reporter = CLIReporter(
            synthetic_summaries,
            profile_name="audit",
            profiles_dir=str(profiles_dir),
            color=False,
        )
        text = reporter.render()
        assert "claude-3.5-sonnet" in text
        assert "gpt-4-turbo" in text
        assert "gemini-1.5-pro" in text

    def test_sensitivity_section(self, synthetic_summaries, profiles_dir):
        from framework.cli_report import CLIReporter

        reporter = CLIReporter(
            synthetic_summaries,
            profile_name="budget",
            profiles_dir=str(profiles_dir),
            color=False,
        )
        text = reporter.render()
        assert "Sensitivity" in text

    def test_print(self, synthetic_summaries, profiles_dir):
        from framework.cli_report import CLIReporter

        reporter = CLIReporter(
            synthetic_summaries,
            profile_name="devops",
            profiles_dir=str(profiles_dir),
            color=False,
        )
        buf = StringIO()
        reporter.print(file=buf)
        assert len(buf.getvalue()) > 100


# =====================================================================
# PDFReporter
# =====================================================================


class TestPDFReporter:
    def test_render_html(self, synthetic_summaries, profiles_dir):
        from framework.pdf_report import PDFReporter

        reporter = PDFReporter(
            synthetic_summaries,
            profile_name="devops",
            profiles_dir=str(profiles_dir),
        )
        html = reporter.render_html()
        assert "<!DOCTYPE html>" in html
        assert "Decision Report" in html
        assert "claude-3.5-sonnet" in html

    def test_html_has_sections(self, synthetic_summaries, profiles_dir):
        from framework.pdf_report import PDFReporter

        reporter = PDFReporter(
            synthetic_summaries,
            profile_name="audit",
            profiles_dir=str(profiles_dir),
        )
        html = reporter.render_html()
        assert "Ranked Decision Matrix" in html
        assert "Recommendation" in html
        assert "Pareto Frontier" in html
        assert "Weight Configuration" in html

    def test_export_html_fallback(
        self, synthetic_summaries, tmp_path, profiles_dir
    ):
        """When weasyprint is not available, should write HTML fallback."""
        from framework.pdf_report import PDFReporter

        reporter = PDFReporter(
            synthetic_summaries,
            profile_name="budget",
            profiles_dir=str(profiles_dir),
        )
        # Try to export — if weasyprint isn't installed, HTML fallback
        p = reporter.export(tmp_path / "test_report.pdf")
        assert p.exists()
        content = p.read_text()
        assert "Decision Report" in content
