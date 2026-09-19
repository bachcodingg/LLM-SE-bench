"""
test_quality_modules.py — Tests for complexity, readability, refactoring,
expert_rating, irr, diff_analysis, and reports modules.
"""

import csv
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quality.ck_metrics import ClassMetrics
from quality.complexity import ComplexityAnalyzer, analyse_file
from quality.diff_analysis import (
    analyse_location_accuracy,
    analyse_patch,
    compute_diff,
    get_changed_line_numbers,
)
from quality.expert_rating import (
    AutomatedRubricScorer,
    Rating,
    create_blank_rating_csv,
    get_rubric_text,
    read_ratings_csv,
    write_ratings_csv,
)
from quality.irr import (
    cohens_kappa,
    fleiss_kappa,
    interpret_kappa,
    percent_agreement,
    weighted_kappa,
)
from quality.readability import ReadabilityScorer
from quality.refactoring import (
    RefactoringEvaluator,
    is_god_class,
)
from quality.reports import (
    QualityReportGenerator,
    _estimate_halstead,
    _maintainability_index,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ===================================================================
# Complexity Analyzer
# ===================================================================

class TestComplexityAnalyzer:

    def test_simple_method(self):
        src = """
        public class Simple {
            public int add(int a, int b) {
                return a + b;
            }
        }
        """
        result = ComplexityAnalyzer().analyse(src)
        assert len(result.methods) == 1
        assert result.methods[0].cyclomatic == 1

    def test_complex_method(self):
        src = """
        public class Complex {
            public String categorize(int x) {
                if (x < 0) {
                    return "negative";
                } else if (x == 0) {
                    return "zero";
                } else if (x < 10) {
                    for (int i = 0; i < x; i++) {
                        if (i % 2 == 0) {
                            continue;
                        }
                    }
                    return "small";
                } else {
                    return "large";
                }
            }
        }
        """
        result = ComplexityAnalyzer().analyse(src)
        assert result.methods[0].cyclomatic >= 5

    def test_cognitive_higher_for_nested(self):
        # Flat conditionals
        flat_src = """
        public class Flat {
            public void process(int a, int b, int c) {
                if (a > 0) { }
                if (b > 0) { }
                if (c > 0) { }
            }
        }
        """
        # Nested conditionals
        nested_src = """
        public class Nested {
            public void process(int a, int b, int c) {
                if (a > 0) {
                    if (b > 0) {
                        if (c > 0) { }
                    }
                }
            }
        }
        """
        flat = ComplexityAnalyzer().analyse(flat_src)
        nested = ComplexityAnalyzer().analyse(nested_src)
        assert nested.methods[0].cognitive > flat.methods[0].cognitive

    def test_aggregation(self):
        src = """
        public class Multi {
            public void a() {}
            public void b() { if (true) {} }
            public void c() { for (;;) { if (true) {} } }
        }
        """
        result = ComplexityAnalyzer().analyse(src)
        result.summarise()
        assert result.total_cyclomatic >= 4
        assert result.avg_cyclomatic > 1.0
        assert result.max_cyclomatic >= 3

    def test_empty_class(self):
        src = "public class Empty {}"
        result = ComplexityAnalyzer().analyse(src)
        assert len(result.methods) == 0
        result.summarise()
        assert result.total_cyclomatic == 0

    def test_unparseable(self):
        result = ComplexityAnalyzer().analyse("not java!!!")
        assert len(result.methods) == 0

    def test_fixture_files(self):
        if not FIXTURES.exists():
            pytest.skip("Fixtures not available")
        high = analyse_file(FIXTURES / "HighQuality.java")
        low = analyse_file(FIXTURES / "LowQuality.java")
        # Low quality file should have higher total complexity
        assert low.total_cyclomatic > high.total_cyclomatic


# ===================================================================
# Readability Scorer
# ===================================================================

class TestReadabilityScorer:

    def test_well_named_code(self):
        src = """
        public class StudentManager {
            private int studentCount;
            public int getStudentCount() { return studentCount; }
            public void setStudentCount(int count) { this.studentCount = count; }
        }
        """
        report = ReadabilityScorer().score(src)
        assert report.naming_score >= 70

    def test_poorly_named_code(self):
        src = """
        public class x {
            public int a;
            public void b(int c) { a = c; }
        }
        """
        report = ReadabilityScorer().score(src)
        assert report.naming_score < 60

    def test_well_documented_code(self):
        src = """
        /**
         * Manages user accounts.
         */
        public class UserManager {
            /** Gets the user name. */
            public String getUserName() { return ""; }
        }
        """
        report = ReadabilityScorer().score(src)
        assert report.documentation_score >= 50

    def test_no_documentation(self):
        src = """
        public class NoDoc {
            public void process() {}
            public void handle() {}
        }
        """
        report = ReadabilityScorer().score(src)
        assert report.documentation_score < 50

    def test_composite_score_computed(self):
        src = """
        public class Sample {
            private int value;
            public int getValue() { return value; }
        }
        """
        report = ReadabilityScorer().score(src)
        report.compute_composite()
        assert 0 <= report.composite_score <= 100

    def test_fixture_ordering(self):
        if not FIXTURES.exists():
            pytest.skip("Fixtures not available")
        scorer = ReadabilityScorer()
        high_src = (FIXTURES / "HighQuality.java").read_text()
        low_src = (FIXTURES / "LowQuality.java").read_text()
        high = scorer.score(high_src, "HighQuality.java")
        low = scorer.score(low_src, "LowQuality.java")
        assert high.composite_score > low.composite_score

    def test_unparseable(self):
        report = ReadabilityScorer().score("{{invalid}}")
        assert report.composite_score == 0.0


# ===================================================================
# Refactoring Evaluator
# ===================================================================

class TestRefactoringEvaluator:

    ORIGINAL = """
    public class GodClass {
        private int a;
        private int b;
        private int c;
        private String name;
        private String desc;

        public int getA() { return a; }
        public int getB() { return b; }
        public int getC() { return c; }
        public String getName() { return name; }
        public String getDesc() { return desc; }
        public void setA(int v) { this.a = v; }
        public void setB(int v) { this.b = v; }
        public void setC(int v) { this.c = v; }
        public void setName(String n) { this.name = n; }
        public void setDesc(String d) { this.desc = d; }
        public int process() {
            if (a > 0) { return a + b; }
            return c;
        }
        public String format() {
            return name + ": " + desc;
        }
    }
    """

    DECOMPOSED_A = """
    public class NumericPart {
        private int a;
        private int b;
        private int c;
        public int getA() { return a; }
        public int getB() { return b; }
        public int getC() { return c; }
        public void setA(int v) { this.a = v; }
        public void setB(int v) { this.b = v; }
        public void setC(int v) { this.c = v; }
        public int process() {
            if (a > 0) { return a + b; }
            return c;
        }
    }
    """

    DECOMPOSED_B = """
    public class DescriptivePart {
        private String name;
        private String desc;
        public String getName() { return name; }
        public String getDesc() { return desc; }
        public void setName(String n) { this.name = n; }
        public void setDesc(String d) { this.desc = d; }
        public String format() {
            return name + ": " + desc;
        }
    }
    """

    def test_basic_refactoring(self):
        evaluator = RefactoringEvaluator()
        result = evaluator.evaluate(
            self.ORIGINAL, [self.DECOMPOSED_A, self.DECOMPOSED_B])
        assert result.original_class == "GodClass"
        assert len(result.decomposed_classes) == 2
        assert result.overall_score > 0

    def test_no_decomposition(self):
        evaluator = RefactoringEvaluator()
        result = evaluator.evaluate(self.ORIGINAL, [self.ORIGINAL])
        # Returning same class isn't real decomposition
        assert result.overall_score < 80

    def test_deltas_computed(self):
        evaluator = RefactoringEvaluator()
        result = evaluator.evaluate(
            self.ORIGINAL, [self.DECOMPOSED_A, self.DECOMPOSED_B])
        delta_names = {d.metric_name for d in result.deltas}
        assert "wmc" in delta_names
        assert "lcom" in delta_names
        assert "cbo" in delta_names

    def test_empty_decomposition(self):
        evaluator = RefactoringEvaluator()
        result = evaluator.evaluate(self.ORIGINAL, [])
        assert result.overall_score == 0
        assert len(result.warnings) > 0


class TestIsGodClass:

    def test_not_god_class(self):
        m = ClassMetrics(class_name="Small", wmc=5, lcom=1, cbo=2, rfc=10)
        assert not is_god_class(m)

    def test_god_class(self):
        m = ClassMetrics(class_name="Big", wmc=60, lcom=20, cbo=20, rfc=60)
        assert is_god_class(m)

    def test_borderline(self):
        # Only 1 threshold exceeded → not god class
        m = ClassMetrics(class_name="Border", wmc=50, lcom=1, cbo=2, rfc=10)
        assert not is_god_class(m)


# ===================================================================
# Expert Rating
# ===================================================================

class TestExpertRating:

    def test_automated_scorer(self):
        scorer = AutomatedRubricScorer()
        rating = scorer.score(
            "resp-1", "prob-1",
            cyclomatic_complexity=3.0,
            wmc=10, cbo=3, lcom=2, rfc=15,
            readability_score=75.0,
            tests_passed=9, tests_total=10,
            loc=80,
        )
        assert rating.rater_id == "auto"
        assert 0 <= rating.correctness <= 10
        assert 0 <= rating.readability <= 10
        assert 0 <= rating.design <= 10
        assert 0 <= rating.overall <= 10

    def test_composite_score(self):
        rating = Rating(
            rating_id="r-1", response_id="resp-1", problem_id="prob-1",
            rater_id="human", correctness=8, readability=7,
            efficiency=6, design=7, overall=7,
        )
        cs = rating.composite_score
        # (8*2 + 7 + 6 + 7) / 5 = 36/5 = 7.2
        assert abs(cs - 7.2) < 0.01

    def test_csv_roundtrip(self):
        ratings = [
            Rating("r-1", "resp-1", "prob-1", "human", 8, 7, 6, 7, 7, "good"),
            Rating("r-2", "resp-2", "prob-2", "human", 5, 4, 5, 4, 5, "ok"),
        ]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            write_ratings_csv(ratings, path)
            loaded = read_ratings_csv(path)
            assert len(loaded) == 2
            assert loaded[0].correctness == 8
            assert loaded[1].correctness == 5
        finally:
            os.unlink(path)

    def test_blank_template(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            create_blank_rating_csv(
                path, ["resp-1", "resp-2"], ["prob-1", "prob-2"])
            loaded = read_ratings_csv(path)
            assert len(loaded) == 2
            assert all(r.correctness == 0 for r in loaded)
        finally:
            os.unlink(path)

    def test_rubric_text(self):
        text = get_rubric_text()
        assert "CORRECTNESS" in text
        assert "READABILITY" in text
        assert "0 to 10" in text


# ===================================================================
# Inter-Rater Reliability
# ===================================================================

class TestIRR:

    def test_perfect_agreement(self):
        r1 = [1, 2, 3, 4, 5]
        r2 = [1, 2, 3, 4, 5]
        result = cohens_kappa(r1, r2)
        assert result.kappa == pytest.approx(1.0)
        assert result.interpretation == "almost_perfect"

    def test_no_agreement(self):
        r1 = [1, 1, 1, 1]
        r2 = [2, 2, 2, 2]
        result = cohens_kappa(r1, r2)
        assert result.kappa < 0.01

    def test_moderate_agreement(self):
        r1 = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
        r2 = [1, 2, 3, 1, 3, 2, 1, 2, 2, 1]
        result = cohens_kappa(r1, r2)
        assert 0.3 < result.kappa < 0.9

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError):
            cohens_kappa([1, 2], [1, 2, 3])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            cohens_kappa([], [])

    def test_weighted_kappa_perfect(self):
        r1 = [1, 2, 3, 4, 5]
        r2 = [1, 2, 3, 4, 5]
        result = weighted_kappa(r1, r2)
        assert result.kappa == pytest.approx(1.0)

    def test_weighted_kappa_quadratic(self):
        r1 = [1, 2, 3, 4, 5]
        r2 = [2, 3, 3, 4, 5]
        result = weighted_kappa(r1, r2, "quadratic")
        assert 0 < result.kappa < 1

    def test_fleiss_kappa(self):
        # 5 items, 3 categories, 3 raters
        matrix = [
            [3, 0, 0],
            [0, 3, 0],
            [2, 1, 0],
            [0, 0, 3],
            [1, 1, 1],
        ]
        result = fleiss_kappa(matrix)
        assert -1 <= result.kappa <= 1

    def test_percent_agreement(self):
        r1 = [1, 2, 3, 4, 5]
        r2 = [1, 2, 3, 3, 5]
        assert percent_agreement(r1, r2) == pytest.approx(0.8)

    def test_interpret_kappa(self):
        assert interpret_kappa(-0.1) == "poor"
        assert interpret_kappa(0.1) == "slight"
        assert interpret_kappa(0.3) == "fair"
        assert interpret_kappa(0.5) == "moderate"
        assert interpret_kappa(0.7) == "substantial"
        assert interpret_kappa(0.9) == "almost_perfect"


# ===================================================================
# Diff Analysis
# ===================================================================

class TestDiffAnalysis:

    def test_identical_files(self):
        text = "line 1\nline 2\nline 3\n"
        diff_lines, stats = compute_diff(text, text)
        assert stats.lines_added == 0
        assert stats.lines_removed == 0
        assert stats.hunks == 0

    def test_added_lines(self):
        orig = "line 1\nline 2\n"
        mod = "line 1\nline 2\nline 3\n"
        _, stats = compute_diff(orig, mod)
        assert stats.lines_added == 1
        assert stats.lines_removed == 0

    def test_removed_lines(self):
        orig = "line 1\nline 2\nline 3\n"
        mod = "line 1\nline 3\n"
        _, stats = compute_diff(orig, mod)
        assert stats.lines_removed == 1

    def test_changed_line_numbers(self):
        orig = "a\nb\nc\nd\n"
        mod = "a\nB\nc\nD\n"
        changed = get_changed_line_numbers(orig, mod)
        assert 2 in changed  # 'b' → 'B'
        assert 4 in changed  # 'd' → 'D'

    def test_location_accuracy_perfect(self):
        orig = "a\nb\nc\nd\n"
        ref = "a\nB\nc\nd\n"
        llm = "a\nB\nc\nd\n"
        result = analyse_location_accuracy(orig, llm, ref)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1_score == 1.0

    def test_location_accuracy_partial(self):
        orig = "a\nb\nc\nd\n"
        ref = "a\nB\nc\nd\n"  # changes line 2
        llm = "a\nb\nC\nd\n"  # changes line 3 (wrong location)
        result = analyse_location_accuracy(orig, llm, ref)
        assert result.precision == 0.0
        assert result.recall == 0.0

    def test_patch_analysis(self):
        orig = "int x = 1;\nint y = 2;\nreturn x + y;\n"
        ref = "int x = 1;\nint y = 3;\nreturn x + y;\n"
        llm = "int x = 1;\nint y = 3;\nreturn x + y;\n"
        result = analyse_patch(orig, llm, ref)
        assert result.location_accuracy.precision == 1.0
        assert result.minimality_score == pytest.approx(1.0)

    def test_patch_analysis_oversized(self):
        orig = "a\nb\nc\n"
        ref = "a\nB\nc\n"  # 1 change
        llm = "A\nB\nC\n"  # 3 changes (overkill)
        result = analyse_patch(orig, llm, ref)
        assert result.minimality_score < 1.0


# ===================================================================
# Reports
# ===================================================================

class TestReports:

    def test_halstead_volume(self):
        src = "int x = 1; int y = 2; return x + y;"
        vol = _estimate_halstead(src)
        assert vol > 0

    def test_halstead_empty(self):
        assert _estimate_halstead("") == 0

    def test_maintainability_index(self):
        mi = _maintainability_index(100, 5, 50)
        assert 0 <= mi <= 100

    def test_maintainability_perfect(self):
        mi = _maintainability_index(0, 0, 0)
        assert mi == 100  # edge case: no code → perfect

    def test_file_quality_report(self):
        src = """
        public class Sample {
            private int value;
            public int getValue() { return value; }
            public void setValue(int v) { this.value = v; }
        }
        """
        gen = QualityReportGenerator()
        report = gen.analyse_file(src, "Sample.java", "resp-1", "prob-1")
        assert report.loc > 0
        assert len(report.ck_metrics) == 1
        assert report.complexity is not None
        assert report.readability is not None
        assert report.halstead_volume > 0
        assert 0 <= report.maintainability_index <= 100

    def test_quality_metrics_csv_row(self):
        gen = QualityReportGenerator()
        src = "public class X { public void m() {} }"
        report = gen.analyse_file(src, "X.java", "resp-1", "prob-1")
        row = report.to_quality_metrics_row()
        assert len(row) == 13  # matches QUALITY_METRICS_HEADER
        assert row[1] == "resp-1"  # response_id
        assert row[2] == "prob-1"  # problem_id

    def test_ck_metrics_csv_rows(self):
        gen = QualityReportGenerator()
        src = "public class Y { private int x; public int getX() { return x; } }"
        report = gen.analyse_file(src, "Y.java", "resp-1", "prob-1")
        rows = report.to_ck_metrics_rows()
        assert len(rows) == 1
        assert len(rows[0]) == 10  # matches CK_METRICS_HEADER

    def test_analyse_directory(self):
        if not FIXTURES.exists():
            pytest.skip("Fixtures not available")
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = QualityReportGenerator()
            reports = gen.analyse_directory(
                FIXTURES, tmpdir, "test_dataset", "test_model")
            assert len(reports) == 3  # 3 fixture files

            # Check CSV was written
            metrics_csv = Path(tmpdir) / "test_dataset" / "test_model" / "metrics.csv"
            assert metrics_csv.exists()
            with open(metrics_csv) as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)
            assert len(rows) == 3

            # Check CK metrics CSV
            ck_csv = Path(tmpdir) / "test_dataset" / "test_model" / "ck_metrics.csv"
            assert ck_csv.exists()

            # Check per-file JSON reports
            reports_dir = Path(tmpdir) / "test_dataset" / "test_model" / "reports"
            assert reports_dir.exists()
            json_files = list(reports_dir.glob("*.json"))
            assert len(json_files) == 3

    def test_fixture_quality_ordering(self):
        """High quality file should have better maintainability than low."""
        if not FIXTURES.exists():
            pytest.skip("Fixtures not available")
        gen = QualityReportGenerator()
        high_src = (FIXTURES / "HighQuality.java").read_text()
        low_src = (FIXTURES / "LowQuality.java").read_text()
        high = gen.analyse_file(high_src, "HighQuality.java")
        low = gen.analyse_file(low_src, "LowQuality.java")
        assert high.maintainability_index > low.maintainability_index
