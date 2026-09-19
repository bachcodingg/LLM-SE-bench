"""
Tests for quality.structural (M5).

The differentiator, and the thing it has to get right: a refactoring that
changes nothing passes every test, so structural scoring must credit real
improvement and refuse to credit the cheap imitations of it.
"""

from __future__ import annotations

import pytest

from quality.structural import (
    SCORE_WEIGHTS,
    diff_public_api,
    extract_public_api,
    score_structure,
)

GOD_CLASS = """\
public class OrderProcessor {
    private java.util.List<String> orders;
    private double taxRate;
    private String reportHeader;
    private int reportCount;
    private java.io.File auditFile;

    public void addOrder(String order) {
        if (order != null) { orders.add(order); }
    }

    public double computeTotal(double amount) {
        if (amount < 0) { return 0; }
        for (int i = 0; i < orders.size(); i++) { amount += i; }
        return amount * (1 + taxRate);
    }

    public int orderCount() {
        return orders.size();
    }

    public String buildReport() {
        reportCount = reportCount + 1;
        while (reportCount > 100) { reportCount--; }
        return reportHeader + " #" + reportCount;
    }

    public void setReportHeader(String header) {
        reportHeader = header;
    }

    public void audit(String message) {
        if (auditFile != null) { auditFile.getName(); }
    }
}
"""

ORDERS_PART = """\
public class OrderManager {
    private java.util.List<String> orders;
    private double taxRate;

    public void addOrder(String order) {
        if (order != null) { orders.add(order); }
    }

    public double computeTotal(double amount) {
        if (amount < 0) { return 0; }
        for (int i = 0; i < orders.size(); i++) { amount += i; }
        return amount * (1 + taxRate);
    }

    public int orderCount() {
        return orders.size();
    }
}
"""

REPORT_PART = """\
public class ReportBuilder {
    private String reportHeader;
    private int reportCount;
    private java.io.File auditFile;

    public String buildReport() {
        reportCount = reportCount + 1;
        while (reportCount > 100) { reportCount--; }
        return reportHeader + " #" + reportCount;
    }

    public void setReportHeader(String header) {
        reportHeader = header;
    }

    public void audit(String message) {
        if (auditFile != null) { auditFile.getName(); }
    }
}
"""

EMPTY_SHELL_A = "public class OrderManager { private java.util.List<String> orders; }"
EMPTY_SHELL_B = "public class ReportBuilder { private String reportHeader; }"


class TestPublicApiExtraction:
    def test_finds_public_methods(self):
        api = extract_public_api(GOD_CLASS)
        names = {signature.name for signature in api}
        assert {"addOrder", "computeTotal", "orderCount", "buildReport"} <= names

    def test_excludes_private_methods(self):
        source = "public class A { private void hidden() {} public void shown() {} }"
        assert {s.name for s in extract_public_api(source)} == {"shown"}

    def test_excludes_package_private_methods(self):
        source = "public class A { void packagePrivate() {} public void shown() {} }"
        assert {s.name for s in extract_public_api(source)} == {"shown"}

    def test_includes_protected_methods(self):
        source = "public class A { protected void shielded() {} }"
        assert {s.name for s in extract_public_api(source)} == {"shielded"}

    def test_excludes_constructors(self):
        source = "public class Widget { public Widget() {} public void go() {} }"
        assert {s.name for s in extract_public_api(source)} == {"go"}

    def test_excludes_control_flow_keywords(self):
        source = "public class A { public void go() { if (x) { } for (;;) { } } }"
        assert {s.name for s in extract_public_api(source)} == {"go"}

    def test_records_parameter_types(self):
        source = "public class A { public int add(int a, int b) { return a; } }"
        signature = next(iter(extract_public_api(source)))
        assert signature.parameter_types == ("int", "int")
        assert signature.return_type == "int"

    def test_package_qualification_is_normalised(self):
        """java.util.List and List are the same dependency to a caller."""
        qualified = "public class A { public java.util.List<String> get() { return null; } }"
        plain = "public class A { public List<String> get() { return null; } }"
        assert extract_public_api(qualified) == extract_public_api(plain)

    def test_static_is_recorded(self):
        source = "public class A { public static void main(String[] args) {} }"
        assert next(iter(extract_public_api(source))).is_static is True


class TestApiDiff:
    def test_a_clean_split_preserves_everything(self):
        diff = diff_public_api(GOD_CLASS, [ORDERS_PART, REPORT_PART])
        assert diff.breaking is False
        assert diff.preserved_fraction == 1.0
        assert diff.removed == []

    def test_a_dropped_method_is_breaking(self):
        diff = diff_public_api(GOD_CLASS, [ORDERS_PART])
        assert diff.breaking is True
        assert any(s.name == "buildReport" for s in diff.removed)

    def test_methods_may_move_between_classes(self):
        """A method that moved is retained, not removed: behaviour survived."""
        diff = diff_public_api(GOD_CLASS, [REPORT_PART, ORDERS_PART])
        assert diff.breaking is False

    def test_new_methods_are_recorded_separately(self):
        extra = ORDERS_PART.replace(
            "public int orderCount()", "public void brandNew() {}\n    public int orderCount()"
        )
        diff = diff_public_api(GOD_CLASS, [extra, REPORT_PART])
        assert any(s.name == "brandNew" for s in diff.added)
        assert diff.breaking is False


class TestScoring:
    def test_a_real_split_scores_positively(self):
        score = score_structure(GOD_CLASS, [ORDERS_PART, REPORT_PART], tests_pass=True)
        assert score.composite > 0
        assert score.guard.gamed is False
        assert score.credited is True

    def test_empty_shells_score_zero(self):
        """Deleting the bodies improves every CK metric. It must not score."""
        score = score_structure(GOD_CLASS, [EMPTY_SHELL_A, EMPTY_SHELL_B], tests_pass=True)
        assert score.guard.gamed is True
        assert score.composite == 0.0
        assert score.credited is False

    def test_empty_shells_are_named_in_the_guard(self):
        score = score_structure(GOD_CLASS, [EMPTY_SHELL_A, EMPTY_SHELL_B])
        assert score.guard.empty_classes
        assert any("empty shells" in note for note in score.guard.notes)

    def test_breaking_the_api_is_not_credited(self):
        score = score_structure(GOD_CLASS, [ORDERS_PART], tests_pass=True)
        assert score.api.breaking is True
        assert score.guard.api_broken is True
        assert score.credited is False

    def test_deleting_most_of_the_code_is_caught(self):
        score = score_structure(GOD_CLASS, [EMPTY_SHELL_A])
        assert score.guard.size_collapsed or score.guard.behaviour_lost

    def test_failing_tests_are_not_credited(self):
        score = score_structure(GOD_CLASS, [ORDERS_PART, REPORT_PART], tests_pass=False)
        assert score.behaviour_preserved is False
        assert score.credited is False

    def test_unknown_test_result_is_not_assumed_to_pass(self):
        """An unrun check is not a passed check."""
        score = score_structure(GOD_CLASS, [ORDERS_PART, REPORT_PART], tests_pass=None)
        assert score.credited is False

    def test_differential_disagreement_blocks_credit(self):
        score = score_structure(
            GOD_CLASS, [ORDERS_PART, REPORT_PART],
            tests_pass=True, differential_agreement=0.8,
        )
        assert score.behaviour_preserved is False
        assert score.credited is False

    def test_full_differential_agreement_allows_credit(self):
        score = score_structure(
            GOD_CLASS, [ORDERS_PART, REPORT_PART],
            tests_pass=True, differential_agreement=1.0,
        )
        assert score.credited is True

    def test_a_single_resulting_class_disperses_nothing(self):
        score = score_structure(GOD_CLASS, [GOD_CLASS], tests_pass=True)
        assert score.complexity_dispersal == 0.0
        assert any("nothing was dispersed" in w for w in score.warnings)

    def test_weights_sum_to_one(self):
        assert sum(SCORE_WEIGHTS.values()) == pytest.approx(1.0)

    def test_axes_are_reported_separately(self):
        """A composite hides disagreement between axes, which is the finding."""
        payload = score_structure(
            GOD_CLASS, [ORDERS_PART, REPORT_PART], tests_pass=True
        ).to_dict()
        assert set(payload["structure"]["axes"]) == set(SCORE_WEIGHTS)
        assert payload["structure"]["weights"] == SCORE_WEIGHTS
        assert "before" in payload["structure"]
        assert "after" in payload["structure"]

    def test_unparseable_original_is_a_warning_not_a_crash(self):
        score = score_structure("this is not java {{{", [ORDERS_PART])
        assert score.composite == 0.0
        assert score.warnings

    def test_no_decomposed_classes_is_a_warning(self):
        score = score_structure(GOD_CLASS, ["// just a comment"])
        assert score.warnings
        assert score.credited is False
