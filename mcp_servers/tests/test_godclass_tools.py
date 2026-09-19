"""
Tests for mcp_servers.tools.godclass.

The two properties worth protecting: the field-cluster planner must actually
find independent responsibilities, and score_decomposition must not reward a
"decomposition" that deleted the code instead of moving it.
"""

from __future__ import annotations

import pytest

from mcp_servers.models import SourceFile
from mcp_servers.tools.godclass import (
    STRATEGIES,
    detect_god_classes,
    propose_decomposition,
    score_decomposition,
)

# The two halves of TWO_CLUSTER_CLASS, split the way a correct
# decomposition would split it.
ORDERS_PART = """\
public class OrderManager {
    private java.util.List<String> orders;
    private double taxRate;

    public void addOrder(String order) {
        if (order != null) {
            orders.add(order);
        }
    }

    public double computeTotal(double amount) {
        if (amount < 0) {
            return 0;
        }
        return amount * orders.size() * (1 + taxRate);
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

    public String buildReport() {
        reportCount = reportCount + 1;
        return reportHeader + " #" + reportCount;
    }

    public void setReportHeader(String header) {
        reportHeader = header;
    }
}
"""

# The gaming attempt: the same two class names, with the bodies deleted.
EMPTY_SHELL_A = """\
public class OrderManager {
    private java.util.List<String> orders;
}
"""

EMPTY_SHELL_B = """\
public class ReportBuilder {
    private String reportHeader;
}
"""


def _sources(**files: str) -> list[SourceFile]:
    return [SourceFile(path=name, content=body) for name, body in files.items()]


class TestDetectGodClasses:
    def test_flags_a_god_class(self, god_class):
        result = detect_god_classes(source_files=_sources(**{"Everything.java": god_class}))
        assert result.ok is True
        assert len(result.god_classes) == 1
        finding = result.god_classes[0]
        assert finding.class_name == "Everything"
        assert len(finding.violations) >= 2
        assert finding.metrics.wmc > 47

    def test_leaves_a_clean_class_alone(self, simple_class):
        result = detect_god_classes(source_files=_sources(**{"Counter.java": simple_class}))
        assert result.ok is True
        assert result.god_classes == []
        assert result.classes_analysed == 1

    def test_reads_a_directory(self, java_dir):
        result = detect_god_classes(project_path=str(java_dir))
        assert result.ok is True
        assert result.files_analysed == 2
        assert result.classes_analysed == 2

    def test_rejects_both_arguments(self, java_dir, simple_class):
        result = detect_god_classes(
            project_path=str(java_dir),
            source_files=_sources(**{"Counter.java": simple_class}),
        )
        assert result.ok is False
        assert "exactly one" in result.error

    def test_rejects_neither_argument(self):
        result = detect_god_classes()
        assert result.ok is False

    def test_zero_findings_from_zero_classes_is_distinguishable(self, broken_class):
        """0 of 0 means nothing parsed; 0 of 40 means the code is clean."""
        result = detect_god_classes(source_files=_sources(**{"Broken.java": broken_class}))
        assert result.ok is True
        assert result.classes_analysed == 0
        assert result.parse_failures


class TestProposeDecomposition:
    def test_finds_the_two_clusters(self, two_cluster_class):
        """The orders methods and the report methods share no field."""
        result = propose_decomposition(source=two_cluster_class)
        assert result.ok is True
        assert result.strategy == "field_clusters"
        assert len(result.extracted_classes) == 2

        clusters = {frozenset(plan.methods) for plan in result.extracted_classes}
        assert frozenset({"addOrder", "computeTotal", "orderCount"}) in clusters
        assert frozenset({"buildReport", "setReportHeader"}) in clusters

    def test_every_method_is_assigned(self, two_cluster_class):
        result = propose_decomposition(source=two_cluster_class)
        assigned = {m for plan in result.extracted_classes for m in plan.methods}
        assert assigned == {
            "addOrder", "computeTotal", "orderCount", "buildReport", "setReportHeader"
        }
        assert not any("unassigned" in w for w in result.warnings)

    def test_estimated_wmc_is_populated(self, two_cluster_class):
        result = propose_decomposition(source=two_cluster_class)
        assert all(plan.estimated_wmc >= 1 for plan in result.extracted_classes)

    def test_cohesive_class_warns_rather_than_splitting_blindly(self, simple_class):
        result = propose_decomposition(source=simple_class)
        assert result.ok is True
        assert len(result.extracted_classes) == 1
        assert any("single cohesive cluster" in w for w in result.warnings)

    def test_warns_when_the_class_is_not_a_god_class(self, simple_class):
        result = propose_decomposition(source=simple_class)
        assert result.is_god_class is False
        assert any("fewer than 2" in w for w in result.warnings)

    def test_recognises_a_god_class(self, god_class):
        result = propose_decomposition(source=god_class)
        assert result.ok is True
        assert result.is_god_class is True
        assert result.metrics_before is not None
        assert result.metrics_before.wmc > 47

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_every_strategy_produces_a_plan(self, two_cluster_class, strategy):
        result = propose_decomposition(source=two_cluster_class, strategy=strategy)
        assert result.ok is True
        assert result.extracted_classes

    def test_layered_separates_accessors_from_behaviour(self, two_cluster_class):
        result = propose_decomposition(source=two_cluster_class, strategy="layered")
        names = {plan.proposed_name for plan in result.extracted_classes}
        assert "OrderProcessorData" in names
        assert "OrderProcessorService" in names

    def test_falls_back_when_no_method_touches_a_field(self):
        """A class of static helpers gives field clustering nothing to work with."""
        static_only = """\
public class MathUtils {
    public static int addNumbers(int a, int b) { return a + b; }
    public static int addThree(int a, int b, int c) { return a + b + c; }
    public static String formatValue(int v) { return String.valueOf(v); }
}
"""
        result = propose_decomposition(source=static_only, strategy="field_clusters")
        assert result.ok is True
        assert result.strategy == "responsibility"
        assert any("fell back" in w for w in result.warnings)

    def test_unknown_strategy_is_rejected(self, simple_class):
        result = propose_decomposition(source=simple_class, strategy="vibes")
        assert result.ok is False
        assert "field_clusters" in result.error

    def test_rejects_both_arguments(self, simple_class, tmp_path):
        result = propose_decomposition(source=simple_class, class_path=str(tmp_path / "x.java"))
        assert result.ok is False
        assert "exactly one" in result.error

    def test_unparseable_source_is_an_error(self, broken_class):
        result = propose_decomposition(source=broken_class)
        assert result.ok is False
        assert "parse" in result.error.lower()

    def test_class_with_no_methods_is_an_error(self):
        result = propose_decomposition(source="public class Empty { private int x; }")
        assert result.ok is False
        assert "no methods" in result.error

    def test_reads_a_file(self, java_dir):
        result = propose_decomposition(class_path=str(java_dir / "OrderProcessor.java"))
        assert result.ok is True
        assert result.class_name == "OrderProcessor"

    def test_is_deterministic(self, two_cluster_class):
        first = propose_decomposition(source=two_cluster_class)
        second = propose_decomposition(source=two_cluster_class)
        assert first.model_dump() == second.model_dump()


class TestScoreDecomposition:
    def test_scores_a_real_split(self, two_cluster_class):
        result = score_decomposition(two_cluster_class, [ORDERS_PART, REPORT_PART])
        assert result.ok is True
        assert result.original_class == "OrderProcessor"
        assert set(result.decomposed_classes) == {"OrderManager", "ReportBuilder"}
        assert result.method_coverage == 1.0
        assert result.field_coverage == 1.0
        assert result.deltas

    def test_catches_the_empty_shell_attack(self, two_cluster_class):
        """Deleting the bodies improves every CK metric. Coverage must catch it."""
        result = score_decomposition(two_cluster_class, [EMPTY_SHELL_A, EMPTY_SHELL_B])
        assert result.ok is True
        assert result.method_coverage == 0.0
        assert any("dropped, not moved" in w or "methods" in w for w in result.warnings)

    def test_coverage_separates_deletion_from_movement(self, two_cluster_class):
        moved = score_decomposition(two_cluster_class, [ORDERS_PART, REPORT_PART])
        deleted = score_decomposition(two_cluster_class, [EMPTY_SHELL_A, EMPTY_SHELL_B])
        assert moved.method_coverage > deleted.method_coverage

    def test_rejects_an_empty_after_entry(self, two_cluster_class):
        result = score_decomposition(two_cluster_class, [ORDERS_PART, "   "])
        assert result.ok is False
        assert "empty extracted class" in result.error

    def test_rejects_empty_before(self):
        result = score_decomposition("", [ORDERS_PART])
        assert result.ok is False

    def test_rejects_empty_after_list(self, two_cluster_class):
        result = score_decomposition(two_cluster_class, [])
        assert result.ok is False

    def test_deltas_cover_the_ck_metrics(self, two_cluster_class):
        result = score_decomposition(two_cluster_class, [ORDERS_PART, REPORT_PART])
        metrics = {delta.metric for delta in result.deltas}
        assert {"wmc", "cbo", "rfc", "lcom", "loc"} <= metrics

    def test_coupling_delta_matches_the_cbo_delta(self, two_cluster_class):
        result = score_decomposition(two_cluster_class, [ORDERS_PART, REPORT_PART])
        cbo = next(delta for delta in result.deltas if delta.metric == "cbo")
        assert result.coupling_delta == cbo.delta

    def test_wmc_drops_when_a_class_is_genuinely_split(self, two_cluster_class):
        result = score_decomposition(two_cluster_class, [ORDERS_PART, REPORT_PART])
        wmc = next(delta for delta in result.deltas if delta.metric == "wmc")
        assert wmc.after < wmc.before
        assert wmc.improved is True
