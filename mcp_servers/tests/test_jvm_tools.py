"""
Tests for mcp_servers.tools.jvm.

Docker is not required: the sandbox degrades to a structural check and the
tools report that with ``executed=False``. Tests therefore assert on the
tool contract — validation, parsing, truncation, the `executed` flag — and
not on compilation outcomes, which would need a container.
"""

from __future__ import annotations

import pytest

from mcp_servers.models import SourceFile
from mcp_servers.tools.jvm import (
    MAX_FILES,
    MAX_TOTAL_SOURCE_CHARS,
    compile_java,
    compute_ck_metrics,
    parse_javac_diagnostics,
    parse_junit_outcomes,
    run_static_analysis,
    run_tests,
)


def _sources(**files: str) -> list[SourceFile]:
    return [SourceFile(path=name, content=body) for name, body in files.items()]


class TestInputValidation:
    def test_empty_source_list_is_rejected(self):
        result = compile_java([])
        assert result.ok is False
        assert "No source files" in result.error

    def test_non_java_extension_is_rejected(self):
        result = compile_java([SourceFile(path="Main.txt", content="x")])
        assert result.ok is False
        assert ".java" in result.error

    @pytest.mark.parametrize("path", ["../escape.java", "/etc/passwd.java", "a/../../b.java"])
    def test_path_traversal_is_rejected_not_sanitised(self, path):
        result = compile_java([SourceFile(path=path, content="class A {}")])
        assert result.ok is False
        assert "escapes the workspace" in result.error

    def test_colliding_basenames_are_rejected(self):
        result = compile_java([
            SourceFile(path="a/Main.java", content="class Main {}"),
            SourceFile(path="b/Main.java", content="class Main {}"),
        ])
        assert result.ok is False
        assert "same name" in result.error

    def test_too_many_files_is_rejected(self):
        files = [
            SourceFile(path=f"C{i}.java", content="class C {}")
            for i in range(MAX_FILES + 1)
        ]
        result = compile_java(files)
        assert result.ok is False
        assert str(MAX_FILES) in result.error

    def test_oversized_payload_is_rejected(self):
        result = compile_java([
            SourceFile(path="Big.java", content="x" * (MAX_TOTAL_SOURCE_CHARS + 1))
        ])
        assert result.ok is False
        assert "Split the call" in result.error

    def test_unsupported_jdk_is_rejected(self, simple_class):
        result = compile_java(_sources(**{"Counter.java": simple_class}), jdk_version=21)
        assert result.ok is False
        assert "jdk_version" in result.error


class TestExclusiveArguments:
    """Tools taking either inline sources or a path must reject both/neither."""

    def test_run_tests_rejects_both(self, simple_class, tmp_path):
        result = run_tests(
            source_files=_sources(**{"Counter.java": simple_class}),
            project_path=str(tmp_path),
        )
        assert result.ok is False
        assert "exactly one" in result.error

    def test_run_tests_rejects_neither(self):
        result = run_tests()
        assert result.ok is False
        assert "exactly one" in result.error

    def test_ck_metrics_rejects_both(self, simple_class, tmp_path):
        result = compute_ck_metrics(
            source_files=_sources(**{"Counter.java": simple_class}),
            class_path=str(tmp_path),
        )
        assert result.ok is False
        assert "exactly one" in result.error


class TestCompileJava:
    def test_valid_source_is_accepted(self, simple_class):
        result = compile_java(_sources(**{"Counter.java": simple_class}))
        assert result.ok is True
        assert result.jdk_version == 17
        assert isinstance(result.executed, bool)

    def test_result_always_carries_a_truncation_record(self, simple_class):
        result = compile_java(_sources(**{"Counter.java": simple_class}))
        assert result.truncation is not None
        assert isinstance(result.truncation.truncated, bool)

    def test_compile_failure_is_ok_true(self, broken_class):
        """A compile failure means the tool worked and the code did not."""
        result = compile_java(_sources(**{"Broken.java": broken_class}))
        assert result.ok is True


class TestDiagnosticParsing:
    def test_parses_javac_error(self):
        output = "Main.java:12: error: cannot find symbol\n  symbol: var x\n1 error"
        diagnostics = parse_javac_diagnostics(output)
        assert len(diagnostics) == 1
        assert diagnostics[0].file == "Main.java"
        assert diagnostics[0].line == 12
        assert diagnostics[0].severity == "error"
        assert "cannot find symbol" in diagnostics[0].message

    def test_errors_sort_before_warnings(self):
        output = (
            "A.java:1: warning: deprecated\n"
            "B.java:2: error: bad\n"
        )
        diagnostics = parse_javac_diagnostics(output)
        assert [d.severity for d in diagnostics] == ["error", "warning"]

    def test_unparseable_output_yields_nothing(self):
        assert parse_javac_diagnostics("something went wrong somewhere") == []

    def test_respects_the_limit(self):
        output = "\n".join(f"F.java:{i}: error: e{i}" for i in range(200))
        assert len(parse_javac_diagnostics(output, limit=10)) == 10


class TestJUnitOutcomeParsing:
    def test_parses_a_failure_block(self):
        output = (
            "There was 1 failure:\n"
            "1) testAdd(com.example.CalcTest)\n"
            "java.lang.AssertionError: expected:<3> but was:<4>\n"
        )
        outcomes = parse_junit_outcomes(output)
        assert len(outcomes) == 1
        assert outcomes[0].test_id == "CalcTest.testAdd"
        assert outcomes[0].passed is False
        assert "expected" in outcomes[0].error_message

    def test_passing_run_reports_no_individual_outcomes(self):
        """JUnit 4's text runner names failures but not passes."""
        assert parse_junit_outcomes("OK (3 tests)") == []


class TestCKMetrics:
    def test_extracts_from_inline_source(self, simple_class):
        result = compute_ck_metrics(source_files=_sources(**{"Counter.java": simple_class}))
        assert result.ok is True
        assert result.files_analysed == 1
        assert [c.class_name for c in result.classes] == ["Counter"]
        counter = result.classes[0]
        assert counter.num_methods == 2
        assert counter.num_fields == 1
        assert counter.wmc >= 2

    def test_reads_a_directory(self, java_dir):
        result = compute_ck_metrics(class_path=str(java_dir))
        assert result.ok is True
        assert result.files_analysed == 2
        assert {c.class_name for c in result.classes} == {"Counter", "OrderProcessor"}

    def test_reads_a_single_file(self, java_dir):
        result = compute_ck_metrics(class_path=str(java_dir / "Counter.java"))
        assert result.ok is True
        assert result.files_analysed == 1

    def test_missing_path_is_an_error_not_an_empty_result(self, tmp_path):
        result = compute_ck_metrics(class_path=str(tmp_path / "nope"))
        assert result.ok is False
        assert "does not exist" in result.error

    def test_unparseable_source_is_reported_not_swallowed(self, broken_class):
        result = compute_ck_metrics(source_files=_sources(**{"Broken.java": broken_class}))
        assert result.ok is True
        assert result.parse_failures
        assert "Broken.java" in result.parse_failures[0]

    def test_god_class_metrics_breach_thresholds(self, god_class):
        result = compute_ck_metrics(source_files=_sources(**{"Everything.java": god_class}))
        assert result.ok is True
        metrics = result.classes[0]
        assert metrics.wmc > 47
        assert metrics.rfc > 50


class TestStaticAnalysis:
    def test_clean_class_has_no_smells(self, simple_class):
        result = run_static_analysis(source_files=_sources(**{"Counter.java": simple_class}))
        assert result.ok is True
        assert result.smells == []

    def test_god_class_is_flagged(self, god_class):
        result = run_static_analysis(source_files=_sources(**{"Everything.java": god_class}))
        assert result.ok is True
        assert any(s.smell == "god_class" for s in result.smells)

    def test_findings_carry_their_evidence(self, god_class):
        result = run_static_analysis(source_files=_sources(**{"Everything.java": god_class}))
        finding = next(s for s in result.smells if s.smell == "god_class")
        assert finding.metrics["wmc"] > 47
        assert "WMC" in finding.detail
        assert finding.severity in {"moderate", "severe", "extreme"}

    def test_smells_are_sorted_most_severe_first(self, god_class, simple_class):
        result = run_static_analysis(source_files=_sources(**{
            "Everything.java": god_class,
            "Counter.java": simple_class,
        }))
        order = {"extreme": 0, "severe": 1, "moderate": 2}
        severities = [order[s.severity] for s in result.smells]
        assert severities == sorted(severities)

    def test_propagates_a_bad_path_as_an_error(self, tmp_path):
        result = run_static_analysis(project_path=str(tmp_path / "missing"))
        assert result.ok is False


class TestRunTests:
    def test_unmatched_filter_is_an_error_not_a_silent_pass(self, simple_class, valid_junit):
        result = run_tests(
            source_files=_sources(**{
                "Counter.java": simple_class,
                "CounterTest.java": valid_junit,
            }),
            test_filter="NoSuchTest",
        )
        assert result.ok is False
        assert "matched no test class" in result.error

    def test_accepts_matching_filter(self, simple_class, valid_junit):
        result = run_tests(
            source_files=_sources(**{
                "Counter.java": simple_class,
                "CounterTest.java": valid_junit,
            }),
            test_filter="CounterTest",
        )
        assert result.ok is True

    def test_empty_directory_is_an_error(self, tmp_path):
        result = run_tests(project_path=str(tmp_path))
        assert result.ok is False
        assert "No .java files" in result.error

    def test_counts_are_internally_consistent(self, simple_class, valid_junit):
        result = run_tests(source_files=_sources(**{
            "Counter.java": simple_class,
            "CounterTest.java": valid_junit,
        }))
        assert result.ok is True
        assert result.passed + result.failed == result.total
