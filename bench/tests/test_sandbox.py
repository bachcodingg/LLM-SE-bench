"""
tests.test_sandbox — Unit tests for DockerSandbox.

These tests use dry-run mode so they don't require Docker.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from bench.sandbox.docker_sandbox import DockerSandbox, SandboxResult


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def sandbox(tmp_path: Path) -> DockerSandbox:
    """Sandbox in dry-run mode."""
    return DockerSandbox(
        jdk_version=17,
        timeout_compile=60,
        timeout_test=120,
        work_dir=tmp_path / "sandbox_work",
        dry_run=True,
    )


@pytest.fixture
def valid_java_code() -> str:
    return (
        "import java.util.List;\n\n"
        "public class Solution {\n"
        "    public static int add(int a, int b) {\n"
        "        return a + b;\n"
        "    }\n"
        "}\n"
    )


@pytest.fixture
def valid_junit_code() -> str:
    return (
        "import org.junit.Test;\n"
        "import static org.junit.Assert.*;\n\n"
        "public class SolutionTest {\n"
        "    @Test\n"
        "    public void testAdd() {\n"
        "        assertEquals(3, Solution.add(1, 2));\n"
        "    }\n\n"
        "    @Test\n"
        "    public void testAddNegative() {\n"
        "        assertEquals(-1, Solution.add(1, -2));\n"
        "    }\n"
        "}\n"
    )


# ======================================================================
# SandboxResult Tests
# ======================================================================

class TestSandboxResult:

    def test_default_values(self):
        r = SandboxResult()
        assert r.compiled is False
        assert r.tests_passed == 0
        assert r.tests_total == 0
        assert r.all_passed is False
        assert r.pass_rate == 0.0

    def test_all_passed(self):
        r = SandboxResult(compiled=True, tests_passed=3, tests_total=3)
        assert r.all_passed is True
        assert r.pass_rate == 1.0

    def test_partial_pass(self):
        r = SandboxResult(compiled=True, tests_passed=2, tests_total=5)
        assert r.all_passed is False
        assert r.pass_rate == pytest.approx(0.4)

    def test_compile_failure(self):
        r = SandboxResult(compiled=False, error_message="syntax error")
        assert r.all_passed is False
        assert r.pass_rate == 0.0

    def test_to_dict(self):
        r = SandboxResult(
            problem_id="test",
            compiled=True,
            tests_passed=2,
            tests_total=3,
            exit_code=0,
        )
        d = r.to_dict()
        assert d["problem_id"] == "test"
        assert d["compiled"] is True
        assert d["tests_passed"] == 2
        assert d["tests_total"] == 3
        assert d["pass_rate"] == pytest.approx(2 / 3)
        assert "all_passed" in d

    def test_to_dict_truncates_stderr(self):
        long_stderr = "x" * 5000
        r = SandboxResult(compile_stderr=long_stderr)
        d = r.to_dict()
        assert len(d["compile_stderr"]) == 2000


# ======================================================================
# DockerSandbox Tests (dry-run mode)
# ======================================================================

class TestDockerSandbox:

    def test_dry_run_valid_code(
        self, sandbox: DockerSandbox, valid_java_code: str, valid_junit_code: str
    ):
        result = sandbox.run(
            source_code=valid_java_code,
            junit_code=valid_junit_code,
            problem_id="test_problem",
        )
        assert isinstance(result, SandboxResult)
        assert result.compiled is True
        assert result.tests_total == 2  # 2 @Test methods
        assert result.tests_passed == 2  # Optimistic in dry-run

    def test_dry_run_empty_code(self, sandbox: DockerSandbox):
        result = sandbox.run(
            source_code="",
            junit_code="",
            problem_id="empty",
        )
        assert result.compiled is False
        assert "No class definition" in result.error_message

    def test_dry_run_no_class(self, sandbox: DockerSandbox):
        result = sandbox.run(
            source_code="int x = 5;",
            junit_code="",
            problem_id="no_class",
        )
        assert result.compiled is False

    def test_dry_run_unbalanced_braces(self, sandbox: DockerSandbox):
        code = "public class Broken { public void foo() {"
        result = sandbox.run(
            source_code=code,
            junit_code="",
            problem_id="broken",
        )
        assert result.compiled is False

    def test_extract_class_name(self):
        assert DockerSandbox._extract_class_name(
            "public class Foo {"
        ) == "Foo"
        assert DockerSandbox._extract_class_name(
            "class Bar extends Base {"
        ) == "Bar"
        assert DockerSandbox._extract_class_name("int x = 5;") is None

    def test_extract_class_name_inner(self):
        code = (
            "public class Outer {\n"
            "    class Inner { }\n"
            "}\n"
        )
        assert DockerSandbox._extract_class_name(code) == "Outer"

    def test_parse_junit_output_ok(self):
        output = "OK (5 tests)"
        total, passed = DockerSandbox._parse_junit_output(output)
        assert total == 5
        assert passed == 5

    def test_parse_junit_output_failures(self):
        output = "Tests run: 5,  Failures: 2"
        total, passed = DockerSandbox._parse_junit_output(output)
        assert total == 5
        assert passed == 3

    def test_parse_junit_output_errors(self):
        output = "Tests run: 10,  Failures: 1,  Errors: 2"
        total, passed = DockerSandbox._parse_junit_output(output)
        assert total == 10
        assert passed == 7

    def test_parse_junit_output_no_match(self):
        total, passed = DockerSandbox._parse_junit_output("garbage output")
        assert total == 0
        assert passed == 0

    def test_parse_junit_single_test(self):
        output = "OK (1 test)"
        total, passed = DockerSandbox._parse_junit_output(output)
        assert total == 1
        assert passed == 1

    def test_configuration(self, sandbox: DockerSandbox):
        assert sandbox.jdk_version == 17
        assert sandbox.timeout_compile == 60
        assert sandbox.timeout_test == 120
        assert sandbox.dry_run is True

    def test_docker_image_default(self):
        sb = DockerSandbox(jdk_version=8, dry_run=True)
        assert sb.docker_image == "llm-se-bench-sandbox:8"

    def test_docker_image_custom(self):
        sb = DockerSandbox(docker_image="custom-image:latest", dry_run=True)
        assert sb.docker_image == "custom-image:latest"

    def test_staging_dir_creation(self, sandbox: DockerSandbox):
        staging = sandbox._create_staging_dir("test_123")
        assert staging.exists()
        assert "test_123" in str(staging)

    def test_write_source_files(self, sandbox: DockerSandbox, valid_java_code: str, valid_junit_code: str):
        staging = sandbox._create_staging_dir("write_test")
        sandbox._write_source_files(
            staging, valid_java_code, valid_junit_code, "Solution", "SolutionTest"
        )
        assert (staging / "src" / "Solution.java").exists()
        assert (staging / "test" / "SolutionTest.java").exists()
        assert (staging / "src" / "Solution.java").read_text() == valid_java_code
