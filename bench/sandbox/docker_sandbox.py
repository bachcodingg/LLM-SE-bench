"""
bench.sandbox.docker_sandbox — Docker-based Java compilation and test runner.

Manages a Docker container with JDK 8 and 17 (selectable), Maven/Gradle
build support, timeout enforcement, and structured output parsing.

Usage
-----
::

    sandbox = DockerSandbox(jdk_version=17, timeout_compile=60, timeout_test=120)
    result = sandbox.run(source_code, junit_code, problem_id="HumanEval_0")
    print(result.compiled, result.tests_passed, result.tests_total)

When Docker is not available (e.g. in CI or unit tests), the sandbox can
operate in ``dry_run`` mode, which performs structural checks only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    """Result of a sandbox compilation + test execution run."""

    problem_id: str = ""
    compiled: bool = False
    compile_stdout: str = ""
    compile_stderr: str = ""
    compile_time_ms: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0
    test_stdout: str = ""
    test_stderr: str = ""
    test_time_ms: float = 0.0
    exit_code: int = -1
    timed_out: bool = False
    error_message: str = ""
    individual_results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.compiled and self.tests_passed == self.tests_total and self.tests_total > 0

    @property
    def pass_rate(self) -> float:
        return self.tests_passed / self.tests_total if self.tests_total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "compiled": self.compiled,
            "compile_time_ms": self.compile_time_ms,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "test_time_ms": self.test_time_ms,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "error_message": self.error_message,
            "all_passed": self.all_passed,
            "pass_rate": self.pass_rate,
            "compile_stderr": self.compile_stderr[:2000],
            "test_stderr": self.test_stderr[:2000],
        }


class DockerSandbox:
    """
    Docker-based sandbox for compiling and testing Java code.

    Parameters
    ----------
    jdk_version : int
        JDK version to use (8 or 17, default 17).
    timeout_compile : int
        Maximum seconds for compilation (default 60).
    timeout_test : int
        Maximum seconds for test execution (default 120).
    docker_image : str | None
        Custom Docker image name.  When ``None``, uses the default
        ``llm-se-bench-sandbox:{jdk_version}`` image.
    work_dir : Path | str | None
        Host directory for staging files.  Auto-creates a temp dir if None.
    dry_run : bool
        When True, skip Docker and perform structural checks only.
    """

    DEFAULT_IMAGE_TEMPLATE = "llm-se-bench-sandbox:{jdk}"

    def __init__(
        self,
        jdk_version: int = 17,
        timeout_compile: int = 60,
        timeout_test: int = 120,
        docker_image: str | None = None,
        work_dir: Path | str | None = None,
        dry_run: bool = False,
    ) -> None:
        self.jdk_version = jdk_version
        self.timeout_compile = timeout_compile
        self.timeout_test = timeout_test
        self.docker_image = docker_image or self.DEFAULT_IMAGE_TEMPLATE.format(
            jdk=jdk_version
        )
        self.work_dir = Path(work_dir) if work_dir else None
        self.dry_run = dry_run
        self._docker_available: bool | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        source_code: str,
        junit_code: str,
        problem_id: str = "unknown",
        class_name: str | None = None,
        test_class_name: str | None = None,
    ) -> SandboxResult:
        """
        Compile and test Java source code in the sandbox.

        Parameters
        ----------
        source_code : str
            The generated Java source code.
        junit_code : str
            The JUnit test class source code.
        problem_id : str
            Problem identifier for tracking.
        class_name : str | None
            Main class name.  Auto-detected from source if None.
        test_class_name : str | None
            Test class name.  Auto-detected from junit_code if None.

        Returns
        -------
        SandboxResult
            Structured result with compilation and test outcomes.
        """
        if self.dry_run or not self.is_docker_available():
            return self._dry_run_check(source_code, junit_code, problem_id)

        # Auto-detect class names
        if class_name is None:
            class_name = self._extract_class_name(source_code)
        if test_class_name is None:
            test_class_name = self._extract_class_name(junit_code)

        if not class_name:
            return SandboxResult(
                problem_id=problem_id,
                error_message="Could not detect class name from source code",
            )

        # Stage files
        staging_dir = self._create_staging_dir(problem_id)
        try:
            self._write_source_files(
                staging_dir, source_code, junit_code, class_name, test_class_name
            )
            # Compile
            compile_result = self._docker_compile(staging_dir, class_name, test_class_name)
            if not compile_result.compiled:
                return compile_result

            # Run tests
            test_result = self._docker_test(staging_dir, test_class_name)
            # Merge results
            compile_result.tests_passed = test_result.tests_passed
            compile_result.tests_total = test_result.tests_total
            compile_result.test_stdout = test_result.test_stdout
            compile_result.test_stderr = test_result.test_stderr
            compile_result.test_time_ms = test_result.test_time_ms
            compile_result.exit_code = test_result.exit_code
            compile_result.timed_out = test_result.timed_out
            compile_result.individual_results = test_result.individual_results

            return compile_result

        except Exception as exc:
            logger.error("Sandbox error for %s: %s", problem_id, exc)
            return SandboxResult(
                problem_id=problem_id,
                error_message=f"Sandbox exception: {exc}",
            )
        finally:
            if self.work_dir is None:
                # Clean up temp dir
                shutil.rmtree(staging_dir, ignore_errors=True)

    def compile_files(
        self,
        files: dict[str, str],
        problem_id: str = "adhoc",
    ) -> SandboxResult:
        """
        Compile an arbitrary set of Java sources, without running anything.

        ``run`` handles the benchmark's shape — one class, one suite.  This
        handles the tool-layer's shape: N files, compile only, report
        diagnostics.  Test sources are staged alongside production sources
        and compiled together, because a suite that does not compile against
        the code under test is a compile failure worth reporting.

        Parameters
        ----------
        files : dict[str, str]
            ``{filename: content}``.  Names are flattened to their basename
            when staged, so ``a/b/C.java`` and ``C.java`` collide.
        problem_id : str
            Label used for the staging directory and in the result.

        Returns
        -------
        SandboxResult
            Only the compilation fields are populated; test counts stay 0.
        """
        java_files = {
            Path(name).name: content
            for name, content in files.items()
            if name.endswith(".java")
        }
        if not java_files:
            return SandboxResult(
                problem_id=problem_id,
                error_message="No .java files supplied",
            )

        if self.dry_run or not self.is_docker_available():
            combined = "\n".join(java_files.values())
            return self._dry_run_check(combined, "", problem_id)

        staging_dir = self._create_staging_dir(problem_id)
        try:
            src_dir = staging_dir / "src"
            src_dir.mkdir(parents=True, exist_ok=True)
            for name, content in java_files.items():
                (src_dir / name).write_text(content, encoding="utf-8")

            result = self._docker_compile_sources(staging_dir)
            result.problem_id = problem_id
            return result
        except Exception as exc:
            logger.error("Sandbox compile error for %s: %s", problem_id, exc)
            return SandboxResult(
                problem_id=problem_id,
                error_message=f"Sandbox exception: {exc}",
            )
        finally:
            if self.work_dir is None:
                shutil.rmtree(staging_dir, ignore_errors=True)

    def run_files(
        self,
        files: dict[str, str],
        test_class_name: str | None = None,
        problem_id: str = "adhoc",
    ) -> SandboxResult:
        """
        Compile an arbitrary set of Java sources and run one JUnit class.

        Parameters
        ----------
        files : dict[str, str]
            ``{filename: content}``, production and test sources together.
        test_class_name : str | None
            Test class to hand to ``JUnitCore``.  When None, the first file
            whose source contains ``@Test`` supplies it.
        problem_id : str
            Label used for the staging directory and in the result.

        Returns
        -------
        SandboxResult
            Compilation *and* test fields.  A compile failure short-circuits:
            no tests are run and ``tests_total`` stays 0.
        """
        java_files = {
            Path(name).name: content
            for name, content in files.items()
            if name.endswith(".java")
        }
        if not java_files:
            return SandboxResult(
                problem_id=problem_id,
                error_message="No .java files supplied",
            )

        if test_class_name is None:
            for name, content in java_files.items():
                if "@Test" in content:
                    test_class_name = self._extract_class_name(content) or Path(name).stem
                    break

        if self.dry_run or not self.is_docker_available():
            sources = "\n".join(
                content for content in java_files.values() if "@Test" not in content
            )
            tests = "\n".join(
                content for content in java_files.values() if "@Test" in content
            )
            return self._dry_run_check(sources or tests, tests, problem_id)

        staging_dir = self._create_staging_dir(problem_id)
        try:
            src_dir = staging_dir / "src"
            src_dir.mkdir(parents=True, exist_ok=True)
            for name, content in java_files.items():
                (src_dir / name).write_text(content, encoding="utf-8")

            result = self._docker_compile_sources(staging_dir)
            result.problem_id = problem_id
            if not result.compiled or not test_class_name:
                if not test_class_name:
                    result.error_message = "No test class found in the supplied sources"
                return result

            test_result = self._docker_test(staging_dir, test_class_name)
            result.tests_passed = test_result.tests_passed
            result.tests_total = test_result.tests_total
            result.test_stdout = test_result.test_stdout
            result.test_stderr = test_result.test_stderr
            result.test_time_ms = test_result.test_time_ms
            result.exit_code = test_result.exit_code
            result.timed_out = test_result.timed_out
            result.individual_results = test_result.individual_results
            return result
        except Exception as exc:
            logger.error("Sandbox run error for %s: %s", problem_id, exc)
            return SandboxResult(
                problem_id=problem_id,
                error_message=f"Sandbox exception: {exc}",
            )
        finally:
            if self.work_dir is None:
                shutil.rmtree(staging_dir, ignore_errors=True)

    def is_docker_available(self) -> bool:
        """Check if Docker daemon is reachable."""
        if self._docker_available is not None:
            return self._docker_available
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )
            self._docker_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._docker_available = False
        if not self._docker_available:
            logger.warning("Docker not available — sandbox will use dry-run mode")
        return self._docker_available

    def build_image(self, dockerfile_path: Path | str | None = None) -> bool:
        """
        Build the sandbox Docker image.

        Parameters
        ----------
        dockerfile_path : Path | str | None
            Path to Dockerfile.  Uses the default bundled Dockerfile if None.

        Returns
        -------
        bool
            True if the build succeeded.
        """
        if dockerfile_path is None:
            dockerfile_path = Path(__file__).parent / "Dockerfile"
        dockerfile_path = Path(dockerfile_path)

        if not dockerfile_path.exists():
            logger.error("Dockerfile not found: %s", dockerfile_path)
            return False

        try:
            result = subprocess.run(
                [
                    "docker", "build",
                    "-t", self.docker_image,
                    "-f", str(dockerfile_path),
                    str(dockerfile_path.parent),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                logger.info("Built Docker image: %s", self.docker_image)
                return True
            logger.error("Docker build failed: %s", result.stderr[:500])
            return False
        except Exception as exc:
            logger.error("Docker build error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _create_staging_dir(self, problem_id: str) -> Path:
        """Create a temporary directory for staging Java files."""
        if self.work_dir:
            staging = self.work_dir / f"sandbox_{problem_id}"
            staging.mkdir(parents=True, exist_ok=True)
            return staging
        return Path(tempfile.mkdtemp(prefix=f"sandbox_{problem_id}_"))

    def _write_source_files(
        self,
        staging_dir: Path,
        source_code: str,
        junit_code: str,
        class_name: str,
        test_class_name: str | None,
    ) -> None:
        """Write source and test files to the staging directory."""
        src_dir = staging_dir / "src"
        test_dir = staging_dir / "test"
        src_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        (src_dir / f"{class_name}.java").write_text(source_code, encoding="utf-8")
        if test_class_name and junit_code:
            (test_dir / f"{test_class_name}.java").write_text(junit_code, encoding="utf-8")

    def _docker_compile(
        self, staging_dir: Path, class_name: str, test_class_name: str | None
    ) -> SandboxResult:
        """Compile Java sources inside Docker container."""
        result = SandboxResult(problem_id=class_name)

        cmd = [
            "docker", "run", "--rm",
            "--network=none",
            f"--memory=256m",
            f"--cpus=1",
            "-v", f"{staging_dir}:/workspace:rw",
            self.docker_image,
            "bash", "-c",
            (
                "cd /workspace && "
                "javac -encoding UTF-8 -cp /usr/share/java/junit4.jar:. "
                "-d out src/*.java test/*.java 2>&1"
            ),
        ]

        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_compile,
            )
            elapsed = (time.monotonic() - start) * 1000
            result.compile_time_ms = elapsed
            result.compile_stdout = proc.stdout
            result.compile_stderr = proc.stderr

            if proc.returncode == 0:
                result.compiled = True
                logger.debug("Compilation succeeded for %s (%.0fms)", class_name, elapsed)
            else:
                result.error_message = f"Compilation failed: {proc.stderr[:500]}"
                logger.debug("Compilation failed for %s: %s", class_name, proc.stderr[:200])

        except subprocess.TimeoutExpired:
            result.timed_out = True
            result.error_message = f"Compilation timed out after {self.timeout_compile}s"
            result.compile_time_ms = self.timeout_compile * 1000

        return result

    def _docker_compile_sources(self, staging_dir: Path) -> SandboxResult:
        """Compile every ``src/*.java`` in *staging_dir* inside the container.

        Differs from :meth:`_docker_compile` only in globbing ``src/`` rather
        than the fixed ``src/ + test/`` pair, so it works for N files.
        """
        result = SandboxResult()

        cmd = [
            "docker", "run", "--rm",
            "--network=none",
            "--memory=256m",
            "--cpus=1",
            "-v", f"{staging_dir}:/workspace:rw",
            self.docker_image,
            "bash", "-c",
            (
                "cd /workspace && "
                "javac -encoding UTF-8 -cp /usr/share/java/junit4.jar:. "
                "-d out src/*.java 2>&1"
            ),
        ]

        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_compile,
            )
            result.compile_time_ms = (time.monotonic() - start) * 1000
            result.compile_stdout = proc.stdout
            result.compile_stderr = proc.stderr
            result.compiled = proc.returncode == 0
            if not result.compiled:
                combined = (proc.stdout + proc.stderr).strip()
                result.error_message = f"Compilation failed: {combined[:500]}"
        except subprocess.TimeoutExpired:
            result.timed_out = True
            result.error_message = f"Compilation timed out after {self.timeout_compile}s"
            result.compile_time_ms = self.timeout_compile * 1000

        return result

    def _docker_test(self, staging_dir: Path, test_class_name: str | None) -> SandboxResult:
        """Run JUnit tests inside Docker container."""
        result = SandboxResult()

        if not test_class_name:
            result.error_message = "No test class to run"
            return result

        cmd = [
            "docker", "run", "--rm",
            "--network=none",
            f"--memory=256m",
            f"--cpus=1",
            "-v", f"{staging_dir}:/workspace:rw",
            self.docker_image,
            "bash", "-c",
            (
                "cd /workspace && "
                f"java -cp /usr/share/java/junit4.jar:out "
                f"org.junit.runner.JUnitCore {test_class_name} 2>&1"
            ),
        ]

        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_test,
            )
            elapsed = (time.monotonic() - start) * 1000
            result.test_time_ms = elapsed
            result.test_stdout = proc.stdout
            result.test_stderr = proc.stderr
            result.exit_code = proc.returncode

            # Parse JUnit output
            result.tests_total, result.tests_passed = self._parse_junit_output(
                proc.stdout + proc.stderr
            )

        except subprocess.TimeoutExpired:
            result.timed_out = True
            result.error_message = f"Tests timed out after {self.timeout_test}s"
            result.test_time_ms = self.timeout_test * 1000

        return result

    def _dry_run_check(
        self, source_code: str, junit_code: str, problem_id: str
    ) -> SandboxResult:
        """Perform structural checks without Docker."""
        result = SandboxResult(problem_id=problem_id)

        # Check if source has a class definition
        class_name = self._extract_class_name(source_code)
        if not class_name:
            result.error_message = "No class definition found in source code"
            return result

        # Check for basic Java syntax markers
        has_method = (
            "public " in source_code
            and ("static " in source_code or "void " in source_code or "int " in source_code
                 or "String " in source_code or "boolean " in source_code
                 or "double " in source_code or "List " in source_code)
        )
        has_braces = source_code.count("{") == source_code.count("}")

        result.compiled = bool(class_name and has_braces)
        if result.compiled:
            result.compile_stdout = f"Dry-run: structural check passed for {class_name}"
        else:
            result.error_message = "Dry-run: structural check failed (unbalanced braces)"

        # Count test methods in junit_code
        test_methods = junit_code.count("@Test")
        result.tests_total = test_methods
        if result.compiled and has_method:
            result.tests_passed = test_methods  # Optimistic in dry-run
        else:
            result.tests_passed = 0

        return result

    @staticmethod
    def _extract_class_name(source: str) -> str | None:
        """Extract the public class name from Java source code."""
        import re
        match = re.search(r"public\s+class\s+(\w+)", source)
        if match:
            return match.group(1)
        # Fallback: any class
        match = re.search(r"class\s+(\w+)", source)
        return match.group(1) if match else None

    @staticmethod
    def _parse_junit_output(output: str) -> tuple[int, int]:
        """
        Parse JUnit 4 text output to extract test counts.

        Returns (tests_total, tests_passed).
        """
        import re

        total, passed = 0, 0

        # JUnit 4: "OK (3 tests)"
        ok_match = re.search(r"OK \((\d+) tests?\)", output)
        if ok_match:
            total = int(ok_match.group(1))
            return total, total

        # JUnit 4: "Tests run: 5,  Failures: 2"
        run_match = re.search(r"Tests run:\s*(\d+)", output)
        fail_match = re.search(r"Failures:\s*(\d+)", output)
        err_match = re.search(r"Errors:\s*(\d+)", output)

        if run_match:
            total = int(run_match.group(1))
            failures = int(fail_match.group(1)) if fail_match else 0
            errors = int(err_match.group(1)) if err_match else 0
            passed = total - failures - errors

        return total, max(0, passed)
