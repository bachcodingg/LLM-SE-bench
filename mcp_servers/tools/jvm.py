"""
mcp_servers.tools.jvm — compile, test and analyse Java, as tool functions.

Wraps :class:`~bench.sandbox.docker_sandbox.DockerSandbox` (C2) and the
quality analyser (C3).  Everything that crosses the tool boundary is
bounded: source input is size-capped, paths are validated, logs are
truncated, and the sandbox's own compile and test timeouts are configurable
per call.

Scope, stated plainly because the tool descriptions promise it: this
compiles with ``javac`` and runs JUnit 4 on a flat set of source files. It
does not drive Maven or Gradle, and it does not resolve dependencies.
Repository-level build support is a separate problem and is not solved here.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

from mcp_servers.models import (
    CKMetricsResult,
    ClassMetricsModel,
    CodeSmell,
    CompileJavaResult,
    Diagnostic,
    RunTestsResult,
    SourceFile,
    StaticAnalysisResult,
    TestOutcome,
)
from mcp_servers.truncation import truncate_log

logger = logging.getLogger(__name__)

__all__ = [
    "compile_java",
    "run_tests",
    "compute_ck_metrics",
    "run_static_analysis",
    "MAX_TOTAL_SOURCE_CHARS",
    "MAX_FILES",
    "SUPPORTED_JDK_VERSIONS",
]

#: Total source characters accepted in one call.  Roughly 25k tokens.
MAX_TOTAL_SOURCE_CHARS = 200_000

#: Files accepted in one call.
MAX_FILES = 50

#: JDK versions the sandbox image supports.
SUPPORTED_JDK_VERSIONS = (8, 17)

#: Files read when a tool is pointed at a directory.
MAX_DIRECTORY_FILES = 500

# javac: ``Foo.java:12: error: cannot find symbol``
_JAVAC_DIAGNOSTIC = re.compile(
    r"^(?P<file>[^\s:][^:]*\.java):(?P<line>\d+):\s*"
    r"(?P<severity>error|warning|note):\s*(?P<message>.*)$"
)

# Thresholds shared with quality/refactoring.py.  Imported lazily there to
# keep this module importable without javalang.
_SMELL_THRESHOLDS = {
    "wmc": 47,
    "cbo": 14,
    "rfc": 50,
    "lcom": 1,
    "loc": 300,
    "dit": 5,
}


# ──────────────────────────────────────────────────────────────────────
# Input validation
# ──────────────────────────────────────────────────────────────────────

class ToolInputError(ValueError):
    """Raised for input a tool refuses, with a message meant for an agent."""


def _validate_sources(source_files: Iterable[SourceFile]) -> dict[str, str]:
    """Return ``{basename: content}``, rejecting unsafe or oversized input.

    Path traversal is rejected rather than sanitised: an agent that wrote
    ``../../etc/passwd`` needs to be told, not quietly corrected.
    """
    files = list(source_files)
    if not files:
        raise ToolInputError("No source files supplied.")
    if len(files) > MAX_FILES:
        raise ToolInputError(
            f"{len(files)} files supplied; at most {MAX_FILES} are accepted "
            f"in one call."
        )

    total = sum(len(f.content) for f in files)
    if total > MAX_TOTAL_SOURCE_CHARS:
        raise ToolInputError(
            f"Sources total {total} characters; the limit is "
            f"{MAX_TOTAL_SOURCE_CHARS}. Split the call."
        )

    staged: dict[str, str] = {}
    for source in files:
        name = source.path.replace("\\", "/")
        if not name.endswith(".java"):
            raise ToolInputError(f"{source.path!r} is not a .java file.")
        if name.startswith("/") or ".." in Path(name).parts:
            raise ToolInputError(
                f"{source.path!r} escapes the workspace. Use a relative path "
                f"with no '..' segment."
            )
        basename = Path(name).name
        if basename in staged:
            raise ToolInputError(
                f"Two files stage to the same name {basename!r}; sources are "
                f"flattened, so file names must be unique."
            )
        staged[basename] = source.content
    return staged


def _read_directory(path: str, suffix: str = ".java") -> tuple[dict[str, str], list[str]]:
    """Read every matching file under *path*. Returns (files, skipped)."""
    root = Path(path).expanduser()
    if not root.exists():
        raise ToolInputError(f"{path!r} does not exist.")
    if not root.is_dir():
        raise ToolInputError(f"{path!r} is not a directory.")

    files: dict[str, str] = {}
    skipped: list[str] = []
    for candidate in sorted(root.rglob(f"*{suffix}")):
        if len(files) >= MAX_DIRECTORY_FILES:
            skipped.append(f"{candidate}: file limit of {MAX_DIRECTORY_FILES} reached")
            continue
        try:
            files[str(candidate.relative_to(root)).replace("\\", "/")] = (
                candidate.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError) as exc:
            skipped.append(f"{candidate}: {exc}")
    if not files:
        raise ToolInputError(f"No {suffix} files found under {path!r}.")
    return files, skipped


def _make_sandbox(jdk_version: int, timeout_s: int, compile_timeout_s: int):
    """Construct a DockerSandbox, importing it lazily."""
    from bench.sandbox.docker_sandbox import DockerSandbox

    return DockerSandbox(
        jdk_version=jdk_version,
        timeout_compile=compile_timeout_s,
        timeout_test=timeout_s,
    )


# ──────────────────────────────────────────────────────────────────────
# Output parsing
# ──────────────────────────────────────────────────────────────────────

def parse_javac_diagnostics(output: str, limit: int = 50) -> list[Diagnostic]:
    """Parse javac output into structured diagnostics, errors first.

    Unparseable lines are dropped, which is why every result also carries
    the raw log: a caller that finds no diagnostics on a failed compile
    needs somewhere to look.
    """
    found: list[Diagnostic] = []
    for line in output.splitlines():
        match = _JAVAC_DIAGNOSTIC.match(line.strip())
        if not match:
            continue
        found.append(
            Diagnostic(
                file=Path(match.group("file")).name,
                line=int(match.group("line")),
                severity=match.group("severity"),  # type: ignore[arg-type]
                message=match.group("message").strip(),
            )
        )
    found.sort(key=lambda d: (d.severity != "error", d.file, d.line))
    return found[:limit]


def parse_junit_outcomes(output: str, limit: int = 100) -> list[TestOutcome]:
    """Extract per-test outcomes from JUnit 4 console output.

    JUnit 4's text runner names failures but not passes, so this reports
    failures individually and leaves passes to the aggregate counts.
    """
    outcomes: list[TestOutcome] = []
    # "1) testFoo(com.example.BarTest)"
    for match in re.finditer(r"^\d+\)\s*(\w+)\(([\w.$]+)\)\s*$", output, re.MULTILINE):
        method, cls = match.group(1), match.group(2)
        tail = output[match.end():match.end() + 400].strip().splitlines()
        detail = tail[0].strip() if tail else ""
        outcomes.append(
            TestOutcome(
                test_id=f"{cls.rsplit('.', 1)[-1]}.{method}",
                passed=False,
                error_message=detail[:300],
            )
        )
        if len(outcomes) >= limit:
            break
    return outcomes


def _to_metrics_model(metrics) -> ClassMetricsModel:
    """Convert a C3 ``ClassMetrics`` dataclass into the wire model."""
    return ClassMetricsModel(
        class_name=metrics.class_name,
        package_name=metrics.package_name,
        loc=metrics.loc,
        wmc=metrics.wmc,
        dit=metrics.dit,
        noc=metrics.noc,
        cbo=metrics.cbo,
        rfc=metrics.rfc,
        lcom=metrics.lcom,
        num_methods=metrics.num_methods,
        num_fields=metrics.num_fields,
        parent_class=metrics.parent_class,
    )


def _severity_for(ratio: float) -> str:
    """Map "how far past threshold" onto the project's 3-level scale."""
    if ratio >= 3.0:
        return "extreme"
    if ratio >= 2.0:
        return "severe"
    return "moderate"


# ──────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────

def compile_java(
    source_files: list[SourceFile],
    jdk_version: int = 17,
    timeout_s: int = 60,
) -> CompileJavaResult:
    """Compile Java sources in the sandbox and return structured diagnostics.

    Compilation only — nothing is executed. Use :func:`run_tests` to run a
    suite.
    """
    if jdk_version not in SUPPORTED_JDK_VERSIONS:
        return CompileJavaResult(
            ok=False,
            error=f"jdk_version must be one of {list(SUPPORTED_JDK_VERSIONS)}; "
                  f"got {jdk_version}.",
        )
    try:
        files = _validate_sources(source_files)
    except ToolInputError as exc:
        return CompileJavaResult(ok=False, error=str(exc))

    sandbox = _make_sandbox(jdk_version, timeout_s=timeout_s, compile_timeout_s=timeout_s)
    executed = sandbox.is_docker_available()
    result = sandbox.compile_files(files, problem_id="mcp-compile")

    raw = "\n".join(
        part for part in (result.compile_stdout, result.compile_stderr, result.error_message)
        if part
    )
    log, truncation = truncate_log(raw)

    return CompileJavaResult(
        ok=True,
        compiled=result.compiled,
        diagnostics=parse_javac_diagnostics(raw),
        log=log,
        truncation=truncation,
        jdk_version=jdk_version,
        duration_ms=result.compile_time_ms,
        timed_out=result.timed_out,
        executed=executed,
    )


def run_tests(
    source_files: list[SourceFile] | None = None,
    project_path: str = "",
    test_filter: str = "",
    timeout_s: int = 120,
    jdk_version: int = 17,
) -> RunTestsResult:
    """Compile sources and run a JUnit 4 suite in the sandbox.

    Supply either *source_files* or *project_path*, not both.
    """
    if bool(source_files) == bool(project_path):
        return RunTestsResult(
            ok=False,
            error="Supply exactly one of source_files or project_path.",
        )

    try:
        if project_path:
            files, skipped = _read_directory(project_path)
            if skipped:
                logger.warning("run_tests skipped %d file(s)", len(skipped))
        else:
            files = _validate_sources(source_files or [])
    except ToolInputError as exc:
        return RunTestsResult(ok=False, error=str(exc))

    if test_filter:
        matching = {
            name: content
            for name, content in files.items()
            if "@Test" not in content or test_filter in name
        }
        if not any("@Test" in content for content in matching.values()):
            return RunTestsResult(
                ok=False,
                error=f"test_filter {test_filter!r} matched no test class. "
                      f"Test classes present: "
                      f"{sorted(n for n, c in files.items() if '@Test' in c)}",
            )
        files = matching

    sandbox = _make_sandbox(jdk_version, timeout_s=timeout_s, compile_timeout_s=timeout_s)
    executed = sandbox.is_docker_available()
    result = sandbox.run_files(files, problem_id="mcp-tests")

    raw = "\n".join(
        part for part in (
            result.compile_stdout, result.compile_stderr,
            result.test_stdout, result.test_stderr, result.error_message,
        ) if part
    )
    log, truncation = truncate_log(raw)

    return RunTestsResult(
        ok=True,
        compiled=result.compiled,
        passed=result.tests_passed,
        failed=max(0, result.tests_total - result.tests_passed),
        total=result.tests_total,
        outcomes=parse_junit_outcomes(raw),
        log=log,
        truncation=truncation,
        duration_ms=result.compile_time_ms + result.test_time_ms,
        timed_out=result.timed_out,
        executed=executed,
    )


def compute_ck_metrics(
    source_files: list[SourceFile] | None = None,
    class_path: str = "",
) -> CKMetricsResult:
    """Extract CK metrics from Java source. Parses only — nothing is run.

    Supply either *source_files* or *class_path* (a .java file or a
    directory), not both.
    """
    if bool(source_files) == bool(class_path):
        return CKMetricsResult(
            ok=False,
            error="Supply exactly one of source_files or class_path.",
        )

    try:
        if class_path:
            path = Path(class_path).expanduser()
            if path.is_dir():
                files, _ = _read_directory(class_path)
            elif path.is_file():
                files = {path.name: path.read_text(encoding="utf-8")}
            else:
                raise ToolInputError(f"{class_path!r} does not exist.")
        else:
            files = _validate_sources(source_files or [])
    except ToolInputError as exc:
        return CKMetricsResult(ok=False, error=str(exc))
    except (OSError, UnicodeDecodeError) as exc:
        return CKMetricsResult(ok=False, error=f"Could not read {class_path!r}: {exc}")

    from quality.ck_metrics import CKMetricsExtractor

    extractor = CKMetricsExtractor()
    classes: list[ClassMetricsModel] = []
    failures: list[str] = []
    for name, content in files.items():
        try:
            extracted = extractor.extract(content, name)
        except Exception as exc:  # javalang raises a wide variety
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if not extracted:
            failures.append(f"{name}: parsed but declared no class")
            continue
        classes.extend(_to_metrics_model(item) for item in extracted)

    return CKMetricsResult(
        ok=True,
        classes=classes,
        files_analysed=len(files),
        parse_failures=failures,
    )


def run_static_analysis(
    project_path: str = "",
    source_files: list[SourceFile] | None = None,
) -> StaticAnalysisResult:
    """Extract CK metrics and flag design smells. Parses only — nothing is run.

    Detects: god_class (2+ of WMC/LCOM/CBO/RFC past threshold),
    high_coupling, low_cohesion, large_class and deep_inheritance.
    """
    metrics = compute_ck_metrics(source_files=source_files, class_path=project_path)
    if not metrics.ok:
        return StaticAnalysisResult(ok=False, error=metrics.error)

    smells: list[CodeSmell] = []
    for cls in metrics.classes:
        breached = {
            name: getattr(cls, name) / _SMELL_THRESHOLDS[name]
            for name in ("wmc", "lcom", "cbo", "rfc")
            if getattr(cls, name) > _SMELL_THRESHOLDS[name]
        }
        cls_metrics = {
            "wmc": float(cls.wmc), "cbo": float(cls.cbo),
            "rfc": float(cls.rfc), "lcom": float(cls.lcom),
            "loc": float(cls.loc), "dit": float(cls.dit),
        }

        if len(breached) >= 2:
            smells.append(
                CodeSmell(
                    smell="god_class",
                    severity=_severity_for(max(breached.values())),  # type: ignore[arg-type]
                    class_name=cls.class_name,
                    detail=(
                        "breaches "
                        + ", ".join(
                            f"{name.upper()}={getattr(cls, name)} "
                            f"(>{_SMELL_THRESHOLDS[name]})"
                            for name in sorted(breached)
                        )
                    ),
                    metrics=cls_metrics,
                )
            )

        if cls.cbo > _SMELL_THRESHOLDS["cbo"]:
            smells.append(
                CodeSmell(
                    smell="high_coupling",
                    severity=_severity_for(cls.cbo / _SMELL_THRESHOLDS["cbo"]),  # type: ignore[arg-type]
                    class_name=cls.class_name,
                    detail=f"CBO={cls.cbo} (>{_SMELL_THRESHOLDS['cbo']}); "
                           f"java.* types are already excluded",
                    metrics=cls_metrics,
                )
            )
        if cls.lcom > _SMELL_THRESHOLDS["lcom"]:
            smells.append(
                CodeSmell(
                    smell="low_cohesion",
                    severity=_severity_for(cls.lcom / _SMELL_THRESHOLDS["lcom"]),  # type: ignore[arg-type]
                    class_name=cls.class_name,
                    detail=f"LCOM={cls.lcom}: the class has that many "
                           f"independent groups of methods and fields",
                    metrics=cls_metrics,
                )
            )
        if cls.loc > _SMELL_THRESHOLDS["loc"]:
            smells.append(
                CodeSmell(
                    smell="large_class",
                    severity=_severity_for(cls.loc / _SMELL_THRESHOLDS["loc"]),  # type: ignore[arg-type]
                    class_name=cls.class_name,
                    detail=f"{cls.loc} logical lines (>{_SMELL_THRESHOLDS['loc']})",
                    metrics=cls_metrics,
                )
            )
        if cls.dit > _SMELL_THRESHOLDS["dit"]:
            smells.append(
                CodeSmell(
                    smell="deep_inheritance",
                    severity=_severity_for(cls.dit / _SMELL_THRESHOLDS["dit"]),  # type: ignore[arg-type]
                    class_name=cls.class_name,
                    detail=f"DIT={cls.dit} (>{_SMELL_THRESHOLDS['dit']})",
                    metrics=cls_metrics,
                )
            )

    order = {"extreme": 0, "severe": 1, "moderate": 2}
    smells.sort(key=lambda s: (order[s.severity], s.class_name, s.smell))

    return StaticAnalysisResult(
        ok=True,
        classes=metrics.classes,
        smells=smells,
        files_analysed=metrics.files_analysed,
        parse_failures=metrics.parse_failures,
    )
