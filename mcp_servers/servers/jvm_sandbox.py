"""
mcp_servers.servers.jvm_sandbox — the ``jvm-sandbox`` MCP server.

Compile Java, run JUnit suites, and analyse structure, all inside the
project's Docker sandbox.
"""

from __future__ import annotations

from mcp_servers._compat import require_sdk
from mcp_servers.models import (
    CKMetricsResult,
    CompileJavaResult,
    RunTestsResult,
    SourceFile,
    StaticAnalysisResult,
)
from mcp_servers.tools import jvm

__all__ = ["build", "INSTRUCTIONS"]

INSTRUCTIONS = """\
Compile and test Java inside a hermetic sandbox, and analyse its structure.

Scope: javac plus JUnit 4 over a flat set of source files. There is no Maven
or Gradle support and no dependency resolution, so a project with external
dependencies will not build here.

Execution happens in a container started with --network=none, 256 MB of
memory and one CPU. Code you compile cannot reach the network.

When Docker is unavailable every tool still returns a result, with
`executed: false`, having run a structural check instead: balanced braces, a
class declaration, a count of @Test methods. Those results are optimistic by
construction. Never report a structural check as a compilation or test
result.
"""


def build(**kwargs):
    """Build the jvm-sandbox server with its four tools registered."""
    server_class = require_sdk()
    server = server_class(
        name="jvm-sandbox",
        instructions=INSTRUCTIONS,
        **kwargs,
    )

    @server.tool(
        description="""\
Compile Java source files in a hermetic sandbox and return structured
compiler diagnostics. Nothing is executed: use run_tests to run a suite.

Each file's `path` must be a relative .java filename matching the public
class it declares ('Calculator.java' for `public class Calculator`). Paths
may not start with '/' or contain '..'. Names are flattened to their
basename, so two files cannot share one. At most 50 files and 200,000
characters per call.

`jdk_version` must be 8 or 17. `timeout_s` bounds compilation; on timeout
the result has `timed_out: true` and `compiled` is meaningless.

Returns ok=true with compiled=false for a normal compile failure — the tool
worked, the code did not. ok=false means the call itself was rejected and
`error` says why. `diagnostics` may be empty on a failure if javac emitted
something unparseable; `log` always holds the raw output, truncated, with
`truncation.truncated` set when anything was cut.""",
    )
    def compile_java(
        source_files: list[SourceFile],
        jdk_version: int = 17,
        timeout_s: int = 60,
    ) -> CompileJavaResult:
        return jvm.compile_java(source_files, jdk_version=jdk_version, timeout_s=timeout_s)

    @server.tool(
        description="""\
Compile Java sources and run their JUnit 4 suite in a hermetic sandbox.

Supply exactly one of `source_files` (inline, same rules as compile_java) or
`project_path` (a directory read recursively for .java files). Supplying
both, or neither, is an error.

Test classes are identified by containing '@Test'; the first such class is
run unless `test_filter` narrows it, in which case `test_filter` is matched
as a substring against test file names.

`timeout_s` bounds the whole run. On timeout `timed_out` is true and the
counts are partial: a non-zero `failed` then does not mean an assertion
failed.

A compile failure short-circuits: `compiled` is false, `total` is 0, and no
test ran. total=0 with compiled=true means no @Test method was found, not
that an empty suite passed. JUnit 4's text runner names failures but not
passes, so `outcomes` lists failures individually and passes are only in the
aggregate counts.""",
    )
    def run_tests(
        source_files: list[SourceFile] | None = None,
        project_path: str = "",
        test_filter: str = "",
        timeout_s: int = 120,
        jdk_version: int = 17,
    ) -> RunTestsResult:
        return jvm.run_tests(
            source_files=source_files,
            project_path=project_path,
            test_filter=test_filter,
            timeout_s=timeout_s,
            jdk_version=jdk_version,
        )

    @server.tool(
        description="""\
Extract Chidamber-Kemerer metrics from Java source. Parses only; nothing is
compiled or executed, so this works without Docker and on code that does not
compile.

Supply exactly one of `source_files` (inline) or `class_path` (a .java file
or a directory read recursively).

Returns one entry per class declared, with WMC, DIT, NOC, CBO, RFC, LCOM,
LOC and member counts. Lower is better for all six metrics. CBO excludes
java.* standard library types, so coupling to String does not count.

A file that fails to parse is listed in `parse_failures` and is silently
absent from `classes` — check that list before treating the metrics as
complete.""",
    )
    def compute_ck_metrics(
        source_files: list[SourceFile] | None = None,
        class_path: str = "",
    ) -> CKMetricsResult:
        return jvm.compute_ck_metrics(source_files=source_files, class_path=class_path)

    @server.tool(
        description="""\
Extract CK metrics and flag design smells in Java source. Parses only;
nothing is compiled or executed.

Supply exactly one of `project_path` (a directory) or `source_files`.

Detects god_class (2 or more of WMC>47, LCOM>1, CBO>14, RFC>50),
high_coupling, low_cohesion, large_class (LOC>300) and deep_inheritance
(DIT>5). Severity is how far past threshold the metrics sit: moderate below
2x, severe below 3x, extreme beyond. Findings are returned most severe
first and each carries the metric values behind it, so the judgement can be
checked rather than trusted.

An empty `smells` list means nothing breached a threshold. It does not mean
the code is good: these are six specific structural measures, not a review.""",
    )
    def run_static_analysis(
        project_path: str = "",
        source_files: list[SourceFile] | None = None,
    ) -> StaticAnalysisResult:
        return jvm.run_static_analysis(
            project_path=project_path, source_files=source_files
        )

    return server


def main() -> None:
    """Run the server on stdio."""
    build().run(transport="stdio")


if __name__ == "__main__":
    main()
