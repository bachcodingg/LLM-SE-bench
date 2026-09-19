"""
mcp_servers.servers.god_class — the ``god-class-tools`` MCP server.

Detect God Classes, plan a decomposition, and score one.
"""

from __future__ import annotations

from mcp_servers._compat import require_sdk
from mcp_servers.models import (
    DetectGodClassesResult,
    ProposeDecompositionResult,
    ScoreDecompositionResult,
    SourceFile,
)
from mcp_servers.tools import godclass

__all__ = ["build", "INSTRUCTIONS"]

INSTRUCTIONS = """\
Detect God Classes in Java, plan a decomposition, and score one.

A class is a God Class when it breaches at least 2 of 4 thresholds:
WMC > 47, LCOM > 1, CBO > 14, RFC > 50.

Every tool here is static analysis. Nothing is compiled, nothing is
executed, no model is called. propose_decomposition is deterministic for a
given input and writes no code — it says what to extract, not how.

Scoring a decomposition does not run tests. A refactoring that changes
nothing passes every test, so behaviour preservation is necessary and not
sufficient. Use the jvm-sandbox run_tests tool separately and require both
a passing suite and a structural improvement.
"""


def build(**kwargs):
    """Build the god-class-tools server with its three tools registered."""
    server_class = require_sdk()
    server = server_class(
        name="god-class-tools",
        instructions=INSTRUCTIONS,
        **kwargs,
    )

    @server.tool(
        description="""\
Find God Classes in Java sources. Parses only; nothing is compiled or
executed.

Supply exactly one of `project_path` (a directory read recursively) or
`source_files` (inline {path, content} entries; .java names, relative, no
'..').

A class is flagged when it breaches at least 2 of: WMC > 47, LCOM > 1,
CBO > 14, RFC > 50. Each finding names which thresholds broke and carries
the full metric set, so the judgement can be checked.

Findings come back most severe first. Severity is how far past threshold the
worst breached metric sits: moderate below 2x, severe below 3x, extreme
beyond. `classes_analysed` counts every class seen, not just flagged ones,
so 0 findings out of 40 classes is a meaningful result and 0 out of 0 means
nothing parsed — check `parse_failures`.""",
    )
    def detect_god_classes(
        project_path: str = "",
        source_files: list[SourceFile] | None = None,
    ) -> DetectGodClassesResult:
        return godclass.detect_god_classes(
            project_path=project_path, source_files=source_files
        )

    @server.tool(
        description="""\
Plan a decomposition of one God Class. Static analysis: deterministic, no
API call, and it writes no code — it returns which members to extract and
why.

Supply exactly one of `source` (Java source text) or `class_path` (a .java
file). The first class declared is analysed.

`strategy`:
  field_clusters (default) — partition methods into connected components
    over the fields they access. This is LCOM4's own definition of cohesion,
    so each component is an independent responsibility by that measure. Best
    for a class with real state.
  responsibility — group by shared method-name verb prefix (loadX/saveX).
    Reads intent from naming, so it is only as good as the naming. Used
    automatically as a fallback when no method touches a field.
  layered — split trivial accessors from behaviour. Useful only for a data
    class that accreted logic.

A single entry in `extracted_classes` means the class does not split along
the axis this strategy measures; try another strategy before splitting it
anyway. Read `warnings`: they say when the class was not a God Class to
begin with, when a fallback was applied, and when members were left
unassigned.""",
    )
    def propose_decomposition(
        source: str = "",
        class_path: str = "",
        strategy: str = "field_clusters",
    ) -> ProposeDecompositionResult:
        return godclass.propose_decomposition(
            source=source, class_path=class_path, strategy=strategy
        )

    @server.tool(
        description="""\
Score a proposed decomposition on structural improvement and behaviour
preservation. Parses only; no tests are run.

`before` is the original class's Java source. `after` is the list of
decomposed classes' sources. An empty entry in `after` is rejected rather
than scored: an empty extracted class is the classic way to game a
structural metric.

`deltas` gives per-metric before/after. For WMC, CBO, RFC, LCOM and DIT the
'after' value is the worst (max) across the decomposed classes, so an
improvement means even the worst new class beats the original; for LOC it is
the sum. Read the deltas alongside `overall_score` — a composite hides
disagreement between axes.

`method_coverage` and `field_coverage` are the fractions of the original's
members that still exist after. Below 1.0 means behaviour was deleted rather
than moved, which is the main thing this tool is guarding against: deleting
code is the easiest way to improve every CK metric at once. A large negative
`loc_delta` with coverage below 1.0 is that failure mode.

This score says nothing about correctness. Run the test suite separately and
require both.""",
    )
    def score_decomposition(before: str, after: list[str]) -> ScoreDecompositionResult:
        return godclass.score_decomposition(before=before, after=after)

    return server


def main() -> None:
    """Run the server on stdio."""
    build().run(transport="stdio")


if __name__ == "__main__":
    main()
