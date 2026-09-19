"""
mcp_servers.servers.bench — the ``llm-se-bench`` MCP server.

List and fetch benchmark tasks, score a candidate solution, launch a run
under a budget ceiling, and read results back.
"""

from __future__ import annotations

from mcp_servers._compat import require_sdk
from mcp_servers.models import (
    EvaluatePatchResult,
    GetRunResultsResult,
    GetTaskResult,
    ListTasksResult,
    RunBenchmarkResult,
)
from mcp_servers.tools import bench

__all__ = ["build", "INSTRUCTIONS"]

INSTRUCTIONS = """\
Drive the llm-se-bench benchmark: 58 self-contained Java tasks across four
datasets — code generation (humaneval-java, mbpp-java), bug fixing
(defects4j) and God Class refactoring (godclass).

A task is one Java class plus one JUnit 4 suite. There is no build system
and no repository context.

No tool here returns a reference solution, for any task.

Only run_benchmark can spend money, and it defaults to dry_run=true. A real
run needs an explicit budget ceiling and is refused before it starts if the
projection exceeds it. Call it with dry_run=true first to see the shape of
the run.

Scoring is by execution: compile, run the suite, count passing tests. No
model judges any output. When Docker is unavailable, evaluate_patch returns
`executed: false` and a structural check — optimistic by construction, and
not a benchmark score.
"""


def build(**kwargs):
    """Build the llm-se-bench server with its five tools registered."""
    server_class = require_sdk()
    server = server_class(
        name="llm-se-bench",
        instructions=INSTRUCTIONS,
        **kwargs,
    )

    @server.tool(
        description="""\
List benchmark tasks, optionally filtered. Returns no solutions and no test
code — call get_task for one task's detail.

`category` is 'codegen', 'bugfix' or 'refactor'. `dataset` is one of
humaneval-java, mbpp-java, defects4j, godclass. `difficulty` is 'easy',
'medium' or 'hard'. An empty string means no filter on that field. An
unrecognised value is an error rather than an empty result, so a typo does
not look like "no tasks match".

`total` is how many matched before `limit` was applied: total > len(tasks)
means the list is truncated.""",
    )
    def list_tasks(
        category: str = "",
        dataset: str = "",
        difficulty: str = "",
        limit: int = 100,
    ) -> ListTasksResult:
        return bench.list_tasks(
            category=category, dataset=dataset, difficulty=difficulty, limit=limit
        )

    @server.tool(
        description="""\
Fetch one benchmark task: its prompt, starting files and test
specification.

The reference solution is never included, for any task.

`prompt` is the exact text the benchmark sends to a model. Use it verbatim
if you want results comparable with the published run.

`repo_state` is {filename: content}. It is empty for codegen tasks, which
start from nothing; it holds the buggy class for bugfix tasks and the God
Class for refactor tasks.

`test_spec.junit_code` is the suite your solution is compiled against, given
so you can match the expected class and method names. You cannot edit it:
the harness uses its own copy. `num_hidden_cases > 0` means some scoring
cases are not shown, so fitting to the visible examples will not pass.""",
    )
    def get_task(task_id: str) -> GetTaskResult:
        return bench.get_task(task_id)

    @server.tool(
        description="""\
Score a candidate solution against a task's test suite. Makes no API call
and costs nothing; `cost_usd` is always 0.0.

`patch` is the complete Java source of the solution class, not a diff and
not a fragment. At most 100,000 characters.

`verdict` is 'pass' (compiled, all tests passed), 'partial' (some passed),
'fail' (none passed) or 'error' (did not compile). `weighted_score` is the
weight-weighted fraction of tests passed.

`metrics` carries CK metrics of the submitted code, empty when it did not
parse. `log` holds sandbox output, truncated — check
`truncation.truncated` before concluding something is absent from it.

`executed: false` means Docker was unavailable and only a structural check
ran. Such a result is optimistic and is not a benchmark score.""",
    )
    def evaluate_patch(task_id: str, patch: str, timeout_s: int = 120) -> EvaluatePatchResult:
        return bench.evaluate_patch(task_id, patch, timeout_s=timeout_s)

    @server.tool(
        description="""\
Run the benchmark for one model over a dataset. THIS TOOL CAN SPEND MONEY.

`dry_run` defaults to true: mock responses, no API calls, nothing spent.
Results from a dry run are structural checks, not benchmark scores. Call it
this way first to see how many evaluations a configuration implies.

A real run (dry_run=false) requires a budget ceiling, from `budget_eur` or
the LLM_SE_BENCH_BUDGET_EUR environment variable. With no ceiling the call
is refused. With one, the projected spend is computed before any client is
built and the run is refused if it exceeds the ceiling — it does not start
and then abort. The projection uses measured cost per evaluation from the
published run; an unmeasured model is assumed to be as expensive as the
most expensive measured one, so the check fails safe.

`task_set` is the dataset name. `limit` caps the number of tasks (0 means
all), `runs_per_task` repeats each one.

Returns `run_id`; pass it to get_run_results. `actual_cost_eur` is the
running total recorded against this model across all runs, not this run's
slice — cost records carry no run id.""",
    )
    def run_benchmark(
        model: str,
        task_set: str = "",
        budget_eur: float = 0.0,
        dry_run: bool = True,
        runs_per_task: int = 1,
        limit: int = 0,
    ) -> RunBenchmarkResult:
        return bench.run_benchmark(
            model=model,
            task_set=task_set,
            budget_eur=budget_eur,
            dry_run=dry_run,
            runs_per_task=runs_per_task,
            limit=limit,
        )

    @server.tool(
        description="""\
Read back the results of a run started by run_benchmark.

`run_id` is what run_benchmark returned. An unknown id lists the most recent
known ids in `error` rather than returning an empty summary.

`dry_run: true` in the response means these are structural-check results,
not benchmark scores. ok=true with an empty `summaries` and a message in
`error` means the run was recorded but its result file is missing or
empty.""",
    )
    def get_run_results(run_id: str) -> GetRunResultsResult:
        return bench.get_run_results(run_id)

    return server


def main() -> None:
    """Run the server on stdio."""
    build().run(transport="stdio")


if __name__ == "__main__":
    main()
