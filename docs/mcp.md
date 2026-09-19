# MCP servers

The harness can be driven by a person through `llm-se-bench`. These three
MCP servers let an agent drive it instead.

```bash
pip install -e ".[mcp]"
python -m mcp_servers --list
python -m mcp_servers --self-test      # build all three, open no transport
python -m mcp_servers jvm-sandbox      # stdio
python -m mcp_servers llm-se-bench --http --port 8931
```

Works with both MCP SDK generations — 1.x (`FastMCP`) and 2.x (`MCPServer`)
— through a shim in `mcp_servers/_compat.py`. The SDK is an optional extra;
the harness, the CLI and the whole test suite work without it.

## Wiring it into a client

Claude Desktop (`claude_desktop_config.json`), Claude Code
(`.mcp.json`) and anything else that speaks stdio take the same shape:

```json
{
  "mcpServers": {
    "jvm-sandbox": {
      "command": "python",
      "args": ["-m", "mcp_servers", "jvm-sandbox"],
      "cwd": "/absolute/path/to/llm-se-bench"
    },
    "llm-se-bench": {
      "command": "python",
      "args": ["-m", "mcp_servers", "llm-se-bench"],
      "cwd": "/absolute/path/to/llm-se-bench",
      "env": {
        "LLM_SE_BENCH_BUDGET_EUR": "1.00"
      }
    },
    "god-class-tools": {
      "command": "python",
      "args": ["-m", "mcp_servers", "god-class-tools"],
      "cwd": "/absolute/path/to/llm-se-bench"
    }
  }
}
```

The same snippet is in [`mcp/claude_desktop_config.json`](../mcp/claude_desktop_config.json);
replace the path and drop it in.

`cwd` matters: the servers resolve `data/`, `results/` and the cache
database relative to it.

Setting `LLM_SE_BENCH_BUDGET_EUR` is the safe default. Without it,
`run_benchmark` refuses every non-dry run rather than guessing a ceiling.

## Servers and tools

### `jvm-sandbox`

Compile and test Java inside the project's Docker sandbox, and analyse its
structure.

| Tool | Does |
|---|---|
| `compile_java(source_files, jdk_version, timeout_s)` | Compiles. Returns `{compiled, diagnostics[], log, truncation}`. |
| `run_tests(source_files \| project_path, test_filter, timeout_s)` | Compiles and runs a JUnit 4 suite. Returns `{passed, failed, total, outcomes[], log}`. |
| `compute_ck_metrics(source_files \| class_path)` | WMC, DIT, NOC, CBO, RFC, LCOM per class. Parses only. |
| `run_static_analysis(project_path \| source_files)` | CK metrics plus smells: `god_class`, `high_coupling`, `low_cohesion`, `large_class`, `deep_inheritance`. |

Scope: `javac` plus JUnit 4 over a flat set of files. **No Maven, no
Gradle, no dependency resolution.** A project with external dependencies
will not build. Repository-level builds are a separate problem — the one
`EnvBench` found automated setup solves for only 29.5% of JVM repositories
— and this does not pretend to solve it.

### `llm-se-bench`

| Tool | Does |
|---|---|
| `list_tasks(category, dataset, difficulty, limit)` | Filtered task list. No code. |
| `get_task(task_id)` | `{prompt, repo_state, test_spec, metadata}`. **Never the reference solution.** |
| `evaluate_patch(task_id, patch)` | Scores a candidate. Costs nothing. |
| `run_benchmark(model, task_set, budget_eur, dry_run, ...)` | **Can spend money.** Defaults to `dry_run=true`. |
| `get_run_results(run_id)` | Reads a run back. |

### `god-class-tools`

| Tool | Does |
|---|---|
| `detect_god_classes(project_path \| source_files)` | Flags classes breaching 2+ of WMC>47, LCOM>1, CBO>14, RFC>50. |
| `propose_decomposition(source \| class_path, strategy)` | A decomposition plan. Static analysis, no model call, no code written. |
| `score_decomposition(before, after[])` | Structural improvement and behaviour preservation. |

`propose_decomposition` has three strategies:

- **`field_clusters`** (default) — partitions methods into connected
  components over the fields they touch. This is LCOM4's own definition of
  cohesion turned into a decomposition: each component is an independent
  responsibility *by that measure*. Falls back to `responsibility` when no
  method touches a field, which is normal for a class of static helpers.
- **`responsibility`** — groups by shared method-name verb prefix. Reads
  intent from naming, so it is only as good as the naming.
- **`layered`** — separates trivial accessors from behaviour.

It is deterministic: the same source gives the same plan every time.

## Design decisions worth knowing

**Tool descriptions are prompts.** They are written for a model, not a
maintainer: preconditions, units, and what each failure mode looks like in
the response. A tool whose description says "runs tests" and not "returns at
most 200 lines of log, truncated, with `truncation.truncated` set" will be
misused. A test asserts every description is over 200 characters, which is a
crude proxy for "someone actually wrote it".

**Failures are values, not exceptions.** Every response carries `ok` and
`error`. An exception crossing the MCP boundary becomes an opaque protocol
error an agent cannot reason about; `ok=false` with a sentence explaining why
is something it can act on. Note the distinction `compile_java` draws:
`ok=true, compiled=false` is a normal compile failure — the tool worked, the
code did not — while `ok=false` means the call itself was rejected.

**Output is bounded and the bound is visible.** A 40,000-line Maven log
destroys an agent's context window. Every log field goes through
`mcp_servers/truncation.py`, which keeps the head, every line that looks
like the start of an error, and as much tail as fits — and records what it
removed in a `truncation` object. That record is not decoration: an agent
that failed because its stack trace was cut is *the harness's* bug, not the
model's, and that is only provable if the cut is visible.

**Unsafe paths are rejected, not sanitised.** `../../etc/passwd` gets an
error naming the problem. Quietly rewriting it teaches the caller nothing.

**No tool returns a reference solution.** A benchmark whose tasks hand out
their own answers measures nothing, and an agent given one has no way to
know it should not look. A test iterates every one of the 58 tasks and
asserts the field is absent.

**Only one tool can spend money, and it defaults to not.** `run_benchmark`
has `dry_run=true` by default. A real run needs an explicit ceiling, from
`budget_eur` or `LLM_SE_BENCH_BUDGET_EUR`; with none, the call is refused.
The projection is computed **before any client is constructed** and the run
is refused rather than started and aborted. A model with no measured cost
history is assumed to be as expensive as the most expensive measured one, so
the check fails safe.

The projection is per-evaluation cost measured in `results/2026-05-run`. It
is an estimate, not a guarantee, and a task mix heavier than that run's will
overshoot it. This is a pre-flight check, not a mid-run abort — adequate for
single-shot evaluation, where per-call cost is bounded and predictable. It
would not be adequate for an agent loop, where episodes run 10–100× more
expensive; see [`cost.md`](cost.md#budgets).

## Known gaps

- **No mid-run budget abort.** Pre-flight only, as above.
- **No Maven or Gradle.** `javac` and JUnit 4 on a flat file set.
- **`actual_cost_eur` is a running total per model**, not this run's slice.
  Cost records carry no run id, so attributing spend to a run would need a
  timestamp window that concurrent runs would break. The field says so.
- **Permission modes are not implemented.** Every tool runs at the same
  access level. Separating read-only from patch-only from full-shell, and
  measuring what each level buys, belongs with the agent harness.
- **No per-tool cost attribution.** Tool output consumes an agent's tokens;
  nothing here attributes that.

## Testing

```bash
pytest mcp_servers/tests -q
```

No SDK, no Docker and no network required. The tool logic in
`mcp_servers/tools/` never imports `mcp`, so it is testable on its own; the
server tests skip themselves when the SDK is absent. `--self-test` covers
what only appears when a client connects: registration, schema generation,
tool names.
