# Architecture

Five components, one shared contract module, one direction of dependency.

```
                       ┌──────────────────────────────┐
                       │      cli.py / mcp_servers    │
                       │  the only entry points       │
                       └──────┬───────────────────────┘
                              │
   ┌──────────┬───────────────┼───────────────┬──────────────┐
   │          │               │               │              │
┌──▼──────┐ ┌─▼───────────┐ ┌─▼───────────┐ ┌─▼──────────┐ ┌─▼──────────┐
│   C1    │ │     C2      │ │     C3      │ │    C4      │ │    C5      │
│ Gateway │ │  Benchmark  │ │  Quality    │ │ Statistics │ │  Decision  │
│         │ │   Engine    │ │  Analyser   │ │  Engine    │ │ Framework  │
│ clients │ │ datasets    │ │ ck_metrics  │ │ hypothesis │ │ mcda       │
│ cache   │ │ orchestrator│ │ complexity  │ │ effect     │ │ recommender│
│ cost    │ │ sandbox     │ │ readability │ │ bootstrap  │ │ dashboard  │
│ limiter │ │ results     │ │ refactoring │ │ cost model │ │ exporters  │
└────┬────┘ └──────┬──────┘ └──────┬──────┘ └─────┬──────┘ └─────┬──────┘
     │             │               │              │              │
     └─────────────┴───────────────┴──────────────┴──────────────┘
                              │
                    ┌─────────▼──────────┐
                    │    contracts.py    │
                    │  Pydantic models   │
                    └────────────────────┘

  LLMResponse  →  EvaluationResult  →  QualityMetrics  →  StatisticalSummary
                                                       →  DecisionMatrix
```

`contracts.py` is the seam. Components import from it and never from each
other, with one exception: C2 constructs prompts through C1's client
interface. Everything else moves as serialised contract objects, which is
why the pipeline can be stopped and resumed between any two stages.

## C1 — LLM Gateway (`llm_gateway/`)

One interface over Anthropic, OpenAI and Google. `LLMClientFactory` maps a
model id to a client; each client turns a `Prompt` into an `LLMResponse`.

Three things wrap every call:

- **Response cache** (`cache.py`) — SQLite, keyed on SHA-256 of the rendered
  prompt plus model id. A repeat of an identical prompt costs nothing. The
  published run has 174 cache entries against 672 calls.
- **Cost tracker** (`cost_tracker.py`) — writes one `CostRecord` per call
  with token counts and the per-token rates in force. Cost is recorded at
  call time, not reconstructed later from a price list that has since moved.
- **Rate limiter** (`rate_limiter.py`) — token bucket per provider.

Prompts are Jinja2 templates in `llm_gateway/templates/`. They are inputs to
the measurement, so `results/2026-05-run/manifest.json` records the SHA-256
of each one as it was at run time.

## C2 — Benchmark Engine (`bench/`)

`Dataset` (in `datasets/base.py`) is a four-method interface: load problems,
get a test suite, format a prompt, verify a solution. Four adapters implement
it. `BenchmarkOrchestrator` walks (model × problem × run), calls C1, extracts
code, hands it to the sandbox, and emits an `EvaluationResult`.

**`DockerSandbox`** is the part that makes the numbers execution-based rather
than model-judged. Each evaluation stages the generated class and its JUnit
suite into a fresh temporary directory and runs two containers:

```
docker run --rm --network=none --memory=256m --cpus=1 \
  -v <staging>:/workspace:rw llm-se-bench-sandbox:17 ...
```

`--network=none` means generated code cannot reach the internet, which is a
correctness property as much as a safety one: a task cannot be solved by
fetching the answer. Compile and test phases have separate timeouts (60 s
and 120 s), and JUnit output is parsed into per-test results rather than
scraped for a pass/fail word.

When Docker is unavailable the sandbox degrades to a structural dry-run:
balanced braces, a class declaration, `@Test` counting. This is what lets the
whole test suite run in CI, and it is **not** a substitute for execution —
dry-run results are optimistic by construction and must never be published.

## C3 — Quality Analyser (`quality/`)

Parses generated Java with `javalang` and computes, per class:

- **CK metrics** (`ck_metrics.py`) — WMC, DIT, NOC, CBO, RFC, LCOM. CBO
  excludes `java.*` standard library types by default, because coupling to
  `String` is not a design smell.
- **Complexity** (`complexity.py`) — cyclomatic, cognitive, max nesting.
- **Readability** (`readability.py`) — naming, comments, structure,
  formatting, documentation, combined into a 0–100 composite.
- **Refactoring** (`refactoring.py`) — God Class detection (a class is
  flagged when it breaches at least 2 of 4 thresholds) and before/after
  metric deltas for a decomposition.

## C4 — Statistical Engine (`stats/`)

Non-parametric throughout, because pass rates are bounded, skewed and paired
across models on the same problems: Friedman across three models, Nemenyi
post-hoc, Wilcoxon signed-rank pairwise, Cliff's delta for effect size,
bootstrap confidence intervals, and unbiased pass@k.

## C5 — Decision Framework (`framework/`)

Min-max normalises five criteria (correctness, quality, speed, cost,
consistency) across models, applies a weighting profile, and ranks. Profiles
live in `framework/profiles/*.yaml` — `devops`, `audit`, `budget` — because
"which model is best" has no answer that is independent of what you are
optimising for. Outputs a Plotly Dash dashboard and PDF/CSV/JSON exports.

## MCP tool layer (`mcp_servers/`)

Three MCP servers expose C1–C3 as tools an agent can call. See
[`mcp.md`](mcp.md).

## Agent loop (`agent/`)

Turns the harness from something that scores a model into something that
drives one: a task, five tools, and a loop that runs until the tests pass or
a budget is exhausted.

```
   agent/loop.py ──── neutral types ────> llm_gateway/adapters/
        │            (conversation.py)     anthropic | openai | gemini
        │                                          │
   agent/tools.py                          llm_gateway/clients/
   read_file, list_dir, grep,              (rate limiting, cost records)
   apply_patch, run_tests
        │
   agent/workspace.py ──materialise──> bench/sandbox (C2)
   in-memory {path: content}
        │
   agent/trajectory.py
   one row per step; replay without an API call
```

`llm_gateway/conversation.py` is the second seam in the project, after
`contracts.py`: the loop speaks only its neutral `Conversation`,
`ToolCall`, `ToolResult` and `AssistantTurn`, so nothing in `agent/` knows
which provider it is talking to. The three adapters translate once each,
and are pure functions over stub objects — no network, no SDK, no state —
which is what lets them be tested without an API key.

Details, and the gaps that are still open, in [`agent.md`](agent.md).

## A note on repository layout

The guide this repository follows recommends a `src/llm_se_bench/` layout.
It uses a flat layout instead: the five top-level packages *are* the five
components, the mapping from directory to architecture diagram is one to
one, and moving them under `src/` would rewrite every import and every
`pytest` path for no behavioural gain. The `src/` layout's real benefit —
you cannot accidentally import the working tree instead of the installed
package — is handled here by `pip install -e .` plus CI installing the
package before running tests.
