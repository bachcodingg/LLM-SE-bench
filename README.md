# llm-se-bench

An execution-based benchmark harness that evaluates LLMs on Java code
generation, bug fixing and architectural refactoring in a reproducible Docker
sandbox, with per-call cost accounting.

[![ci](https://github.com/<OWNER>/llm-se-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/<OWNER>/llm-se-bench/actions/workflows/ci.yml)
[![licence](https://img.shields.io/badge/licence-Apache--2.0-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

## Results

58 Java tasks, 3 runs each, 3 models. 522 evaluations, 672 API calls, $3.52.
May 2026. Full artifact in [`results/2026-05-run/`](results/2026-05-run/).

| Model | Tasks | Pass rate | Mean cost / eval | Mean latency | Total cost |
|---|---|---|---|---|---|
| `claude-sonnet-4-6` | 58 | 100.0% | $0.0148 | 6.5 s | $2.58 |
| `gpt-4o`            | 58 | 100.0% | $0.0049 | 3.7 s | $0.85 |
| `gemini-2.5-flash`  | 58 | 97.1%  | $0.0005 | 21.0 s | $0.09 |

**Read that table as a cost result, not a capability result.** The three
models are not statistically distinguishable on correctness — Friedman on
weighted score gives χ² = 4.00, p = 0.135; Cochran's Q on pass/fail gives
p = 0.368. They *are* separated by 28× on cost and, significantly, on latency
(Friedman χ² = 63.03, p < 0.001, Kendall's W = 0.54).

At a ~100% pass rate the benchmark is saturated: it confirms all three models
clear the bar and cannot rank them above it. That is the honest headline, and
it is why the tasks in the next version are repository-level rather than
single-file. See [Limitations](#limitations).

Per dataset:

| Dataset | Type | Tasks | Claude | GPT-4o | Gemini |
|---|---|---|---|---|---|
| `humaneval-java` | Code generation | 23 | 100% | 100% | 100% |
| `mbpp-java` | Code generation | 18 | 100% | 100% | 94.4% |
| `defects4j` | Bug fixing | 10 | 100% | 100% | 100% |
| `godclass` | Refactoring | 7 | 100% | 100% | 90.5% |

<!--
  TODO(before first push): record a 10-15 s GIF of the Dash dashboard and
  drop it here, under 5 MB.
    llm-se-bench dashboard --port 8050
    peek / asciinema to record, gifsicle -O3 --lossy=80 to compress
  Put it at docs/assets/dashboard.gif and reference it as:
    ![dashboard](docs/assets/dashboard.gif)
  A broken image link is worse than no image, so this stays a comment
  until the file exists.
-->

## Quickstart

```bash
git clone https://github.com/<OWNER>/llm-se-bench && cd llm-se-bench
cp .env.example .env                                  # add your API keys
docker build -t llm-se-bench-sandbox:17 bench/sandbox/
pip install -e ".[dev]"
llm-se-bench run --dataset humaneval --models claude --limit 3 --dry-run
```

The last command makes no API calls and costs nothing. Drop `--dry-run` when
you want real results. Without Docker the sandbox degrades to a structural
check whose results are optimistic and must not be published.

```bash
llm-se-bench datasets validate --dataset all    # 58 tasks, all with suites
llm-se-bench gateway test                       # one trivial call per provider
llm-se-bench gateway costs --summary            # what you have spent so far
llm-se-bench dashboard --port 8050              # interactive comparison
```

Exact commands to reproduce the published run: [`docs/reproducing.md`](docs/reproducing.md).

## What it does

**Execution, not judgment.** Every task is scored by compiling the generated
Java and running its JUnit suite inside a container started with
`--network=none --memory=256m --cpus=1`. No LLM judges any output, and
generated code cannot reach the internet to look up an answer.

**Cost recorded at call time.** Each API call writes a row with token counts
*and the per-token rates in force*. A cost reconstructed later from a price
list that has since moved is a guess; these rows are a historical record.

**Refactoring scored on structure, not just tests.** A refactoring that
changes nothing passes every test, so pass/fail is the wrong objective. The
`godclass` tasks are scored on CK metric deltas — WMC, CBO, RFC, LCOM — with
method and field coverage checks to stop a class being "decomposed" into
empty shells.

**Non-parametric statistics.** Pass rates are bounded, skewed and paired
across models. Friedman, Nemenyi post-hoc, Holm-corrected Wilcoxon, Cliff's
delta, 10,000-resample bootstrap CIs, and unbiased pass@k — the
combinatorial estimator, not "run it k times and take the max".

## Components

| | Package | What it owns |
|---|---|---|
| **C1** | `llm_gateway/` | Claude, GPT-4o and Gemini behind one interface. SQLite response cache, per-call cost records, token-bucket rate limiting, Jinja2 prompt templates. |
| **C2** | `bench/` | Four dataset adapters, the run orchestrator, and the Docker sandbox that compiles and tests generated Java. |
| **C3** | `quality/` | CK metrics (WMC, DIT, NOC, CBO, RFC, LCOM), cyclomatic and cognitive complexity, readability scoring, God Class detection and before/after refactoring deltas — via `javalang` AST parsing. |
| **C4** | `stats/` | Hypothesis tests, effect sizes, bootstrap CIs, cost modelling, consistency analysis. 23 figures and 8 LaTeX tables. |
| **C5** | `framework/` | MCDA over five criteria with weighting profiles (`devops`, `audit`, `budget`), Pareto frontier, recommender, Dash dashboard, PDF/CSV/JSON export. |
| | `mcp_servers/` | Three MCP servers exposing C1–C3 as tools an agent can call. |
| | `agent/` | The tool-calling agent loop: workspace, five tools, six termination conditions, trajectory recording, replay. |

`contracts.py` holds every shared Pydantic model and is the only thing all
components import. Details in [`docs/architecture.md`](docs/architecture.md).

## MCP servers

The harness can be driven by an agent rather than by a person:

```bash
pip install -e ".[mcp]"
python -m mcp_servers --list
```

| Server | Tools |
|---|---|
| `jvm-sandbox` | `compile_java`, `run_tests`, `run_static_analysis`, `compute_ck_metrics` |
| `llm-se-bench` | `list_tasks`, `get_task`, `evaluate_patch`, `run_benchmark`, `get_run_results` |
| `god-class-tools` | `detect_god_classes`, `propose_decomposition`, `score_decomposition` |

Every tool takes and returns a Pydantic model, enforces its own timeout, and
truncates its output with the truncation recorded as metadata — a 40,000-line
build log destroys an agent's context window, and an agent that failed
because its stack trace was cut is the harness's bug, not the model's.
Configuration snippet and full tool reference: [`docs/mcp.md`](docs/mcp.md).

## Agent mode

The harness can also *drive* a model rather than just score one: give it a
failing task and five tools, and let it work until the tests pass or a
budget runs out.

```bash
llm-se-bench agent run --task D4J_Lang_1 --model claude --dry-run
llm-se-bench agent run --dataset defects4j --model claude \
  --budget-eur 0.50 --total-budget-eur 5.00 --compare
llm-se-bench agent show ep-20260919T034026-0aa688
```

All six termination conditions are enforced — success, max steps, max
tokens, max cost, no progress, wall clock — and the cost ceiling is checked
*before* each call, because the call is what spends the money. There is no
default ceiling and a real run refuses to start without one: agent episodes
cost 10–100× a single-shot completion.

Success requires two things that are easy to skip: **no regression** against
a baseline measured before the agent touched anything, and **real
execution** — the sandbox's structural fallback credits every `@Test` as
passing, so a machine without Docker would otherwise report a 100% resolve
rate.

Every episode produces a trajectory: one row per step with the thought, the
tool call, the result hash, tokens (cache reads counted separately), cost,
files touched and test state. Recorded episodes can be **re-scored without
calling any API**, so changing a scoring rule costs a CPU second rather than
another run's spend.

Provider differences — Anthropic's `tool_use` blocks, OpenAI's JSON-string
arguments, Gemini's id-less function calls — are normalised once in
`llm_gateway/adapters/`. Details and known gaps: [`docs/agent.md`](docs/agent.md).

## Tests

```bash
pytest -q          # 963 tests, no network, no API calls, no Docker required
ruff check .
python scripts/security_sweep.py    # before every push to a public remote
```

## Limitations

The full list is in [`docs/methodology.md`](docs/methodology.md#limitations).
The four that matter most:

1. **The benchmark is saturated.** 517 of 522 evaluations passed. It cannot
   rank models on capability, only confirm they clear a low bar.
2. **Contamination is near-certain** for the 41 `humaneval-java` and
   `mbpp-java` tasks — both derive from public benchmarks that predate every
   model evaluated. Treat those results as an upper bound.
3. **Tasks are single-file.** No build system, no cross-file context, no
   dependency resolution. This sidesteps the JVM environment-setup problem
   that dominates repository-level Java benchmarks, and means these numbers
   say nothing about performance on a real codebase.
4. **One run, one scaffold, one point in time.** Three runs per task detects
   gross instability and nothing finer, and every result is a single-shot
   completion with one prompt template. These are conclusions about
   model-plus-this-scaffold, not about models.

## Licence

Apache-2.0 for the harness — see [`LICENSE`](LICENSE). The task data in
`data/` is CC-BY-4.0 with its own [`LICENSE`](data/LICENSE) and
[`PROVENANCE.md`](data/PROVENANCE.md), so the problems can be reused without
pulling in the harness.

Citation metadata: [`CITATION.cff`](CITATION.cff).
Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md).
