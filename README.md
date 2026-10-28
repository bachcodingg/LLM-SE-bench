# llm-se-bench

An execution-based benchmark harness that evaluates LLMs on Java code
generation, bug fixing and architectural refactoring in a reproducible Docker
sandbox, with per-call cost accounting.

[![ci](https://github.com/bachcodingg/LLM-SE-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/bachcodingg/LLM-SE-bench/actions/workflows/ci.yml)
[![licence](https://img.shields.io/badge/licence-Apache--2.0-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

## Results

![evaluate_patch on three refactorings of one God Class, then the dashboard](docs/assets/demo.gif)

**Refactoring is where the models differ from what the tests say.** On the
seven God Class tasks, 82 of 84 refactorings keep the original behaviour
(every JUnit test passes), but only **48 earn structural credit**. Two tasks,
`GC_004` and `GC_007`, were never credited for any model in any sample. A
pass/fail harness would have scored nearly all of them as solved.

October 2026: 58 Java tasks, 3 independent samples each, 4 models, 696
evaluations, every one compiled and run under `javac` and JUnit in Docker.
$2.20 in total. Full artifact in [`results/2026-10-run/`](results/2026-10-run/).

| Model | Code tasks passed (51) | Refactoring: behaviour kept | Refactoring: structure credited | Cost / task | Run cost |
|---|---|---|---|---|---|
| `claude-sonnet-5-5` | 95% (90–100) | 100% | 62% (29–90) | $0.0068 | $1.17 |
| `claude-haiku-5-5`  | 96% (90–100) | 100% | 57% (14–86) | $0.0004 | $0.07 |
| `gpt-5.6-terra`     | 95% (88–100) | 100% | 57% (24–86) | $0.0049 | $0.85 |
| `gpt-5.6-luna`      | 95% (90–100) | 90% (71–100) | 52% (19–86) | $0.0006 | $0.11 |

95% bootstrap intervals over tasks in brackets. The axes are deliberately
not summed into one score: [`leaderboard.md`](results/2026-10-run/leaderboard.md),
[`leaderboard.png`](results/2026-10-run/leaderboard.png).

What the run shows:

- **The code-generation tasks do not separate these models.** HumanEval is
  100% for all four; the intervals on the 51 code tasks overlap completely.
  The cheapest model matches the most expensive one at about 1/17th of the
  cost.
- **Structural credit comes almost entirely from cohesion.** Across the 48
  credited refactorings, cohesion improvement averages 0.93, complexity
  dispersal 0.26, coupling 0.10, and size reduction is zero in every one.
  Models split classes; they do not make them smaller.
- **Most uncredited refactorings made the structure slightly worse.** Of
  the 34 that kept behaviour but were not credited, 24 have a composite
  score at or below zero, and 10 were stopped by an anti-gaming guard,
  almost all of them a facade that still holds most of the original's
  branching.
- **The refactoring intervals are wide.** Seven tasks are too few to rank
  models on structure; the gap between kept and credited is the finding,
  not the order.

### How these numbers were checked

The run host ran out of memory partway through, and containers killed
mid-`javac` were recorded as compile errors against the model: 98 MBPP
evaluations for the two cheap models, plus 2 whose JUnit run was killed.
Every recorded response was then re-executed from the response cache on a
healthy host, with no new API calls
([`scripts/reexecute.py`](scripts/reexecute.py)). Exactly those 100 verdicts
changed (97 to pass, 3 to partial); nothing else moved. The sandbox now records a silently killed container as *not
executed* instead of as a verdict, and reports what `javac` actually said.

Two overlapping run processes also recorded 27 God Class samples twice.
The later row of each pair is kept, because its code is the one on disk,
whatever its verdict. Both counts, and the changed verdicts, are in
[`manifest.json`](results/2026-10-run/manifest.json).

The earlier May 2026 run is withdrawn. Its sandbox had silently fallen back
to a structural check that credits every `@Test`, its repeated runs were
mostly cache hits on run 1, and its God Class suites asserted `true`. All
three are fixed and covered by tests.

### Not run yet

The agent study ([`scripts/agent_study.py`](scripts/agent_study.py)) is
built and tested but has not been run, for lack of budget. It plots solve
rate against a per-episode euro budget for the same four models, against
the single-shot baseline above. Its second sweep lets the agent write to
the test files and reports a **clean solve rate** next to the raw one.
Estimated cost at the configured budgets is a few euros, hard-capped in code.

## Quickstart

```bash
git clone https://github.com/bachcodingg/LLM-SE-bench && cd LLM-SE-bench
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
changes nothing passes every test, so pass/fail is the wrong objective. A
`godclass` task passes only when its behaviour suite passes *and*
`quality/structural.py` credits the CK metric deltas (CBO, LCOM, WMC
dispersal, size), with guards against empty-shell classes, deleted code and
a broken public API.

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
| | `agent/` | The tool-calling agent loop: workspace, five tools, six termination conditions, four scaffolds, trajectory recording and replay, failure taxonomy. |
| | `taskfactory/` | Mining, building and validating JVM repository tasks: reverse patches, build-system detection, six admission gates, contamination checks. |

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

## Where this is going

v2 is repository-level agentic evaluation for JVM refactoring, scored on
three independent axes rather than one:

| | |
|---|---|
| **Behaviour preserved** | Tests pass, public API unchanged, pre/post agree on generated inputs |
| **Structure improved** | CK metric deltas, with published weights and anti-gaming guards |
| **Euros spent** | Cost per solve, not just resolve rate |

Plus automatic detection of test tampering — an immutable test manifest,
static detectors for the known cheats, and held-out suites — reported as a
**clean solve rate** alongside the raw one. The gap between them is a result
in its own right.

Four scaffolds run behind one interface (`single_shot`, `react`,
`plan_then_execute`, `external`), so "the model is better" can be separated
from "my loop is better" — a confound most papers carry silently.

```bash
llm-se-bench agent run --dataset defects4j --model claude \
  --scaffold plan_then_execute --permissions read_only \
  --budget-eur 0.50 --total-budget-eur 5.00 --workers 4
```

Positioning and what is deliberately absent: [`docs/v2.md`](docs/v2.md).
Schedule: [`docs/roadmap.md`](docs/roadmap.md). Failure modes:
[`docs/risks.md`](docs/risks.md). Full index: [`docs/`](docs/README.md).

The honest summary of where it stands: nine of the ten modules have a
working, tested core. The tenth is the task factory, and running it —
mining, building and validating 60–300 real repository tasks — is the
remaining project.

```bash
python scripts/project_status.py
```

reports which modules are built, which are tested, and which have never
been exercised end to end. Four currently fall in the last category (the
agent harness, task factory, trajectory analysis and trajectory dashboard),
and the script says so rather than letting "built" read as "finished". It
also blocks publication of any run that lacks execution evidence.

## Tests

```bash
pytest -q          # ~1,350 tests, no network, no API calls, no Docker required
ruff check .
python scripts/security_sweep.py       # before every push to a public remote
python scripts/regression_gate.py --check-prompts   # published numbers still valid?
```

## Limitations

The full list is in [`docs/methodology.md`](docs/methodology.md#limitations).
The ones that matter most:

0. **Seven refactoring tasks.** Enough to show that kept and credited
   differ; not enough to rank models on structure. The intervals in the
   table span 60 to 70 points.
1. **The structural score is one operationalisation.** It scores coupling
   and cohesion on the extracted classes and requires the API-preserving
   facade to give up more than half of the original's branching. Those
   choices are documented and tested in `quality/structural.py`, but a
   different reasonable choice would move the credited rates.
2. **Contamination is near-certain** for the 41 `humaneval-java` and
   `mbpp-java` tasks — both derive from public benchmarks that predate every
   model evaluated. Treat those results as an upper bound.
3. **Tasks are single-file.** No build system, no cross-file context, no
   dependency resolution. This sidesteps the JVM environment-setup problem
   that dominates repository-level Java benchmarks, and means these numbers
   say nothing about performance on a real codebase.
4. **One run, one scaffold, one point in time.** Three samples per task
   detect gross instability and nothing finer, and every result is a
   single-shot completion with one prompt template. These are conclusions about
   model-plus-this-scaffold, not about models.

## Licence

Apache-2.0 for the harness — see [`LICENSE`](LICENSE). The task data in
`data/` is CC-BY-4.0 with its own [`LICENSE`](data/LICENSE) and
[`PROVENANCE.md`](data/PROVENANCE.md), so the problems can be reused without
pulling in the harness.

Citation metadata: [`CITATION.cff`](CITATION.cff).
Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md).
