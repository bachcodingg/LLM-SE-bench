# Documentation

Start with whichever question you arrived with. For the whole project in one
place, read [`report.md`](report.md).

## Using it

| | |
|---|---|
| [`reproducing.md`](reproducing.md) | The exact commands that produced `results/2026-10-run`, how it was re-executed after the host ran out of memory, and what will not reproduce exactly. |
| [`cost.md`](cost.md) | Where the May run's $3.52 came from, how caching and rate limiting work, and what the budget controls do and do not cover. |
| [`mcp.md`](mcp.md) | The three MCP servers, their twelve tools, and a config snippet to drop into a client. |
| [`agent.md`](agent.md) | The agent loop: five tools, six termination conditions, the trajectory table, replay, and the provider differences the adapters absorb. |
| [`demo.md`](demo.md) | The five-minute live demo: an agent submits a refactoring that passes every test, and the harness says why it is not one. |

## Understanding it

| | |
|---|---|
| [`report.md`](report.md) | The research report: design, results of the October 2026 run, how the measurement was validated, threats to validity, references. |
| [`architecture.md`](architecture.md) | Five components, one contract module, one direction of dependency — and why the layout is flat rather than `src/`. |
| [`methodology.md`](methodology.md) | How a task is defined and scored, which statistics are used and why, what the published run actually found, and ten limitations. |
| [`v2.md`](v2.md) | Where this is going: the three research gaps, the module breakdown, and what is deliberately absent. |

## Working on it

| | |
|---|---|
| [`roadmap.md`](roadmap.md) | The schedule, against a 47-credit course load, with the checkpoint rule. |
| [`risks.md`](risks.md) | Seven failure modes, each mapped to whether code prevents it or only a decision does. |
| [`research-statement.md`](research-statement.md) | A two-page draft, assembled from the evidence. Needs your voice before it is usable. |

## Checks that answer questions faster than reading

```bash
python scripts/project_status.py      # what is built, tested, and actually run
python scripts/security_sweep.py      # anything about to be leaked
python scripts/build_run_artifact.py --check   # is the published run stale
python scripts/regression_gate.py --check-prompts   # are published numbers still valid
python -m mcp_servers --self-test     # do the servers still register their tools
```

## The short version

An execution-based benchmark harness for LLMs on Java, with per-call cost
accounting. The October 2026 run, four models, every evaluation executed in
Docker, found that the code-generation tasks do not separate the models:
the cheapest matches the most expensive at about 1/17th of the cost. The
refactoring tasks do separate behaviour from structure: 82 of 84 God Class
refactorings keep behaviour, and only 48 earn structural credit.

v2 is repository-level agentic refactoring scored on three independent axes
— behaviour preservation, structural quality, euros — with tamper
detection. The agent study that measures the euro axis is built and has
not been run yet. The task factory is built and has never been run; doing
so is the remaining project.
