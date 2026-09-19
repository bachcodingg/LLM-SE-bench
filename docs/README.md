# Documentation

Eleven pages. Start with whichever question you arrived with.

## Using it

| | |
|---|---|
| [`reproducing.md`](reproducing.md) | The exact commands that produced `results/2026-05-run`, and an honest account of what will not reproduce exactly. |
| [`cost.md`](cost.md) | Where the $3.52 came from, how caching and rate limiting work, and what the budget controls do and do not cover. |
| [`mcp.md`](mcp.md) | The three MCP servers, their twelve tools, and a config snippet to drop into a client. |
| [`agent.md`](agent.md) | The agent loop: five tools, six termination conditions, the trajectory table, replay, and the provider differences the adapters absorb. |

## Understanding it

| | |
|---|---|
| [`architecture.md`](architecture.md) | Five components, one contract module, one direction of dependency — and why the layout is flat rather than `src/`. |
| [`methodology.md`](methodology.md) | How a task is defined and scored, which statistics are used and why, what the published run actually found, and ten limitations. |
| [`v2.md`](v2.md) | Where this is going: the three research gaps, the module breakdown, and what is deliberately absent. |

## Working on it

| | |
|---|---|
| [`roadmap.md`](roadmap.md) | The schedule, against a 47-credit course load, with the checkpoint rule. |
| [`risks.md`](risks.md) | Seven failure modes, each mapped to whether code prevents it or only a decision does. |
| [`positioning.md`](positioning.md) | Which numbers to quote and which to stop quoting. |
| [`describing-it.md`](describing-it.md) | How to phrase it in an application, on a CV, and in a research statement. |
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
accounting. The published run found three frontier models statistically
indistinguishable on correctness and separated by 28x on cost — because the
tasks are saturated, which is the finding that motivates v2.

v2 is repository-level agentic refactoring scored on three independent axes
— behaviour preservation, structural quality, euros — with tamper
detection. Nine of its ten modules have a working tested core. The tenth,
the task factory, is built and has never been run; doing so is the
remaining project.
