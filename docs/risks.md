# Risks and failure modes

Seven ways this project fails. Each one is mapped to whether a mechanism
prevents it or whether only a decision does — because those need different
kinds of attention, and conflating them is how a risk register becomes
decoration.

**Mechanically enforced** means code refuses the failure and a test pins the
refusal. **Behavioural** means nothing stops you; you have to choose.

`python scripts/project_status.py` reports on the mechanical ones.

---

## 1. Scope collapse

**The most likely outcome.** You build M1 and M9 — loops and dashboards are
fun — and never finish M3, because mining repositories is tedious. The
project ends up with a beautiful trajectory viewer and nothing real to view
in it.

**Status: partly mitigated, mostly behavioural.**

The guide's mitigation is "build M3 before M9, force the order". That has
already been violated: M9 exists and M3 has never been run. The mitigation
that remains is honesty about it —
[`roadmap.md`](roadmap.md) and [`v2.md`](v2.md) both say M3 is machinery
only, `project_status.py` reports it mechanically, and the README says it in
the third paragraph.

**What to actually do:** the checkpoint rule. If the previous phase is not
published and working, do not start the next one. The next phase is one real
agent run on the existing 58 tasks — not more modules.

**Early warning:** you are adding a module while an earlier one has never
been executed end to end.

---

## 2. Cost blowup

Agent episodes cost 10-100x a single-shot completion. Multi-step loops over
300 tasks across three models run into hundreds of euros, and the failure
mode is not a big bill at the end — it is a loop that does not terminate
while you are asleep.

**Status: mechanically enforced.**

| Mechanism | Where |
|---|---|
| No default cost ceiling; a real run refuses to start without one | `agent/termination.py`, `TerminationPolicy.__post_init__` |
| The ceiling is checked *before* each call, not after | `BudgetGovernor.check` |
| Worst-case projection refuses a step that could cross | same |
| Run-level budget shared and locked across parallel workers | `agent/parallel.py`, `RunBudget.reserve` |
| MCP `run_benchmark` defaults to `dry_run=true` and refuses without a budget | `mcp_servers/tools/bench.py` |
| An unmeasured model is assumed to be the most expensive one | `MEAN_COST_USD_PER_EVALUATION` fallback |
| CI sets a hard `LLM_SE_BENCH_BUDGET_EUR` in the workflow, not in a script | `.github/workflows/regression.yml` |

**Residual risk:** the pre-flight projection is an estimate from the v1 run.
A task mix heavier than that run's will overshoot by up to one call. There
is no mid-episode abort — documented in [`agent.md`](agent.md#known-gaps).

**Never run an unbounded loop against a paid API.** The mechanisms above
exist so that this is not a matter of remembering.

---

## 3. Flaky tasks poisoning results

A task that fails 30% of the time for environmental reasons looks exactly
like a model capability difference. Once it is in the dataset, every
aggregate computed from it is contaminated, and no amount of statistical
care downstream undoes it.

**Status: mechanically enforced — in the factory that has not run yet.**

The five-run flake gate is implemented and tested
(`taskfactory/validation.py`, `check_flakiness`). A task whose results vary
across five runs is **rejected**, not down-weighted and not flagged.

**The specific temptation:** this gate costs five times what every other
gate costs put together. It is the obvious place to save time and it is the
one place where saving time is unrecoverable, because the damage is
invisible until someone tries to reproduce a result and cannot.

**Also already enforced:** an unexecuted sandbox result can never be
reported as a pass (`agent/termination.py`, `tests_executed` is part of the
success condition). That closes the adjacent hole where a machine without
Docker produces a 100% resolve rate.

---

## 4. Contamination

Popular Apache and Spring repositories are almost certainly in every model's
training data. A task mined from a 2019 Commons Lang commit measures
whether a model remembers the fix, not whether it can produce one.

**Status: measured and disclosed. Not solvable.**

| Mechanism | Where |
|---|---|
| Commit date recorded per task, cutoff per model | `taskfactory/contamination.py` |
| Results stratified pre-cutoff / post-cutoff, with the gap reported | `stratify_by_cutoff` |
| Tasks within 90 days of a cutoff go in neither stratum | `CUTOFF_MARGIN_DAYS` |
| High-exposure repositories flagged even post-cutoff | `HIGH_EXPOSURE_REPOS` |
| Stated loudly in the existing limitations | [`methodology.md`](methodology.md#limitations) |

**The honest position**, and the one the current README already takes: the
41 HumanEval and MBPP tasks predate every model evaluated, so results on
them are an upper bound, not a measurement. Say it before anyone asks.

**Residual risk:** training cutoffs are published approximations and
providers update models behind a stable id. A cutoff can move without the
name changing, which is why the near-cutoff band exists and why a clean
stratification is weaker evidence than it looks.

---

## 5. Scaffold confound

If your harness is weak, every model looks weak, and you publish conclusions
about models that are really conclusions about your loop. This is the
failure most papers in this area actually commit.

**Status: mitigated structurally, not yet demonstrated.**

Four scaffolds sit behind one interface — `single_shot` (the v1 baseline),
`react`, `plan_then_execute`, and `external`, which shells out to an
off-the-shelf agent. Every trajectory records which one ran, and a test pins
that all four return the same shape so a comparison is like-for-like
(`agent/tests/test_scaffolds.py::TestInterfaceParity`).

**What is missing:** none of them have been run against each other on real
tasks. The confound is *addressable* and not yet *addressed*, and any claim
about model quality made before that comparison exists carries the confound.

**The decisive step** is `ExternalHarnessScaffold` pointed at Claude Code or
Aider. Comparing against a loop you did not write is the only real answer,
and it is a command-line argument away.

---

## 6. Academic collision

Someone publishes something similar while you build. The field is moving:
SWE-Refactor, SpecBench and ProjDevBench all appeared in 2026.

**Status: behavioural, and the clock is running.**

**Mitigation: publish early and incrementally.** A public repository with a
dated commit history is a priority claim. A private one is not, and no
amount of finished work converts one into the other retroactively.

A remote exists (`github.com/bachcodingg/LLM-SE-bench`), so the claim is
partly staked. What matters now is that the *current* state is pushed:
`scripts/project_status.py` reports the remote but cannot tell you whether
what is on it is a year old. Check with `git log origin/main..HEAD`.

**Also worth knowing:** collision is survivable if the positioning is
specific. "A repository-level agentic benchmark for JVM refactoring scored
on behaviour preservation, structural quality and euros, with tamper
detection" is narrow enough that a generic issue-resolution benchmark does
not subsume it.

---

## 7. Burnout against course load

47 credits — Integral Calculus, Differential Calculus, Matrix Algebra,
Statistics II, Functional Programming, Advanced Networks, Foundations of AI
and ML, two digitalisation courses — plus a November hackathon plus this.
All three at full intensity is not possible.

**Status: behavioural. Decide now, not in February.**

The guide's instruction is to decide in advance which one gives.
[`roadmap.md`](roadmap.md) makes the call explicitly: **this project is the
thing that slips**, because the courses are not optional and the hackathon
is a fixed date.

That decision is only useful if the project is built to survive slipping,
which is what the phase structure is for — each phase independently
publishable, so a bad month leaves a smaller finished thing rather than a
larger unfinished one.

**Early warning:** you are working on this instead of coursework, rather
than after it.

---

## Summary

| Risk | Enforced by | Residual |
|---|---|---|
| 1. Scope collapse | Honest reporting only | **High** — already partly realised |
| 2. Cost blowup | Code, tested | Low — no mid-episode abort |
| 3. Flaky tasks | Code, untested in anger | Medium — gate is skippable by choice |
| 4. Contamination | Measurement and disclosure | Permanent — not solvable |
| 5. Scaffold confound | Structure, not yet demonstrated | Medium until the comparison runs |
| 6. Academic collision | A remote exists; keep it current | Medium |
| 7. Burnout | A decision recorded in the roadmap | Medium |

Risk 1 is the one to watch, and it is the one this document cannot fix.
`scripts/project_status.py` currently reports **three modules built and
tested but never exercised end to end** — the agent harness, the trajectory
analysis, and the task factory. That is the scope-collapse pattern in
progress, visible early rather than in a post-mortem.

The action that reduces it is not another module. It is one real agent run
on the 58 tasks that already exist, which turns two of those three into
`working` and costs a few euros.
