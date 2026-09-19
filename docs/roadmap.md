# Roadmap and time budget

The sequencing from Part 5 of the upgrade guide, rewritten against what is
actually built. Parts 0–3 are done and most of Part 4's modules have a
working core, which moves the schedule forward — but it moves the *hard*
work forward too, and the hard work is not the code.

## The constraint that does not move

Roughly 47 credits of coursework: Integral Calculus, Differential Calculus,
Matrix Algebra, Statistics II, Functional Programming, Advanced Networks,
Foundations of AI and ML, and two digitalisation courses. A hackathon in
November sits inside that.

This is a **3–6 month part-time project**. It is not a six-week project, and
the fact that the scaffolding now exists does not make it one. Do not
promise v2 to anyone before it exists.

## Where things stand

| | Module | State | What remains |
|---|---|---|---|
| — | Publish (Part 1) | **Done** | Record the dashboard GIF |
| M2 | Tools + MCP | **Done** | — |
| M1 | Agent harness | **Core done** | Mid-episode resume; validate scaffolds against each other |
| M4 | Trajectory analysis | **Core done** | LLM-judge fallback for ambiguous cases |
| M5 | Structural scoring | **Core done** | Differential testing needs a real run against Docker |
| M6 | Tamper detection | **Core done** | Held-out suites need task data that has them |
| M7 | Cost governance | **Core done** | Batch API, once a provider's is wired |
| M8 | Agent statistics | **Core done** | — |
| M9 | Dashboard v2 | **Core done** | Needs real episodes to be worth looking at |
| M10 | CI + regression | **Done** | Needs a repository with secrets configured |
| M3 | **Task factory** | **Machinery only** | **Run it. This is the whole remaining project.** |

The asymmetry is the point. Nine modules are code and code is the part that
finishes. M3 is code *plus* thousands of GitHub API calls, a Maven or Gradle
build per candidate, five runs each for the flake gate, and a human reading
rejection logs to find out why 70% of candidates failed. That is measured in
weekends, not hours.

## Sequence

### Now — publish what exists

Push it. A public repository with a dated commit history is a priority claim
on the idea; a private one is not, and Part 7's risk 6 — somebody publishes
something similar while you build — is the one you cannot mitigate after the
fact.

Before every push, in order:

```bash
python scripts/security_sweep.py     # nothing about to be leaked
python scripts/check_packaging.py    # a fresh clone can actually run
pytest -q && ruff check .
```

The packaging check is not optional and is not paranoia: `.gitignore` once
listed `quality/` under "outputs", which left the whole C3 component out of
the repository while four tracked modules imported it. Every test passed
locally. A clone could not start.

Still outstanding, non-blocking: the dashboard GIF.

### Weeks 1–2 — one real agent run

The smallest thing that makes all of this real: run the agent loop against
the existing 58 problems, with a budget, and publish the comparison.

```bash
llm-se-bench agent run --dataset defects4j --model claude \
  --budget-eur 0.50 --total-budget-eur 5.00 --compare
```

Expect this to be unglamorous and to surface bugs the test suite cannot: a
provider returning a shape no adapter handles, a context window filling
faster than the estimate, a task whose JUnit class name does not match its
file. That is what the two weeks are for.

**The deliverable is a blog post**, not a module: *agent mode versus
single-shot, same 58 tasks, with cost*. It is publishable on its own, it is
the first thing in this project that is genuinely novel, and it takes a
weekend once the loop works.

### Weeks 3–5 — scaffold comparison

Run `single_shot`, `react` and `plan_then_execute` against the same tasks
with the same model. This is the contribution the guide singles out: most
papers compare models, publish conclusions about models, and have actually
compared model-plus-their-scaffold.

Then wire `ExternalHarnessScaffold` to Claude Code or Aider and run it too.
Comparing against a loop you did not write is the only real answer to the
confound, and it is a command-line argument away.

Budget it properly: three scaffolds × 58 tasks × 3 repeats is 522 episodes,
and agent episodes cost 10–100× a single-shot completion. Use
`BudgetSimulator` before, not after.

### November — the hackathon

**Do not start M3.** Use the hackathon to stress-test M1 and M2 on somebody
else's code, under time pressure, with people watching. Take notes on what
broke. That list is worth more than a fortnight of speculative hardening.

### December–March — the task factory

The contiguous-time module, done when you have contiguous time. Not in
fragments between exams: mining is stateful, slow, and unrewarding in
30-minute slices.

1. **Mine and reject.** Run `taskfactory.pipeline` over a handful of
   repositories with `--dry-run`. Read `rejections.json`. The first-failure
   distribution tells you what the pipeline cannot parse, and the first pass
   will reject almost everything for reasons that are your bugs, not the
   repositories'.
2. **Fix the environment detection** against what you actually find. This is
   where the time goes and it is the module's external value: EnvBench
   measured 29.5% success at automated JVM setup, and every point above that
   is a contribution.
3. **Validate.** The five-run flake gate is not optional and costs five
   times everything else put together.
4. **Target 60–80 instances, not 300.** The guide's trim, and it is right:
   80 validated instances beats 300 half-validated ones, and SWE-bench-java
   has 91.

Then M5's differential testing and M6's held-out suites, both of which need
real repository tasks to be worth running.

### March–May — analysis and writeup

M4's process metrics and M8's statistics against real trajectories, then the
writeup. Treat it as a draft research statement: this project is a stronger
artifact in front of an admissions committee than the thesis alone, because
it is public, reproducible, and addresses a documented gap.

## The checkpoint rule

**If the previous phase is not published and working, do not start the next
one.**

Three finished modules beat ten half-finished ones, and a public repository
full of stubs is worse than no repository. The rule exists because Part 7's
first risk is the likely one: you build M1 and M9 because loops and
dashboards are fun, and never finish M3 because mining repositories is
tedious. The order above forces M3 before the interesting analysis, which is
the only way that risk gets managed.

## Cutting, if it comes to that

The guide's priority order, which still holds:

- **Keep**: M1, M2, M6, M7, and M5 — the differentiator.
- **Trim**: M3 to 60–80 instances. M4 to the failure taxonomy, dropping the
  deeper process metrics.
- **Drop**: M8 beyond pass@k, M9 beyond a static report, M10 entirely.

A cut version still supports the core claim:

> A repository-level agentic benchmark for JVM refactoring that scores on
> three independent axes — behaviour preservation, structural quality
> change, and euros spent — with automatic detection of test tampering.

## Deciding in advance what gives

47 credits, a hackathon and this project cannot all run at full intensity.
Something gives, and the only question is whether you choose it now or
discover it in February.

The honest ranking, given that the courses are not optional and the
hackathon is a fixed date: **this project is the thing that slips.** Plan
for it to slip, keep each phase independently publishable, and it survives
the slip. Plan for it not to, and a single bad month leaves an unfinished
v2 and a v1 that was never published either.
