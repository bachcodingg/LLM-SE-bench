# Research statement (draft)

*Gia Bach Vo — draft of 2026-10-28. Roughly two pages. Every number here is
checkable in the repository; see the provenance note at the end.*

*This is a draft in the literal sense: the argument is assembled from
evidence, the voice is not yours yet. Read it aloud and rewrite the
sentences that sound like someone else wrote them, because they were.*

---

## The problem

Large language models are now evaluated on software engineering almost
entirely through benchmarks that ask one question: did the tests pass?

That question is the right one for bug fixing. It is the wrong one for
refactoring, and wrong in a way that is quietly exploitable — **a
refactoring that changes nothing passes every test.** Behaviour preservation
is a necessary condition that any benchmark scoring on tests alone treats as
sufficient. A model that returns the input unchanged scores as well as one
that genuinely improves the design.

Two further gaps compound this.

**The JVM is badly served.** SWE-bench-java contains 91 instances;
SWE-PolyBench contains 2,110 across all languages and 165 in Java. Reported
Java resolve rates sit far below Python, and the bottleneck is largely
environmental rather than cognitive: EnvBench measured that the best
automated environment-setup method succeeded on 29.5% of JVM repositories.
Seven times in ten, nothing is learned about the model because nothing
compiled. A Java benchmark that does not solve environment setup is
measuring its own harness.

**Cost is absent from the scoring.** Leaderboards report resolve rate and
rarely report money. An agent that resolves five percentage points more
tasks at eight times the price is not obviously better, and for any team
operating under a budget it is worse. The omission is not neutral: it
systematically favours approaches that spend more.

## What I have built, and what it already shows

I built an execution-based benchmark harness for Java code generation, bug
fixing and God Class refactoring. Generated code is compiled and tested
inside a Docker container started with `--network=none`, so a task cannot be
solved by fetching the answer, and every API call writes a cost record
carrying the per-token rates in force at call time — which makes the
reported total a historical fact rather than a reconstruction from a price
list that has since moved.

**My first published result was wrong, and finding out how shaped the
harness.** The May 2026 run reported that three frontier models passed
nearly every task. An audit showed the sandbox had silently fallen back to a
structural check that credits every test as passing, that repeated runs
were mostly cache hits on the first, and that the refactoring suites
asserted `true`. I withdrew the numbers. The harness now records whether
each test actually executed, refuses to publish a run that contains an
unexecuted result, and treats a container killed for memory as *not
executed* rather than as the model's failure — a case the October run hit
for real, and recovered from by re-executing every recorded response
without a single new API call. An evaluation pipeline that cannot fail
loudly reports what its author hoped.

The October 2026 run covered 58 tasks across four datasets, four models from
two vendors at two price tiers, three independent samples each: **696
evaluations, every one compiled and run under JUnit, $2.20 in total.**

**On code generation and bug fixing, the tasks cannot rank the models.** On
47 of the 51 tasks, every sample from every model passes. Two tasks fail
for everyone and only two separate the models at all; a Friedman test over
per-task pass rates gives χ² = 1.40, p = 0.71. Meanwhile the cheapest model
costs about one seventeenth of the most expensive one. These tasks are
single-file, self-contained, and drawn from problem sets that predate every
model evaluated. What they measure is not capability but whether capability
exceeds a threshold that was crossed some time ago.

**On refactoring, passing the tests and refactoring come apart.** Of 84
God Class refactorings, 82 keep the original behaviour, but only **48 earn
structural credit**. Two of the seven tasks were never credited for any
model in any sample. The credit that is earned comes almost entirely from
cohesion: the models split classes, and in none of the 48 is the result
smaller. A benchmark scoring on tests alone would have reported 82 of 84
as solved. Seven tasks are too few to rank models on this axis (p = 0.80),
which is the case for building many more.

## The proposal

I am extending the harness from single-shot evaluation to **repository-level
agentic evaluation of JVM refactoring, scored on three independent axes
rather than one**:

1. **Behaviour preserved** — the suite passes, the public API is unchanged,
   and the pre- and post-refactoring versions agree on generated inputs.
   Three signals, reported separately, because the test suite alone cannot
   see an API break and differential testing alone cannot see intent.

2. **Structure improved** — Chidamber–Kemerer metric deltas with published
   weights, scored on the classes the refactoring extracted. Crucially, with
   explicit anti-gaming guards: deleting code improves every CK metric
   simultaneously, empty classes improve coupling and cohesion for free, and
   a facade can keep all the original logic while wrapping it in new names.
   Each of these is detected, and the October run shows the guards firing
   on real model output, not only on constructed examples.

3. **Euros spent** — resolve rate per euro alongside raw resolve rate, with
   cache reads and writes priced at their own tiers rather than folded into
   the input rate.

Alongside these, **automatic detection of reward hacking**. Long-horizon
agents optimise the reward signal, and there are cheaper routes to "the
tests pass" than fixing the code: editing the test, deleting the assertion,
swallowing the exception, hardcoding the expected value. Detection is
layered — an immutable test manifest that cannot produce a false positive, a
set of static detectors that can and are therefore never allowed to strike a
result on their own, and a held-out suite the agent never sees. The number
this produces is a **clean solve rate**, and the gap between it and the raw
rate is a result in itself.

The harness is also built to separate model quality from scaffold quality.
Four agent strategies — single-shot, ReAct, plan-then-execute, and a
passthrough to an off-the-shelf harness — sit behind one interface and are
recorded on every trajectory. Most work in this area compares models,
publishes conclusions about models, and has in fact compared
model-plus-the-authors'-scaffold. Holding the model fixed and varying the
scaffold is the only way out of that confound, and comparing against a loop
one did not write is the only convincing version of it.

## State of the work

The harness, the agent loop, the three scoring axes, the tamper detection,
the cost governance and the statistical machinery are implemented and
tested offline, without network, API calls or Docker. The October run is
committed as an artifact with model identifiers, prompt-template hashes,
dataset hashes, the sandbox image digest and wall-clock dates; a script
verifies it against a fresh rebuild, and another blocks publication of any
run without execution evidence.

Two things are built and **have not been run**, and I state both plainly
because the distinction between *built* and *demonstrated* is exactly the
one this project argues benchmarks should be careful about.

**The agent study** measures solve rate against a per-episode euro budget
for the same four models, and in a second sweep lets the agent write to
the test files to measure the gap between raw and clean solve rate. It is
capped in code at a few euros; it has not been run for lack of budget.

**The task factory** mines merged pull requests, constructs reverse
patches, derives fail-to-pass and pass-to-pass sets, detects build systems
across Maven and both Gradle DSLs, infers JDK versions, prewarms dependency
caches for hermetic offline execution, and applies six admission gates
including a five-run flake check — all implemented, all tested against
fixtures, none exercised against a real repository. Producing 60–300
validated instances requires network access, a GitHub token, and many
hours of builds.

A script in the repository reports which modules have and have not been
exercised end to end.

## Why this is the right next problem for me

The three gaps above are not independent of my background; they are where it
already points. My thesis work is on God Class detection and decomposition,
which is precisely the instrument the second axis needs — CK metrics and a
severity classifier that already exist and already work. The deterministic
JDK 17 sandbox, which is the part of JVM benchmarking that actually
defeats people, is built and running. The cost accounting that the third
axis needs is not a design; it is a row per API call in a database.

What I want to do next is the part that cannot be done alone in fragments:
mine and validate a real Java task set large enough to rank models on
structure, run the scaffold comparison against an external harness, and
publish the three-axis result with its clean solve rate and its
contamination stratification. That work is slow, unglamorous, and the
reason the field's Java numbers are thin.

I would rather spend a doctorate making one benchmark honest than making
several of them larger.

---

### Provenance

Every figure is reproducible from the repository at
`github.com/bachcodingg/LLM-SE-bench`.

| Claim | Where |
|---|---|
| 696 evaluations, all executed; $2.20 | `results/2026-10-run/manifest.json` (`counts`, `cost_usd`) |
| 47 of 51 code tasks passed by every sample; 2 by none | `results/2026-10-run/evaluations.csv` |
| Friedman χ² = 1.40, p = 0.71 (code tasks); p = 0.80 (structural credit) | per-task pass rates from `evaluations.csv`, `scipy.stats.friedmanchisquare` |
| ~1/17th cost (Haiku $0.0004 vs Sonnet $0.0068 per task) | `results/2026-10-run/summary.csv` |
| 82 of 84 refactorings keep behaviour, 48 credited | `results/2026-10-run/leaderboard.md`, `evaluations.csv` |
| Credit from cohesion, size reduction zero | `structural_*` columns of `evaluations.csv` |
| 100 verdicts changed by re-execution, all MBPP | `manifest.json` (`reexecuted`) |
| Why the May run was withdrawn | `README.md`, "How these numbers were checked" |
| Module completion, including what has not run | `python scripts/project_status.py` |

External figures — SWE-bench-java's 91 instances, SWE-PolyBench's 165 Java
instances, EnvBench's 29.5% — are from the respective papers and should be
cited properly before this goes anywhere.

### Before sending this

- [ ] Rewrite in your own voice. It currently reads as competent and
      anonymous.
- [ ] Add proper citations for SWE-bench-java, SWE-PolyBench, EnvBench,
      SpecBench, Chidamber & Kemerer, and Chen et al. (2021).
- [ ] Name the specific group or supervisor and say why them, in one
      sentence. A statement with no addressee reads as a form letter.
- [ ] Cut to two pages if it has grown. It has a tendency to.
- [ ] Check the two code tasks no model passes (`D4J_Lang_7`, `MBPP_1`).
      If the task is at fault rather than the models, say so or fix it
      before quoting "two fail for everyone".
- [ ] Re-run `scripts/project_status.py` and update the state-of-the-work
      section. If the agent study or the task factory has run by then, that
      paragraph becomes the strongest one here instead of the most honest
      one.
