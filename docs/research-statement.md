# Research statement (draft)

*Gia Bach Vo — draft of 2026-09-19. Roughly two pages. Every number here is
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

The first full run covered 58 tasks across four datasets, three frontier
models, three repeats each: **522 evaluations, 672 API calls, $3.52 total.**

The result was negative, and it is the reason for everything that follows.
The three models were statistically indistinguishable on correctness — a
Friedman test over weighted score gives χ² = 4.00, p = 0.135, and Cochran's
Q over pass/fail gives p = 0.368 — while differing by a factor of 27 in
cost. 517 of 522 evaluations passed.

A benchmark on which every model scores near 100% cannot rank models. It can
only confirm they clear a low bar. The saturation is the finding: these
tasks are single-file, self-contained, and drawn from problem sets that
predate every model evaluated. What they measure is not capability but
whether capability exceeds a threshold that was crossed some time ago.

## The proposal

I am extending the harness from single-shot evaluation to **repository-level
agentic evaluation of JVM refactoring, scored on three independent axes
rather than one**:

1. **Behaviour preserved** — the suite passes, the public API is unchanged,
   and the pre- and post-refactoring versions agree on generated inputs.
   Three signals, reported separately, because the test suite alone cannot
   see an API break and differential testing alone cannot see intent.

2. **Structure improved** — Chidamber–Kemerer metric deltas with published
   weights. Crucially, with explicit anti-gaming guards: deleting code
   improves every CK metric simultaneously, so a decomposition is credited
   only when the extracted classes retain the original's members and carry
   non-trivial behaviour.

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
tested: 1,340 tests, run offline, green in CI on every push. The published
run is committed as an artifact with model identifiers, prompt-template
hashes, dataset hashes and wall-clock dates, and a script verifies the
artifact against a fresh rebuild.

**The task factory is built and has not yet been run.** Mining merged pull
requests, constructing reverse patches, deriving fail-to-pass and
pass-to-pass sets, detecting build systems across Maven and both Gradle
DSLs, inferring JDK versions, prewarming dependency caches for hermetic
offline execution, and six admission gates including a five-run flake check
— all implemented, all tested against fixtures, none exercised against a
real repository. Producing 60–300 validated instances requires network
access, a GitHub token, and many hours of builds.

I state this plainly because the alternative is a reader discovering it, and
because the distinction between *built* and *demonstrated* is exactly the
distinction this project argues benchmarks should be careful about. A script
in the repository reports which modules have and have not been exercised end
to end.

## Why this is the right next problem for me

The three gaps above are not independent of my background; they are where it
already points. My thesis work is on God Class detection and decomposition,
which is precisely the instrument the second axis needs — CK metrics and a
severity classifier that already exist and already work. The deterministic
JDK 17 sandbox, which is the part of JVM benchmarking that actually
defeats people, is built and running. The cost accounting that the third
axis needs is not a design; it is 672 rows in a database.

What I want to do next is the part that cannot be done alone in fragments:
mine and validate a real Java task set, run the scaffold comparison against
an external harness, and publish the three-axis result with its clean solve
rate and its contamination stratification. That work is slow, unglamorous,
and the reason the field's Java numbers are thin.

I would rather spend a doctorate making one benchmark honest than making
several of them larger.

---

### Provenance

Every figure is reproducible from the repository at
`github.com/bachcodingg/LLM-SE-bench`.

| Claim | Where |
|---|---|
| 522 evaluations, 672 calls, $3.5241 | `results/2026-05-run/manifest.json` |
| Friedman χ² = 4.00, p = 0.135; Cochran's Q p = 0.368 | `analysis/statistical_summary.json` |
| 517 of 522 passed; 27× cost spread (27.5, rounded down) | `results/2026-05-run/summary.csv` |
| 1,340 tests | `pytest -q` |
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
- [ ] Re-run `scripts/project_status.py` and update the state-of-the-work
      section. If the task factory has run by then, that paragraph becomes
      the strongest one here instead of the most honest one.
