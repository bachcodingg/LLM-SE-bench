# Methodology

How a task is defined, how an answer is scored, and which claims the
resulting numbers do and do not support.

## Task design

58 Java tasks across four datasets, all self-contained: one class, one JUnit
4 suite, no build system, no repository context.

| Dataset | Tasks | Task type | The model is given | It must produce |
|---|---|---|---|---|
| `humaneval-java` | 23 | Code generation | Signature, docstring, up to 3 visible examples | A complete class |
| `mbpp-java` | 18 | Code generation | Natural-language description, visible examples | A complete class |
| `defects4j` | 10 | Bug fixing | Buggy class, defect description, failing test name | A corrected class |
| `godclass` | 7 | Refactoring | A God Class and its CK metrics | A decomposition |

Provenance and licensing are in [`../data/PROVENANCE.md`](../data/PROVENANCE.md).

Hidden test cases exist (`TestCase.is_hidden`) and are excluded from the
prompt but included in scoring, so a solution cannot be fitted to the
examples it was shown.

## Scoring

**Execution, not judgment.** Every task is scored by compiling the generated
class with `javac` and running its JUnit suite in a network-isolated Docker
container. No LLM judges any output. A task passes when it compiles and every
test in its suite passes.

`weighted_score` is the weight-weighted fraction of tests passed:

```
weighted_score = Σ(weight of passing tests) / Σ(weight of all tests)
```

All current tasks use uniform weights, so this equals the plain pass fraction.
`verdict` is `pass` (all tests), `partial` (some), `fail` (none) or `error`
(did not compile, or the sandbox itself failed).

**Refactoring is scored differently, and that is the point.** A refactoring
that changes nothing passes every test. For the `godclass` tasks the test
suite establishes behaviour preservation only, and a `godclass` task passes
only when its suite passes *and* `quality/structural.py` credits the
structure. That scorer combines four axes with published weights: coupling
reduction and cohesion improvement, measured on the classes the refactoring
*extracted*; complexity dispersal (how evenly WMC is spread across the
resulting classes); and size reduction, penalised when it is just deletion. The class that keeps the original's name is treated as a facade:
it may delegate, but it must give up more than half of the original's
branching (`max(0, WMC - methods)`), or the result is flagged as gaming.
The other guards reject extracted classes with no methods, lost behaviour,
a collapsed size and a broken public API. Extracted classes holding a single
trivial method are noted as a warning but do not block credit on their own.
Every refusal comes with a plain-language reason (`explain_score`).

## Sampling

Temperature 0 where the provider accepts it, three samples per (model,
problem). All four models in the published run reject a temperature, so all
ran at their providers' default sampling; the manifest records this per
model. Greedy decoding is not deterministic
in practice either: no provider exposes a seed. The run index is part of the
response-cache key, so the three samples are three real API calls, not one
call scored three times. The published run found a verdict that differs
between samples in 8 of 232 problem-model pairs (3.4%): 3 each for
`gpt-5.6-terra` and `gpt-5.6-luna`, 2 for `claude-sonnet-5-5`, none for
`claude-haiku-5-5`.

Three runs is enough to detect gross instability and nowhere near enough for
`pass@k` at k > 1 to mean much. `pass@5` is reported because the estimator
extrapolates, not because five samples were drawn.

## Statistics

Non-parametric throughout. Pass rates are bounded in [0, 1], heavily
left-skewed here, and paired — every model sees every problem — so the
parametric assumptions do not hold and the pairing is worth keeping.

| Question | Test |
|---|---|
| Do the models differ at all on a continuous metric? | Friedman |
| Which pairs differ, given a significant Friedman? | Nemenyi post-hoc |
| Does a specific pair differ? | Wilcoxon signed-rank, Holm-corrected |
| Do the models differ on a binary pass/fail outcome? | Cochran's Q |
| How large is the difference? | Cliff's delta |
| What is the uncertainty on a point estimate? | Bootstrap CI, 10,000 resamples |
| How often does the same model give a different verdict? | Flip rate, intra-problem SD |

α = 0.05. Pairwise comparisons are Holm-corrected within a metric family.

**Unbiased pass@k** uses the combinatorial correction from Chen et al.
(2021), not "run it k times and take the max", which is biased upward:

```
pass@k = 1 - C(n - c, k) / C(n, k)
```

for `n` samples of which `c` pass.

## What the published run actually found

Read this before quoting a winner. The run is `results/2026-10-run/`: 58
tasks, four models, three samples, 696 evaluations, every one executed in
Docker. The May 2026 run is withdrawn (see the README for why); only its
cost figures stand.

**Correctness on the code tasks does not separate the four models.** On 47
of the 51 code-generation and bug-fixing tasks, every sample from every
model passes. Two tasks (`D4J_Lang_7`, `MBPP_1`) are failed by every
model, which may say more about the task than the models and has not been
checked. Only two tasks separate the models at all. Friedman over per-task
pass rates: χ² = 1.40, p = 0.71.

**Refactoring separates behaviour from structure, not models from each
other.** 82 of 84 God Class refactorings keep the original behaviour, and
48 earn structural credit. `GC_004` and `GC_007` were never credited for any
model. The credit comes almost entirely from cohesion (mean 0.93 across the
credited 48), then complexity dispersal (0.26) and coupling (0.10); size
reduction is zero in every one. Of the 34 that kept behaviour but were not
credited, 24 have a composite at or below zero and 10 were stopped by a
gaming guard, nearly all a facade still holding the original's branching.
Across models, the credited rate does not differ significantly (Friedman
p = 0.80 over seven tasks), and the bootstrap intervals span 60 to 70
points.

**Cost separates the models by more than an order of magnitude.** $0.0068
per task for `claude-sonnet-5-5` against $0.0004 for `claude-haiku-5-5`,
for the same pass rate.

**Latency separates them too.** Friedman χ² = 79.1, p < 0.001. Mean rank,
lower is faster: Haiku 1.60, Terra 2.28, Sonnet 2.43, Luna 3.69.

So the defensible claims from this run are two. On these code tasks, choose
on price, because quality does not differ. On refactoring, passing the tests
is a poor proxy for refactoring, for every model tested.

**How the numbers were checked.** The run host ran out of memory, and
containers killed mid-`javac` were first recorded as compile errors against
the model. Every recorded response was re-executed from the response cache
on a healthy host (`scripts/reexecute.py`, no API calls); exactly the 100
affected MBPP verdicts changed. 27 God Class samples recorded twice by
overlapping processes were deduplicated by keeping the later row, whose code
is the one on disk. Both are recorded in the manifest.

## Limitations

Stated plainly, because a reviewer who finds an unstated limitation assumes
you missed it.

1. **The code tasks are saturated.** On 47 of 51, every sample of every
   model passes. A benchmark where every model scores ~100% cannot rank
   models; it can only confirm they clear the bar. This is why the
   interesting numbers here are refactoring, cost and latency rather than
   correctness.

2. **58 tasks is a small sample, and 7 refactoring tasks is very small.**
   With a ceiling effect on top of it, the statistical power to detect a
   real 5-point capability difference is very low. A non-significant
   Friedman result is consistent with "no difference" and equally
   consistent with "not enough data".

3. **Training-data contamination is near-certain** for `humaneval-java` and
   `mbpp-java`. Both derive from public benchmarks published years before
   every model evaluated. Treat those 41 tasks as an upper bound on
   capability, not a measurement of it.

4. **Tasks are single-file, not repository-level.** No build system, no
   cross-file context, no dependency resolution. This deliberately avoids the
   JVM environment-setup problem that dominates failures in repository-level
   Java benchmarks — and it means these results say nothing about how a model
   performs on a real codebase.

5. **The `defects4j` tasks are reconstructions**, not the upstream defects.
   They reproduce the described fault in a small standalone class. A model
   that fixes `D4J_Lang_1` here has not fixed Commons Lang.

6. **Three runs per task** measures gross instability only.

7. **The structural score is one operationalisation, and can still be
   gamed.** The guards stop empty shells, trivial one-method classes, a
   facade that keeps the logic, deleted code and a broken API. A
   decomposition that mechanically moves methods into classes by name can
   still score on cohesion without being better code, and there is no
   differential testing yet. A different reasonable weighting would move
   the credited rates.

8. **Cost is recorded at call time and converted from USD.** Provider prices
   change; the rates in force during the run are frozen in the `CostRecord`
   rows, but euro figures depend on `LLM_SE_BENCH_USD_PER_EUR` and are only
   as good as the rate you pin.

9. **One run of the whole benchmark, one point in time.** October 2026,
   four model versions from two vendors; Gemini is absent because only a
   free-tier key was available. Nothing here is longitudinal.

10. **Single scaffold.** Every result is a single-shot completion with one
    prompt template per task type. A different prompt, or an agentic loop,
    would produce different numbers. Conclusions drawn here are about
    model-plus-this-scaffold, not about models.
