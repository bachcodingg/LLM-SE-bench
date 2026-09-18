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
suite establishes behaviour preservation only; structural improvement is
measured separately by `quality/refactoring.py`, which compares CK metrics
before and after: WMC, CBO, RFC and LCOM are lower-is-better, LOC and NOC are
treated as neutral, and a decomposition is credited only if the extracted
classes retain the original's methods and fields (`method_coverage`,
`field_coverage`). This is necessary but not sufficient as an anti-gaming
measure — see Limitations.

## Sampling

Temperature 0, three runs per (model, problem). Greedy decoding is not
deterministic in practice: none of the three providers exposes a seed, so
repeats measure residual provider-side variance rather than sampling
variance. The published run found a single verdict flip across 174
problem-model pairs, all of it in `gemini-2.5-flash` (flip rate 1.7%).

Three runs is enough to detect gross instability and nowhere near enough for
`pass@k` at k > 1 to mean much. `pass@5` is reported because the estimator
extrapolates, not because five samples were drawn.

## Statistics

Non-parametric throughout. Pass rates are bounded in [0, 1], heavily
left-skewed here, and paired — every model sees every problem — so the
parametric assumptions do not hold and the pairing is worth keeping.

| Question | Test |
|---|---|
| Do the three models differ at all on a continuous metric? | Friedman |
| Which pairs differ, given a significant Friedman? | Nemenyi post-hoc |
| Does a specific pair differ? | Wilcoxon signed-rank, Holm-corrected |
| Do the three models differ on a binary pass/fail outcome? | Cochran's Q |
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

Read this before quoting a winner.

**Correctness does not separate the three models on these tasks.** Friedman on
`weighted_score`: χ² = 4.00, p = 0.135. Cochran's Q on pass/fail: Q = 2.00,
p = 0.368. Neither is significant. Claude Sonnet 4.6 and GPT-4o both solved
174/174; Gemini 2.5 Flash solved 169/174. The Wilcoxon comparison between
Claude and GPT-4o has literally zero non-zero differences.

**Latency does separate them, strongly.** Friedman χ² = 63.03, p < 0.001,
Kendall's W = 0.54. Nemenyi puts GPT-4o (mean rank 1.41) and Claude (1.76)
within the critical difference of each other, and Gemini (2.83) clearly
slower than both.

**Cost separates them by more than an order of magnitude.** $2.58 / $0.85 /
$0.09 for the same 174 evaluations.

So the defensible claim from this run is about cost and latency, not
capability. The composite MCDA ranking in `framework/` puts Claude first by
0.003 points — that gap is noise, and the ranking should be read as "these
three are tied on quality, choose on price".

## Limitations

Stated plainly, because a reviewer who finds an unstated limitation assumes
you missed it.

1. **The benchmark is saturated.** 517 of 522 evaluations passed. A benchmark
   where every model scores ~100% cannot rank models; it can only confirm
   they clear the bar. The tasks are too easy. This is the single most
   important thing to fix, and it is why the interesting numbers here are
   cost and latency rather than correctness.

2. **58 tasks is a small sample.** With a ceiling effect on top of it, the
   statistical power to detect a real 5-point capability difference is very
   low. The non-significant Friedman result is consistent with "no
   difference" and equally consistent with "not enough data".

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

7. **Refactoring scoring can be gamed.** Method and field coverage stop the
   crudest attack — splitting a class into empty shells — but a decomposition
   that mechanically moves each method into its own class scores well on CBO
   and LCOM without being better code. There is no differential testing and
   no public-API-signature check yet.

8. **Cost is recorded at call time and converted from USD.** Provider prices
   change; the rates in force during the run are frozen in the `CostRecord`
   rows, but euro figures depend on `LLM_SE_BENCH_USD_PER_EUR` and are only
   as good as the rate you pin.

9. **One run of the whole benchmark, one point in time.** May 2026, three
   model versions. Nothing here is longitudinal.

10. **Single scaffold.** Every result is a single-shot completion with one
    prompt template per task type. A different prompt, or an agentic loop,
    would produce different numbers. Conclusions drawn here are about
    model-plus-this-scaffold, not about models.
