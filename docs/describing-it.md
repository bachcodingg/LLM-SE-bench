# Describing this project

What to say about it, in the three places it will be described, with every
number checkable. Companion to [`positioning.md`](positioning.md), which
covers which numbers to quote; this covers how to phrase them.

The rule underneath all of it: **say only what survives a reader opening the
repository and checking.** Every figure below is verified against
`results/2026-05-run/manifest.json` or `pytest -q`, and anything that is not
yet true is marked as such rather than quietly rounded up.

## For a hackathon or team application

One paragraph. Lead with the artifact, close with the problem you solve for
them.

> I built an LLM evaluation harness that ran 522 evaluations across Claude
> Sonnet 4.6, GPT-4o and Gemini 2.5 Flash for a total API spend of $3.52,
> with per-call cost tracking, response caching and rate limiting.
> Repo: github.com/bachcodingg/LLM-SE-bench. Given the token budget
> constraint, I can instrument the group's spend from day one.

Why that shape:

- **The spend figure is the hook.** $3.52 for 522 evaluations is unusual
  enough to be remembered and small enough to be obviously honest. Nobody
  inflates a number downward.
- **"Per-call cost tracking" is the claim that matters** and it is
  load-bearing: `llm_gateway/cost_tracker.py` writes a row per call with the
  rates in force at call time, which is why the total is a historical fact
  rather than a reconstruction.
- **The last sentence answers a concern before it is raised.** If the person
  reading has mentioned a token budget, this is the sentence that turns you
  from a participant into the person who handles it. Only include it if the
  cost plumbing actually works — it does, and `llm-se-bench gateway costs
  --summary` demonstrates it in one command.

If you get a follow-up question, the strongest available answer is the
finding rather than the feature:

> The interesting result was negative: across those 58 tasks the three
> models were statistically indistinguishable on correctness — Friedman
> p = 0.135 — and separated by 28x on cost. The benchmark is saturated at
> about a 100% pass rate, which is itself the finding, and it is why the
> next version targets repository-level refactoring instead.

Volunteering the saturation is the move. A reviewer who finds an unstated
limitation assumes the rest was missed too; one who is told it up front
reads everything else as more credible.

## For a CV

**Now**, with what exists:

> Built an execution-based LLM benchmark harness for Java: 522 evaluations
> across three frontier models in a network-isolated Docker sandbox, with
> per-call cost attribution ($3.52 total), non-parametric significance
> testing, and a reproducible published run artifact.

**After v2**, when the task factory has actually been run:

> Extended an LLM evaluation framework from single-shot to agentic
> repository-level evaluation: built a multi-provider tool-calling harness,
> mined and validated N Java repository tasks with hermetic JVM
> environments, and introduced behaviour-preservation and structural-quality
> scoring plus automatic test-tampering detection. Reported resolve rate per
> euro alongside raw resolve rate.

**Do not write the second line until N is a real number.** It is the one
claim on this page that a reader could not verify by opening the repository,
and writing it early is how the whole thing stops being checkable.

No LOC. No file counts. No class counts. See
[`positioning.md`](positioning.md) for why.

## For a master's application

This project is a stronger single artifact than the thesis alone — it is
public, reproducible, and addresses a documented gap — so the writeup is
worth treating as a draft research statement rather than a README.

**A draft exists: [`research-statement.md`](research-statement.md).** It is
assembled from evidence and written in nobody's voice in particular, which
is the one thing you have to fix before it is usable. The skeleton below is
the argument it follows, kept here because the structure outlives any
particular draft.

**1. The gap.** Java and the JVM are thin in agentic benchmarks:
SWE-bench-java has 91 instances, SWE-PolyBench 165 in Java, and EnvBench
measured 29.5% success at automated JVM environment setup. Most failures
happen before any code is written, so most of what these benchmarks measure
is the harness.

**2. Why refactoring specifically.** Pass/fail on tests is the wrong
objective function for refactoring, and wrong in an exploitable way: a
refactoring that changes nothing passes every test. Behaviour preservation
is necessary and not sufficient, which means structural improvement has to
be measured independently — and CK metrics plus a severity classifier are
exactly the instrument, already built for the thesis.

**3. Why cost belongs in the scoring.** Leaderboards report resolve rate and
rarely report euros. An agent solving 5% more tasks at 8x the cost is worse
for anyone with a budget. The v1 run already demonstrates the asymmetry: a
28x cost spread with no significant difference in correctness.

**4. Why tampering has to be instrumented.** Long-horizon agents optimise the
reward signal, and there are cheaper routes to "the tests pass" than fixing
the code. Reporting clean solve rate alongside raw pass rate makes the gap
between them a result.

**5. What is already built, and what is not.** Nine modules with a working
tested core; the task factory built but not yet run. Saying this plainly is
better than implying a finished system — a committee that discovers the gap
themselves discounts everything else.

Keep it to two pages. The repository is the evidence; the statement is the
argument for why the evidence matters.

## Phrases to avoid

| Instead of | Say |
|---|---|
| "24,858 lines of code" | "522 evaluations for $3.52, with per-call cost attribution" |
| "0 test failures" | "1340 tests, green in CI on every push" |
| "0.012% integration defect rate" | *nothing — the repository does not measure this* |
| "Comprehensive benchmark" | "58 Java tasks across four datasets, saturated at ~100%" |
| "Production-ready" | "Reproducible: `scripts/build_run_artifact.py --check` verifies the published artifact against a rebuild" |
| "Supports agentic evaluation" | "Four agent scaffolds behind one interface, so model quality can be separated from scaffold quality" |
| "Mined N repository tasks" | *not yet true — see the CV section* |

The pattern: every replacement is a claim a reader can check in under a
minute. That is what makes the rest of the description worth trusting.

## A note on the negative result

The temptation is to bury it. The saturation — 517 of 522 evaluations
passing — reads like a failure of the benchmark, and it is one.

It is also the most defensible thing in the project. A benchmark author who
reports their own ceiling effect, names it as the reason for the next
version's design, and shows the statistics behind it is demonstrating
exactly the judgement the work is supposed to exhibit. The finding is not
"three models are equally good at Java"; it is "these tasks cannot tell
three models apart, and here is what would".

Lead with it.
