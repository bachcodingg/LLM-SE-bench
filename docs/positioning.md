# Positioning: which numbers to quote

A note to the author, kept in the repository because the temptation it
guards against recurs every time this project gets described.

## The rule

**Never target lines of code. Target capability.**

LOC is an input metric. "24,858 LOC, 98 files, 82 classes, 474 functions" on
a CV does not read as *accomplished* to a senior engineer; a meaningful
fraction read it as *padded, or does not know which numbers matter*. If code
is ever added to hit a number, the project is worse for it.

The LOC belongs in the repository, where anyone can count it. It does not
belong in the description.

## The numbers that survive scrutiny

All verified against the committed artifact and the cost database, not
recalled from memory. Recompute with `python scripts/build_run_artifact.py`.

| Claim | Value | Where to check it |
|---|---|---|
| Evaluations | 522 across 3 models and 4 datasets | `results/2026-05-run/manifest.json` |
| Tasks | 58 Java problems, 3 runs each | `counts` in the manifest |
| API calls | 672 | `cost_records` table |
| Total spend | $3.5241 (€3.26 at 1.08) | `cost_usd.total` in the manifest |
| Cost attribution | Per call, at call time, with the rates in force | `llm_gateway/cost_tracker.py` |
| Cost spread | $2.58 / $0.85 / $0.09 for identical work | `summary.csv` |
| Execution | Network-isolated Docker, per-test results | `bench/sandbox/docker_sandbox.py` |
| Statistics | Friedman / Nemenyi / Wilcoxon-Holm / Cliff's δ / 10k bootstrap | `analysis/statistical_summary.json` |
| Tests | 963, all passing | `pytest -q` |

These are output metrics. They are checkable by a reader in under a minute,
which is what makes them worth stating.

## Claims to stop making

Two figures that have circulated about this project do not hold up, and both
are the kind a reviewer checks first:

- **"0 test failures."** As of the publication sweep the suite had three
  failures — two stale assertions in `bench/tests/test_datasets.py` that
  predated built-in examples being merged with JSONL, and one in
  `framework/tests/test_integration.py` that constructed a
  `StatisticalSummary` without `n`, which the aggregator then skipped. All
  three are fixed and the suite is green. The honest form of the claim is
  "963 tests, green in CI on every push", with the CI badge as evidence.

- **"0.012% integration defect rate."** Nothing in this repository measures
  that. Without a written definition of a defect, a denominator, and the
  data behind it, the number cannot be defended under questioning and should
  be dropped rather than restated.

## The honest one-sentence description

> An execution-based benchmark harness that evaluates LLMs on Java code
> generation, bug fixing and God Class refactoring in a reproducible Docker
> sandbox, with per-call cost accounting.

And for what the published run found, which is more interesting than a
leaderboard position:

> Across 58 Java tasks, three frontier models were statistically
> indistinguishable on correctness (Friedman p = 0.135) and separated by
> 28× on cost. The tasks are saturated at ~100% pass rate — which is itself
> the finding, and the reason the next version targets repository-level
> refactoring rather than single-file generation.

Volunteering the saturation is not a weakness. A reviewer who spots an
unstated limitation assumes the rest was missed too.

## After v2

The eventual CV line, when the agentic work exists:

> Extended an LLM evaluation framework from single-shot to agentic
> repository-level evaluation: built a multi-provider tool-calling harness,
> mined and validated N Java repository tasks with hermetic JVM environments,
> and introduced behaviour-preservation and structural-quality scoring plus
> automatic test-tampering detection. Reported resolve rate per euro
> alongside raw resolve rate.

No LOC. No file counts. No class counts.
