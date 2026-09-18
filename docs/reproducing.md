# Reproducing the published run

`results/2026-05-run/` is the artifact of a real run: 522 evaluations across
3 models, 672 API calls, $3.52. This page gives the exact commands, and is
honest about which parts reproduce exactly and which cannot.

## What is pinned

`results/2026-05-run/manifest.json` records:

| Field | Why it is there |
|---|---|
| `git_commit` | Pins the harness *and* the 12 tasks embedded in `bench/datasets/*.py` |
| `prompt_template_sha256` | Prompts are inputs to the measurement; a changed template invalidates comparison |
| `dataset_sha256` | Detects a silent edit to the 46 JSONL tasks |
| `models` | Exact provider model ids |
| `wall_clock` | First and last API call, so a reader can tell which model version was live |
| `cost_usd`, `tokens` | Per-model spend and token counts, recorded at call time |
| `sampling` | Temperature 0, no seed available, 3 runs per problem |
| `sandbox` | Dockerfile hash (see the caveat below) |
| `python_lockfile` | Hash of `requirements.lock` |

## Prerequisites

- Python 3.10+
- Docker, for real execution. Without it the sandbox silently degrades to a
  structural dry-run whose results are optimistic and must not be published.
- API keys for the providers you want to call.

```bash
git clone https://github.com/<OWNER>/llm-se-bench
cd llm-se-bench
pip install -e ".[dev]"
cp .env.example .env          # fill in the keys you need
docker build -t llm-se-bench-sandbox:17 bench/sandbox/
```

## Verify before you spend anything

```bash
python scripts/security_sweep.py       # no credentials about to be published
pytest -q                              # the harness itself works
llm-se-bench datasets validate --dataset all   # 58 tasks load, all with suites
llm-se-bench gateway test              # one trivial call per provider (~$0.001)
```

Then a full dry-run, which makes no API calls at all:

```bash
llm-se-bench run --dataset humaneval --models claude gpt4 gemini \
  --runs 3 --dry-run --output-dir /tmp/dry
```

## The run

Four datasets, three models, three runs each. This is the command that
produced the published artifact:

```bash
for ds in humaneval mbpp defects4j godclass; do
  llm-se-bench run --dataset "$ds" \
    --models claude-sonnet-4-6 gpt-4o gemini-2.5-flash \
    --runs 3 --max-concurrency 3 --output-dir results
done
```

Expect roughly **$3.50 and about 90 minutes of wall clock**. The response
cache means a re-run against an unchanged prompt costs nothing, so an
interrupted run resumes cheaply — delete `llm_cache.db` only if you want to
pay for the whole thing again.

Then quality, statistics, report:

```bash
llm-se-bench quality --input results/ --output quality/
llm-se-bench stats  --input results/ --quality quality/ --output analysis/
llm-se-bench report --input analysis/statistical_summary.json --format pdf
python scripts/build_run_artifact.py --name my-run
```

## What will not reproduce exactly, and why

**Model outputs.** Temperature 0 is not determinism. No provider in this run
exposes a seed, providers silently update model weights behind a stable id,
and batching and hardware affect floating-point reduction order. Expect the
same *verdicts* on almost every task and different *text* on many.

**Cost.** Provider prices move. `CostRecord` rows store the per-token rates
in force at call time, so the published total is a historical fact; your
total will differ if prices have changed. Euro figures additionally depend on
`LLM_SE_BENCH_USD_PER_EUR`.

**Latency.** Depends on your network, your region and provider load. The
published latency comparison is internally valid — all three models were
measured under the same conditions in the same window — and is not a claim
about absolute speed.

**The sandbox image.** The manifest records the *Dockerfile* hash, not an
image digest: no digest was captured at run time. The base image
`eclipse-temurin:17-jdk-jammy` is a moving tag, so a rebuild today gets a
newer JDK 17 patch level and newer `junit4` and `maven` packages from apt.
For these tasks — single-file classes, JUnit 4 assertions — that is very
unlikely to change a verdict, but it is a real gap. **If you run this
yourself, record the digest**:

```bash
docker image inspect llm-se-bench-sandbox:17 \
  --format '{{index .RepoDigests 0}}'
```

and pin the base image by digest in `bench/sandbox/Dockerfile` before the
run, not after.

**The response cache is not shipped.** `llm_cache.db` holds full prompts and
model outputs and is git-ignored. This means a fresh clone cannot rebuild the
cost columns of the artifact from scratch, which is exactly why the artifact
is committed rather than derived on demand.

## Checking the committed artifact

```bash
python scripts/build_run_artifact.py --check
```

Rebuilds into a temporary directory and diffs `summary.csv` and
`evaluations.csv` against what is committed. A difference means either
`results/` has changed or the artifact was edited by hand — it should never
be edited by hand.

## Known gap in the published analysis

`analysis/statistical_summary.json` was generated without `--db`, so it
reports `"n_cost": 0` and zeros in the `avg_cost_usd` column of the model
rankings. The cost numbers quoted in the README and in
`results/2026-05-run/summary.csv` come from the cost database directly and
are correct. `--db` now defaults to the configured cache path, so
regenerating the analysis picks the cost data up:

```bash
llm-se-bench stats --input results/ --quality quality/ --output analysis/
```
