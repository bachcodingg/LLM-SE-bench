# Reproducing the published run

`results/2026-10-run/` is the artifact of a real run: 696 evaluations across
4 models, 58 tasks, 3 samples each, every one executed under `javac` and
JUnit in Docker, $2.20. This page gives the exact commands, and is honest
about which parts reproduce exactly and which cannot.

## What is pinned

`results/2026-10-run/manifest.json` records:

| Field | Why it is there |
|---|---|
| `git_commit` | Pins the harness *and* the 12 tasks embedded in `bench/datasets/*.py` |
| `prompt_template_sha256` | Prompts are inputs to the measurement; a changed template invalidates comparison |
| `dataset_sha256` | Detects a silent edit to the 46 JSONL tasks |
| `models` | Exact provider model ids |
| `wall_clock` | First and last API call, so a reader can tell which model version was live |
| `cost_usd`, `tokens` | Per-model spend and token counts, recorded at call time |
| `sampling` | Temperature 0 where the provider accepts it, no seed, 3 samples per task |
| `sandbox` | Image digest of the sandbox the run executed in |
| `python_lockfile` | Hash of `requirements.lock` |
| `counts.evaluations_not_executed` | Must be 0; the artifact is refused otherwise |
| `duplicates_dropped`, `reexecuted` | What was deduplicated and re-executed, and which verdicts changed |

## Prerequisites

- Python 3.10+
- Docker, for real execution. Without it the sandbox silently degrades to a
  structural dry-run whose results are optimistic and must not be published.
- API keys for the providers you want to call.

```bash
git clone https://github.com/bachcodingg/LLM-SE-bench
cd LLM-SE-bench
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
llm-se-bench run --dataset humaneval -m claude-sonnet-5-5 -m gpt-5.6-terra \
  --runs 3 --dry-run --output-dir /tmp/dry
```

`--models` / `-m` is repeated once per model; `--models a b c` does not work.

## The run

Four datasets, four models, three samples each. This is the script that
produced the published artifact:

```bash
scripts/run_2026_10.sh            # all four models
scripts/run_2026_10.sh claude-haiku-5-5   # or a subset
```

It loops over the datasets with `--runs 3 --max-concurrency 3`. Expect
roughly **$2.20**. A real run resumes from `results.jsonl`, so re-running
after an interruption skips finished samples. **Run one process at a
time**: two overlapping processes evaluate the same samples twice and
overwrite each other's code files (this happened once in the published
run; see `duplicates_dropped` in the manifest).

Docker needs memory. On a 16 GB Windows host, cap WSL 2 in
`%USERPROFILE%\.wslconfig` (`[wsl2]`, `memory=4GB`) and close what you can:
a container killed for memory used to be recorded as a compile error
against the model. It is now recorded as *not executed*, and the artifact
builder refuses a run that has any.

Then re-execute, build the artifact and the leaderboard. None of these
makes an API call:

```bash
python scripts/reexecute.py          # every recorded response, from the cache
python scripts/build_run_artifact.py --name 2026-10-run
python scripts/leaderboard.py --run results/2026-10-run
```

`reexecute.py` is optional on a healthy host; it is how the published run
recovered from one that was not. It keeps each file's original rows as
`results.first-pass.jsonl` and records the changed verdicts in the
manifest.

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
published latency comparison is internally valid — all four models were
measured under the same conditions in the same window — and is not a claim
about absolute speed.

**The sandbox image.** The manifest records the digest of the image the
run executed in. The base image `eclipse-temurin:17-jdk-jammy` is a moving
tag, so a rebuild today may get a newer JDK 17 patch level and newer
`junit4` packages. For single-file classes and JUnit 4 assertions that is
very unlikely to change a verdict. Compare your digest with the manifest's:

```bash
docker image inspect llm-se-bench-sandbox:17 \
  --format '{{index .RepoDigests 0}}'
```

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
