# Cost model

The published run cost **$3.52** for 672 API calls. This page explains how
that number is produced, why it is trustworthy, and where it is soft.

## Cost is recorded, not estimated

Every call through the gateway writes one row to `cost_records`:

```
record_id  response_id  model_id  prompt_tokens  completion_tokens
cost_per_prompt_token  cost_per_completion_token  total_cost_usd  created_at
```

The per-token rates are written into the row, not looked up later. This is
the whole design: a price list changes, and a cost reconstructed from
today's prices for a call made in May is a guess. These rows are a
historical record.

Token counts come from the provider's own usage response, not from a
tokeniser estimate.

```
total_cost_usd = prompt_tokens  × cost_per_prompt_token
               + completion_tokens × cost_per_completion_token
```

## Rates used in the published run

USD per token, from `llm_gateway/config.py`:

| Model | Input | Output |
|---|---|---|
| `claude-sonnet-4-6` | $3.00 / 1M | $15.00 / 1M |
| `gpt-4o` | $2.50 / 1M | $10.00 / 1M |
| `gemini-2.5-flash` | $0.15 / 1M | $0.60 / 1M |

Override per run with a YAML config passed to `--config`; a model with no
pricing tier gets a zero-cost tier, so an unpriced model silently reports
$0.00. Check `gateway costs --summary` against your provider dashboard the
first time you use a new model id.

## What the published run actually spent

| Model | Calls | Input tokens | Output tokens | Cost | Per evaluation |
|---|---|---|---|---|---|
| `claude-sonnet-4-6` | 222 | 79,689 | 156,227 | $2.58 | $0.0148 |
| `gpt-4o` | 217 | 63,766 | 68,817 | $0.85 | $0.0049 |
| `gemini-2.5-flash` | 233 | 84,034 | 135,683 | $0.09 | $0.0005 |
| **Total** | **672** | **227,489** | **360,727** | **$3.52** | |

Output tokens dominate: 61% of the tokens and, at 4–5× the input rate, the
overwhelming majority of the spend. Prompt engineering that shortens prompts
saves almost nothing here. Anything that shortens *answers* — asking for a
method body instead of a whole file, capping `max_tokens` — is where the
money is.

**Cost per evaluation is not cost per call.** 672 calls produced 522
evaluations: retries and re-prompts do not map one-to-one onto results. The
artifact reports cost per model and leaves per-dataset cost blank rather than
splitting by a ratio that would look precise and not be.

## Caching

`ResponseCache` keys on SHA-256 of the rendered prompt plus the model id. An
identical prompt to the same model returns the stored response and writes no
cost record. The published run has 174 distinct cached responses — one per
(model, problem) — against 672 calls, so re-running a dataset after an
interruption is free for everything already done.

Consequences worth knowing:

- **A cache hit costs nothing and is recorded as nothing.** Total spend is
  the cost of *distinct* prompts, not of evaluations.
- **Deleting `llm_cache.db` means paying again.** It is git-ignored because
  it holds full prompts and model outputs, not because it is disposable.
- **Changing a prompt template invalidates every entry it produced**, since
  the hash covers the rendered text. That is intended: a changed prompt is a
  different measurement.

## Rate limiting

Token-bucket per provider, defaults in `llm_gateway/config.py`:

| Provider | Requests/min | Tokens/min |
|---|---|---|
| Anthropic | 50 | 80,000 |
| OpenAI | 60 | 150,000 |
| Google | 60 | 120,000 |

These are conservative floors, well under most paid tiers. Raise them if your
account allows; the bucket protects against burning quota on a runaway loop
as much as against 429s.

## Budgets

Two controls exist today:

- `LLM_SE_BENCH_BUDGET_EUR` — default ceiling, read by `settings.py`.
- The MCP `run_benchmark` tool takes `budget_eur` and refuses to start when
  the projected spend exceeds it. It also defaults to `dry_run=true`, so an
  agent cannot spend money by calling it without meaning to.

Both are **pre-flight** checks: they estimate before starting and refuse to
begin. There is no mid-run abort that stops a loop once it has crossed a
ceiling. For single-shot evaluation, where per-call cost is bounded and
predictable, that is adequate. For the agent loop this repository does not
yet have, it would not be: agent episodes run 10–100× more expensive than
single-shot completions, and a hard mid-run ceiling has to exist before the
first unbounded loop is ever pointed at a paid API.

## Euros

Provider price lists are in USD. Every euro figure is a conversion at
`LLM_SE_BENCH_USD_PER_EUR` (default 1.08). Pin it for a published run — an
unpinned rate makes a reported euro cost irreproducible, and the conversion
is the only part of the cost pipeline that is not a recorded fact.

At the default rate the published run cost **€3.26**.

## Cost-normalised comparison

Raw resolve rate answers "which is most capable". For a team with a budget
the question is usually "which is most capable per euro". The decision
framework carries cost as a first-class criterion (`framework/profiles/
budget.yaml` weights it heaviest) and `analysis/figures/
cost_per_correct_bar.pdf` plots cost per passing evaluation.

On this run that comparison is stark and slightly unfair: with all three
models at or near 100%, cost per correct answer is just cost, and Gemini
wins by 28×. The honest reading is that these tasks are too easy to
distinguish capability, so price is the only axis left. See
[`methodology.md`](methodology.md#limitations).
