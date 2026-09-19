# Agent mode

Single-shot evaluation asks a model for an answer and scores it. Agent mode
asks it to *work*: here is a failing task, here are tools, keep going until
the tests pass or you run out of budget.

```bash
# Costs nothing. No API call, no Docker, no model.
llm-se-bench agent run --task D4J_Lang_1 --model claude --dry-run

# A real episode. The budget ceiling is not optional.
llm-se-bench agent run --task D4J_Lang_1 --model claude --budget-eur 0.50

# A dataset, with a ceiling on the whole run as well as on each episode.
llm-se-bench agent run --dataset defects4j --model claude \
  --budget-eur 0.50 --total-budget-eur 5.00 --compare

# Read an episode back, step by step.
llm-se-bench agent show ep-20260919T034026-0aa688
```

## The loop

```
give the agent a failing task and a tool set
while not terminated:
    check the budget            <- BEFORE the call, not after
    ask the model
    run the tools it asked for
    feed the results back as tool results
record everything
```

Five tools, held fixed: `read_file`, `list_dir`, `grep`, `apply_patch`,
`run_tests`. Small on purpose — every extra tool is another schema in the
context and another way for a weak model to get lost, and a comparison
between models only means something if the toolset is the same for both.

The workspace is an in-memory dict of `{path: content}`, not a directory.
That makes an episode serialisable, makes escape impossible rather than
merely forbidden, and makes every mutation observable — which is what the
no-progress rule and the churn metric are computed from.

**Test files are read-only.** `apply_patch` refuses them and says why. An
agent that can edit the tests it is judged by is not solving the task, and
catching that afterwards is much harder than preventing it.

## Termination

All six conditions, all enforced:

| Condition | Fires when |
|---|---|
| `success` | Target tests pass **and** nothing previously passing broke |
| `max_steps` | Tool calls exhausted (default 30) |
| `max_tokens` | Token ceiling reached (default 500k) |
| `max_cost_eur` | Spend ceiling reached — **no default** |
| `no_progress` | N consecutive steps with no file change and no test delta |
| `wall_clock` | Seconds elapsed (default 900) |

Plus `abandoned` (the model stopped calling tools without solving) and
`error` (the provider or the harness failed — kept separate so it cannot be
counted as a capability result).

Three details that are load-bearing:

**Success requires no regression.** A patch that fixes the target by
breaking two other tests is not a solution. The loop runs the suite once
*before* the agent touches anything, and compares. Without that baseline
there is nothing to compare against and the check is silently vacuous.

**Success requires real execution.** The Docker sandbox falls back to a
structural check when Docker is missing, and that check optimistically
credits every `@Test` it can see as passing. `tests_executed` is part of the
success condition, not a footnote — without it, a machine with no Docker
produces a 100% resolve rate that looks exactly like a real one. Episodes
that never executed are flagged in the trajectory and in the CLI output.

**The budget is checked before each call.** Steps and tokens can be counted
afterwards; money cannot, because the call is what spends it. The governor
projects the worst case for the next call and refuses if it would cross. The
projection is deliberately pessimistic: an agent that stops one step early
has cost you a result, an agent that overruns a budget has cost you trust.

There is no default cost ceiling and `llm-se-bench agent run` refuses a real
run without one. Agent episodes cost 10–100× a single-shot completion.
Never point an unbounded loop at a paid API.

## Provider abstraction

The three providers disagree about every part of tool calling:

| | Anthropic | OpenAI | Google |
|---|---|---|---|
| Tool schema | `input_schema` | `function.parameters` | `function_declarations[].parameters` |
| Call arrives as | `tool_use` block | `message.tool_calls[]` | `function_call` part |
| Arguments | dict | **JSON string** | dict |
| Result goes back as | `user` + `tool_result` | `role: "tool"` message | `function_response` part |
| Linked by | `tool_use_id` | `tool_call_id` | function **name** |
| System prompt | top-level `system` | a message | `system_instruction` |
| Cached tokens | separate field | **inside** `prompt_tokens` | separate field |
| Assistant role | `assistant` | `assistant` | `model` |

Every one of those is a chance to be silently wrong. So the loop speaks only
the neutral types in `llm_gateway/conversation.py`, and each adapter in
`llm_gateway/adapters/` translates once, in one place, with a test per
difference. The adapters are pure — no network, no SDK client, no state —
which is what lets 40 tests cover them without an API key.

Two consequences worth stating rather than discovering:

- **OpenAI sends arguments as a model-generated string, so it can be invalid
  JSON.** That is not hypothetical. A malformed call is flagged
  (`ToolCall.malformed`), the raw text is kept, and the policy is
  configurable: `reprompt` (default — tell the model, costs one step),
  `fail_step`, or `abort`. The rate is recorded on every trajectory.
- **Gemini has no per-call id and links results by function name.** The
  adapter synthesises deterministic ids so a recorded trajectory replays
  identically. The real consequence: two concurrent calls to the same
  function in one turn could not be told apart on the way back. The loop
  runs tool calls sequentially, which sidesteps it.

Cache reads and writes are tracked separately from base input tokens
throughout. Folding them together is the single easiest way to make a cost
comparison wrong in a way that looks right.

## The trajectory table

One row per step. This is the raw material for everything that comes after —
failure taxonomy, localisation accuracy, trajectory diffing — so the schema
is fixed now rather than grown later.

| Column | |
|---|---|
| `step_index` | Position in the episode |
| `thought_text` | Prose the model emitted with the call |
| `tool_name`, `tool_args` | What it asked for |
| `tool_result_hash` | SHA-256 of the result |
| `result_preview` | First 400 characters, for a human reader |
| `tokens_in`, `tokens_out`, `cache_read_tokens` | Priced separately |
| `cost_eur`, `latency_ms` | What the step cost |
| `files_touched` | Paths this step wrote |
| `tests_passing_after` | Test state after the step |
| `tool_error`, `malformed_call` | Failure flags |

The hash rather than the result: tool output is large and often repeated,
and hashing makes a loop — five identical calls — visible at a glance
without storing the payload five times.

Derived metrics come free: `steps_to_first_edit`, `repeated_calls`,
`tool_error_rate`, `malformed_call_rate`, `cache_hit_rate`, `edit_churn`
(lines written and then written away again — real work that leaves no trace
in the final diff).

**Cost attribution.** A turn requesting three tools made *one* API call. Its
cost and tokens are attributed to the first of the three steps, not spread
across them, so per-step cost figures stay honest.

## Replay

```python
from agent.trajectory import TrajectoryStore, replay_scoring

store = TrajectoryStore("results/trajectories")
trajectory = store.load("ep-20260919T034026-0aa688")

result = replay_scoring(
    trajectory,
    score=lambda files: my_new_scoring_rule(files),
)
```

Re-scores a recorded episode **without calling any API**, using the stored
final workspace. Most harnesses lack this, and without it every change to a
scoring rule costs another full run's spend to evaluate — which in practice
means the question stops being asked.

## Agent versus single-shot

```bash
llm-se-bench agent run --dataset defects4j --model claude \
  --budget-eur 0.50 --total-budget-eur 5.00 --compare
```

Prints resolve rate and euros for both modes on the same tasks. The point is
not that the agent wins — on a task it can iterate against, it usually does
— but **what the win cost**. An agent that solves 5% more tasks at 8× the
price is worse for anyone with a budget, and a leaderboard reporting only
resolve rate hides that completely.

Read the caveat the comparison prints: the agent iterates against the same
suite it is judged by and single-shot does not get to. That is "what
iteration buys", not a like-for-like capability comparison.

## Known gaps

Stated because an unstated limitation reads as one that was missed.

- **No context compaction.** A long episode will eventually exceed the
  context window and the provider will error, which the loop records as
  `error`. Summarising old turns is not implemented.
- **No parallel episodes.** One at a time. Parallelism needs a Docker
  container per worker.
- **No mid-episode resume.** A trajectory is serialisable and an episode is
  not; the loop cannot be restarted from step 12.
- **One scaffold.** `--scaffold` is recorded on every trajectory so results
  can be separated later, but only the ReAct-style loop is implemented.
  Plan-then-execute and an external-harness passthrough are what would let
  "the model is better" be told apart from "my loop is better", and that
  confound is real until they exist.
- **Cache reads are priced at the full input rate.** That overstates cost,
  which is the safe direction for a budget check, but it means a reported
  agent cost is an upper bound rather than a measurement until per-tier
  cache pricing exists.
- **No reward-hacking detection beyond read-only test files.** An agent
  cannot edit the tests, but nothing yet detects a hardcoded return value
  matching the expected output, a swallowed exception, or a weakened
  assertion.
