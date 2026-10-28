# Live demo: a refactoring that passes every test and still fails

Five minutes. One claim: **passing the tests is not the same as refactoring**,
and the harness can tell the difference while an agent is using it.

## Before going on stage

```bash
docker info                                   # the sandbox needs Docker
pip install -e ".[mcp]"
python -m mcp_servers --self-test             # three servers build
python scripts/demo.py                        # dry run of everything below
llm-se-bench dashboard --port 8050            # leave running in a tab
```

Register the server with Claude Code once, from the repository root:

```bash
claude mcp add llm-se-bench -- python -m mcp_servers llm-se-bench
```

(or drop [`mcp/claude_desktop_config.json`](../mcp/claude_desktop_config.json)
into `.mcp.json` with your path filled in.)

## 1. The cheat (Claude Code, ~90 s)

Ask Claude Code:

> Use the llm-se-bench MCP server. Get task GC_001, then submit the
> original class completely unchanged to evaluate_patch and tell me the
> verdict and why.

What comes back: **3/3 tests pass, verdict `partial`, not credited**, with
the reason in plain words: the class *still holds 100% of the original's
branching*. A pass/fail harness would have scored this as solved.

## 2. The sneakier cheat (~60 s)

> Now add three empty classes to the same file and evaluate it again.

Every CK metric improves (coupling and cohesion both go to +1.00), and it
is still not credited: *3 extracted classes declare no methods*. This is
the attack the anti-gaming guard exists for.

## 3. The real thing (~90 s)

> Now actually decompose OrderProcessor: extract classes that own the
> logic, keep OrderProcessor as a thin facade with the same public API, and
> evaluate until it is credited.

Claude Code iterates against `evaluate_patch`. The finish line is
`verdict: pass` *and* `structural.credited: true`, with the per-axis scores
showing where the improvement came from.

## 4. Why, at scale (dashboard, ~60 s)

Open <http://localhost:8050/refactoring>:

- **By model:** behaviour kept against structure credited, for every model
  in the October 2026 run. The gap between the two bars is the finding.
- **Check a refactoring:** paste any of the three versions above; same
  code path as the MCP tool.
- **Every recorded refactoring:** each model output with its per-axis
  scores and the reasons it was or was not credited.

## If something breaks on stage

`python scripts/demo.py` replays steps 1–3 without an agent and without any
API call, in about ten seconds. It is the fallback, and it is what the
README GIF shows.
