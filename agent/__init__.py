"""
agent — the tool-using agent loop.

Single-shot evaluation asks a model for an answer and scores it. This asks
it to *work*: here is a failing task, here are tools, keep going until the
tests pass or you run out of budget.

::

    1. Give the agent a failing task and a tool set, not a prompt-and-pray
       single completion.
    2. The agent calls tools: read_file, list_dir, grep, apply_patch, run_tests.
    3. Tool results go back as tool_result blocks.
    4. Repeat until the tests pass, or a budget is exhausted.

Modules
-------
``agent.workspace``
    The files the agent edits, and the audit trail of what it touched.
``agent.tools``
    The five tools, their schemas, and their bounded output.
``agent.termination``
    All six stopping conditions, and the budget governor that enforces the
    money one *before* each call rather than after.
``agent.trajectory``
    One row per step, the raw material for failure analysis; plus replay,
    which re-scores a recorded episode without calling any API.
``agent.loop``
    The loop itself.

Provider differences live in :mod:`llm_gateway.adapters`, behind the
neutral types in :mod:`llm_gateway.conversation`, so nothing here knows
which provider it is talking to.

Nothing in this package spends money except :class:`~agent.loop.AgentLoop`
running against a real client, and it will not start without a cost
ceiling.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
