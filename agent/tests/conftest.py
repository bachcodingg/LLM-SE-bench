"""
Shared fixtures for the agent tests.

Everything here runs offline. The model is a scripted stub, the sandbox is
never started, and no test touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agent.termination import TerminationPolicy
from agent.tools import ToolOutcome
from agent.workspace import Workspace
from llm_gateway.conversation import AssistantTurn, TokenUsage, ToolCall

BUGGY_SOURCE = """\
public class Calculator {
    public static int add(int a, int b) {
        return a - b;
    }
}
"""

FIXED_SOURCE = """\
public class Calculator {
    public static int add(int a, int b) {
        return a + b;
    }
}
"""

TEST_SOURCE = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class CalculatorTest {
    @Test
    public void testAdd() {
        assertEquals(5, Calculator.add(2, 3));
    }

    @Test
    public void testAddZero() {
        assertEquals(2, Calculator.add(2, 0));
    }
}
"""


@pytest.fixture
def workspace() -> Workspace:
    """A two-file workspace whose test file is read-only."""
    return Workspace(
        files={"Calculator.java": BUGGY_SOURCE, "CalculatorTest.java": TEST_SOURCE},
        read_only={"CalculatorTest.java"},
    )


class ScriptedTestRunner:
    """A ``run_tests`` stand-in whose verdict depends on the source.

    Keyed on a substring so it models the real property that matters: the
    tests pass once, and only once, the code is actually fixed.
    """

    def __init__(
        self,
        passing_marker: str = "a + b",
        total: int = 2,
        source_path: str = "Calculator.java",
    ) -> None:
        self.passing_marker = passing_marker
        self.total = total
        self.source_path = source_path
        self.calls = 0

    def __call__(self, workspace: Workspace) -> ToolOutcome:
        self.calls += 1
        source = workspace.read(self.source_path) if self.source_path in workspace else ""
        passed = self.total if self.passing_marker in source else 0
        return ToolOutcome(
            content=(
                f"All {self.total} tests passed."
                if passed == self.total
                else f"0/{self.total} tests passed.\nexpected:<5> but was:<-1>"
            ),
            is_error=passed != self.total,
            tests_passed=passed,
            tests_total=self.total,
            compiled=True,
        )


@pytest.fixture
def test_runner() -> ScriptedTestRunner:
    return ScriptedTestRunner()


@dataclass
class ScriptedClient:
    """A client that replays a fixed list of assistant turns.

    This is what lets the loop be tested end to end without an API key: the
    loop cannot tell a scripted turn from a real one, because both arrive
    as the same neutral :class:`AssistantTurn`.
    """

    turns: list[AssistantTurn]
    calls: list[dict] = field(default_factory=list)
    raise_on_call: Exception | None = None

    def send_conversation(self, conversation, tools, model_id, max_tokens, temperature):
        if self.raise_on_call is not None:
            raise self.raise_on_call
        self.calls.append({
            "messages": len(conversation.messages),
            "tools": [tool.name for tool in tools],
            "model_id": model_id,
        })
        if not self.turns:
            # Out of script: behave like a model that has finished talking.
            return AssistantTurn(text="Done.", stop_reason="end_turn", model_id=model_id)
        return self.turns.pop(0)


def tool_turn(name: str, arguments: dict, text: str = "", **usage) -> AssistantTurn:
    """An assistant turn requesting one tool call."""
    return AssistantTurn(
        text=text,
        tool_calls=[ToolCall(call_id=f"call-{name}-{id(arguments)}", name=name, arguments=arguments)],
        stop_reason="tool_use",
        usage=TokenUsage(input_tokens=usage.get("input", 100),
                         output_tokens=usage.get("output", 50),
                         cache_read_tokens=usage.get("cached", 0)),
        model_id="test-model",
    )


def final_turn(text: str = "All tests pass.") -> AssistantTurn:
    """An assistant turn that stops calling tools."""
    return AssistantTurn(
        text=text,
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=100, output_tokens=20),
        model_id="test-model",
    )


@pytest.fixture
def policy() -> TerminationPolicy:
    """A permissive-but-bounded policy for tests."""
    return TerminationPolicy(
        max_cost_eur=1.0,
        max_steps=20,
        max_tokens=1_000_000,
        no_progress_steps=3,
        wall_clock_seconds=60.0,
    )


@pytest.fixture
def free() -> "object":
    """A cost function that charges nothing, for tests about other things."""
    return lambda turn: 0.0
