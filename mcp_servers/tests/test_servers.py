"""
Tests for the MCP server wiring.

These cover what only shows up when a client connects: a tool that failed to
register, a Pydantic model the SDK cannot turn into a schema, a description
that is missing or too thin to steer a model.

Skipped in full when the SDK is absent, since the harness is usable without
it. The tool logic is covered by the other test modules, which need no SDK.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from mcp_servers import SERVER_NAMES
from mcp_servers._compat import MCPServer
from mcp_servers.servers import BUILDERS, build_server

pytestmark = pytest.mark.skipif(
    MCPServer is None,
    reason="MCP SDK not installed; install the [mcp] extra",
)

EXPECTED_TOOLS = {
    "jvm-sandbox": {
        "compile_java", "run_tests", "run_static_analysis", "compute_ck_metrics",
    },
    "llm-se-bench": {
        "list_tasks", "get_task", "evaluate_patch", "run_benchmark", "get_run_results",
    },
    "god-class-tools": {
        "detect_god_classes", "propose_decomposition", "score_decomposition",
    },
}


def _tools(server):
    """Tool objects registered on *server*, across SDK versions."""
    result = server.list_tools()
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    return list(result)


def _input_schema(tool):
    """The tool's JSON Schema. 1.x calls it inputSchema, 2.x input_schema."""
    for attribute in ("input_schema", "inputSchema"):
        schema = getattr(tool, attribute, None)
        if schema is not None:
            return schema
    raise AssertionError(f"{tool.name} exposes no input schema")


@pytest.fixture(scope="module", params=SERVER_NAMES)
def server_and_name(request):
    return build_server(request.param), request.param


class TestRegistration:
    def test_every_server_is_in_the_builder_table(self):
        assert set(BUILDERS) == set(SERVER_NAMES)

    def test_unknown_server_name_raises(self):
        with pytest.raises(KeyError):
            build_server("not-a-server")

    def test_registers_exactly_the_documented_tools(self, server_and_name):
        server, name = server_and_name
        assert {tool.name for tool in _tools(server)} == EXPECTED_TOOLS[name]

    def test_server_carries_instructions(self, server_and_name):
        server, _ = server_and_name
        assert server.instructions
        assert len(server.instructions) > 100


class TestToolSchemas:
    def test_every_tool_exposes_an_input_schema(self, server_and_name):
        server, _ = server_and_name
        for tool in _tools(server):
            schema = _input_schema(tool)
            assert isinstance(schema, dict)
            assert schema.get("type") == "object"

    def test_required_arguments_appear_in_the_schema(self, server_and_name):
        """A tool whose required argument is missing from the schema is uncallable."""
        server, name = server_and_name
        required_by_tool = {
            "get_task": "task_id",
            "evaluate_patch": "task_id",
            "run_benchmark": "model",
            "get_run_results": "run_id",
            "compile_java": "source_files",
            "score_decomposition": "before",
        }
        for tool in _tools(server):
            expected = required_by_tool.get(tool.name)
            if expected is None:
                continue
            properties = _input_schema(tool).get("properties", {})
            assert expected in properties, f"{tool.name} is missing {expected}"

    def test_every_tool_has_a_substantial_description(self, server_and_name):
        """Tool descriptions are prompts. A one-liner will get the tool misused."""
        server, _ = server_and_name
        for tool in _tools(server):
            assert tool.description, f"{tool.name} has no description"
            assert len(tool.description) > 200, f"{tool.name}'s description is too thin"

    def test_spending_tool_warns_about_spending(self, server_and_name):
        server, name = server_and_name
        if name != "llm-se-bench":
            pytest.skip("only run_benchmark can spend money")
        tool = next(t for t in _tools(server) if t.name == "run_benchmark")
        assert "SPEND MONEY" in tool.description
        assert "dry_run" in tool.description
        assert "budget" in tool.description.lower()

    def test_sandbox_tools_document_the_dry_run_fallback(self, server_and_name):
        """An optimistic structural check must never be mistaken for execution."""
        server, name = server_and_name
        if name != "jvm-sandbox":
            pytest.skip("applies to the sandbox server")
        for tool_name in ("compile_java", "run_tests"):
            tool = next(t for t in _tools(server) if t.name == tool_name)
            combined = tool.description + server.instructions
            assert "executed" in combined

    def test_task_tools_state_that_solutions_are_withheld(self, server_and_name):
        server, name = server_and_name
        if name != "llm-se-bench":
            pytest.skip("applies to the benchmark server")
        tool = next(t for t in _tools(server) if t.name == "get_task")
        assert "reference solution is never included" in tool.description


class TestToolInvocation:
    """Drive a tool the way a client would, through the server."""

    def _call(self, server, name, arguments):
        result = server.call_tool(name, arguments)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        return result

    def test_list_tasks_round_trips_through_the_server(self):
        server = build_server("llm-se-bench")
        result = self._call(server, "list_tasks", {"limit": 3})
        assert result is not None

    def test_ck_metrics_round_trips_through_the_server(self, simple_class):
        server = build_server("jvm-sandbox")
        result = self._call(
            server,
            "compute_ck_metrics",
            {"source_files": [{"path": "Counter.java", "content": simple_class}]},
        )
        assert result is not None

    def test_detect_god_classes_round_trips_through_the_server(self, god_class):
        server = build_server("god-class-tools")
        result = self._call(
            server,
            "detect_god_classes",
            {"source_files": [{"path": "Everything.java", "content": god_class}]},
        )
        assert result is not None


class TestSelfTest:
    def test_self_test_passes(self, capsys):
        from mcp_servers.__main__ import self_test

        assert self_test() == 0
        captured = capsys.readouterr().out
        for name in SERVER_NAMES:
            assert name in captured

    def test_list_servers_names_every_tool(self, capsys):
        from mcp_servers.__main__ import list_servers

        assert list_servers() == 0
        captured = capsys.readouterr().out
        for tools in EXPECTED_TOOLS.values():
            for tool in tools:
                assert tool in captured
