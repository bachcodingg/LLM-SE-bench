"""
agent.tools — the five tools an agent gets, and their bounded output.

``read_file``, ``list_dir``, ``grep``, ``apply_patch``, ``run_tests``.

Small on purpose. Every extra tool is another schema in the system prompt
and another way for a weak model to get lost; and a comparison between
models is only meaningful if the toolset is held fixed, so this one is
fixed and versioned rather than grown per experiment.

Two rules hold for all five:

**Output is bounded, and the bound is visible.** Everything goes through
:mod:`mcp_servers.truncation`, and a truncated result says so in the text
the model reads. An agent that gave up because its stack trace was cut is
the harness's failure, not the model's, and the distinction has to be
provable from the trajectory.

**A tool failure is a result, not an exception.** A tool that raises ends
the episode; a tool that returns "that file does not exist, here are the
files that do" lets the agent recover, which is the behaviour being
measured. Only a bug in the harness itself propagates.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from agent.workspace import Workspace, WorkspaceError
from llm_gateway.conversation import ToolSchema
from mcp_servers.truncation import truncate_log

logger = logging.getLogger(__name__)

__all__ = [
    "ToolContext",
    "ToolOutcome",
    "AgentToolset",
    "TOOL_SCHEMAS",
    "MAX_TOOL_OUTPUT_LINES",
]

#: Lines of tool output the agent sees. Deliberately tighter than the MCP
#: layer's default: an agent pays for this text on every subsequent turn,
#: not just once.
MAX_TOOL_OUTPUT_LINES = 120

#: Matches per file in a grep.
MAX_GREP_MATCHES = 40


@dataclass
class ToolOutcome:
    """What running one tool produced."""

    content: str
    is_error: bool = False
    truncated: bool = False
    #: Set by ``run_tests`` so the loop can detect a test delta without
    #: re-parsing the text it just handed the model.
    tests_passed: int | None = None
    tests_total: int | None = None
    compiled: bool | None = None
    executed: bool = True


@dataclass
class ToolContext:
    """What the tools act on for one episode."""

    workspace: Workspace
    #: Runs the workspace's tests. Injected so the loop can supply a real
    #: Docker sandbox, a dry-run sandbox, or a stub, without the tools
    #: knowing which.
    test_runner: Callable[[Workspace], ToolOutcome]
    step_index: int = 0


TOOL_SCHEMAS: list[ToolSchema] = [
    ToolSchema(
        name="read_file",
        description=(
            "Read one file from the workspace.\n\n"
            "Returns the file with 1-based line numbers prefixed, so you can "
            "quote a line number back in apply_patch. Long files are "
            "truncated from the middle and the output says so; use "
            "start_line and end_line to page through one instead of "
            "re-reading the whole thing.\n\n"
            "If the file does not exist, the error lists the files that do. "
            "Do not guess a path twice — call list_dir."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path, e.g. 'Calculator.java'.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to return, 1-based. Omit for the start.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to return, inclusive. Omit for the end.",
                },
            },
            "required": ["path"],
        },
    ),
    ToolSchema(
        name="list_dir",
        description=(
            "List the files in the workspace.\n\n"
            "The workspace is flat and small — a handful of files, not a "
            "repository. One call is normally enough for the whole episode.\n\n"
            "Read-only files are marked. Those are the test files you are "
            "judged by: apply_patch will refuse to write them, and trying is "
            "a wasted step."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Optional glob such as '*.java'. Omit for everything.",
                },
            },
        },
    ),
    ToolSchema(
        name="grep",
        description=(
            "Search the workspace for a regular expression.\n\n"
            "Python regex syntax. Returns 'path:line: text' for each match, "
            "at most 40 per file. Use it to find a symbol before reading a "
            "whole file — it is much cheaper than read_file.\n\n"
            "An invalid pattern returns the regex error rather than failing "
            "the episode. No matches is a normal result, not an error."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Python regular expression.",
                },
                "path_glob": {
                    "type": "string",
                    "description": "Optional glob limiting which files are searched.",
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Case-insensitive match. Default false.",
                },
            },
            "required": ["pattern"],
        },
    ),
    ToolSchema(
        name="apply_patch",
        description=(
            "Write a file in the workspace, replacing it entirely.\n\n"
            "This is a whole-file write, not a diff: `content` must be the "
            "complete new contents of the file. Partial content silently "
            "destroys the rest of the file.\n\n"
            "Creates the file if it does not exist. Refuses read-only paths "
            "— the test files — and says so.\n\n"
            "Writing content identical to what is already there is reported "
            "as a no-op. It still costs a step and still counts toward the "
            "no-progress limit, so read before you write."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path to write.",
                },
                "content": {
                    "type": "string",
                    "description": "The complete new file contents.",
                },
            },
            "required": ["path", "content"],
        },
    ),
    ToolSchema(
        name="run_tests",
        description=(
            "Compile the workspace and run its test suite.\n\n"
            "This is the only tool that tells you whether you are finished, "
            "and it is the most expensive one: it starts a container. Do not "
            "call it after every edit — make a coherent change, then run.\n\n"
            "Returns compile status, passed/total counts, and the failure "
            "output, truncated with the first errors kept.\n\n"
            "A compile failure means no test ran: 0/0 is not 'all passing'. "
            "If the result says execution was skipped, the counts are from a "
            "structural check and mean nothing — report that rather than "
            "treating it as a pass."
        ),
        parameters={"type": "object", "properties": {}},
    ),
]


class AgentToolset:
    """Dispatches tool calls against a :class:`ToolContext`."""

    def __init__(self, context: ToolContext) -> None:
        self.context = context
        self._handlers: dict[str, Callable[..., ToolOutcome]] = {
            "read_file": self.read_file,
            "list_dir": self.list_dir,
            "grep": self.grep,
            "apply_patch": self.apply_patch,
            "run_tests": self.run_tests,
        }

    @property
    def names(self) -> list[str]:
        return sorted(self._handlers)

    def dispatch(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        """Run tool *name* with *arguments*.

        Never raises for anything the agent did: an unknown tool, a missing
        argument and a tool-level failure all come back as an error
        ``ToolOutcome`` the agent can read and recover from.
        """
        handler = self._handlers.get(name)
        if handler is None:
            return ToolOutcome(
                content=(
                    f"No such tool: {name!r}. Available tools: "
                    f"{', '.join(self.names)}."
                ),
                is_error=True,
            )
        try:
            return handler(**arguments)
        except TypeError as exc:
            # Wrong or missing arguments — the model's mistake, and one it
            # can fix if told precisely what was wrong.
            return ToolOutcome(
                content=f"Invalid arguments for {name}: {exc}",
                is_error=True,
            )
        except WorkspaceError as exc:
            return ToolOutcome(content=str(exc), is_error=True)
        except Exception as exc:  # a harness bug, not the agent's
            logger.exception("Tool %s raised", name)
            return ToolOutcome(
                content=f"Tool {name} failed internally: {type(exc).__name__}: {exc}",
                is_error=True,
            )

    # ── tools ─────────────────────────────────────────────────────────

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> ToolOutcome:
        content = self.context.workspace.read(path)
        lines = content.splitlines()

        first = max(1, start_line or 1)
        last = min(len(lines), end_line or len(lines))
        if first > len(lines):
            return ToolOutcome(
                content=f"{path} has {len(lines)} lines; start_line {first} is past the end.",
                is_error=True,
            )

        numbered = "\n".join(
            f"{number:>5}  {lines[number - 1]}" for number in range(first, last + 1)
        )
        header = f"{path} (lines {first}-{last} of {len(lines)})"
        if self.context.workspace.is_read_only(path):
            header += "  [read-only]"

        text, truncation = truncate_log(numbered, max_lines=MAX_TOOL_OUTPUT_LINES)
        if truncation.truncated:
            text += (
                f"\n\n[{truncation.dropped_lines} lines omitted. Use start_line "
                f"and end_line to read a specific range.]"
            )
        return ToolOutcome(content=f"{header}\n{text}", truncated=truncation.truncated)

    def list_dir(self, pattern: str = "") -> ToolOutcome:
        paths = self.context.workspace.list_paths(pattern)
        if not paths:
            where = f" matching {pattern!r}" if pattern else ""
            return ToolOutcome(content=f"No files in the workspace{where}.")

        rows = []
        for path in paths:
            lines = len(self.context.workspace.read(path).splitlines())
            marker = "  [read-only]" if self.context.workspace.is_read_only(path) else ""
            rows.append(f"  {path}  ({lines} lines){marker}")
        return ToolOutcome(content=f"{len(paths)} file(s):\n" + "\n".join(rows))

    def grep(
        self,
        pattern: str,
        path_glob: str = "",
        ignore_case: bool = False,
    ) -> ToolOutcome:
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            return ToolOutcome(
                content=f"Invalid regular expression {pattern!r}: {exc}",
                is_error=True,
            )

        hits: list[str] = []
        searched = self.context.workspace.list_paths(path_glob)
        for path in searched:
            per_file = 0
            for number, line in enumerate(
                self.context.workspace.read(path).splitlines(), start=1
            ):
                if regex.search(line):
                    hits.append(f"{path}:{number}: {line.strip()}")
                    per_file += 1
                    if per_file >= MAX_GREP_MATCHES:
                        hits.append(f"{path}: [more than {MAX_GREP_MATCHES} matches, stopped]")
                        break

        if not hits:
            return ToolOutcome(
                content=f"No match for {pattern!r} in {len(searched)} file(s)."
            )
        text, truncation = truncate_log("\n".join(hits), max_lines=MAX_TOOL_OUTPUT_LINES)
        return ToolOutcome(
            content=f"{len(hits)} match(es):\n{text}",
            truncated=truncation.truncated,
        )

    def apply_patch(self, path: str, content: str) -> ToolOutcome:
        edit = self.context.workspace.write(
            path, content, step_index=self.context.step_index
        )
        if edit.operation == "no-op":
            return ToolOutcome(
                content=(
                    f"{path} already contained exactly this content; nothing "
                    f"changed. Read the file before rewriting it."
                )
            )
        verb = "Created" if edit.operation == "create" else "Updated"
        return ToolOutcome(
            content=(
                f"{verb} {path} ({len(content.splitlines())} lines, "
                f"+{edit.lines_added}/-{edit.lines_removed}). "
                f"Run run_tests to check it."
            )
        )

    def run_tests(self) -> ToolOutcome:
        return self.context.test_runner(self.context.workspace)
