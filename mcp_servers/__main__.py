"""
mcp_servers.__main__ — launch one of the MCP servers.

::

    python -m mcp_servers jvm-sandbox            # stdio, the usual case
    python -m mcp_servers llm-se-bench --http --port 8931
    python -m mcp_servers --list
    python -m mcp_servers --self-test            # build all three, no I/O

``--self-test`` builds every server and lists the tools each registered,
without opening a transport.  It is what CI runs: it catches a broken
registration, a schema a Pydantic model cannot generate, and a typo in a
tool name, none of which show up until a client connects.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from mcp_servers import SERVER_NAMES
from mcp_servers._compat import SDK_VARIANT, MCPUnavailable
from mcp_servers.servers import build_server


def _tool_names(server) -> list[str]:
    """Tool names registered on *server*, across SDK versions.

    ``list_tools`` is async in both 1.x and 2.x, so this drives it through
    ``asyncio.run`` rather than assuming a sync accessor exists.
    """
    import asyncio
    import inspect

    lister = getattr(server, "list_tools", None)
    if lister is None:
        return []
    result = lister()
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    return sorted(getattr(tool, "name", str(tool)) for tool in result)


def self_test() -> int:
    """Build every server and report its tools. Returns a process exit code."""
    print(f"MCP SDK: {SDK_VARIANT}")
    failures = 0
    for name in SERVER_NAMES:
        try:
            server = build_server(name)
            tools = _tool_names(server)
        except Exception as exc:
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        if not tools:
            print(f"  FAIL {name}: built, but registered no tools")
            failures += 1
            continue
        print(f"  ok   {name}: {len(tools)} tools — {', '.join(tools)}")

    if failures:
        print(f"\n{failures} of {len(SERVER_NAMES)} server(s) failed to build.")
        return 1
    print(f"\nAll {len(SERVER_NAMES)} servers built.")
    return 0


def list_servers(as_json: bool = False) -> int:
    """Print the available servers and their tools."""
    rows = []
    for name in SERVER_NAMES:
        try:
            rows.append({"server": name, "tools": _tool_names(build_server(name))})
        except Exception as exc:
            rows.append({"server": name, "error": f"{type(exc).__name__}: {exc}"})

    if as_json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            if "error" in row:
                print(f"{row['server']}: {row['error']}")
                continue
            print(f"{row['server']}")
            for tool in row["tools"]:
                print(f"  - {tool}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mcp_servers",
        description="Run an llm-se-bench MCP server.",
    )
    parser.add_argument("server", nargs="?", choices=list(SERVER_NAMES),
                        help="Which server to run. Omit with --list or --self-test.")
    parser.add_argument("--list", action="store_true",
                        help="List the servers and their tools, then exit.")
    parser.add_argument("--json", action="store_true",
                        help="With --list, emit JSON.")
    parser.add_argument("--self-test", action="store_true",
                        help="Build every server without opening a transport.")
    parser.add_argument("--http", action="store_true",
                        help="Serve over streamable HTTP instead of stdio.")
    parser.add_argument("--port", type=int, default=8931,
                        help="Port for --http (default: 8931).")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    # stderr, not stdout: stdout is the stdio transport's channel and a log
    # line written there corrupts the protocol stream.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)-8s %(name)s — %(message)s",
    )

    try:
        if args.self_test:
            return self_test()
        if args.list:
            return list_servers(as_json=args.json)
        if not args.server:
            parser.print_help()
            return 2

        server = build_server(args.server)
        if args.http:
            server.run(transport="streamable-http", port=args.port)
        else:
            server.run(transport="stdio")
        return 0
    except MCPUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
