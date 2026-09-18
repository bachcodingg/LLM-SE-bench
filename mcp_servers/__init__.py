"""
mcp_servers — Model Context Protocol servers over the llm-se-bench harness.

The harness can be driven by a person through ``cli.py``.  These servers
let an agent drive it instead, which is the difference between a tool and
infrastructure.

Three servers, each a thin, typed wrapper over one component:

``jvm-sandbox``
    Compile Java, run JUnit suites, extract CK metrics and detect smells.
    Wraps C2's :class:`~bench.sandbox.docker_sandbox.DockerSandbox` and C3.

``llm-se-bench``
    List and fetch benchmark tasks, evaluate a candidate patch, launch a
    benchmark run under a budget ceiling, read back results.  Wraps C2.

``god-class-tools``
    Detect God Classes, plan a decomposition, score one.  Wraps C3's
    refactoring evaluator — the thesis work, exposed as tools.

Layout
------
``mcp_servers.tools``
    The tool *logic*: plain functions taking and returning Pydantic models
    from :mod:`mcp_servers.models`.  No MCP dependency, so they are
    importable and testable without the SDK installed.

``mcp_servers.servers``
    The MCP *wiring*: registration, descriptions, transport.  Needs the
    SDK.

Nothing in :mod:`mcp_servers.tools` makes a paid API call except
:func:`~mcp_servers.tools.bench.run_benchmark`, which defaults to
``dry_run=True`` and refuses to start above its budget ceiling.
"""

from __future__ import annotations

__all__ = ["SERVER_NAMES", "__version__"]

__version__ = "0.1.0"

#: Server names accepted by ``python -m mcp_servers <name>``.
SERVER_NAMES = ("jvm-sandbox", "llm-se-bench", "god-class-tools")
