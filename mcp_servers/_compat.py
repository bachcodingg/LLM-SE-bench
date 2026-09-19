"""
mcp_servers._compat — one import point for the MCP SDK.

The SDK renamed its high-level server class between major versions:
``mcp.server.fastmcp.FastMCP`` in 1.x, ``mcp.server.mcpserver.MCPServer``
in 2.x.  The decorator API (``@server.tool()``) and ``server.run(transport)``
are the same in both, so one alias covers everything these servers use.

Importing this module without the SDK installed raises
:class:`MCPUnavailable` with the install command, rather than a bare
``ModuleNotFoundError`` naming a submodule that moved.
"""

from __future__ import annotations

__all__ = ["MCPServer", "SDK_VARIANT", "MCPUnavailable", "require_sdk"]


class MCPUnavailable(ImportError):
    """The MCP SDK is not installed, or is a version this shim cannot use."""


_INSTALL_HINT = (
    "The MCP SDK is not installed. Install the optional extra:\n"
    "    pip install -e \".[mcp]\"\n"
    "The harness and the CLI work without it; only the MCP servers need it."
)

MCPServer = None
SDK_VARIANT = "none"

try:  # SDK 2.x
    from mcp.server.mcpserver import MCPServer as _MCPServer

    MCPServer = _MCPServer
    SDK_VARIANT = "mcp>=2 (MCPServer)"
except ImportError:
    try:  # SDK 1.x
        from mcp.server.fastmcp import FastMCP as _MCPServer

        MCPServer = _MCPServer
        SDK_VARIANT = "mcp<2 (FastMCP)"
    except ImportError:
        MCPServer = None
        SDK_VARIANT = "none"


def require_sdk() -> type:
    """Return the server class, or raise :class:`MCPUnavailable`."""
    if MCPServer is None:
        raise MCPUnavailable(_INSTALL_HINT)
    return MCPServer
