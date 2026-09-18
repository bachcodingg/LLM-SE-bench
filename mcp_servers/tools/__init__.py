"""
mcp_servers.tools — tool logic, independent of the MCP SDK.

Each function here takes primitives or Pydantic models and returns a model
from :mod:`mcp_servers.models`.  None of them import ``mcp``, so the whole
tool layer is importable and testable without the SDK installed — which is
also what lets the test suite cover it in CI on a machine with neither the
SDK nor Docker.

The MCP wiring in :mod:`mcp_servers.servers` is a thin registration layer
over these.
"""

from __future__ import annotations
