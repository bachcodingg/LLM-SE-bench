"""
mcp_servers.servers — MCP wiring for the three tool groups.

Each module here exposes ``build()``, which returns a configured server with
its tools registered.  The tool bodies are one-line delegations to
:mod:`mcp_servers.tools`; everything interesting is there, and everything
an agent reads — names, descriptions, schemas — is here.

Tool descriptions are prompts.  They are written for a model: preconditions,
units, and what each failure mode looks like in the response.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["BUILDERS", "build_server"]


def _jvm_sandbox():
    from mcp_servers.servers.jvm_sandbox import build

    return build


def _bench():
    from mcp_servers.servers.bench import build

    return build


def _god_class():
    from mcp_servers.servers.god_class import build

    return build


#: ``{server name: () -> build function}``.  Lazy so that importing this
#: package does not require the SDK.
BUILDERS: dict[str, Callable[[], Callable[..., Any]]] = {
    "jvm-sandbox": _jvm_sandbox,
    "llm-se-bench": _bench,
    "god-class-tools": _god_class,
}


def build_server(name: str, **kwargs: Any) -> Any:
    """Build the named server. Raises KeyError for an unknown name."""
    if name not in BUILDERS:
        raise KeyError(f"unknown server {name!r}; expected one of {sorted(BUILDERS)}")
    return BUILDERS[name]()(**kwargs)
