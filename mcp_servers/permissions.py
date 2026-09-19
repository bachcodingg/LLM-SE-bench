"""
mcp_servers.permissions — access levels, tool versioning, cost attribution (M2).

**Permission modes.** Run the same task at read-only, patch-only and full
access, and measure what access actually buys. The usual assumption is that
more access is better; it is an assumption, it is cheap to test, and the
answer is a result. A read-only agent that scores nearly as well as a
full-access one is telling you the task did not need the access — and an
agent that scores *worse* with more access is telling you something more
interesting than that.

**Tool versioning.** A tool's schema and its description are both inputs to
the measurement. Changing either invalidates comparison with results
produced under the old one, in exactly the way changing a prompt template
does. Versions and a registry hash make the break detectable instead of
silent.

**Cost attribution.** Tool output is not free: it enters the conversation
and is re-sent on every subsequent turn. A tool returning 3,000 tokens at
step 2 of a 20-step episode costs those tokens nineteen more times. Nothing
usually attributes this, which makes verbose tools look free and makes
truncation look like a nicety rather than the cost control it is.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

logger = logging.getLogger(__name__)

__all__ = [
    "PermissionMode",
    "Capability",
    "ToolPermissions",
    "ToolRegistration",
    "ToolRegistry",
    "ToolCostAttribution",
    "attribute_tool_costs",
]


class Capability(str, Enum):
    """What a tool needs to be allowed to do."""

    READ = "read"           # inspect files, list, search
    ANALYSE = "analyse"     # parse and compute metrics; still read-only
    PATCH = "patch"         # modify files inside the workspace
    EXECUTE = "execute"     # compile and run code in the sandbox
    SPEND = "spend"         # make a paid API call
    SHELL = "shell"         # arbitrary commands; nothing here grants it


class PermissionMode(str, Enum):
    """How much a given run lets an agent do.

    Deliberately coarse. Three levels that differ in kind are more useful
    for an experiment than twelve that differ in degree, because the
    comparison has to be legible in a results table.
    """

    READ_ONLY = "read_only"
    PATCH_ONLY = "patch_only"
    FULL = "full"
    #: Everything, including paid calls and shell access. Never a default.
    UNRESTRICTED = "unrestricted"

    @property
    def capabilities(self) -> frozenset[Capability]:
        return {
            PermissionMode.READ_ONLY: frozenset({Capability.READ, Capability.ANALYSE}),
            PermissionMode.PATCH_ONLY: frozenset({
                Capability.READ, Capability.ANALYSE, Capability.PATCH,
            }),
            PermissionMode.FULL: frozenset({
                Capability.READ, Capability.ANALYSE, Capability.PATCH,
                Capability.EXECUTE,
            }),
            PermissionMode.UNRESTRICTED: frozenset(Capability),
        }[self]

    @property
    def description(self) -> str:
        return {
            PermissionMode.READ_ONLY: (
                "Inspect and analyse. Cannot change or run anything. An agent "
                "here can diagnose but not fix."
            ),
            PermissionMode.PATCH_ONLY: (
                "Inspect, analyse and edit, but not execute. The agent must "
                "reason about whether its patch is correct rather than "
                "finding out. Isolates reasoning from trial and error."
            ),
            PermissionMode.FULL: (
                "The default for benchmark runs: read, edit, compile, test. "
                "No paid calls and no shell."
            ),
            PermissionMode.UNRESTRICTED: (
                "Everything, including spending money and running arbitrary "
                "commands. Never a default; opt in per run."
            ),
        }[self]


@dataclass
class ToolPermissions:
    """Decides whether a tool call is allowed in this run."""

    mode: PermissionMode = PermissionMode.FULL
    #: Tools denied regardless of mode, by name.
    denied: frozenset[str] = field(default_factory=frozenset)
    #: Calls that were refused, for the trajectory. An agent that spent five
    #: steps trying to run tests in read-only mode is a finding about the
    #: prompt, not about the model.
    refusals: list[dict[str, str]] = field(default_factory=list)

    def allows(self, capabilities: Iterable[Capability]) -> bool:
        return set(capabilities) <= self.mode.capabilities

    def check(self, tool_name: str, capabilities: Iterable[Capability]) -> str | None:
        """Return a refusal message, or None when the call is allowed.

        The message is written for the model: it says what is not permitted
        and what to do instead, because an agent told only "denied" will
        retry.
        """
        if tool_name in self.denied:
            self.refusals.append({"tool": tool_name, "reason": "explicitly denied"})
            return (
                f"{tool_name} is not available in this run. Work with the "
                f"tools you have."
            )

        missing = set(capabilities) - self.mode.capabilities
        if not missing:
            return None

        self.refusals.append({
            "tool": tool_name,
            "reason": f"needs {sorted(c.value for c in missing)}",
        })
        advice = {
            Capability.PATCH: "You cannot modify files in this run. Report "
                              "what you would change and why.",
            Capability.EXECUTE: "You cannot compile or run anything in this "
                                "run. Reason about correctness by reading.",
            Capability.SPEND: "You cannot make paid calls in this run.",
            Capability.SHELL: "Shell access is not available.",
        }
        hints = " ".join(advice.get(capability, "") for capability in sorted(missing, key=str))
        return (
            f"{tool_name} requires "
            f"{', '.join(sorted(c.value for c in missing))}, which this run "
            f"({self.mode.value}) does not grant. {hints}".strip()
        )

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "description": self.mode.description,
            "capabilities": sorted(c.value for c in self.mode.capabilities),
            "denied_tools": sorted(self.denied),
            "refusals": len(self.refusals),
            "refused_tools": sorted({r["tool"] for r in self.refusals}),
        }


# ──────────────────────────────────────────────────────────────────────
# Tool registry
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolRegistration:
    """One tool, versioned, with the capabilities it needs."""

    name: str
    version: str
    capabilities: frozenset[Capability]
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    #: Set when a tool is superseded. Kept registered so old trajectories
    #: remain interpretable — a result produced by a tool that no longer
    #: exists is uninterpretable, not merely stale.
    deprecated_by: str = ""

    def fingerprint(self) -> str:
        """Hash of everything a model sees or a caller depends on.

        Covers the description as well as the schema, because the
        description *is* the prompt: rewording it changes behaviour as
        surely as changing a parameter does.
        """
        payload = json.dumps(
            {
                "name": self.name,
                "version": self.version,
                "description": self.description,
                "parameters": self.parameters,
                "capabilities": sorted(c.value for c in self.capabilities),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ToolRegistry:
    """Versioned tools, with a hash over the whole set.

    The registry hash goes in the run manifest next to the prompt template
    hashes. Two runs with different registry hashes were not given the same
    tools, and comparing their numbers is comparing two experiments.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolRegistration] = {}

    def register(self, registration: ToolRegistration) -> ToolRegistration:
        """Add a tool. Re-registering the same name replaces it, loudly."""
        existing = self._tools.get(registration.name)
        if existing and existing.fingerprint() != registration.fingerprint():
            logger.warning(
                "Tool %r re-registered with a different fingerprint "
                "(%s -> %s). Results from before this change are not "
                "comparable with results after it.",
                registration.name,
                existing.fingerprint()[:12],
                registration.fingerprint()[:12],
            )
        self._tools[registration.name] = registration
        return registration

    def get(self, name: str) -> ToolRegistration | None:
        return self._tools.get(name)

    def available(self, permissions: ToolPermissions) -> list[ToolRegistration]:
        """Tools this permission mode allows, deprecated ones excluded."""
        return sorted(
            (
                tool for tool in self._tools.values()
                if not tool.deprecated_by
                and tool.name not in permissions.denied
                and permissions.allows(tool.capabilities)
            ),
            key=lambda tool: tool.name,
        )

    def fingerprint(self) -> str:
        """Hash over every registered tool. Belongs in the run manifest."""
        combined = "".join(
            self._tools[name].fingerprint() for name in sorted(self._tools)
        )
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def manifest(self) -> dict[str, Any]:
        """What went in the run manifest, so a run can be identified later."""
        return {
            "registry_fingerprint": self.fingerprint(),
            "tool_count": len(self._tools),
            "tools": {
                name: {
                    "version": tool.version,
                    "fingerprint": tool.fingerprint()[:16],
                    "capabilities": sorted(c.value for c in tool.capabilities),
                    "deprecated_by": tool.deprecated_by or None,
                }
                for name, tool in sorted(self._tools.items())
            },
        }


def default_registry() -> ToolRegistry:
    """The agent toolset, registered with capabilities and versions."""
    from agent.tools import TOOL_SCHEMAS

    capabilities = {
        "read_file": frozenset({Capability.READ}),
        "list_dir": frozenset({Capability.READ}),
        "grep": frozenset({Capability.READ}),
        "apply_patch": frozenset({Capability.PATCH}),
        "run_tests": frozenset({Capability.EXECUTE}),
    }

    registry = ToolRegistry()
    for schema in TOOL_SCHEMAS:
        registry.register(ToolRegistration(
            name=schema.name,
            version="1.0.0",
            capabilities=capabilities.get(schema.name, frozenset({Capability.READ})),
            description=schema.description,
            parameters=schema.parameters,
        ))
    return registry


# ──────────────────────────────────────────────────────────────────────
# Tool call cost attribution
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ToolCostAttribution:
    """What one tool's output cost over the whole episode.

    ``carried_tokens`` is the number that matters and the one nobody
    reports: a result produced at step 2 of a 20-step episode sits in the
    conversation for 18 more turns and is charged for on each of them.
    """

    tool_name: str
    calls: int = 0
    output_tokens: int = 0
    carried_tokens: int = 0
    estimated_cost_eur: float = 0.0
    truncated_calls: int = 0

    @property
    def mean_output_tokens(self) -> float:
        return self.output_tokens / self.calls if self.calls else 0.0

    @property
    def amplification(self) -> float:
        """Carried tokens per token produced.

        1.0 means the output was charged once. 15.0 means it was charged
        fifteen times, which is what an early verbose call actually costs.
        """
        return self.carried_tokens / self.output_tokens if self.output_tokens else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool_name,
            "calls": self.calls,
            "output_tokens": self.output_tokens,
            "mean_output_tokens": round(self.mean_output_tokens, 1),
            "carried_tokens": self.carried_tokens,
            "amplification": round(self.amplification, 2),
            "estimated_cost_eur": round(self.estimated_cost_eur, 6),
            "truncated_calls": self.truncated_calls,
        }


def attribute_tool_costs(
    trajectory: Any,
    cost_per_input_token_eur: float = 3e-6 / 1.08,
    cache_read_multiplier: float = 0.1,
) -> dict[str, Any]:
    """Attribute an episode's input-token cost to the tools that caused it.

    A tool result produced at step *i* is re-sent on every turn after it, so
    its cost is its size multiplied by the turns that follow. Later turns
    are usually cache reads, so those are priced at the discounted rate.

    The result answers "which tool is eating my budget", which is not the
    same question as "which tool did I call most" and frequently has a
    different answer.
    """
    from agent.context import estimate_tokens

    steps = getattr(trajectory, "steps", [])
    if not steps:
        return {"tools": [], "total_carried_tokens": 0}

    by_tool: dict[str, ToolCostAttribution] = {}
    total_steps = len(steps)

    for index, step in enumerate(steps):
        name = step.tool_name or "(no tool call)"
        attribution = by_tool.setdefault(name, ToolCostAttribution(tool_name=name))
        attribution.calls += 1

        # The preview is a sample of the real result; scale it back up using
        # the ratio the truncation recorded, when there is one.
        produced = estimate_tokens(step.result_preview)
        attribution.output_tokens += produced

        # Charged once fresh, then on every subsequent turn as a cache read.
        remaining_turns = total_steps - index - 1
        carried = produced * (1 + remaining_turns * cache_read_multiplier)
        attribution.carried_tokens += int(carried)
        attribution.estimated_cost_eur += carried * cost_per_input_token_eur

    rows = sorted(
        (attribution.to_dict() for attribution in by_tool.values()),
        key=lambda row: -row["carried_tokens"],
    )
    return {
        "tools": rows,
        "total_carried_tokens": sum(row["carried_tokens"] for row in rows),
        "total_estimated_cost_eur": round(
            sum(row["estimated_cost_eur"] for row in rows), 6
        ),
        "dominant_tool": rows[0]["tool"] if rows else None,
        "caveat": (
            "Estimated from the stored result previews, not the full tool "
            "output, so absolute sizes are floors. The relative ordering — "
            "which tool dominates — is the usable part."
        ),
    }
