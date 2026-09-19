"""
Tests for mcp_servers.permissions (M2).

Permission modes exist so "more access is better" can be tested rather than
assumed. The tests pin the two things that make that experiment valid: a
mode grants exactly what it says, and a refusal tells the agent what to do
instead — an agent told only "denied" retries until its budget is gone, and
the resulting score measures the refusal message.
"""

from __future__ import annotations

import pytest

from agent.tools import ToolOutcome
from agent.workspace import Workspace
from mcp_servers.permissions import (
    Capability,
    PermissionMode,
    ToolCostAttribution,
    ToolPermissions,
    ToolRegistration,
    ToolRegistry,
    attribute_tool_costs,
    default_registry,
)


@pytest.fixture
def workspace() -> Workspace:
    """A small workspace whose test file is read-only."""
    return Workspace(
        files={
            "Calculator.java": "public class Calculator { int add(int a, int b) { return a - b; } }",
            "CalculatorTest.java": "public class CalculatorTest { @Test void t() {} }",
        },
        read_only={"CalculatorTest.java"},
    )


class _CountingRunner:
    """Records whether run_tests was reached, which is what permission tests check."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _workspace: Workspace) -> ToolOutcome:
        self.calls += 1
        return ToolOutcome(content="2/2 tests passed.", tests_passed=2, tests_total=2)


@pytest.fixture
def test_runner() -> _CountingRunner:
    return _CountingRunner()


class TestPermissionModes:
    def test_read_only_grants_no_write_or_execute(self):
        capabilities = PermissionMode.READ_ONLY.capabilities
        assert Capability.READ in capabilities
        assert Capability.PATCH not in capabilities
        assert Capability.EXECUTE not in capabilities

    def test_patch_only_allows_editing_but_not_running(self):
        capabilities = PermissionMode.PATCH_ONLY.capabilities
        assert Capability.PATCH in capabilities
        assert Capability.EXECUTE not in capabilities

    def test_full_allows_execution_but_not_spending_or_shell(self):
        capabilities = PermissionMode.FULL.capabilities
        assert Capability.EXECUTE in capabilities
        assert Capability.SPEND not in capabilities
        assert Capability.SHELL not in capabilities

    def test_unrestricted_grants_everything(self):
        assert PermissionMode.UNRESTRICTED.capabilities == frozenset(Capability)

    def test_modes_are_ordered_by_inclusion(self):
        assert (
            PermissionMode.READ_ONLY.capabilities
            < PermissionMode.PATCH_ONLY.capabilities
            < PermissionMode.FULL.capabilities
            < PermissionMode.UNRESTRICTED.capabilities
        )

    def test_every_mode_describes_itself(self):
        for mode in PermissionMode:
            assert len(mode.description) > 40


class TestToolPermissions:
    def test_allows_a_permitted_tool(self):
        permissions = ToolPermissions(mode=PermissionMode.FULL)
        assert permissions.check("run_tests", [Capability.EXECUTE]) is None

    def test_refuses_a_tool_the_mode_does_not_grant(self):
        permissions = ToolPermissions(mode=PermissionMode.READ_ONLY)
        refusal = permissions.check("apply_patch", [Capability.PATCH])
        assert refusal is not None
        assert "patch" in refusal

    def test_the_refusal_says_what_to_do_instead(self):
        """An agent told only 'denied' retries until its budget is gone."""
        permissions = ToolPermissions(mode=PermissionMode.READ_ONLY)
        refusal = permissions.check("apply_patch", [Capability.PATCH])
        assert "Report what you would change" in refusal

    def test_patch_only_refuses_execution_with_advice(self):
        permissions = ToolPermissions(mode=PermissionMode.PATCH_ONLY)
        refusal = permissions.check("run_tests", [Capability.EXECUTE])
        assert "Reason about correctness by reading" in refusal

    def test_explicitly_denied_tools_are_refused_at_any_mode(self):
        permissions = ToolPermissions(
            mode=PermissionMode.UNRESTRICTED, denied=frozenset({"run_tests"})
        )
        assert permissions.check("run_tests", [Capability.EXECUTE]) is not None

    def test_refusals_are_recorded(self):
        """Five steps spent fighting the harness is a finding about the prompt."""
        permissions = ToolPermissions(mode=PermissionMode.READ_ONLY)
        for _ in range(3):
            permissions.check("apply_patch", [Capability.PATCH])
        assert permissions.summary()["refusals"] == 3
        assert permissions.summary()["refused_tools"] == ["apply_patch"]


class TestToolsetIntegration:
    def test_read_only_blocks_apply_patch(self, workspace, test_runner):
        from agent.tools import AgentToolset, ToolContext

        toolset = AgentToolset(ToolContext(
            workspace=workspace, test_runner=test_runner,
            permissions=ToolPermissions(mode=PermissionMode.READ_ONLY),
        ))
        outcome = toolset.dispatch("apply_patch", {"path": "A.java", "content": "x"})
        assert outcome.is_error
        assert "read_only" in outcome.content
        assert "A.java" not in workspace

    def test_read_only_still_allows_reading(self, workspace, test_runner):
        from agent.tools import AgentToolset, ToolContext

        toolset = AgentToolset(ToolContext(
            workspace=workspace, test_runner=test_runner,
            permissions=ToolPermissions(mode=PermissionMode.READ_ONLY),
        ))
        assert toolset.dispatch("read_file", {"path": "Calculator.java"}).is_error is False

    def test_patch_only_blocks_run_tests(self, workspace, test_runner):
        from agent.tools import AgentToolset, ToolContext

        toolset = AgentToolset(ToolContext(
            workspace=workspace, test_runner=test_runner,
            permissions=ToolPermissions(mode=PermissionMode.PATCH_ONLY),
        ))
        assert toolset.dispatch("run_tests", {}).is_error is True
        assert test_runner.calls == 0

    def test_allowed_names_reflect_the_mode(self, workspace, test_runner):
        from agent.tools import AgentToolset, ToolContext

        toolset = AgentToolset(ToolContext(
            workspace=workspace, test_runner=test_runner,
            permissions=ToolPermissions(mode=PermissionMode.READ_ONLY),
        ))
        assert set(toolset.allowed_names()) == {"read_file", "list_dir", "grep"}

    def test_no_permissions_means_full_access(self, workspace, test_runner):
        from agent.tools import AgentToolset, ToolContext

        toolset = AgentToolset(ToolContext(workspace=workspace, test_runner=test_runner))
        assert len(toolset.allowed_names()) == 5


class TestToolRegistry:
    def _registration(self, name: str = "read_file", version: str = "1.0.0") -> ToolRegistration:
        return ToolRegistration(
            name=name, version=version,
            capabilities=frozenset({Capability.READ}),
            description="Read a file.",
            parameters={"type": "object", "properties": {}},
        )

    def test_fingerprint_is_stable(self):
        assert self._registration().fingerprint() == self._registration().fingerprint()

    def test_a_changed_description_changes_the_fingerprint(self):
        """The description is the prompt: rewording it changes behaviour."""
        first = self._registration()
        second = ToolRegistration(
            name=first.name, version=first.version,
            capabilities=first.capabilities,
            description="Read a file. Now with more words.",
            parameters=first.parameters,
        )
        assert first.fingerprint() != second.fingerprint()

    def test_a_changed_schema_changes_the_fingerprint(self):
        first = self._registration()
        second = ToolRegistration(
            name=first.name, version=first.version,
            capabilities=first.capabilities, description=first.description,
            parameters={"type": "object", "properties": {"extra": {"type": "string"}}},
        )
        assert first.fingerprint() != second.fingerprint()

    def test_registry_fingerprint_covers_every_tool(self):
        registry = ToolRegistry()
        registry.register(self._registration("a"))
        before = registry.fingerprint()
        registry.register(self._registration("b"))
        assert registry.fingerprint() != before

    def test_re_registering_with_a_different_shape_warns(self, caplog):
        registry = ToolRegistry()
        registry.register(self._registration())
        with caplog.at_level("WARNING"):
            registry.register(ToolRegistration(
                name="read_file", version="2.0.0",
                capabilities=frozenset({Capability.READ}),
                description="Completely different.",
                parameters={},
            ))
        assert "not comparable" in caplog.text

    def test_available_filters_by_permission(self):
        registry = ToolRegistry()
        registry.register(self._registration("read_file"))
        registry.register(ToolRegistration(
            name="apply_patch", version="1.0.0",
            capabilities=frozenset({Capability.PATCH}),
            description="Write a file.",
        ))
        available = registry.available(ToolPermissions(mode=PermissionMode.READ_ONLY))
        assert [tool.name for tool in available] == ["read_file"]

    def test_deprecated_tools_stay_registered_but_unavailable(self):
        """A result from a tool that no longer exists is uninterpretable."""
        registry = ToolRegistry()
        registry.register(ToolRegistration(
            name="old_tool", version="1.0.0",
            capabilities=frozenset({Capability.READ}),
            description="Superseded.", deprecated_by="new_tool",
        ))
        assert registry.get("old_tool") is not None
        assert registry.available(ToolPermissions()) == []

    def test_default_registry_covers_the_agent_toolset(self):
        registry = default_registry()
        assert set(registry.manifest()["tools"]) == {
            "read_file", "list_dir", "grep", "apply_patch", "run_tests"
        }

    def test_manifest_carries_the_fingerprint(self):
        manifest = default_registry().manifest()
        assert len(manifest["registry_fingerprint"]) == 64
        assert manifest["tool_count"] == 5


class TestCostAttribution:
    def test_amplification_reflects_being_re_sent(self):
        """A result produced at step 2 of 20 is charged for 18 more turns."""
        attribution = ToolCostAttribution(
            tool_name="run_tests", calls=1, output_tokens=1000, carried_tokens=3000
        )
        assert attribution.amplification == pytest.approx(3.0)

    def test_attributes_across_a_trajectory(self):
        from agent.trajectory import Trajectory, TrajectoryStep

        trajectory = Trajectory(
            episode_id="ep",
            steps=[
                TrajectoryStep(step_index=1, tool_name="run_tests",
                               result_preview="x" * 2000),
                *[
                    TrajectoryStep(step_index=i, tool_name="list_dir",
                                   result_preview="y" * 50)
                    for i in range(2, 12)
                ],
            ],
        )
        result = attribute_tool_costs(trajectory)
        assert result["dominant_tool"] == "run_tests"
        assert result["total_carried_tokens"] > 0

    def test_an_early_verbose_call_costs_more_than_a_late_one(self):
        from agent.trajectory import Trajectory, TrajectoryStep

        def carried(position: int) -> int:
            steps = [
                TrajectoryStep(step_index=i, tool_name="filler", result_preview="z")
                for i in range(1, 11)
            ]
            steps[position] = TrajectoryStep(
                step_index=position + 1, tool_name="big", result_preview="x" * 4000
            )
            result = attribute_tool_costs(Trajectory(episode_id="e", steps=steps))
            return next(row for row in result["tools"] if row["tool"] == "big")["carried_tokens"]

        assert carried(0) > carried(8)

    def test_empty_trajectory(self):
        from agent.trajectory import Trajectory

        assert attribute_tool_costs(Trajectory(episode_id="e"))["tools"] == []
