"""
agent.trajectory — one row per step, plus storage and replay.

Outcome metrics say what happened. Process metrics say why, and they can
only be computed from a record made while the episode ran. The guide this
project follows is blunt about it: design this table carefully now and the
failure taxonomy, the localisation metrics and the trajectory diff all get
much easier later. So the schema is fixed here and written once.

One :class:`TrajectoryStep` per tool call, carrying:

===========================  ===================================================
``step_index``               Position in the episode, from 1.
``thought_text``             Prose the model emitted with the call.
``tool_name`` / ``tool_args``  What it asked for.
``tool_result_hash``         SHA-256 of the result, not the result.
``tokens_in`` / ``tokens_out`` / ``cache_read_tokens``  Priced separately.
``cost_eur`` / ``latency_ms``  What the step cost.
``files_touched``            Paths this step wrote.
``tests_passing_after``      Test state after the step, or None.
===========================  ===================================================

``tool_result_hash`` rather than the result itself: tool output is large and
often repeated, and the hash makes a loop — five identical calls — visible
at a glance without storing the payload five times. ``result_preview`` keeps
the first few hundred characters so a human reading the trajectory is not
reduced to reading hashes.

Replay re-runs scoring over a recorded episode **without calling any API**.
Most harnesses lack it, and without it every change to a scoring rule costs
another full run's worth of money to evaluate.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

from agent.termination import StopCondition

__all__ = ["TrajectoryStep", "Trajectory", "TrajectoryStore", "replay_scoring"]

#: Characters of tool output kept inline for a human reader.
PREVIEW_CHARS = 400


class TrajectoryStep(BaseModel):
    """One step: one tool call and its result."""

    step_index: int = Field(description="Position in the episode, from 1.")
    timestamp: str = Field(default="", description="ISO 8601 UTC.")

    thought_text: str = Field(
        default="",
        description="Prose the model emitted alongside the call. Usually its "
                    "reasoning, and the thing that makes a trajectory readable.",
    )
    tool_name: str = Field(default="")
    tool_args: dict[str, Any] = Field(default_factory=dict)
    tool_result_hash: str = Field(
        default="", description="SHA-256 of the result content."
    )
    result_preview: str = Field(
        default="", description=f"First {PREVIEW_CHARS} characters of the result."
    )
    tool_error: bool = Field(default=False, description="The tool reported a failure.")
    malformed_call: bool = Field(
        default=False,
        description="The provider's arguments would not parse as JSON. Models "
                    "do emit invalid JSON and the rate is worth measuring.",
    )

    tokens_in: int = Field(default=0, description="Uncached input tokens.")
    tokens_out: int = Field(default=0)
    cache_read_tokens: int = Field(
        default=0,
        description="Input tokens served from the provider's prompt cache. "
                    "Priced differently from tokens_in; folding them together "
                    "makes a cost comparison quietly wrong.",
    )
    cost_eur: float = Field(default=0.0, description="Cost of the call behind this step.")
    latency_ms: float = Field(default=0.0)

    files_touched: list[str] = Field(
        default_factory=list, description="Paths written during this step."
    )
    tests_passing_after: int | None = Field(
        default=None, description="Tests passing after the step; None if not run."
    )
    tests_total_after: int | None = Field(default=None)

    @property
    def is_edit(self) -> bool:
        return bool(self.files_touched)

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out + self.cache_read_tokens


class Trajectory(BaseModel):
    """A complete episode: what was attempted, every step, how it ended."""

    episode_id: str = Field(description="Unique identifier for this episode.")
    task_id: str = Field(default="")
    dataset: str = Field(default="")
    model_id: str = Field(default="")
    scaffold: str = Field(
        default="react",
        description="Which agent scaffold ran. Separates 'the model is "
                    "better' from 'my loop is better'.",
    )

    started_at: str = Field(default="")
    finished_at: str = Field(default="")

    steps: list[TrajectoryStep] = Field(default_factory=list)

    stop_condition: str = Field(default="", description="A StopCondition value.")
    solved: bool = Field(
        default=False,
        description="Target tests pass and nothing previously passing broke.",
    )
    regressed: bool = Field(
        default=False,
        description="Fewer tests pass than before the agent started. A "
                    "'solution' that regresses is not one.",
    )
    tests_executed: bool = Field(
        default=True,
        description="False when the sandbox fell back to a structural check "
                    "instead of compiling and running. Such an episode can "
                    "never be `solved` and its numbers are not a benchmark "
                    "score — exclude it from any results table.",
    )

    total_cost_eur: float = Field(default=0.0)
    total_tokens: int = Field(default=0)
    wall_clock_seconds: float = Field(default=0.0)

    baseline_passed: int | None = Field(default=None)
    baseline_total: int | None = Field(default=None)
    final_passed: int | None = Field(default=None)
    final_total: int | None = Field(default=None)

    files_changed: dict[str, str] = Field(
        default_factory=dict,
        description="{path: created|modified|deleted} against the start state.",
    )
    edit_churn: int = Field(
        default=0, description="Lines written and then written away again."
    )
    final_workspace: dict[str, str] = Field(
        default_factory=dict,
        description="Every file as the episode left it. This is what makes "
                    "replay possible: scoring can be redone from here with no "
                    "API call.",
    )
    error: str = Field(default="")

    # ── derived process metrics ───────────────────────────────────────

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def steps_to_first_edit(self) -> int | None:
        """How long the agent explored before changing anything.

        None when it never edited — which is itself a finding, usually a
        localisation failure.
        """
        for step in self.steps:
            if step.is_edit:
                return step.step_index
        return None

    @property
    def tool_call_distribution(self) -> dict[str, int]:
        """How many times each tool was called."""
        counts: dict[str, int] = {}
        for step in self.steps:
            counts[step.tool_name] = counts.get(step.tool_name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    @property
    def repeated_calls(self) -> int:
        """Calls identical to an earlier one: same tool, same arguments.

        The signature of an agent going in circles. Counts repeats, so three
        identical calls score 2.
        """
        seen: set[str] = set()
        repeats = 0
        for step in self.steps:
            signature = f"{step.tool_name}:{json.dumps(step.tool_args, sort_keys=True, default=str)}"
            if signature in seen:
                repeats += 1
            seen.add(signature)
        return repeats

    @property
    def malformed_call_rate(self) -> float:
        """Fraction of steps whose tool arguments would not parse."""
        return (
            sum(1 for step in self.steps if step.malformed_call) / len(self.steps)
            if self.steps else 0.0
        )

    @property
    def tool_error_rate(self) -> float:
        """Fraction of steps whose tool reported a failure."""
        return (
            sum(1 for step in self.steps if step.tool_error) / len(self.steps)
            if self.steps else 0.0
        )

    @property
    def cache_hit_rate(self) -> float:
        """Share of input tokens served from the provider's prompt cache.

        Directly a cost number: cached input is discounted everywhere it is
        offered, and a long agent conversation is mostly a repeat of itself.
        """
        cached = sum(step.cache_read_tokens for step in self.steps)
        fresh = sum(step.tokens_in for step in self.steps)
        return cached / (cached + fresh) if (cached + fresh) else 0.0

    def cost_per_step(self) -> float:
        return self.total_cost_eur / len(self.steps) if self.steps else 0.0

    def summary(self) -> dict[str, Any]:
        """A flat row, for a results table."""
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "dataset": self.dataset,
            "model_id": self.model_id,
            "scaffold": self.scaffold,
            "solved": self.solved,
            "regressed": self.regressed,
            "tests_executed": self.tests_executed,
            "stop_condition": self.stop_condition,
            "steps": self.num_steps,
            "steps_to_first_edit": self.steps_to_first_edit,
            "repeated_calls": self.repeated_calls,
            "tool_error_rate": round(self.tool_error_rate, 4),
            "malformed_call_rate": round(self.malformed_call_rate, 4),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "edit_churn": self.edit_churn,
            "files_changed": len(self.files_changed),
            "total_tokens": self.total_tokens,
            "total_cost_eur": round(self.total_cost_eur, 6),
            "wall_clock_seconds": round(self.wall_clock_seconds, 2),
            "final_passed": self.final_passed,
            "final_total": self.final_total,
        }


class TrajectoryStore:
    """Trajectories on disk: one JSON file per episode, one JSONL index.

    Parameters
    ----------
    root
        Directory for the episodes. Defaults to ``results/trajectories``.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root else Path("results") / "trajectories"

    def _path(self, episode_id: str) -> Path:
        if not episode_id or "/" in episode_id or "\\" in episode_id or ".." in episode_id:
            raise ValueError(f"invalid episode_id: {episode_id!r}")
        return self.root / f"{episode_id}.json"

    def save(self, trajectory: Trajectory) -> Path:
        """Write *trajectory* and append its summary to the index."""
        path = self._path(trajectory.episode_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            trajectory.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

        index = self.root / "index.jsonl"
        with index.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trajectory.summary(), default=str) + "\n")
        return path

    def load(self, episode_id: str) -> Trajectory | None:
        """Read one trajectory back, or None when it is not there."""
        try:
            path = self._path(episode_id)
        except ValueError:
            return None
        if not path.exists():
            return None
        return Trajectory.model_validate_json(path.read_text(encoding="utf-8"))

    def load_all(self) -> Iterator[Trajectory]:
        """Every stored trajectory, oldest first."""
        if not self.root.exists():
            return
        for path in sorted(self.root.glob("*.json")):
            try:
                yield Trajectory.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:  # a corrupt file must not hide the rest
                continue

    def episode_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(path.stem for path in self.root.glob("*.json"))


def replay_scoring(
    trajectory: Trajectory,
    score: Any,
) -> dict[str, Any]:
    """Re-score a recorded episode without calling any API.

    *score* is called with the final workspace — ``{path: content}`` — and
    returns whatever the new scoring rule produces.

    This is why :attr:`Trajectory.final_workspace` is stored. Changing a
    scoring rule should cost a CPU second, not another full run's API
    spend, and a harness without replay silently makes every scoring
    question expensive enough not to ask.

    Raises:
        ValueError: when the trajectory carries no workspace, which means
            it predates this field and cannot be replayed.
    """
    if not trajectory.final_workspace:
        raise ValueError(
            f"Episode {trajectory.episode_id} stored no final workspace and "
            f"cannot be replayed. Re-run it, or score it from files_changed."
        )
    return {
        "episode_id": trajectory.episode_id,
        "task_id": trajectory.task_id,
        "model_id": trajectory.model_id,
        "original_solved": trajectory.solved,
        "replayed_at": datetime.now(timezone.utc).isoformat(),
        "score": score(dict(trajectory.final_workspace)),
    }


def stop_condition_of(trajectory: Trajectory) -> StopCondition:
    """The stored stop condition as an enum, or ERROR when unrecognised."""
    try:
        return StopCondition(trajectory.stop_condition)
    except ValueError:
        return StopCondition.ERROR
