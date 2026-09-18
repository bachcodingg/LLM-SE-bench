"""
mcp_servers.registry — task lookup and run bookkeeping for the tool layer.

The CLI knows which dataset it is working on because the user said so. An
agent calling ``get_task("D4J_Lang_1")`` does not, so this builds the
reverse index once and caches it.

It also owns the run registry: ``run_benchmark`` writes a record here and
``get_run_results`` reads it back, which is the only state the MCP servers
keep between calls.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DATASET_CATEGORY",
    "TaskLocation",
    "TaskRegistry",
    "get_registry",
    "RunRegistry",
    "default_runs_dir",
]

#: What each dataset asks a model to do.  Used to filter in ``list_tasks``
#: and to pick the right prompt shape.
DATASET_CATEGORY: dict[str, str] = {
    "humaneval-java": "codegen",
    "mbpp-java": "codegen",
    "defects4j": "bugfix",
    "godclass": "refactor",
}


@dataclass(frozen=True)
class TaskLocation:
    """Where a task lives: which adapter holds it, under which dataset."""

    dataset_name: str
    category: str
    adapter: Any  # bench.datasets.base.Dataset; untyped to avoid the import


class TaskRegistry:
    """Index of every task across every dataset, built once and cached.

    Loading all four adapters costs a few hundred milliseconds and parses
    every JSONL file, so an MCP server does it on first use and keeps it.

    Parameters
    ----------
    data_dir : Path | str
        Root directory holding the dataset files.
    """

    def __init__(self, data_dir: Path | str = "data") -> None:
        self.data_dir = Path(data_dir)
        self._by_task_id: dict[str, TaskLocation] = {}
        self._adapters: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load every adapter and index its task ids. Idempotent."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return

            from bench.datasets.defects4j import Defects4JDataset
            from bench.datasets.godclass import GodClassDataset
            from bench.datasets.humaneval import HumanEvalDataset
            from bench.datasets.mbpp import MBPPDataset

            for cls in (HumanEvalDataset, MBPPDataset, Defects4JDataset, GodClassDataset):
                try:
                    adapter = cls(data_dir=self.data_dir)
                    task_ids = adapter.list_problem_ids()
                except Exception as exc:
                    # One broken dataset must not make the others unreachable.
                    logger.error("Could not load %s: %s", cls.__name__, exc)
                    continue

                name = adapter.name
                category = DATASET_CATEGORY.get(name, "unknown")
                self._adapters[name] = adapter
                for task_id in task_ids:
                    if task_id in self._by_task_id:
                        logger.warning(
                            "Duplicate task id %s in %s and %s; keeping the first",
                            task_id, self._by_task_id[task_id].dataset_name, name,
                        )
                        continue
                    self._by_task_id[task_id] = TaskLocation(name, category, adapter)

            self._loaded = True
            logger.debug("Task registry: %d tasks across %d datasets",
                         len(self._by_task_id), len(self._adapters))

    def locate(self, task_id: str) -> TaskLocation | None:
        """Return where *task_id* lives, or None when it is unknown."""
        self._ensure_loaded()
        return self._by_task_id.get(task_id)

    def task_ids(self) -> list[str]:
        """Every known task id, in dataset load order."""
        self._ensure_loaded()
        return list(self._by_task_id)

    def adapters(self) -> dict[str, Any]:
        """``{dataset_name: adapter}`` for every dataset that loaded."""
        self._ensure_loaded()
        return dict(self._adapters)

    def dataset_names(self) -> list[str]:
        """Names of the datasets that loaded successfully."""
        self._ensure_loaded()
        return list(self._adapters)


_REGISTRY: TaskRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_registry(data_dir: Path | str = "data") -> TaskRegistry:
    """Return the process-wide registry, creating it on first call.

    A second call with a different *data_dir* rebuilds it, which is what
    the tests want and what a server reconfigured mid-process would need.
    """
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None or _REGISTRY.data_dir != Path(data_dir):
            _REGISTRY = TaskRegistry(data_dir)
        return _REGISTRY


def default_runs_dir() -> Path:
    """Directory holding run records written by ``run_benchmark``."""
    return Path("results") / "runs"


class RunRegistry:
    """Run records on disk, one JSON file per run.

    Deliberately not a database: a run record is small, written once, and
    read by a human as often as by a tool.

    Parameters
    ----------
    runs_dir : Path | str | None
        Where records live.  Defaults to ``results/runs``.
    """

    def __init__(self, runs_dir: Path | str | None = None) -> None:
        self.runs_dir = Path(runs_dir) if runs_dir else default_runs_dir()

    def _path(self, run_id: str) -> Path:
        """Path for *run_id*, rejecting anything that could escape the dir."""
        if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
            raise ValueError(f"invalid run_id: {run_id!r}")
        return self.runs_dir / f"{run_id}.json"

    def save(self, run_id: str, record: dict[str, Any]) -> Path:
        """Write *record* for *run_id* and return the file path."""
        path = self._path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        return path

    def load(self, run_id: str) -> dict[str, Any] | None:
        """Read the record for *run_id*, or None when there is none."""
        try:
            path = self._path(run_id)
        except ValueError:
            return None
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Could not read run record %s: %s", run_id, exc)
            return None

    def list_run_ids(self) -> list[str]:
        """Every stored run id, newest first."""
        if not self.runs_dir.exists():
            return []
        records = sorted(
            self.runs_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [p.stem for p in records]
