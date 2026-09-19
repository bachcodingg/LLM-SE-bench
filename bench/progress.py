"""
bench.progress — Checkpoint/resume and progress tracking.

``ProgressTracker`` maintains a set of completed run keys, enabling
the orchestrator to skip already-completed evaluations after a crash
or interruption.

``CheckpointManager`` handles the persistence of checkpoint files
to disk, supporting both JSON and SQLite backends.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Tracks which evaluation runs have been completed.

    Each run is identified by a unique key string of the form
    ``{dataset}/{model}/{problem_id}/run_{N}``.

    Parameters
    ----------
    checkpoint_path : Path | str | None
        Path to the checkpoint file.  When None, tracking is in-memory
        only (no persistence across process restarts).
    backend : str
        Storage backend: ``"json"`` or ``"sqlite"`` (default ``"json"``).
    auto_save_interval : int
        Number of completions between auto-saves (default 10).
    """

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        backend: str = "json",
        auto_save_interval: int = 10,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.backend = backend
        self.auto_save_interval = auto_save_interval
        self._completed: set[str] = set()
        self._timestamps: dict[str, str] = {}
        self._unsaved_count: int = 0
        self._started_at: str = datetime.utcnow().isoformat()

        if self.checkpoint_path and self.checkpoint_path.exists():
            self._load_checkpoint()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def mark_completed(self, run_key: str) -> None:
        """
        Mark a run as completed.

        Parameters
        ----------
        run_key : str
            Unique run identifier (e.g.
            ``"humaneval-java/gpt-4o/HumanEval_0/run_1"``).
        """
        self._completed.add(run_key)
        self._timestamps[run_key] = datetime.utcnow().isoformat()
        self._unsaved_count += 1

        if (
            self.checkpoint_path
            and self._unsaved_count >= self.auto_save_interval
        ):
            self.save()

    def is_completed(self, run_key: str) -> bool:
        """Check whether a run has already been completed."""
        return run_key in self._completed

    def completion_count(self) -> int:
        """Return the number of completed runs."""
        return len(self._completed)

    def get_completed_keys(self) -> set[str]:
        """Return the full set of completed run keys."""
        return set(self._completed)

    def get_progress_summary(self) -> dict[str, Any]:
        """
        Return a summary of progress by dataset and model.

        Returns
        -------
        dict
            Nested summary: ``{dataset: {model: count}}``.
        """
        summary: dict[str, dict[str, int]] = {}
        for key in self._completed:
            parts = key.split("/")
            if len(parts) >= 2:
                dataset = parts[0]
                model = parts[1]
                if dataset not in summary:
                    summary[dataset] = {}
                summary[dataset][model] = summary[dataset].get(model, 0) + 1
        return {
            "started_at": self._started_at,
            "total_completed": len(self._completed),
            "by_dataset_model": summary,
        }

    def reset(self) -> None:
        """Clear all progress (useful for re-runs)."""
        self._completed.clear()
        self._timestamps.clear()
        self._unsaved_count = 0
        self._started_at = datetime.utcnow().isoformat()
        if self.checkpoint_path and self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        logger.info("Progress tracker reset")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save current progress to the checkpoint file."""
        if not self.checkpoint_path:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        if self.backend == "sqlite":
            self._save_sqlite()
        else:
            self._save_json()

        self._unsaved_count = 0
        logger.debug(
            "Saved checkpoint (%d completed) → %s",
            len(self._completed),
            self.checkpoint_path,
        )

    def _load_checkpoint(self) -> None:
        """Load progress from existing checkpoint file."""
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            return

        try:
            if self.backend == "sqlite":
                self._load_sqlite()
            else:
                self._load_json()
            logger.info(
                "Resumed from checkpoint: %d completed runs",
                len(self._completed),
            )
        except Exception as exc:
            logger.warning("Failed to load checkpoint: %s", exc)

    def _save_json(self) -> None:
        """Write checkpoint as JSON."""
        data = {
            "started_at": self._started_at,
            "saved_at": datetime.utcnow().isoformat(),
            "total_completed": len(self._completed),
            "completed": {k: self._timestamps.get(k, "") for k in sorted(self._completed)},
        }
        # Atomic write
        tmp_path = self.checkpoint_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        tmp_path.replace(self.checkpoint_path)

    def _load_json(self) -> None:
        """Read checkpoint from JSON."""
        data = json.loads(self.checkpoint_path.read_text())
        completed = data.get("completed", {})
        if isinstance(completed, dict):
            self._completed = set(completed.keys())
            self._timestamps = completed
        elif isinstance(completed, list):
            self._completed = set(completed)
        self._started_at = data.get("started_at", self._started_at)

    def _save_sqlite(self) -> None:
        """Write checkpoint to SQLite."""
        conn = sqlite3.connect(str(self.checkpoint_path))
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS progress "
                "(run_key TEXT PRIMARY KEY, completed_at TEXT)"
            )
            conn.executemany(
                "INSERT OR REPLACE INTO progress (run_key, completed_at) "
                "VALUES (?, ?)",
                [
                    (k, self._timestamps.get(k, ""))
                    for k in self._completed
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def _load_sqlite(self) -> None:
        """Read checkpoint from SQLite."""
        conn = sqlite3.connect(str(self.checkpoint_path))
        try:
            rows = conn.execute(
                "SELECT run_key, completed_at FROM progress"
            ).fetchall()
            for key, ts in rows:
                self._completed.add(key)
                self._timestamps[key] = ts
        finally:
            conn.close()


class CheckpointManager:
    """
    High-level checkpoint manager for multi-dataset benchmark runs.

    Wraps ``ProgressTracker`` with additional conveniences:
    - Creates per-run checkpoint directories
    - Manages run metadata (config snapshot, timing)
    - Supports listing and resuming past runs

    Parameters
    ----------
    checkpoints_dir : Path | str
        Root directory for checkpoint storage.
    """

    def __init__(self, checkpoints_dir: Path | str = "checkpoints") -> None:
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def create_tracker(
        self,
        run_name: str,
        config: dict[str, Any] | None = None,
        backend: str = "json",
    ) -> ProgressTracker:
        """
        Create a new ``ProgressTracker`` for a named run.

        Parameters
        ----------
        run_name : str
            Unique name for this benchmark run.
        config : dict | None
            Run configuration to snapshot alongside the checkpoint.
        backend : str
            Storage backend (``"json"`` or ``"sqlite"``).

        Returns
        -------
        ProgressTracker
            Tracker with persistence enabled.
        """
        run_dir = self.checkpoints_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config snapshot
        if config:
            config_path = run_dir / "config.json"
            config_path.write_text(json.dumps(config, indent=2, default=str))

        ext = ".db" if backend == "sqlite" else ".json"
        checkpoint_path = run_dir / f"progress{ext}"

        return ProgressTracker(
            checkpoint_path=checkpoint_path,
            backend=backend,
        )

    def list_runs(self) -> list[dict[str, Any]]:
        """
        List all checkpoint runs.

        Returns
        -------
        list[dict]
            Each entry has ``name``, ``created_at``, ``config_exists``,
            and ``checkpoint_exists`` keys.
        """
        runs = []
        if not self.checkpoints_dir.exists():
            return runs

        for d in sorted(self.checkpoints_dir.iterdir()):
            if not d.is_dir():
                continue
            config_path = d / "config.json"
            checkpoint_json = d / "progress.json"
            checkpoint_db = d / "progress.db"
            runs.append({
                "name": d.name,
                "created_at": datetime.fromtimestamp(d.stat().st_ctime).isoformat(),
                "config_exists": config_path.exists(),
                "checkpoint_exists": checkpoint_json.exists() or checkpoint_db.exists(),
            })

        return runs

    def resume_tracker(
        self, run_name: str, backend: str = "json"
    ) -> ProgressTracker | None:
        """
        Resume a ``ProgressTracker`` from an existing checkpoint.

        Returns None if the checkpoint doesn't exist.
        """
        run_dir = self.checkpoints_dir / run_name
        if not run_dir.exists():
            logger.warning("No checkpoint directory for run '%s'", run_name)
            return None

        ext = ".db" if backend == "sqlite" else ".json"
        checkpoint_path = run_dir / f"progress{ext}"

        if not checkpoint_path.exists():
            logger.warning("No checkpoint file at %s", checkpoint_path)
            return None

        return ProgressTracker(
            checkpoint_path=checkpoint_path,
            backend=backend,
        )

    def delete_run(self, run_name: str) -> bool:
        """Delete a checkpoint run directory."""
        import shutil
        run_dir = self.checkpoints_dir / run_name
        if run_dir.exists():
            shutil.rmtree(run_dir)
            logger.info("Deleted checkpoint run: %s", run_name)
            return True
        return False
