"""
tests.test_progress — Unit tests for ProgressTracker and CheckpointManager.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.progress import CheckpointManager, ProgressTracker

# ======================================================================
# ProgressTracker Tests
# ======================================================================

class TestProgressTracker:

    def test_mark_and_check(self):
        tracker = ProgressTracker()
        assert not tracker.is_completed("run_1")
        tracker.mark_completed("run_1")
        assert tracker.is_completed("run_1")

    def test_completion_count(self):
        tracker = ProgressTracker()
        assert tracker.completion_count() == 0
        tracker.mark_completed("a")
        tracker.mark_completed("b")
        tracker.mark_completed("c")
        assert tracker.completion_count() == 3

    def test_duplicate_marks_idempotent(self):
        tracker = ProgressTracker()
        tracker.mark_completed("run_1")
        tracker.mark_completed("run_1")
        assert tracker.completion_count() == 1

    def test_get_completed_keys(self):
        tracker = ProgressTracker()
        tracker.mark_completed("a")
        tracker.mark_completed("b")
        keys = tracker.get_completed_keys()
        assert keys == {"a", "b"}
        # Should be a copy
        keys.add("c")
        assert "c" not in tracker.get_completed_keys()

    def test_progress_summary(self):
        tracker = ProgressTracker()
        tracker.mark_completed("humaneval/gpt-4/p1/run_1")
        tracker.mark_completed("humaneval/gpt-4/p2/run_1")
        tracker.mark_completed("humaneval/claude-3/p1/run_1")
        tracker.mark_completed("mbpp/gpt-4/p1/run_1")

        summary = tracker.get_progress_summary()
        assert summary["total_completed"] == 4
        assert summary["by_dataset_model"]["humaneval"]["gpt-4"] == 2
        assert summary["by_dataset_model"]["humaneval"]["claude-3"] == 1
        assert summary["by_dataset_model"]["mbpp"]["gpt-4"] == 1

    def test_reset(self):
        tracker = ProgressTracker()
        tracker.mark_completed("a")
        tracker.mark_completed("b")
        tracker.reset()
        assert tracker.completion_count() == 0
        assert not tracker.is_completed("a")

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------

    def test_json_save_and_load(self, tmp_path: Path):
        cp = tmp_path / "progress.json"
        tracker = ProgressTracker(checkpoint_path=cp, backend="json")
        tracker.mark_completed("run_a")
        tracker.mark_completed("run_b")
        tracker.save()
        assert cp.exists()

        # Load into new tracker
        tracker2 = ProgressTracker(checkpoint_path=cp, backend="json")
        assert tracker2.is_completed("run_a")
        assert tracker2.is_completed("run_b")
        assert tracker2.completion_count() == 2

    def test_json_auto_save(self, tmp_path: Path):
        cp = tmp_path / "progress.json"
        tracker = ProgressTracker(
            checkpoint_path=cp, backend="json", auto_save_interval=3
        )
        tracker.mark_completed("a")
        tracker.mark_completed("b")
        assert not cp.exists()  # Not yet at interval
        tracker.mark_completed("c")
        assert cp.exists()  # Should auto-save after 3

    def test_json_file_format(self, tmp_path: Path):
        cp = tmp_path / "progress.json"
        tracker = ProgressTracker(checkpoint_path=cp, backend="json")
        tracker.mark_completed("x")
        tracker.save()

        data = json.loads(cp.read_text())
        assert "started_at" in data
        assert "saved_at" in data
        assert "total_completed" in data
        assert data["total_completed"] == 1
        assert "x" in data["completed"]

    def test_json_atomic_write(self, tmp_path: Path):
        """Save should not corrupt the file on partial writes."""
        cp = tmp_path / "progress.json"
        tracker = ProgressTracker(checkpoint_path=cp, backend="json")
        tracker.mark_completed("a")
        tracker.save()
        # File should be valid JSON
        json.loads(cp.read_text())

    def test_reset_deletes_file(self, tmp_path: Path):
        cp = tmp_path / "progress.json"
        tracker = ProgressTracker(checkpoint_path=cp, backend="json")
        tracker.mark_completed("a")
        tracker.save()
        assert cp.exists()
        tracker.reset()
        assert not cp.exists()

    # ------------------------------------------------------------------
    # SQLite persistence
    # ------------------------------------------------------------------

    def test_sqlite_save_and_load(self, tmp_path: Path):
        cp = tmp_path / "progress.db"
        tracker = ProgressTracker(checkpoint_path=cp, backend="sqlite")
        tracker.mark_completed("run_x")
        tracker.mark_completed("run_y")
        tracker.save()

        tracker2 = ProgressTracker(checkpoint_path=cp, backend="sqlite")
        assert tracker2.is_completed("run_x")
        assert tracker2.is_completed("run_y")
        assert tracker2.completion_count() == 2

    def test_sqlite_idempotent_save(self, tmp_path: Path):
        cp = tmp_path / "progress.db"
        tracker = ProgressTracker(checkpoint_path=cp, backend="sqlite")
        tracker.mark_completed("a")
        tracker.save()
        tracker.mark_completed("b")
        tracker.save()  # Should not error on existing rows

        tracker2 = ProgressTracker(checkpoint_path=cp, backend="sqlite")
        assert tracker2.completion_count() == 2

    # ------------------------------------------------------------------
    # In-memory only
    # ------------------------------------------------------------------

    def test_in_memory_no_file(self):
        tracker = ProgressTracker()  # No checkpoint_path
        tracker.mark_completed("x")
        assert tracker.is_completed("x")
        tracker.save()  # Should be a no-op

    def test_load_nonexistent_file(self, tmp_path: Path):
        cp = tmp_path / "nonexistent.json"
        tracker = ProgressTracker(checkpoint_path=cp)
        assert tracker.completion_count() == 0


# ======================================================================
# CheckpointManager Tests
# ======================================================================

class TestCheckpointManager:

    @pytest.fixture
    def manager(self, tmp_path: Path) -> CheckpointManager:
        return CheckpointManager(checkpoints_dir=tmp_path / "checkpoints")

    def test_create_tracker(self, manager: CheckpointManager):
        tracker = manager.create_tracker("run_001")
        assert isinstance(tracker, ProgressTracker)
        assert tracker.checkpoint_path is not None

    def test_create_tracker_with_config(self, manager: CheckpointManager):
        config = {"models": ["gpt-4"], "runs_per_problem": 3}
        tracker = manager.create_tracker("run_002", config=config)
        config_path = manager.checkpoints_dir / "run_002" / "config.json"
        assert config_path.exists()
        saved_config = json.loads(config_path.read_text())
        assert saved_config["models"] == ["gpt-4"]

    def test_list_runs(self, manager: CheckpointManager):
        manager.create_tracker("run_a")
        manager.create_tracker("run_b")
        runs = manager.list_runs()
        names = [r["name"] for r in runs]
        assert "run_a" in names
        assert "run_b" in names

    def test_list_runs_empty(self, manager: CheckpointManager):
        assert manager.list_runs() == []

    def test_resume_tracker(self, manager: CheckpointManager):
        tracker1 = manager.create_tracker("run_resume")
        tracker1.mark_completed("task_1")
        tracker1.save()

        tracker2 = manager.resume_tracker("run_resume")
        assert tracker2 is not None
        assert tracker2.is_completed("task_1")

    def test_resume_nonexistent(self, manager: CheckpointManager):
        result = manager.resume_tracker("nonexistent_run")
        assert result is None

    def test_delete_run(self, manager: CheckpointManager):
        manager.create_tracker("run_to_delete")
        assert manager.delete_run("run_to_delete") is True
        assert manager.resume_tracker("run_to_delete") is None

    def test_delete_nonexistent_run(self, manager: CheckpointManager):
        assert manager.delete_run("nope") is False

    def test_sqlite_backend(self, manager: CheckpointManager):
        tracker = manager.create_tracker("sqlite_run", backend="sqlite")
        tracker.mark_completed("a")
        tracker.save()
        assert (
            manager.checkpoints_dir / "sqlite_run" / "progress.db"
        ).exists()

    def test_run_metadata(self, manager: CheckpointManager):
        manager.create_tracker("meta_run", config={"key": "value"})
        runs = manager.list_runs()
        meta_run = [r for r in runs if r["name"] == "meta_run"][0]
        assert meta_run["config_exists"] is True
