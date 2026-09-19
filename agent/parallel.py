"""
agent.parallel — run episodes concurrently under one budget (M1).

Episodes are independent: different task, different workspace, different
container. The only thing they share is the money, and that is exactly the
thing that makes naive parallelism dangerous — N workers each checking a
per-episode ceiling will happily spend N times the run budget.

So the run-level budget is shared and locked, checked before a worker
starts an episode rather than after it finishes, and the pool stops
dispatching the moment the remaining budget will not cover another episode.

Threads rather than processes: an episode is dominated by waiting on an API
call and on Docker, both of which release the GIL, and threads keep the
shared budget a simple lock instead of an IPC problem.

Docker isolation is per episode already — each ``run_tests`` stages into a
fresh temporary directory and starts its own container — so concurrent
episodes cannot see each other's files.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from agent.trajectory import Trajectory, TrajectoryStore

logger = logging.getLogger(__name__)

__all__ = ["RunBudget", "ParallelRunner", "ParallelResult"]


class RunBudget:
    """A spend ceiling shared by every worker in a run.

    The per-episode ceiling bounds one episode. This bounds the run. Without
    it, eight workers with a €0.50 episode ceiling can spend €4.00 while
    every individual check passes.

    Thread-safe. ``reserve`` and ``settle`` bracket an episode: reserve the
    worst case before starting, settle the actual cost after, and the
    difference goes back to the pool.
    """

    def __init__(self, total_eur: float) -> None:
        if total_eur <= 0:
            raise ValueError("total_eur must be positive.")
        self.total_eur = total_eur
        self._spent = 0.0
        self._reserved = 0.0
        self._lock = threading.Lock()

    @property
    def spent_eur(self) -> float:
        with self._lock:
            return self._spent

    @property
    def available_eur(self) -> float:
        """What is left after spending *and* outstanding reservations."""
        with self._lock:
            return max(0.0, self.total_eur - self._spent - self._reserved)

    def reserve(self, amount_eur: float) -> bool:
        """Hold *amount_eur* for an episode. False when it will not fit.

        Reserving before the work — rather than checking after — is what
        stops two workers each seeing "€0.40 left" and both starting a
        €0.30 episode.
        """
        with self._lock:
            if self._spent + self._reserved + amount_eur > self.total_eur:
                return False
            self._reserved += amount_eur
            return True

    def settle(self, reserved_eur: float, actual_eur: float) -> None:
        """Release the reservation and book what was really spent."""
        with self._lock:
            self._reserved = max(0.0, self._reserved - reserved_eur)
            self._spent += max(0.0, actual_eur)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return {
                "total_eur": round(self.total_eur, 6),
                "spent_eur": round(self._spent, 6),
                "reserved_eur": round(self._reserved, 6),
                "available_eur": round(
                    max(0.0, self.total_eur - self._spent - self._reserved), 6
                ),
            }


@dataclass
class ParallelResult:
    """Outcome of a parallel run."""

    trajectories: list[Trajectory] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    budget: dict[str, float] = field(default_factory=dict)

    @property
    def solved(self) -> int:
        return sum(1 for t in self.trajectories if t.solved)

    def summary(self) -> dict[str, Any]:
        attempted = len(self.trajectories)
        return {
            "attempted": attempted,
            "solved": self.solved,
            "resolve_rate": round(self.solved / attempted, 4) if attempted else 0.0,
            "skipped_for_budget": self.skipped,
            "failed": self.failed,
            "budget": self.budget,
        }


class ParallelRunner:
    """Runs episodes across a worker pool under a shared budget.

    Parameters
    ----------
    workers
        Concurrent episodes. Each holds a Docker container during
        ``run_tests``, so this is bounded by the host, not by the API.
    budget
        Shared run-level ceiling.
    per_episode_reserve_eur
        Reserved before an episode starts. Should be the per-episode
        ceiling: reserving less lets the pool start an episode it cannot
        afford to finish.
    store
        Trajectories are saved as each episode completes, not at the end.
        A run that dies at episode 40 of 50 should not lose 39 results.
    """

    def __init__(
        self,
        workers: int = 4,
        budget: RunBudget | None = None,
        per_episode_reserve_eur: float = 0.5,
        store: TrajectoryStore | None = None,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1.")
        self.workers = workers
        self.budget = budget
        self.per_episode_reserve_eur = per_episode_reserve_eur
        self.store = store
        self._store_lock = threading.Lock()

    def run(
        self,
        task_ids: Iterable[str],
        episode_fn: Callable[[str], Trajectory],
    ) -> ParallelResult:
        """Run *episode_fn* over *task_ids*, concurrently and within budget.

        ``episode_fn`` must be safe to call from several threads at once. It
        gets a task id and returns a trajectory; everything else it needs is
        the caller's to close over.
        """
        result = ParallelResult()
        tasks = list(task_ids)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures: dict[Future[Trajectory], str] = {}

            for task_id in tasks:
                reserved = 0.0
                if self.budget is not None:
                    if not self.budget.reserve(self.per_episode_reserve_eur):
                        logger.warning(
                            "Run budget exhausted; not dispatching %s "
                            "(or any task after it).", task_id,
                        )
                        remaining = tasks[tasks.index(task_id):]
                        result.skipped.extend(remaining)
                        break
                    reserved = self.per_episode_reserve_eur

                futures[pool.submit(self._one, task_id, episode_fn, reserved)] = task_id

            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    trajectory = future.result()
                except Exception as exc:
                    logger.exception("Episode for %s raised", task_id)
                    result.failed[task_id] = f"{type(exc).__name__}: {exc}"
                    continue
                result.trajectories.append(trajectory)

        result.trajectories.sort(key=lambda t: t.task_id)
        result.budget = self.budget.snapshot() if self.budget else {}
        return result

    def _one(
        self,
        task_id: str,
        episode_fn: Callable[[str], Trajectory],
        reserved_eur: float,
    ) -> Trajectory:
        """Run one episode, settle its budget, persist it."""
        actual = 0.0
        try:
            trajectory = episode_fn(task_id)
            actual = trajectory.total_cost_eur
            if self.store is not None:
                # Saving appends to a shared index file, so it is serialised.
                with self._store_lock:
                    self.store.save(trajectory)
            return trajectory
        finally:
            if self.budget is not None and reserved_eur:
                self.budget.settle(reserved_eur, actual)
