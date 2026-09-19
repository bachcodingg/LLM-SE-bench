"""
bench.datasets.base — Abstract base class for all benchmark dataset adapters.

Every dataset adapter (HumanEval, MBPP, Defects4J, God Class) inherits from
``Dataset`` and implements the four core methods.  This ensures that the
``BenchmarkOrchestrator`` can treat every dataset uniformly.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from contracts import (
    Problem,
    Severity,
    TestSuite,
    VerificationResult,
)

logger = logging.getLogger(__name__)


class Dataset(ABC):
    """
    Abstract interface that every benchmark dataset adapter must implement.

    Parameters
    ----------
    name : str
        Human-readable dataset name (e.g. ``"humaneval-java"``).
    data_dir : Path
        Root directory where raw dataset files live.
    language : str
        Target programming language (default ``"java"``).
    """

    def __init__(
        self,
        name: str,
        data_dir: Path | str,
        language: str = "java",
    ) -> None:
        self.name = name
        self.data_dir = Path(data_dir)
        self.language = language
        self._problems: dict[str, Problem] = {}
        self._test_suites: dict[str, TestSuite] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Core abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_problems(self) -> list[Problem]:
        """
        Load all problems from the dataset files.

        Returns
        -------
        list[Problem]
            Ordered list of ``Problem`` contract objects.
        """

    @abstractmethod
    def get_test_suite(self, problem_id: str) -> TestSuite:
        """
        Return the ``TestSuite`` for a given problem.

        Parameters
        ----------
        problem_id : str
            The unique problem identifier.

        Returns
        -------
        TestSuite
            Contract object containing all ``TestCase`` entries.

        Raises
        ------
        KeyError
            If ``problem_id`` does not exist in this dataset.
        """

    @abstractmethod
    def format_prompt(self, problem_id: str) -> str:
        """
        Build the prompt text that will be sent to the LLM for this problem.

        Parameters
        ----------
        problem_id : str
            The unique problem identifier.

        Returns
        -------
        str
            Fully rendered prompt string.
        """

    @abstractmethod
    def verify_solution(
        self,
        problem_id: str,
        generated_code: str,
    ) -> list[VerificationResult]:
        """
        Check generated code against the problem's test suite.

        This method may delegate to the ``DockerSandbox`` for actual
        compilation and execution, or it may perform in-process verification
        for lightweight checks.

        Parameters
        ----------
        problem_id : str
            The unique problem identifier.
        generated_code : str
            Source code produced by the LLM.

        Returns
        -------
        list[VerificationResult]
            One entry per test case.
        """

    # ------------------------------------------------------------------
    # Convenience helpers shared by all adapters
    # ------------------------------------------------------------------

    def get_problem(self, problem_id: str) -> Problem:
        """Retrieve a single ``Problem`` by ID (loads lazily)."""
        self._ensure_loaded()
        if problem_id not in self._problems:
            raise KeyError(
                f"Problem '{problem_id}' not found in dataset '{self.name}'. "
                f"Available: {list(self._problems.keys())[:10]}..."
            )
        return self._problems[problem_id]

    def list_problem_ids(self) -> list[str]:
        """Return all problem IDs in load order."""
        self._ensure_loaded()
        return list(self._problems.keys())

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._problems)

    def __contains__(self, problem_id: str) -> bool:
        self._ensure_loaded()
        return problem_id in self._problems

    # ------------------------------------------------------------------
    # Validation utilities
    # ------------------------------------------------------------------

    def validate_dataset(self) -> dict[str, Any]:
        """
        Run basic integrity checks on every loaded problem and its tests.

        Returns
        -------
        dict
            Summary with keys ``total``, ``valid``, ``errors``.
        """
        self._ensure_loaded()
        errors: list[str] = []
        for pid, prob in self._problems.items():
            if not prob.description.strip():
                errors.append(f"{pid}: empty description")
            try:
                suite = self.get_test_suite(pid)
                if not suite.cases:
                    errors.append(f"{pid}: test suite has no cases")
            except KeyError:
                errors.append(f"{pid}: no test suite registered")

        summary = {
            "dataset": self.name,
            "total": len(self._problems),
            "valid": len(self._problems) - len(errors),
            "errors": errors,
        }
        if errors:
            logger.warning(
                "Dataset '%s' validation found %d issues", self.name, len(errors)
            )
        else:
            logger.info(
                "Dataset '%s' validation passed (%d problems)",
                self.name,
                len(self._problems),
            )
        return summary

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def export_metadata(self, out_path: Path | str) -> Path:
        """
        Write a JSON manifest of all problems (without solutions).

        Parameters
        ----------
        out_path : Path | str
            Destination file path.

        Returns
        -------
        Path
            The written file.
        """
        self._ensure_loaded()
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for prob in self._problems.values():
            rec = prob.model_dump()
            rec.pop("reference_solution", None)
            records.append(rec)
        out.write_text(json.dumps(records, indent=2, default=str))
        logger.info("Exported %d problem metadata records → %s", len(records), out)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load problems on first access (lazy initialisation)."""
        if not self._loaded:
            problems = self.load_problems()
            for p in problems:
                self._problems[p.problem_id] = p
            self._loaded = True
            logger.debug(
                "Loaded %d problems for dataset '%s'", len(self._problems), self.name
            )

    def _register_test_suite(self, suite: TestSuite) -> None:
        """Store a ``TestSuite`` in the internal lookup table."""
        self._test_suites[suite.problem_id] = suite

    def _make_verification_result(
        self,
        test_id: str,
        passed: bool,
        actual_output: Any = None,
        error_message: str = "",
        execution_time_ms: float = 0.0,
    ) -> VerificationResult:
        """Factory for creating ``VerificationResult`` contract objects."""
        severity = Severity.INFO if passed else Severity.ERROR
        return VerificationResult(
            test_id=test_id,
            passed=passed,
            actual_output=actual_output,
            error_message=error_message,
            execution_time_ms=execution_time_ms,
            severity=severity,
        )
