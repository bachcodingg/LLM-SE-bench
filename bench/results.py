"""
bench.results — Result collection and pass@k computation.

``ResultCollector`` writes ``EvaluationResult`` objects to structured
JSONL files under ``results/{dataset}/{model}/results.jsonl``.

``PassAtKCalculator`` computes unbiased pass@k estimates following
the formula from Chen et al. (2021) — Evaluating Large Language Models
Trained on Code (Codex paper).
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from contracts import EvaluationResult, Verdict

logger = logging.getLogger(__name__)


class ResultCollector:
    """
    Collects and persists evaluation results to JSONL files.

    Directory structure::

        results/
        └── {dataset}/
            └── {model}/
                ├── results.jsonl
                └── code/
                    └── {problem_id}_run{N}.java

    Parameters
    ----------
    results_dir : Path | str
        Root directory for output files.
    """

    def __init__(self, results_dir: Path | str = "results") -> None:
        self.results_dir = Path(results_dir)
        self._counts: dict[str, int] = defaultdict(int)

    def record(
        self,
        dataset_name: str,
        model_id: str,
        result: EvaluationResult,
    ) -> Path:
        """
        Append an ``EvaluationResult`` to the appropriate JSONL file.

        Parameters
        ----------
        dataset_name : str
            Dataset identifier.
        model_id : str
            Model identifier.
        result : EvaluationResult
            The evaluation result to record.

        Returns
        -------
        Path
            Path to the JSONL file written.
        """
        out_dir = self.results_dir / dataset_name / model_id
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_dir / "results.jsonl"

        record = result.model_dump(mode="json")
        record["recorded_at"] = datetime.utcnow().isoformat()

        with open(jsonl_path, "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

        key = f"{dataset_name}/{model_id}"
        self._counts[key] += 1
        logger.debug(
            "Recorded result #%d for %s/%s → %s",
            self._counts[key],
            dataset_name,
            model_id,
            jsonl_path,
        )
        return jsonl_path

    def load_results(
        self,
        dataset_name: str,
        model_id: str,
    ) -> list[EvaluationResult]:
        """
        Load all results from a JSONL file.

        Parameters
        ----------
        dataset_name : str
            Dataset identifier.
        model_id : str
            Model identifier.

        Returns
        -------
        list[EvaluationResult]
            Parsed evaluation results.
        """
        jsonl_path = self.results_dir / dataset_name / model_id / "results.jsonl"
        if not jsonl_path.exists():
            logger.warning("No results file at %s", jsonl_path)
            return []

        results: list[EvaluationResult] = []
        with open(jsonl_path) as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    data.pop("recorded_at", None)
                    results.append(EvaluationResult(**data))
                except (json.JSONDecodeError, Exception) as exc:
                    logger.warning(
                        "Skipping malformed record at %s:%d: %s",
                        jsonl_path, line_no, exc,
                    )

        logger.info(
            "Loaded %d results from %s/%s",
            len(results),
            dataset_name,
            model_id,
        )
        return results

    def get_summary(
        self,
        dataset_name: str,
        model_id: str,
    ) -> dict[str, Any]:
        """
        Compute summary statistics for a dataset/model combination.

        Returns
        -------
        dict
            Summary with total, passed, failed counts and pass rate.
        """
        results = self.load_results(dataset_name, model_id)
        if not results:
            return {
                "dataset": dataset_name,
                "model": model_id,
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "partial": 0,
                "pass_rate": 0.0,
            }

        verdicts = defaultdict(int)
        for r in results:
            verdicts[r.verdict.value] += 1

        total = len(results)
        return {
            "dataset": dataset_name,
            "model": model_id,
            "total": total,
            "passed": verdicts.get("pass", 0),
            "failed": verdicts.get("fail", 0),
            "errors": verdicts.get("error", 0),
            "partial": verdicts.get("partial", 0),
            "pass_rate": verdicts.get("pass", 0) / total if total > 0 else 0.0,
        }

    def list_datasets(self) -> list[str]:
        """List all dataset directories under results_dir."""
        if not self.results_dir.exists():
            return []
        return [
            d.name for d in sorted(self.results_dir.iterdir())
            if d.is_dir()
        ]

    def list_models(self, dataset_name: str) -> list[str]:
        """List all model directories for a given dataset."""
        ds_dir = self.results_dir / dataset_name
        if not ds_dir.exists():
            return []
        return [
            d.name for d in sorted(ds_dir.iterdir())
            if d.is_dir()
        ]


class PassAtKCalculator:
    """
    Computes unbiased pass@k estimates.

    Uses the formula from Chen et al. (2021):

        pass@k = 1 - C(n-c, k) / C(n, k)

    where:
        n = total number of samples per problem
        c = number of correct samples
        k = k value for pass@k

    This is computed in log-space to avoid numerical overflow for
    large n and k.
    """

    @staticmethod
    def pass_at_k(n: int, c: int, k: int) -> float:
        """
        Compute pass@k for a single problem.

        Parameters
        ----------
        n : int
            Total number of code samples generated for this problem.
        c : int
            Number of samples that pass all tests.
        k : int
            The k value (e.g. 1 for pass@1, 5 for pass@5).

        Returns
        -------
        float
            Estimated pass@k probability in [0, 1].
        """
        if n < k:
            # Not enough samples to compute pass@k
            logger.warning(
                "n=%d < k=%d — returning pass@k based on available samples",
                n, k,
            )
            return 1.0 if c > 0 else 0.0

        if c == 0:
            return 0.0
        if c >= n:
            return 1.0
        if n - c < k:
            return 1.0

        # Compute 1 - C(n-c, k) / C(n, k) in log-space
        # log C(n-c, k) - log C(n, k) = sum of log terms
        log_prod = 0.0
        for i in range(k):
            log_prod += math.log(n - c - i) - math.log(n - i)

        return 1.0 - math.exp(log_prod)

    @classmethod
    def compute_from_results(
        cls,
        results: list[EvaluationResult],
        k_values: list[int] | None = None,
    ) -> dict[str, dict[int, float]]:
        """
        Compute pass@k for each problem from a list of results.

        Parameters
        ----------
        results : list[EvaluationResult]
            All evaluation results (possibly multiple runs per problem).
        k_values : list[int] | None
            k values to compute (default [1, 5, 10]).

        Returns
        -------
        dict[str, dict[int, float]]
            Mapping of ``problem_id`` → ``{k: pass@k_value}``.
        """
        if k_values is None:
            k_values = [1, 5, 10]

        # Group by problem_id
        by_problem: dict[str, list[EvaluationResult]] = defaultdict(list)
        for r in results:
            by_problem[r.problem_id].append(r)

        pass_at_k_results: dict[str, dict[int, float]] = {}

        for pid, problem_results in by_problem.items():
            n = len(problem_results)
            c = sum(1 for r in problem_results if r.verdict == Verdict.PASS)

            pass_at_k_results[pid] = {}
            for k in k_values:
                pass_at_k_results[pid][k] = cls.pass_at_k(n, c, k)

        return pass_at_k_results

    @classmethod
    def compute_aggregate(
        cls,
        results: list[EvaluationResult],
        k_values: list[int] | None = None,
    ) -> dict[int, float]:
        """
        Compute aggregate (mean) pass@k across all problems.

        Parameters
        ----------
        results : list[EvaluationResult]
            All evaluation results.
        k_values : list[int] | None
            k values to compute (default [1, 5, 10]).

        Returns
        -------
        dict[int, float]
            Mapping of ``k`` → mean pass@k across all problems.
        """
        if k_values is None:
            k_values = [1, 5, 10]

        per_problem = cls.compute_from_results(results, k_values)

        if not per_problem:
            return {k: 0.0 for k in k_values}

        aggregates: dict[int, float] = {}
        for k in k_values:
            values = [scores[k] for scores in per_problem.values()]
            aggregates[k] = sum(values) / len(values) if values else 0.0

        return aggregates
