"""
bench.orchestrator — Evaluation orchestration engine.

Coordinates the end-to-end benchmark pipeline:

1. Load problems from a dataset adapter
2. For each problem × model × run:
   a. Format prompt (via dataset adapter)
   b. Send prompt to LLM (via C1 LLMClient)
   c. Write generated code to ``results/{dataset}/{model}/code/``
   d. Execute code in sandbox
   e. Record ``EvaluationResult`` to JSONL

Supports configurable concurrency, retry on transient failures, and
checkpoint/resume via ``ProgressTracker``.
"""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from bench.datasets.base import Dataset
from bench.progress import ProgressTracker
from bench.results import ResultCollector
from bench.sandbox.docker_sandbox import DockerSandbox, SandboxResult
from contracts import (
    EvaluationResult,
    LLMResponse,
    Prompt,
    Severity,
    Verdict,
    VerificationResult,
)

logger = logging.getLogger(__name__)


# ======================================================================
# LLMClient protocol — matches the C1 abstract interface
# ======================================================================

class LLMClient(Protocol):
    """
    Protocol matching the C1 LLMClient abstract interface.

    Any class implementing ``send_prompt(Prompt) → LLMResponse``
    satisfies this protocol.
    """

    def send_prompt(self, prompt: Prompt) -> LLMResponse: ...


# ======================================================================
# Mock LLM client for testing
# ======================================================================

class MockLLMClient:
    """
    Mock implementation of the C1 LLMClient for testing.

    Returns configurable ``LLMResponse`` objects without making real
    API calls.  Supports per-problem custom responses and configurable
    latency simulation.

    Parameters
    ----------
    default_code : str
        Default generated code returned for any problem.
    latency_ms : float
        Simulated latency in milliseconds (default 100).
    model_id : str
        Model identifier used in responses.
    custom_responses : dict[str, str] | None
        Mapping of ``problem_id`` → custom code to return.
    """

    def __init__(
        self,
        default_code: str = "// mock generated code\npublic class Solution {\n    // TODO\n}\n",
        latency_ms: float = 100.0,
        model_id: str = "mock-model",
        custom_responses: dict[str, str] | None = None,
    ) -> None:
        self.default_code = default_code
        self.latency_ms = latency_ms
        self.model_id = model_id
        self.custom_responses = custom_responses or {}
        self._call_count = 0
        self._call_log: list[dict[str, Any]] = []

    def send_prompt(self, prompt: Prompt) -> LLMResponse:
        """
        Return a mock ``LLMResponse``.

        If ``prompt.problem_id`` has a custom response configured,
        that code is returned; otherwise ``default_code`` is used.
        """
        self._call_count += 1
        start = time.monotonic()

        # Simulate latency
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

        code = self.custom_responses.get(prompt.problem_id, self.default_code)
        elapsed_ms = (time.monotonic() - start) * 1000

        response = LLMResponse(
            response_id=f"mock-resp-{self._call_count:04d}",
            prompt_id=prompt.prompt_id,
            model_id=self.model_id,
            raw_text=f"```java\n{code}\n```",
            extracted_code=code,
            finish_reason="stop",
            prompt_tokens=len(prompt.user_message.split()),
            completion_tokens=len(code.split()),
            latency_ms=elapsed_ms,
        )

        self._call_log.append({
            "call_number": self._call_count,
            "prompt_id": prompt.prompt_id,
            "problem_id": prompt.problem_id,
            "model_id": prompt.model_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

        return response

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def call_log(self) -> list[dict[str, Any]]:
        return list(self._call_log)

    def reset(self) -> None:
        """Reset call counter and log."""
        self._call_count = 0
        self._call_log.clear()


# ======================================================================
# Run configuration
# ======================================================================

@dataclass
class RunConfig:
    """Configuration for a benchmark run."""

    models: list[str] = field(default_factory=lambda: ["mock-model"])
    runs_per_problem: int = 3
    temperature: float = 0.0
    max_tokens: int = 16384
    max_concurrency: int = 3
    max_retries: int = 2
    retry_delay_seconds: float = 5.0
    results_dir: Path = field(default_factory=lambda: Path("results"))
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": self.models,
            "runs_per_problem": self.runs_per_problem,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_concurrency": self.max_concurrency,
            "max_retries": self.max_retries,
            "results_dir": str(self.results_dir),
            "dry_run": self.dry_run,
        }


# ======================================================================
# Orchestrator
# ======================================================================

class BenchmarkOrchestrator:
    """
    Orchestrates benchmark evaluation runs across datasets and models.

    Parameters
    ----------
    dataset : Dataset
        The dataset adapter to evaluate against.
    llm_clients : dict[str, LLMClient]
        Mapping of ``model_id`` → LLM client instance.
    sandbox : DockerSandbox
        The sandbox for code compilation/testing.
    config : RunConfig
        Run configuration parameters.
    progress : ProgressTracker | None
        Optional progress tracker for checkpoint/resume.
    """

    def __init__(
        self,
        dataset: Dataset,
        llm_clients: dict[str, LLMClient],
        sandbox: DockerSandbox,
        config: RunConfig | None = None,
        progress: ProgressTracker | None = None,
    ) -> None:
        self.dataset = dataset
        self.llm_clients = llm_clients
        self.sandbox = sandbox
        self.config = config or RunConfig()
        self.progress = progress or ProgressTracker()
        self.result_collector = ResultCollector(self.config.results_dir)
        self._run_stats: dict[str, Any] = {
            "started_at": None,
            "completed_at": None,
            "total_evaluations": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
        }

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def run_all(self) -> dict[str, Any]:
        """
        Run the full benchmark: all problems × all models × N runs.

        Returns
        -------
        dict
            Summary statistics for the entire run.
        """
        self._run_stats["started_at"] = datetime.utcnow().isoformat()
        problem_ids = self.dataset.list_problem_ids()
        models = self.config.models

        logger.info(
            "Starting benchmark: %d problems × %d models × %d runs = %d evaluations",
            len(problem_ids),
            len(models),
            self.config.runs_per_problem,
            len(problem_ids) * len(models) * self.config.runs_per_problem,
        )

        for model_id in models:
            if model_id not in self.llm_clients:
                logger.error("No client configured for model '%s'", model_id)
                continue
            self._run_model(model_id, problem_ids)

        self._run_stats["completed_at"] = datetime.utcnow().isoformat()
        logger.info(
            "Benchmark complete: %d evaluations (%d passed, %d failed, %d skipped)",
            self._run_stats["total_evaluations"],
            self._run_stats["successful"],
            self._run_stats["failed"],
            self._run_stats["skipped"],
        )
        return dict(self._run_stats)

    def run_single(
        self,
        problem_id: str,
        model_id: str,
        run_id: int = 1,
    ) -> EvaluationResult:
        """
        Run a single evaluation: one problem, one model, one attempt.

        Parameters
        ----------
        problem_id : str
            Problem to evaluate.
        model_id : str
            Model to use.
        run_id : int
            Run number (1-based).

        Returns
        -------
        EvaluationResult
            The evaluation result contract object.
        """
        run_key = f"{self.dataset.name}/{model_id}/{problem_id}/run_{run_id}"

        # Check if already completed (resume support)
        if self.progress.is_completed(run_key):
            logger.debug("Skipping completed run: %s", run_key)
            self._run_stats["skipped"] += 1
            return self._create_skipped_result(problem_id, model_id, run_id)

        client = self.llm_clients.get(model_id)
        if client is None:
            raise ValueError(f"No LLM client for model '{model_id}'")

        # 1. Format prompt
        prompt_text = self.dataset.format_prompt(problem_id)
        prompt = Prompt(
            prompt_id=f"prompt-{uuid.uuid4().hex[:8]}",
            problem_id=problem_id,
            model_id=model_id,
            user_message=prompt_text,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # 2. Send to LLM (with retries)
        response = self._send_with_retry(client, prompt, model_id)
        if response is None:
            return self._create_error_result(
                problem_id, model_id, run_id,
                "LLM request failed after retries",
            )

        # 3. Write generated code
        code = response.extracted_code or response.raw_text
        self._write_generated_code(problem_id, model_id, run_id, code)

        # 4. Run in sandbox
        suite = self.dataset.get_test_suite(problem_id)
        junit_code = ""
        try:
            junit_code = self.dataset.get_junit_code(problem_id)  # type: ignore[attr-defined]
        except (AttributeError, KeyError):
            logger.debug("No JUnit code available for %s", problem_id)

        sandbox_result = self.sandbox.run(
            source_code=code,
            junit_code=junit_code,
            problem_id=problem_id,
        )

        # 5. Build EvaluationResult
        eval_result = self._build_evaluation_result(
            problem_id=problem_id,
            model_id=model_id,
            run_id=run_id,
            response=response,
            suite=suite,
            sandbox_result=sandbox_result,
        )

        # 6. Record
        self.result_collector.record(
            dataset_name=self.dataset.name,
            model_id=model_id,
            result=eval_result,
        )
        self.progress.mark_completed(run_key)

        self._run_stats["total_evaluations"] += 1
        if eval_result.verdict == Verdict.PASS:
            self._run_stats["successful"] += 1
        else:
            self._run_stats["failed"] += 1

        return eval_result

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _run_model(self, model_id: str, problem_ids: list[str]) -> None:
        """Run all problems for a single model."""
        if self.config.max_concurrency <= 1:
            # Sequential
            for pid in problem_ids:
                for run_id in range(1, self.config.runs_per_problem + 1):
                    try:
                        self.run_single(pid, model_id, run_id)
                    except Exception as exc:
                        logger.error(
                            "Error evaluating %s/%s/run_%d: %s",
                            model_id, pid, run_id, exc,
                        )
                        self._run_stats["failed"] += 1
        else:
            # Parallel
            tasks = [
                (pid, model_id, run_id)
                for pid in problem_ids
                for run_id in range(1, self.config.runs_per_problem + 1)
            ]
            with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as pool:
                futures = {
                    pool.submit(self.run_single, pid, mid, rid): (pid, mid, rid)
                    for pid, mid, rid in tasks
                }
                for future in as_completed(futures):
                    pid, mid, rid = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error(
                            "Error evaluating %s/%s/run_%d: %s",
                            mid, pid, rid, exc,
                        )
                        self._run_stats["failed"] += 1

    def _send_with_retry(
        self, client: LLMClient, prompt: Prompt, model_id: str
    ) -> LLMResponse | None:
        """Send prompt to LLM with retry logic."""
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = client.send_prompt(prompt)
                return response
            except Exception as exc:
                logger.warning(
                    "LLM request failed (attempt %d/%d) for %s/%s: %s",
                    attempt,
                    self.config.max_retries,
                    model_id,
                    prompt.problem_id,
                    exc,
                )
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_seconds)
        return None

    def _write_generated_code(
        self,
        problem_id: str,
        model_id: str,
        run_id: int,
        code: str,
    ) -> Path:
        """Write generated code to results directory."""
        code_dir = (
            self.config.results_dir
            / self.dataset.name
            / model_id
            / "code"
        )
        code_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{problem_id}_run{run_id}.java"
        code_path = code_dir / filename
        code_path.write_text(code, encoding="utf-8")
        logger.debug("Wrote generated code → %s", code_path)
        return code_path

    def _build_evaluation_result(
        self,
        problem_id: str,
        model_id: str,
        run_id: int,
        response: LLMResponse,
        suite: Any,
        sandbox_result: SandboxResult,
    ) -> EvaluationResult:
        """Construct an EvaluationResult from sandbox output."""
        verification_results: list[VerificationResult] = []

        if not sandbox_result.compiled:
            # All tests fail on compilation error
            for tc in suite.cases:
                verification_results.append(VerificationResult(
                    test_id=tc.test_id,
                    passed=False,
                    error_message=sandbox_result.error_message or "Compilation failed",
                    severity=Severity.ERROR,
                ))
            verdict = Verdict.ERROR
        elif sandbox_result.all_passed:
            for tc in suite.cases:
                verification_results.append(VerificationResult(
                    test_id=tc.test_id,
                    passed=True,
                    execution_time_ms=sandbox_result.test_time_ms / max(1, len(suite.cases)),
                    severity=Severity.INFO,
                ))
            verdict = Verdict.PASS
        elif sandbox_result.tests_passed > 0:
            # Partial pass — mark first N as passed, rest as failed
            for i, tc in enumerate(suite.cases):
                passed = i < sandbox_result.tests_passed
                verification_results.append(VerificationResult(
                    test_id=tc.test_id,
                    passed=passed,
                    severity=Severity.INFO if passed else Severity.ERROR,
                ))
            verdict = Verdict.PARTIAL
        else:
            for tc in suite.cases:
                verification_results.append(VerificationResult(
                    test_id=tc.test_id,
                    passed=False,
                    error_message=sandbox_result.error_message or "Test failed",
                    severity=Severity.ERROR,
                ))
            verdict = Verdict.FAIL

        tests_total = len(suite.cases)
        tests_passed = sum(1 for vr in verification_results if vr.passed)
        weighted_score = (
            sum(
                tc.weight
                for tc, vr in zip(suite.cases, verification_results)
                if vr.passed
            )
            / suite.total_weight
            if suite.total_weight > 0
            else 0.0
        )

        runtime_errors: list[str] = []
        if sandbox_result.compile_stderr:
            runtime_errors.append(sandbox_result.compile_stderr[:500])
        if sandbox_result.test_stderr:
            runtime_errors.append(sandbox_result.test_stderr[:500])

        return EvaluationResult(
            evaluation_id=f"eval-{uuid.uuid4().hex[:8]}",
            response_id=response.response_id,
            problem_id=problem_id,
            suite_id=suite.suite_id,
            verdict=verdict,
            tests_total=tests_total,
            tests_passed=tests_passed,
            weighted_score=weighted_score,
            results=verification_results,
            compile_success=sandbox_result.compiled,
            runtime_errors=runtime_errors,
            latency_ms=response.latency_ms,
        )

    def _create_error_result(
        self,
        problem_id: str,
        model_id: str,
        run_id: int,
        error_message: str,
    ) -> EvaluationResult:
        """Create an error EvaluationResult when the pipeline fails."""
        suite = self.dataset.get_test_suite(problem_id)
        return EvaluationResult(
            evaluation_id=f"eval-err-{uuid.uuid4().hex[:8]}",
            response_id="error",
            problem_id=problem_id,
            suite_id=suite.suite_id,
            verdict=Verdict.ERROR,
            tests_total=len(suite.cases),
            tests_passed=0,
            weighted_score=0.0,
            compile_success=False,
            runtime_errors=[error_message],
        )

    def _create_skipped_result(
        self,
        problem_id: str,
        model_id: str,
        run_id: int,
    ) -> EvaluationResult:
        """Create a placeholder result for skipped (already completed) runs."""
        suite = self.dataset.get_test_suite(problem_id)
        return EvaluationResult(
            evaluation_id=f"eval-skip-{uuid.uuid4().hex[:8]}",
            response_id="skipped",
            problem_id=problem_id,
            suite_id=suite.suite_id,
            verdict=Verdict.PASS,
            tests_total=0,
            tests_passed=0,
            weighted_score=0.0,
        )
