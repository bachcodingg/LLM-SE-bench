"""
tests.test_orchestrator — Unit tests for BenchmarkOrchestrator and MockLLMClient.

Uses MockLLMClient and dry-run DockerSandbox so no external services are needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.datasets.defects4j import Defects4JDataset
from bench.datasets.godclass import GodClassDataset
from bench.datasets.humaneval import HumanEvalDataset
from bench.datasets.mbpp import MBPPDataset
from bench.orchestrator import (
    BenchmarkOrchestrator,
    MockLLMClient,
    RunConfig,
)
from bench.progress import ProgressTracker
from bench.sandbox.docker_sandbox import DockerSandbox, SandboxResult
from contracts import (
    EvaluationResult,
    LLMResponse,
    Prompt,
    Verdict,
)

# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def tmp_results(tmp_path: Path) -> Path:
    return tmp_path / "results"


@pytest.fixture
def mock_client() -> MockLLMClient:
    return MockLLMClient(
        default_code=(
            "import java.util.List;\n\n"
            "public class Solution {\n"
            "    public static boolean hasCloseElements(List<Double> numbers, double threshold) {\n"
            "        for (int i = 0; i < numbers.size(); i++) {\n"
            "            for (int j = i + 1; j < numbers.size(); j++) {\n"
            "                if (Math.abs(numbers.get(i) - numbers.get(j)) < threshold)\n"
            "                    return true;\n"
            "            }\n"
            "        }\n"
            "        return false;\n"
            "    }\n"
            "}\n"
        ),
        latency_ms=0,
        model_id="test-model",
    )


@pytest.fixture
def sandbox(tmp_path: Path) -> DockerSandbox:
    return DockerSandbox(
        work_dir=tmp_path / "sandbox",
        dry_run=True,
    )


@pytest.fixture
def humaneval(tmp_path: Path) -> HumanEvalDataset:
    return HumanEvalDataset(data_dir=tmp_path / "no_data")


@pytest.fixture
def config(tmp_results: Path) -> RunConfig:
    return RunConfig(
        models=["test-model"],
        runs_per_problem=1,
        temperature=0.0,
        max_concurrency=1,
        max_retries=1,
        retry_delay_seconds=0,
        results_dir=tmp_results,
        dry_run=True,
    )


@pytest.fixture
def orchestrator(
    humaneval: HumanEvalDataset,
    mock_client: MockLLMClient,
    sandbox: DockerSandbox,
    config: RunConfig,
) -> BenchmarkOrchestrator:
    return BenchmarkOrchestrator(
        dataset=humaneval,
        llm_clients={"test-model": mock_client},
        sandbox=sandbox,
        config=config,
    )


# ======================================================================
# MockLLMClient Tests
# ======================================================================

class TestMockLLMClient:

    def test_basic_response(self, mock_client: MockLLMClient):
        prompt = Prompt(
            prompt_id="p-001",
            problem_id="test",
            model_id="test-model",
            user_message="Generate code",
        )
        response = mock_client.send_prompt(prompt)
        assert isinstance(response, LLMResponse)
        assert response.model_id == "test-model"
        assert response.prompt_id == "p-001"
        assert response.finish_reason == "stop"
        assert "class Solution" in response.extracted_code

    def test_call_count(self, mock_client: MockLLMClient):
        prompt = Prompt(
            prompt_id="p-001",
            problem_id="test",
            model_id="test-model",
            user_message="Code",
        )
        assert mock_client.call_count == 0
        mock_client.send_prompt(prompt)
        assert mock_client.call_count == 1
        mock_client.send_prompt(prompt)
        assert mock_client.call_count == 2

    def test_call_log(self, mock_client: MockLLMClient):
        prompt = Prompt(
            prompt_id="p-002",
            problem_id="prob_x",
            model_id="test-model",
            user_message="Code",
        )
        mock_client.send_prompt(prompt)
        log = mock_client.call_log
        assert len(log) == 1
        assert log[0]["prompt_id"] == "p-002"
        assert log[0]["problem_id"] == "prob_x"

    def test_custom_responses(self):
        client = MockLLMClient(
            custom_responses={
                "prob_a": "public class A { }",
                "prob_b": "public class B { }",
            },
            latency_ms=0,
        )
        prompt_a = Prompt(
            prompt_id="p1", problem_id="prob_a",
            model_id="mock", user_message="Code",
        )
        prompt_b = Prompt(
            prompt_id="p2", problem_id="prob_b",
            model_id="mock", user_message="Code",
        )
        prompt_c = Prompt(
            prompt_id="p3", problem_id="prob_c",
            model_id="mock", user_message="Code",
        )

        assert "class A" in client.send_prompt(prompt_a).extracted_code
        assert "class B" in client.send_prompt(prompt_b).extracted_code
        assert "mock generated" in client.send_prompt(prompt_c).extracted_code

    def test_reset(self, mock_client: MockLLMClient):
        prompt = Prompt(
            prompt_id="p1", problem_id="test",
            model_id="test", user_message="Code",
        )
        mock_client.send_prompt(prompt)
        assert mock_client.call_count == 1
        mock_client.reset()
        assert mock_client.call_count == 0
        assert mock_client.call_log == []

    def test_response_ids_unique(self, mock_client: MockLLMClient):
        prompt = Prompt(
            prompt_id="p1", problem_id="test",
            model_id="test", user_message="Code",
        )
        r1 = mock_client.send_prompt(prompt)
        r2 = mock_client.send_prompt(prompt)
        assert r1.response_id != r2.response_id

    def test_token_counts(self, mock_client: MockLLMClient):
        prompt = Prompt(
            prompt_id="p1", problem_id="test",
            model_id="test", user_message="Write a Java class",
        )
        response = mock_client.send_prompt(prompt)
        assert response.prompt_tokens > 0
        assert response.completion_tokens > 0
        assert response.total_tokens == response.prompt_tokens + response.completion_tokens

    def test_raw_text_wraps_code(self, mock_client: MockLLMClient):
        prompt = Prompt(
            prompt_id="p1", problem_id="test",
            model_id="test", user_message="Code",
        )
        response = mock_client.send_prompt(prompt)
        assert response.raw_text.startswith("```java")
        assert response.raw_text.endswith("```")


# ======================================================================
# RunConfig Tests
# ======================================================================

class TestRunConfig:

    def test_defaults(self):
        config = RunConfig()
        assert config.runs_per_problem == 3
        assert config.temperature == 0.0
        assert config.max_concurrency == 3
        assert config.max_retries == 2

    def test_to_dict(self):
        config = RunConfig(models=["gpt-4", "claude-3"])
        d = config.to_dict()
        assert d["models"] == ["gpt-4", "claude-3"]
        assert "results_dir" in d

    def test_custom_values(self, tmp_path: Path):
        config = RunConfig(
            models=["model-a"],
            runs_per_problem=5,
            temperature=0.8,
            max_concurrency=10,
            results_dir=tmp_path / "out",
        )
        assert config.runs_per_problem == 5
        assert config.temperature == 0.8
        assert config.max_concurrency == 10


# ======================================================================
# Orchestrator Tests
# ======================================================================

class TestBenchmarkOrchestrator:

    def test_run_single(self, orchestrator: BenchmarkOrchestrator):
        result = orchestrator.run_single(
            problem_id="HumanEval_0",
            model_id="test-model",
            run_id=1,
        )
        assert isinstance(result, EvaluationResult)
        assert result.problem_id == "HumanEval_0"
        assert result.tests_total > 0

    def test_run_single_writes_code(
        self, orchestrator: BenchmarkOrchestrator, tmp_results: Path,
    ):
        orchestrator.run_single("HumanEval_0", "test-model", 1)
        code_dir = tmp_results / "humaneval-java" / "test-model" / "code"
        assert code_dir.exists()
        files = list(code_dir.glob("*.java"))
        assert len(files) == 1
        assert "HumanEval_0" in files[0].name

    def test_run_single_writes_jsonl(
        self, orchestrator: BenchmarkOrchestrator, tmp_results: Path,
    ):
        orchestrator.run_single("HumanEval_0", "test-model", 1)
        jsonl = tmp_results / "humaneval-java" / "test-model" / "results.jsonl"
        assert jsonl.exists()
        lines = jsonl.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["problem_id"] == "HumanEval_0"

    def test_run_all(self, orchestrator: BenchmarkOrchestrator):
        stats = orchestrator.run_all()
        assert stats["total_evaluations"] == 3  # 3 problems × 1 model × 1 run
        assert stats["started_at"] is not None
        assert stats["completed_at"] is not None

    def test_run_all_multiple_runs(
        self, humaneval: HumanEvalDataset, mock_client: MockLLMClient,
        sandbox: DockerSandbox, tmp_results: Path,
    ):
        config = RunConfig(
            models=["test-model"],
            runs_per_problem=3,
            max_concurrency=1,
            results_dir=tmp_results,
            dry_run=True,
        )
        orch = BenchmarkOrchestrator(
            dataset=humaneval,
            llm_clients={"test-model": mock_client},
            sandbox=sandbox,
            config=config,
        )
        stats = orch.run_all()
        assert stats["total_evaluations"] == 9  # 3 problems × 1 model × 3 runs

    def test_checkpoint_resume(
        self, humaneval: HumanEvalDataset, mock_client: MockLLMClient,
        sandbox: DockerSandbox, tmp_results: Path, tmp_path: Path,
    ):
        """Completed runs should be skipped on resume."""
        progress = ProgressTracker(
            checkpoint_path=tmp_path / "progress.json"
        )
        # Pre-mark one run as completed
        progress.mark_completed("humaneval-java/test-model/HumanEval_0/run_1")
        progress.save()

        config = RunConfig(
            models=["test-model"],
            runs_per_problem=1,
            max_concurrency=1,
            results_dir=tmp_results,
            dry_run=True,
        )
        orch = BenchmarkOrchestrator(
            dataset=humaneval,
            llm_clients={"test-model": mock_client},
            sandbox=sandbox,
            config=config,
            progress=progress,
        )
        stats = orch.run_all()
        # HumanEval_0 should be skipped
        assert stats["skipped"] >= 1
        assert stats["total_evaluations"] == 2  # Only 2 of 3 problems ran

    def test_missing_model_client(self, orchestrator: BenchmarkOrchestrator):
        with pytest.raises(ValueError, match="No LLM client"):
            orchestrator.run_single("HumanEval_0", "nonexistent-model", 1)

    def test_error_result_on_llm_failure(
        self, humaneval: HumanEvalDataset, sandbox: DockerSandbox, tmp_results: Path,
    ):
        """LLM failure should produce an ERROR verdict."""

        class FailingClient:
            def send_prompt(self, prompt):
                raise ConnectionError("API down")

        config = RunConfig(
            models=["fail-model"],
            runs_per_problem=1,
            max_retries=1,
            retry_delay_seconds=0,
            results_dir=tmp_results,
            dry_run=True,
        )
        orch = BenchmarkOrchestrator(
            dataset=humaneval,
            llm_clients={"fail-model": FailingClient()},
            sandbox=sandbox,
            config=config,
        )
        result = orch.run_single("HumanEval_0", "fail-model", 1)
        assert result.verdict == Verdict.ERROR
        assert "failed" in result.runtime_errors[0].lower() or "retry" in result.runtime_errors[0].lower()

    def test_parallel_execution(
        self, humaneval: HumanEvalDataset, mock_client: MockLLMClient,
        sandbox: DockerSandbox, tmp_results: Path,
    ):
        """Parallel execution should produce same results as sequential."""
        config = RunConfig(
            models=["test-model"],
            runs_per_problem=1,
            max_concurrency=3,
            results_dir=tmp_results,
            dry_run=True,
        )
        orch = BenchmarkOrchestrator(
            dataset=humaneval,
            llm_clients={"test-model": mock_client},
            sandbox=sandbox,
            config=config,
        )
        stats = orch.run_all()
        assert stats["total_evaluations"] == 3


# ======================================================================
# Integration: Orchestrator with all datasets
# ======================================================================

class TestOrchestratorWithAllDatasets:

    @pytest.fixture
    def setup_clients(self):
        return {"test-model": MockLLMClient(latency_ms=0)}

    def _run_dataset(
        self, dataset, clients, sandbox, tmp_results: Path
    ) -> dict:
        config = RunConfig(
            models=["test-model"],
            runs_per_problem=1,
            max_concurrency=1,
            results_dir=tmp_results,
            dry_run=True,
        )
        orch = BenchmarkOrchestrator(
            dataset=dataset,
            llm_clients=clients,
            sandbox=sandbox,
            config=config,
        )
        return orch.run_all()

    def test_humaneval_integration(
        self, setup_clients, sandbox, tmp_path: Path,
    ):
        ds = HumanEvalDataset(data_dir=tmp_path / "no_data")
        stats = self._run_dataset(ds, setup_clients, sandbox, tmp_path / "results")
        assert stats["total_evaluations"] == 3

    def test_mbpp_integration(
        self, setup_clients, sandbox, tmp_path: Path,
    ):
        ds = MBPPDataset(data_dir=tmp_path / "no_data")
        stats = self._run_dataset(ds, setup_clients, sandbox, tmp_path / "results")
        assert stats["total_evaluations"] == 3

    def test_defects4j_integration(
        self, setup_clients, sandbox, tmp_path: Path,
    ):
        ds = Defects4JDataset(data_dir=tmp_path / "no_data")
        stats = self._run_dataset(ds, setup_clients, sandbox, tmp_path / "results")
        assert stats["total_evaluations"] == 3

    def test_godclass_integration(
        self, setup_clients, sandbox, tmp_path: Path,
    ):
        ds = GodClassDataset(data_dir=tmp_path / "no_data")
        stats = self._run_dataset(ds, setup_clients, sandbox, tmp_path / "results")
        assert stats["total_evaluations"] == 3


# ======================================================================
# Unexecuted tests are never a pass
# ======================================================================

class TestExecutionIsRequired:
    """The May 2026 run published structural-fallback passes as results.

    These pin the guards that stop that happening again.
    """

    def _orchestrator(self, dataset, sandbox, tmp_results: Path, dry_run: bool):
        return BenchmarkOrchestrator(
            dataset=dataset,
            llm_clients={"test-model": MockLLMClient(latency_ms=0)},
            sandbox=sandbox,
            config=RunConfig(
                models=["test-model"], runs_per_problem=1, max_concurrency=1,
                results_dir=tmp_results, dry_run=dry_run,
            ),
        )

    def _all_passed(self, executed: bool) -> SandboxResult:
        return SandboxResult(
            problem_id="p", compiled=True, tests_passed=3, tests_total=3,
            executed=executed,
        )

    def _response(self) -> LLMResponse:
        return LLMResponse(
            response_id="r", prompt_id="p", model_id="test-model", raw_text="",
        )

    def test_a_real_run_refuses_to_start_without_docker(
        self, humaneval, sandbox, tmp_results: Path,
    ):
        orch = self._orchestrator(humaneval, sandbox, tmp_results, dry_run=False)
        with pytest.raises(RuntimeError, match="Docker is unavailable"):
            orch.run_all()

    def test_an_unexecuted_pass_is_recorded_as_an_error(
        self, humaneval, sandbox, tmp_results: Path,
    ):
        orch = self._orchestrator(humaneval, sandbox, tmp_results, dry_run=False)
        pid = humaneval.list_problem_ids()[0]
        result = orch._build_evaluation_result(
            problem_id=pid, model_id="test-model", run_id=1,
            response=self._response(), suite=humaneval.get_test_suite(pid),
            sandbox_result=self._all_passed(executed=False),
        )
        assert result.verdict == Verdict.ERROR
        assert result.tests_executed is False
        assert result.weighted_score == 0.0

    def test_an_executed_pass_is_a_pass(
        self, humaneval, sandbox, tmp_results: Path,
    ):
        orch = self._orchestrator(humaneval, sandbox, tmp_results, dry_run=False)
        pid = humaneval.list_problem_ids()[0]
        result = orch._build_evaluation_result(
            problem_id=pid, model_id="test-model", run_id=2,
            response=self._response(), suite=humaneval.get_test_suite(pid),
            sandbox_result=self._all_passed(executed=True),
        )
        assert result.verdict == Verdict.PASS
        assert result.tests_executed is True
        assert result.sample_index == 2

    def test_runs_carry_their_sample_index_into_the_prompt(
        self, orchestrator: BenchmarkOrchestrator,
    ):
        seen: list[int] = []
        client = orchestrator.llm_clients["test-model"]
        original = client.send_prompt

        def spy(prompt: Prompt) -> LLMResponse:
            seen.append(prompt.metadata["sample_index"])
            return original(prompt)

        client.send_prompt = spy  # type: ignore[method-assign]
        pid = orchestrator.dataset.list_problem_ids()[0]
        for run_id in (1, 2, 3):
            orchestrator.run_single(pid, "test-model", run_id)
        assert seen == [1, 2, 3]


class TestRefactoringNeedsStructure:

    def test_an_unchanged_god_class_is_not_a_pass(
        self, sandbox, tmp_results: Path, tmp_path: Path,
    ):
        """Behaviour tests pass on the original. That is not a refactoring."""
        dataset = GodClassDataset(data_dir=tmp_path / "no_data")
        orch = BenchmarkOrchestrator(
            dataset=dataset,
            llm_clients={"test-model": MockLLMClient(latency_ms=0)},
            sandbox=sandbox,
            config=RunConfig(models=["test-model"], results_dir=tmp_results),
        )
        pid = dataset.list_problem_ids()[0]
        result = orch._build_evaluation_result(
            problem_id=pid, model_id="test-model", run_id=1,
            response=LLMResponse(
                response_id="r", prompt_id="p", model_id="test-model", raw_text="",
            ),
            suite=dataset.get_test_suite(pid),
            sandbox_result=SandboxResult(
                problem_id=pid, compiled=True, tests_passed=3, tests_total=3,
                executed=True,
            ),
            code=dataset.get_original_code(pid),
        )
        assert result.verdict == Verdict.PARTIAL
        assert result.structural["credited"] is False

    def test_god_class_suites_assert_behaviour(self, tmp_path: Path):
        """No suite may contain an assertion that cannot fail."""
        import re

        for data_dir in (tmp_path / "no_data", Path("data")):
            dataset = GodClassDataset(data_dir=data_dir)
            for pid in dataset.list_problem_ids():
                junit = dataset.get_junit_code(pid)
                assert not re.search(r"assertTrue\([^;]*,\s*true\)", junit), pid
                assert dataset.get_problem(pid).metadata["entry_point"] in junit, pid


class TestResume:
    """A run stopped part-way must resume, not re-record what it finished."""

    def _orchestrator(self, dataset, sandbox, tmp_results: Path):
        return BenchmarkOrchestrator(
            dataset=dataset,
            llm_clients={"test-model": MockLLMClient(latency_ms=0)},
            sandbox=sandbox,
            config=RunConfig(
                models=["test-model"], runs_per_problem=2, max_concurrency=1,
                results_dir=tmp_results, dry_run=True,
            ),
        )

    def _results(self, dataset, tmp_results: Path) -> Path:
        return tmp_results / dataset.name / "test-model" / "results.jsonl"

    def test_recorded_samples_are_skipped_on_restart(
        self, humaneval, sandbox, tmp_results: Path,
    ):
        pid = humaneval.list_problem_ids()[0]
        first = self._orchestrator(humaneval, sandbox, tmp_results)
        first.run_single(pid, "test-model", 1)
        lines_before = self._results(humaneval, tmp_results).read_text().count("\n")

        second = self._orchestrator(humaneval, sandbox, tmp_results)
        assert second.resume_from_results() == 1
        second.run_single(pid, "test-model", 1)   # already recorded: skipped
        second.run_single(pid, "test-model", 2)   # new sample: recorded
        assert self._results(humaneval, tmp_results).read_text().count("\n") == lines_before + 1

    def test_a_partial_last_line_is_dropped(self, humaneval, sandbox, tmp_results: Path):
        pid = humaneval.list_problem_ids()[0]
        self._orchestrator(humaneval, sandbox, tmp_results).run_single(pid, "test-model", 1)
        path = self._results(humaneval, tmp_results)
        path.write_text(path.read_text() + '{"problem_id": "trunc', encoding="utf-8")

        assert self._orchestrator(humaneval, sandbox, tmp_results).resume_from_results() == 1
        assert path.read_text().endswith("\n")
