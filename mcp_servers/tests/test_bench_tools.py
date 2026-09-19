"""
Tests for mcp_servers.tools.bench.

Two invariants get the most attention, because both are safety properties
rather than behaviour:

* no tool returns a reference solution;
* ``run_benchmark`` cannot spend money without an explicit ceiling.
"""

from __future__ import annotations

import pytest

from mcp_servers.registry import DATASET_CATEGORY, RunRegistry, get_registry
from mcp_servers.tools.bench import (
    MEAN_COST_USD_PER_EVALUATION,
    evaluate_patch,
    get_run_results,
    get_task,
    list_tasks,
    run_benchmark,
)

DATASETS = sorted(DATASET_CATEGORY)


@pytest.fixture(scope="module")
def registry():
    return get_registry("data")


class TestListTasks:
    def test_lists_every_task(self):
        result = list_tasks(limit=1000)
        assert result.ok is True
        assert result.total == 58
        assert len(result.tasks) == 58

    def test_limit_truncates_but_total_does_not(self):
        result = list_tasks(limit=5)
        assert len(result.tasks) == 5
        assert result.total == 58

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_filters_by_dataset(self, dataset):
        result = list_tasks(dataset=dataset, limit=1000)
        assert result.ok is True
        assert result.tasks
        assert {task.dataset for task in result.tasks} == {dataset}

    @pytest.mark.parametrize("category", sorted(set(DATASET_CATEGORY.values())))
    def test_filters_by_category(self, category):
        result = list_tasks(category=category, limit=1000)
        assert result.ok is True
        assert {task.category for task in result.tasks} == {category}

    def test_unknown_filter_is_an_error_not_an_empty_list(self):
        """A typo must not look like 'no tasks match'."""
        result = list_tasks(category="nonsense")
        assert result.ok is False
        assert "Unknown category" in result.error

        result = list_tasks(dataset="nonsense")
        assert result.ok is False
        assert "Unknown dataset" in result.error

    def test_summaries_carry_no_code(self):
        result = list_tasks(limit=5)
        for task in result.tasks:
            payload = task.model_dump()
            assert "reference_solution" not in payload
            assert "junit_code" not in payload


class TestGetTask:
    def test_returns_a_prompt(self):
        result = get_task("HumanEval_0")
        assert result.ok is True
        assert result.dataset == "humaneval-java"
        assert result.category == "codegen"
        assert result.prompt.strip()

    def test_never_leaks_the_reference_solution(self):
        """The whole benchmark is void if a task hands out its own answer."""
        for task_id in list_tasks(limit=1000).tasks:
            result = get_task(task_id.task_id)
            assert result.ok is True
            payload = result.model_dump()
            assert "reference_solution" not in payload

    def test_codegen_starts_from_nothing(self):
        result = get_task("HumanEval_0")
        assert result.repo_state == {}

    def test_bugfix_supplies_the_buggy_class(self):
        result = get_task("D4J_Lang_1")
        assert result.category == "bugfix"
        assert result.repo_state
        assert any("class" in content for content in result.repo_state.values())

    def test_refactor_supplies_the_god_class(self):
        result = get_task("GC_001")
        assert result.category == "refactor"
        assert result.repo_state

    def test_test_spec_counts_visible_and_hidden(self):
        result = get_task("HumanEval_0")
        spec = result.test_spec
        assert spec.num_visible_cases + spec.num_hidden_cases > 0
        assert len(spec.visible_cases) <= 5
        assert len(spec.visible_cases) <= spec.num_visible_cases

    def test_hidden_cases_are_not_in_visible_cases(self):
        for task_id in ("HumanEval_0", "MBPP_1", "D4J_Lang_1", "GC_001"):
            spec = get_task(task_id).test_spec
            assert len(spec.visible_cases) <= spec.num_visible_cases

    def test_unknown_task_says_how_to_find_the_real_ones(self):
        result = get_task("NoSuchTask_999")
        assert result.ok is False
        assert "list_tasks" in result.error

    def test_every_task_is_fetchable(self):
        failures = [
            task.task_id
            for task in list_tasks(limit=1000).tasks
            if not get_task(task.task_id).ok
        ]
        assert failures == []


class TestEvaluatePatch:
    def test_rejects_an_empty_patch(self):
        result = evaluate_patch("HumanEval_0", "")
        assert result.ok is False
        assert "empty" in result.error

    def test_rejects_an_oversized_patch(self):
        result = evaluate_patch("HumanEval_0", "x" * 200_000)
        assert result.ok is False
        assert "limit" in result.error

    def test_rejects_an_unknown_task(self):
        result = evaluate_patch("NoSuchTask_999", "public class A {}")
        assert result.ok is False
        assert "list_tasks" in result.error

    def test_costs_nothing(self):
        result = evaluate_patch("HumanEval_0", "public class A { void f() {} }")
        assert result.ok is True
        assert result.cost_usd == 0.0

    def test_reports_ck_metrics_of_the_submission(self):
        patch = """\
public class HasCloseElements {
    public static boolean check(int x) {
        if (x > 0) { return true; }
        return false;
    }
}
"""
        result = evaluate_patch("HumanEval_0", patch)
        assert result.ok is True
        assert result.metrics
        assert result.metrics["wmc"] >= 1

    def test_unparseable_patch_still_returns_a_verdict(self, broken_class):
        result = evaluate_patch("HumanEval_0", broken_class)
        assert result.ok is True
        assert result.verdict in {"pass", "partial", "fail", "error"}
        assert result.metrics == {}


class TestRunBenchmarkBudget:
    """A tool a model can call must not be able to spend money by accident."""

    def test_defaults_to_dry_run(self, tmp_path):
        result = run_benchmark(
            model="claude-sonnet-4-6",
            task_set="humaneval-java",
            limit=2,
            output_dir=str(tmp_path / "results"),
            runs_dir=str(tmp_path / "runs"),
        )
        assert result.ok is True
        assert result.dry_run is True
        assert result.actual_cost_eur == 0.0
        assert result.estimated_cost_eur == 0.0

    def test_real_run_without_a_ceiling_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LLM_SE_BENCH_BUDGET_EUR", raising=False)
        result = run_benchmark(
            model="claude-sonnet-4-6",
            task_set="humaneval-java",
            limit=2,
            dry_run=False,
            output_dir=str(tmp_path / "results"),
            runs_dir=str(tmp_path / "runs"),
        )
        assert result.ok is False
        assert result.started is False
        assert "budget ceiling" in result.error

    def test_real_run_over_ceiling_is_refused_before_starting(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LLM_SE_BENCH_BUDGET_EUR", raising=False)
        result = run_benchmark(
            model="claude-sonnet-4-6",
            task_set="humaneval-java",
            budget_eur=0.0001,
            dry_run=False,
            output_dir=str(tmp_path / "results"),
            runs_dir=str(tmp_path / "runs"),
        )
        assert result.ok is False
        assert result.started is False
        assert "Refused" in result.error
        assert result.estimated_cost_eur > 0

    def test_unknown_model_is_assumed_expensive(self, tmp_path, monkeypatch):
        """An unmeasured model must over-estimate, so the check fails safe."""
        monkeypatch.delenv("LLM_SE_BENCH_BUDGET_EUR", raising=False)
        cheapest = min(MEAN_COST_USD_PER_EVALUATION.values())
        result = run_benchmark(
            model="some-unreleased-model",
            task_set="humaneval-java",
            budget_eur=0.0001,
            dry_run=False,
            output_dir=str(tmp_path / "results"),
            runs_dir=str(tmp_path / "runs"),
        )
        assert result.ok is False
        assert "assumed" in result.error
        assert result.estimated_cost_eur > cheapest

    def test_env_var_supplies_the_ceiling(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_SE_BENCH_BUDGET_EUR", "0.0001")
        result = run_benchmark(
            model="claude-sonnet-4-6",
            task_set="humaneval-java",
            dry_run=False,
            output_dir=str(tmp_path / "results"),
            runs_dir=str(tmp_path / "runs"),
        )
        assert result.ok is False
        assert result.budget_eur == pytest.approx(0.0001)

    def test_rejects_zero_runs(self, tmp_path):
        result = run_benchmark(
            model="claude-sonnet-4-6",
            runs_per_task=0,
            output_dir=str(tmp_path / "results"),
            runs_dir=str(tmp_path / "runs"),
        )
        assert result.ok is False
        assert "at least 1" in result.error

    def test_rejects_an_unknown_task_set(self, tmp_path):
        result = run_benchmark(
            model="claude-sonnet-4-6",
            task_set="nonsense",
            output_dir=str(tmp_path / "results"),
            runs_dir=str(tmp_path / "runs"),
        )
        assert result.ok is False
        assert "Unknown task_set" in result.error


class TestRunRoundTrip:
    def test_dry_run_results_can_be_read_back(self, tmp_path):
        results_dir = tmp_path / "results"
        runs_dir = tmp_path / "runs"
        started = run_benchmark(
            model="claude-sonnet-4-6",
            task_set="humaneval-java",
            limit=3,
            output_dir=str(results_dir),
            runs_dir=str(runs_dir),
        )
        assert started.ok is True
        assert started.run_id
        assert started.task_count == 3
        assert started.evaluations == 3

        read_back = get_run_results(
            started.run_id,
            results_dir=str(results_dir),
            runs_dir=str(runs_dir),
        )
        assert read_back.ok is True
        assert read_back.run_id == started.run_id
        assert read_back.dry_run is True
        assert read_back.summaries
        summary = read_back.summaries[0]
        assert summary.evaluations == 3
        assert 0.0 <= summary.pass_rate <= 1.0

    def test_unknown_run_id_lists_what_exists(self, tmp_path):
        result = get_run_results("run-does-not-exist", runs_dir=str(tmp_path / "runs"))
        assert result.ok is False
        assert "Unknown run_id" in result.error


class TestRunRegistry:
    def test_round_trips_a_record(self, tmp_path):
        registry = RunRegistry(tmp_path)
        registry.save("run-abc", {"run_id": "run-abc", "model": "m"})
        assert registry.load("run-abc") == {"run_id": "run-abc", "model": "m"}

    def test_missing_record_is_none(self, tmp_path):
        assert RunRegistry(tmp_path).load("nope") is None

    @pytest.mark.parametrize("run_id", ["../escape", "a/b", "..", ""])
    def test_rejects_path_traversal(self, tmp_path, run_id):
        registry = RunRegistry(tmp_path)
        assert registry.load(run_id) is None
        with pytest.raises(ValueError):
            registry.save(run_id, {})

    def test_lists_newest_first(self, tmp_path):
        registry = RunRegistry(tmp_path)
        for name in ("run-1", "run-2", "run-3"):
            registry.save(name, {"run_id": name})
        assert set(registry.list_run_ids()) == {"run-1", "run-2", "run-3"}


class TestTaskRegistry:
    def test_indexes_every_task(self, registry):
        assert len(registry.task_ids()) == 58

    def test_knows_all_four_datasets(self, registry):
        assert set(registry.dataset_names()) == set(DATASETS)

    def test_locate_resolves_a_task_to_its_dataset(self, registry):
        location = registry.locate("D4J_Lang_1")
        assert location is not None
        assert location.dataset_name == "defects4j"
        assert location.category == "bugfix"

    def test_locate_returns_none_for_an_unknown_task(self, registry):
        assert registry.locate("NoSuchTask_999") is None
