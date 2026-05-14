"""
tests.test_datasets — Unit tests for all four dataset adapters.

Tests cover:
- Loading built-in examples
- Problem and TestSuite contracts
- Prompt formatting
- Structural verification
- Validation
- Edge cases
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from contracts import Problem, TestCase, TestSuite, VerificationResult
from bench.datasets.base import Dataset
from bench.datasets.humaneval import HumanEvalDataset
from bench.datasets.mbpp import MBPPDataset
from bench.datasets.defects4j import Defects4JDataset
from bench.datasets.godclass import GodClassDataset


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary data directory with no JSONL files (forces built-in examples)."""
    return tmp_path / "data"


@pytest.fixture
def humaneval(tmp_data_dir: Path) -> HumanEvalDataset:
    return HumanEvalDataset(data_dir=tmp_data_dir)


@pytest.fixture
def mbpp(tmp_data_dir: Path) -> MBPPDataset:
    return MBPPDataset(data_dir=tmp_data_dir)


@pytest.fixture
def defects4j(tmp_data_dir: Path) -> Defects4JDataset:
    return Defects4JDataset(data_dir=tmp_data_dir)


@pytest.fixture
def godclass(tmp_data_dir: Path) -> GodClassDataset:
    return GodClassDataset(data_dir=tmp_data_dir)


# ======================================================================
# Base Dataset interface tests (shared assertions)
# ======================================================================

class DatasetInterfaceTests:
    """Mixin with reusable dataset interface assertions."""

    def assert_dataset_loads(self, ds: Dataset, expected_count: int):
        problems = ds.load_problems()
        assert isinstance(problems, list)
        assert len(problems) == expected_count
        for p in problems:
            assert isinstance(p, Problem)
            assert p.problem_id
            assert p.language == "java"

    def assert_problems_have_test_suites(self, ds: Dataset):
        for pid in ds.list_problem_ids():
            suite = ds.get_test_suite(pid)
            assert isinstance(suite, TestSuite)
            assert suite.problem_id == pid
            assert len(suite.cases) > 0
            for tc in suite.cases:
                assert isinstance(tc, TestCase)
                assert tc.test_id
                assert tc.timeout_seconds > 0

    def assert_prompt_formatting(self, ds: Dataset):
        for pid in ds.list_problem_ids():
            prompt = ds.format_prompt(pid)
            assert isinstance(prompt, str)
            assert len(prompt) > 50
            # Prompt should contain the problem title or description keywords
            problem = ds.get_problem(pid)
            # At least mention Java or code
            assert "java" in prompt.lower() or "class" in prompt.lower()

    def assert_verification(self, ds: Dataset):
        for pid in ds.list_problem_ids():
            # Verify with good-looking code
            problem = ds.get_problem(pid)
            results = ds.verify_solution(pid, problem.reference_solution)
            assert isinstance(results, list)
            assert len(results) > 0
            for vr in results:
                assert isinstance(vr, VerificationResult)
                assert vr.test_id

    def assert_validation(self, ds: Dataset):
        summary = ds.validate_dataset()
        assert "total" in summary
        assert "valid" in summary
        assert "errors" in summary
        assert summary["total"] > 0

    def assert_metadata_export(self, ds: Dataset, tmp_path: Path):
        out = ds.export_metadata(tmp_path / "meta.json")
        assert out.exists()
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == len(ds)
        # Should not contain reference solutions
        for entry in data:
            assert "reference_solution" not in entry


# ======================================================================
# HumanEval Tests
# ======================================================================

class TestHumanEvalDataset(DatasetInterfaceTests):

    def test_load_builtin_examples(self, humaneval: HumanEvalDataset):
        self.assert_dataset_loads(humaneval, 3)

    def test_dataset_name(self, humaneval: HumanEvalDataset):
        assert humaneval.name == "humaneval-java"
        assert humaneval.EXPECTED_SIZE == 164

    def test_problem_ids(self, humaneval: HumanEvalDataset):
        ids = humaneval.list_problem_ids()
        assert "HumanEval_0" in ids
        assert "HumanEval_1" in ids
        assert "HumanEval_4" in ids

    def test_get_problem(self, humaneval: HumanEvalDataset):
        p = humaneval.get_problem("HumanEval_0")
        assert p.title == "Has Close Elements"
        assert p.difficulty == "easy"
        assert "hasCloseElements" in p.metadata.get("entry_point", "")

    def test_test_suites(self, humaneval: HumanEvalDataset):
        self.assert_problems_have_test_suites(humaneval)

    def test_humaneval_0_test_suite(self, humaneval: HumanEvalDataset):
        suite = humaneval.get_test_suite("HumanEval_0")
        assert len(suite.cases) == 4
        visible = [tc for tc in suite.cases if not tc.is_hidden]
        hidden = [tc for tc in suite.cases if tc.is_hidden]
        assert len(visible) == 2
        assert len(hidden) == 2

    def test_prompt_formatting(self, humaneval: HumanEvalDataset):
        self.assert_prompt_formatting(humaneval)

    def test_prompt_contains_problem_info(self, humaneval: HumanEvalDataset):
        prompt = humaneval.format_prompt("HumanEval_0")
        assert "Has Close Elements" in prompt
        assert "threshold" in prompt.lower()

    def test_verification_with_reference(self, humaneval: HumanEvalDataset):
        self.assert_verification(humaneval)

    def test_verification_with_empty_code(self, humaneval: HumanEvalDataset):
        results = humaneval.verify_solution("HumanEval_0", "")
        assert all(not vr.passed for vr in results)
        assert any("class" in vr.error_message.lower() for vr in results)

    def test_verification_with_missing_method(self, humaneval: HumanEvalDataset):
        code = "public class Foo { public static int bar() { return 0; } }"
        results = humaneval.verify_solution("HumanEval_0", code)
        assert all(not vr.passed for vr in results)

    def test_junit_code(self, humaneval: HumanEvalDataset):
        junit = humaneval.get_junit_code("HumanEval_0")
        assert "HasCloseElementsTest" in junit
        assert "@Test" in junit

    def test_validation(self, humaneval: HumanEvalDataset):
        self.assert_validation(humaneval)

    def test_metadata_export(self, humaneval: HumanEvalDataset, tmp_path: Path):
        self.assert_metadata_export(humaneval, tmp_path)

    def test_len_and_contains(self, humaneval: HumanEvalDataset):
        assert len(humaneval) == 3
        assert "HumanEval_0" in humaneval
        assert "HumanEval_999" not in humaneval

    def test_get_problem_not_found(self, humaneval: HumanEvalDataset):
        with pytest.raises(KeyError, match="HumanEval_999"):
            humaneval.get_problem("HumanEval_999")

    def test_get_test_suite_not_found(self, humaneval: HumanEvalDataset):
        with pytest.raises(KeyError):
            humaneval.get_test_suite("nonexistent")

    def test_load_from_jsonl(self, tmp_path: Path):
        """Test loading from a real JSONL file."""
        jsonl_dir = tmp_path / "data" / "humaneval"
        jsonl_dir.mkdir(parents=True)
        jsonl_path = jsonl_dir / "problems.jsonl"

        problem_data = {
            "problem_id": "HumanEval_test",
            "title": "Test Problem",
            "description": "A test problem",
            "difficulty": "easy",
            "reference_solution": "class Test {}",
            "tags": ["test"],
            "metadata": {"entry_point": "solve"},
            "test_cases": [
                {
                    "test_id": "t1",
                    "input_data": {"x": 1},
                    "expected_output": 2,
                }
            ],
            "junit_code": "import org.junit.Test;\npublic class TestTest {}",
        }
        jsonl_path.write_text(json.dumps(problem_data) + "\n")

        ds = HumanEvalDataset(data_dir=tmp_path / "data")
        problems = ds.load_problems()
        assert len(problems) == 1
        assert problems[0].problem_id == "HumanEval_test"
        suite = ds.get_test_suite("HumanEval_test")
        assert len(suite.cases) == 1


# ======================================================================
# MBPP Tests
# ======================================================================

class TestMBPPDataset(DatasetInterfaceTests):

    def test_load_builtin_examples(self, mbpp: MBPPDataset):
        self.assert_dataset_loads(mbpp, 3)

    def test_dataset_name(self, mbpp: MBPPDataset):
        assert mbpp.name == "mbpp-java"
        assert mbpp.EXPECTED_SIZE == 200

    def test_difficulty_distribution(self, mbpp: MBPPDataset):
        expected = {"easy": 80, "medium": 80, "hard": 40}
        assert mbpp.DIFFICULTY_DISTRIBUTION == expected

    def test_problem_ids(self, mbpp: MBPPDataset):
        ids = mbpp.list_problem_ids()
        assert "MBPP_1" in ids
        assert "MBPP_2" in ids
        assert "MBPP_3" in ids

    def test_get_problem(self, mbpp: MBPPDataset):
        p = mbpp.get_problem("MBPP_1")
        assert p.title == "Minimum Cost Path"
        assert p.difficulty == "medium"

    def test_test_suites(self, mbpp: MBPPDataset):
        self.assert_problems_have_test_suites(mbpp)

    def test_prompt_formatting(self, mbpp: MBPPDataset):
        self.assert_prompt_formatting(mbpp)

    def test_verification(self, mbpp: MBPPDataset):
        self.assert_verification(mbpp)

    def test_junit_code(self, mbpp: MBPPDataset):
        junit = mbpp.get_junit_code("MBPP_1")
        assert "@Test" in junit
        assert "MinCostPathTest" in junit

    def test_get_problems_by_difficulty(self, mbpp: MBPPDataset):
        easy = mbpp.get_problems_by_difficulty("easy")
        medium = mbpp.get_problems_by_difficulty("medium")
        assert len(easy) == 2  # MBPP_2 and MBPP_3 are easy
        assert len(medium) == 1  # MBPP_1 is medium

    def test_validation(self, mbpp: MBPPDataset):
        self.assert_validation(mbpp)

    def test_len(self, mbpp: MBPPDataset):
        assert len(mbpp) == 3


# ======================================================================
# Defects4J Tests
# ======================================================================

class TestDefects4JDataset(DatasetInterfaceTests):

    def test_load_builtin_examples(self, defects4j: Defects4JDataset):
        self.assert_dataset_loads(defects4j, 3)

    def test_dataset_name(self, defects4j: Defects4JDataset):
        assert defects4j.name == "defects4j"
        assert defects4j.EXPECTED_SIZE == 50

    def test_projects(self, defects4j: Defects4JDataset):
        assert "Chart" in defects4j.PROJECTS
        assert "Lang" in defects4j.PROJECTS
        assert "Math" in defects4j.PROJECTS

    def test_get_problem(self, defects4j: Defects4JDataset):
        p = defects4j.get_problem("D4J_Lang_1")
        assert "NumberUtils" in p.title
        assert p.metadata["project"] == "Lang"
        assert p.metadata["bug_id"] == 1

    def test_test_suites(self, defects4j: Defects4JDataset):
        self.assert_problems_have_test_suites(defects4j)

    def test_buggy_code(self, defects4j: Defects4JDataset):
        buggy = defects4j.get_buggy_code("D4J_Lang_1")
        assert "BUG" in buggy  # Has a BUG comment
        assert "createNumber" in buggy

    def test_prompt_contains_buggy_code(self, defects4j: Defects4JDataset):
        prompt = defects4j.format_prompt("D4J_Lang_1")
        assert "Fix the bug" in prompt or "fix" in prompt.lower()
        assert "NumberUtils" in prompt or "createNumber" in prompt

    def test_prompt_formatting(self, defects4j: Defects4JDataset):
        self.assert_prompt_formatting(defects4j)

    def test_verification(self, defects4j: Defects4JDataset):
        self.assert_verification(defects4j)

    def test_get_bugs_by_project(self, defects4j: Defects4JDataset):
        lang_bugs = defects4j.get_bugs_by_project("Lang")
        assert len(lang_bugs) == 1
        assert lang_bugs[0].problem_id == "D4J_Lang_1"

    def test_junit_code(self, defects4j: Defects4JDataset):
        junit = defects4j.get_junit_code("D4J_Lang_1")
        assert "@Test" in junit
        assert "NumberFormatException" in junit

    def test_weighted_test_cases(self, defects4j: Defects4JDataset):
        """D4J bugs should have weighted test cases (failing test counts more)."""
        suite = defects4j.get_test_suite("D4J_Lang_1")
        weights = [tc.weight for tc in suite.cases]
        assert max(weights) > 1.0  # At least one has weight > 1

    def test_validation(self, defects4j: Defects4JDataset):
        self.assert_validation(defects4j)


# ======================================================================
# God Class Tests
# ======================================================================

class TestGodClassDataset(DatasetInterfaceTests):

    def test_load_builtin_examples(self, godclass: GodClassDataset):
        self.assert_dataset_loads(godclass, 3)

    def test_dataset_name(self, godclass: GodClassDataset):
        assert godclass.name == "godclass"
        assert godclass.EXPECTED_SIZE == 10

    def test_severity_levels(self, godclass: GodClassDataset):
        assert "mild" in godclass.SEVERITY_LEVELS
        assert "severe" in godclass.SEVERITY_LEVELS

    def test_get_problem(self, godclass: GodClassDataset):
        p = godclass.get_problem("GC_001")
        assert "OrderProcessor" in p.title
        assert p.metadata["severity"] == "severe"

    def test_original_code(self, godclass: GodClassDataset):
        original = godclass.get_original_code("GC_001")
        assert "class OrderProcessor" in original
        assert "validateOrder" in original
        assert "processPayment" in original

    def test_baseline_metrics(self, godclass: GodClassDataset):
        metrics = godclass.get_baseline_metrics("GC_001")
        assert metrics["loc"] == 450
        assert metrics["wmc"] == 38
        assert metrics["cbo"] == 22
        assert metrics["lcom"] == 0.85

    def test_prompt_contains_metrics(self, godclass: GodClassDataset):
        prompt = godclass.format_prompt("GC_001")
        assert "Refactor" in prompt or "refactor" in prompt
        assert "WMC" in prompt
        assert "CBO" in prompt

    def test_prompt_formatting(self, godclass: GodClassDataset):
        self.assert_prompt_formatting(godclass)

    def test_verification_class_count(self, godclass: GodClassDataset):
        """God Class verification should check class count."""
        # Code with multiple classes should pass the class_count check
        multi_class_code = (
            "public class A { }\n"
            "public class B { }\n"
            "public class C { }\n"
        )
        results = godclass.verify_solution("GC_001", multi_class_code)
        # Find the class_count test result
        class_count_results = [
            vr for vr in results
            if "class_count" not in vr.error_message and vr.error_message != "PENDING_SANDBOX_EXECUTION"
        ]
        # At least one should pass due to class count >= 3
        passed = [vr for vr in results if vr.passed]
        assert len(passed) >= 1

    def test_verification_single_class(self, godclass: GodClassDataset):
        """Single class should fail the class_count check."""
        single_class_code = "public class Foo { }"
        results = godclass.verify_solution("GC_001", single_class_code)
        # The class_count test should fail (expects >= 3)
        class_count_test = results[0]  # First test is class_count
        assert not class_count_test.passed

    def test_get_classes_by_severity(self, godclass: GodClassDataset):
        severe = godclass.get_classes_by_severity("severe")
        assert len(severe) == 2  # GC_001 and GC_003
        moderate = godclass.get_classes_by_severity("moderate")
        assert len(moderate) == 1  # GC_002

    def test_test_suites(self, godclass: GodClassDataset):
        self.assert_problems_have_test_suites(godclass)

    def test_validation(self, godclass: GodClassDataset):
        self.assert_validation(godclass)

    def test_junit_code(self, godclass: GodClassDataset):
        junit = godclass.get_junit_code("GC_001")
        assert "@Test" in junit


# ======================================================================
# Cross-cutting edge case tests
# ======================================================================

class TestDatasetEdgeCases:

    def test_lazy_loading(self, tmp_data_dir: Path):
        """Dataset should not load until first access."""
        ds = HumanEvalDataset(data_dir=tmp_data_dir)
        assert not ds._loaded
        _ = ds.list_problem_ids()
        assert ds._loaded

    def test_double_load_idempotent(self, humaneval: HumanEvalDataset):
        """Calling load_problems twice should not duplicate."""
        problems1 = humaneval.load_problems()
        humaneval._loaded = False
        problems2 = humaneval.load_problems()
        assert len(problems1) == len(problems2)

    def test_test_suite_total_weight(self, humaneval: HumanEvalDataset):
        suite = humaneval.get_test_suite("HumanEval_0")
        assert suite.total_weight == pytest.approx(4.0)

    def test_test_case_timeouts(self, defects4j: Defects4JDataset):
        """D4J test cases should have 15s timeout."""
        suite = defects4j.get_test_suite("D4J_Lang_1")
        for tc in suite.cases:
            assert tc.timeout_seconds == 15.0

    def test_godclass_test_case_timeouts(self, godclass: GodClassDataset):
        """God Class test cases should have 30s timeout."""
        suite = godclass.get_test_suite("GC_001")
        for tc in suite.cases:
            assert tc.timeout_seconds == 30.0

    def test_load_malformed_jsonl(self, tmp_path: Path):
        """Malformed JSONL should be skipped with a warning."""
        jsonl_dir = tmp_path / "data" / "humaneval"
        jsonl_dir.mkdir(parents=True)
        jsonl_path = jsonl_dir / "problems.jsonl"
        jsonl_path.write_text(
            '{"problem_id": "good", "title": "Good"}\n'
            'NOT VALID JSON\n'
            '{"problem_id": "also_good", "title": "Also Good"}\n'
        )
        ds = HumanEvalDataset(data_dir=tmp_path / "data")
        problems = ds.load_problems()
        assert len(problems) == 2
