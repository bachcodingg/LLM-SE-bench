"""
bench.datasets.mbpp — MBPP-Java dataset adapter (200 problems).

Adapts a 200-problem subset of the Mostly Basic Programming Problems
benchmark (Austin et al., 2021), translated to Java and stratified by
difficulty (80 Easy / 80 Medium / 40 Hard).

Ships with three built-in example problems; the full dataset is loaded
from ``data_dir/mbpp/problems.jsonl``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from bench.datasets.base import Dataset
from contracts import (
    Problem,
    TestCase,
    TestSuite,
    VerificationResult,
)

logger = logging.getLogger(__name__)

# ======================================================================
# Built-in example problems (3 of 200)
# ======================================================================

_EXAMPLE_PROBLEMS: list[dict[str, Any]] = [
    {
        "problem_id": "MBPP_1",
        "title": "Minimum Cost Path",
        "description": (
            "Write a Java function to find the minimum cost path in a 2D "
            "grid from (0,0) to (m-1,n-1). You can only move right or down.\n"
            "Each cell has a non-negative integer cost."
        ),
        "difficulty": "medium",
        "language": "java",
        "reference_solution": (
            "public class MinCostPath {\n"
            "    public static int minCostPath(int[][] cost, int m, int n) {\n"
            "        int[][] dp = new int[m][n];\n"
            "        dp[0][0] = cost[0][0];\n"
            "        for (int i = 1; i < m; i++) dp[i][0] = dp[i-1][0] + cost[i][0];\n"
            "        for (int j = 1; j < n; j++) dp[0][j] = dp[0][j-1] + cost[0][j];\n"
            "        for (int i = 1; i < m; i++) {\n"
            "            for (int j = 1; j < n; j++) {\n"
            "                dp[i][j] = Math.min(dp[i-1][j], dp[i][j-1]) + cost[i][j];\n"
            "            }\n"
            "        }\n"
            "        return dp[m-1][n-1];\n"
            "    }\n"
            "}\n"
        ),
        "tags": ["dp", "grid", "medium"],
        "metadata": {"original_index": 1, "entry_point": "minCostPath", "difficulty_tier": "medium"},
        "test_cases": [
            {
                "test_id": "MBPP_1_test_1",
                "input_data": {
                    "cost": [[1, 2, 3], [4, 8, 2], [1, 5, 3]],
                    "m": 3, "n": 3,
                },
                "expected_output": 8,
                "is_hidden": False,
                "weight": 1.0,
                "timeout_seconds": 10.0,
            },
            {
                "test_id": "MBPP_1_test_2",
                "input_data": {
                    "cost": [[2, 3, 4], [5, 8, 1], [6, 2, 5]],
                    "m": 3, "n": 3,
                },
                "expected_output": 12,
                "is_hidden": False,
                "weight": 1.0,
                "timeout_seconds": 10.0,
            },
            {
                "test_id": "MBPP_1_test_3",
                "input_data": {
                    "cost": [[5]],
                    "m": 1, "n": 1,
                },
                "expected_output": 5,
                "is_hidden": True,
                "weight": 1.0,
                "timeout_seconds": 10.0,
            },
        ],
        "junit_code": (
            "import org.junit.Test;\n"
            "import static org.junit.Assert.*;\n\n"
            "public class MinCostPathTest {\n"
            "    @Test\n"
            "    public void testBasicGrid() {\n"
            "        int[][] cost = {{1,2,3},{4,8,2},{1,5,3}};\n"
            "        assertEquals(8, MinCostPath.minCostPath(cost, 3, 3));\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testAnotherGrid() {\n"
            "        int[][] cost = {{2,3,4},{5,8,1},{6,2,5}};\n"
            "        assertEquals(12, MinCostPath.minCostPath(cost, 3, 3));\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testSingleCell() {\n"
            "        int[][] cost = {{5}};\n"
            "        assertEquals(5, MinCostPath.minCostPath(cost, 1, 1));\n"
            "    }\n"
            "}\n"
        ),
    },
    {
        "problem_id": "MBPP_2",
        "title": "Similar Triangles",
        "description": (
            "Write a Java function to check whether two given triangles "
            "are similar. Triangles are given as arrays of three side "
            "lengths. Two triangles are similar if the ratios of their "
            "corresponding sorted sides are equal."
        ),
        "difficulty": "easy",
        "language": "java",
        "reference_solution": (
            "import java.util.Arrays;\n\n"
            "public class SimilarTriangles {\n"
            "    public static boolean areSimilar(int[] t1, int[] t2) {\n"
            "        int[] s1 = t1.clone();\n"
            "        int[] s2 = t2.clone();\n"
            "        Arrays.sort(s1);\n"
            "        Arrays.sort(s2);\n"
            "        double r = (double) s1[0] / s2[0];\n"
            "        return Math.abs((double) s1[1] / s2[1] - r) < 1e-9\n"
            "            && Math.abs((double) s1[2] / s2[2] - r) < 1e-9;\n"
            "    }\n"
            "}\n"
        ),
        "tags": ["geometry", "math", "easy"],
        "metadata": {"original_index": 2, "entry_point": "areSimilar", "difficulty_tier": "easy"},
        "test_cases": [
            {
                "test_id": "MBPP_2_test_1",
                "input_data": {"t1": [3, 4, 5], "t2": [6, 8, 10]},
                "expected_output": True,
                "is_hidden": False,
                "weight": 1.0,
                "timeout_seconds": 10.0,
            },
            {
                "test_id": "MBPP_2_test_2",
                "input_data": {"t1": [3, 4, 5], "t2": [6, 8, 11]},
                "expected_output": False,
                "is_hidden": False,
                "weight": 1.0,
                "timeout_seconds": 10.0,
            },
            {
                "test_id": "MBPP_2_test_3",
                "input_data": {"t1": [1, 1, 1], "t2": [7, 7, 7]},
                "expected_output": True,
                "is_hidden": True,
                "weight": 1.0,
                "timeout_seconds": 10.0,
            },
        ],
        "junit_code": (
            "import org.junit.Test;\n"
            "import static org.junit.Assert.*;\n\n"
            "public class SimilarTrianglesTest {\n"
            "    @Test\n"
            "    public void testSimilar() {\n"
            "        assertTrue(SimilarTriangles.areSimilar(\n"
            "            new int[]{3,4,5}, new int[]{6,8,10}));\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testNotSimilar() {\n"
            "        assertFalse(SimilarTriangles.areSimilar(\n"
            "            new int[]{3,4,5}, new int[]{6,8,11}));\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testEquilateral() {\n"
            "        assertTrue(SimilarTriangles.areSimilar(\n"
            "            new int[]{1,1,1}, new int[]{7,7,7}));\n"
            "    }\n"
            "}\n"
        ),
    },
    {
        "problem_id": "MBPP_3",
        "title": "Remove Duplicates from Sorted Array",
        "description": (
            "Write a Java function that takes a sorted array of integers "
            "and removes duplicates in-place such that each element appears "
            "only once. Return the new length."
        ),
        "difficulty": "easy",
        "language": "java",
        "reference_solution": (
            "public class RemoveDuplicates {\n"
            "    public static int removeDuplicates(int[] nums) {\n"
            "        if (nums.length == 0) return 0;\n"
            "        int i = 0;\n"
            "        for (int j = 1; j < nums.length; j++) {\n"
            "            if (nums[j] != nums[i]) {\n"
            "                i++;\n"
            "                nums[i] = nums[j];\n"
            "            }\n"
            "        }\n"
            "        return i + 1;\n"
            "    }\n"
            "}\n"
        ),
        "tags": ["array", "two-pointers", "easy"],
        "metadata": {"original_index": 3, "entry_point": "removeDuplicates", "difficulty_tier": "easy"},
        "test_cases": [
            {
                "test_id": "MBPP_3_test_1",
                "input_data": {"nums": [1, 1, 2]},
                "expected_output": 2,
                "is_hidden": False,
                "weight": 1.0,
                "timeout_seconds": 10.0,
            },
            {
                "test_id": "MBPP_3_test_2",
                "input_data": {"nums": [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]},
                "expected_output": 5,
                "is_hidden": False,
                "weight": 1.0,
                "timeout_seconds": 10.0,
            },
            {
                "test_id": "MBPP_3_test_3",
                "input_data": {"nums": []},
                "expected_output": 0,
                "is_hidden": True,
                "weight": 1.0,
                "timeout_seconds": 10.0,
            },
        ],
        "junit_code": (
            "import org.junit.Test;\n"
            "import static org.junit.Assert.*;\n\n"
            "public class RemoveDuplicatesTest {\n"
            "    @Test\n"
            "    public void testBasic() {\n"
            "        assertEquals(2, RemoveDuplicates.removeDuplicates(\n"
            "            new int[]{1,1,2}));\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testLonger() {\n"
            "        assertEquals(5, RemoveDuplicates.removeDuplicates(\n"
            "            new int[]{0,0,1,1,1,2,2,3,3,4}));\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testEmpty() {\n"
            "        assertEquals(0, RemoveDuplicates.removeDuplicates(\n"
            "            new int[]{}));\n"
            "    }\n"
            "}\n"
        ),
    },
]


class MBPPDataset(Dataset):
    """
    Adapter for the MBPP-Java benchmark (200 problems, stratified).

    Stratification: 80 Easy / 80 Medium / 40 Hard.

    Parameters
    ----------
    data_dir : Path | str
        Directory containing ``mbpp/problems.jsonl``.
    """

    DATASET_NAME = "mbpp-java"
    EXPECTED_SIZE = 200
    DIFFICULTY_DISTRIBUTION = {"easy": 80, "medium": 80, "hard": 40}

    def __init__(self, data_dir: Path | str = "data") -> None:
        super().__init__(
            name=self.DATASET_NAME,
            data_dir=data_dir,
            language="java",
        )
        self._junit_code: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def load_problems(self) -> list[Problem]:
        builtin = self._load_builtin_examples()
        jsonl_path = self.data_dir / "mbpp" / "problems.jsonl"
        if jsonl_path.exists():
            return builtin + self._load_from_jsonl(jsonl_path)
        return builtin

    def get_test_suite(self, problem_id: str) -> TestSuite:
        self._ensure_loaded()
        if problem_id not in self._test_suites:
            raise KeyError(f"No test suite for '{problem_id}' in {self.name}")
        return self._test_suites[problem_id]

    def format_prompt(self, problem_id: str) -> str:
        problem = self.get_problem(problem_id)
        suite = self.get_test_suite(problem_id)

        visible_tests = [tc for tc in suite.cases if not tc.is_hidden]
        test_examples = ""
        for tc in visible_tests[:3]:
            test_examples += (
                f"  Input:    {json.dumps(tc.input_data)}\n"
                f"  Expected: {json.dumps(tc.expected_output)}\n\n"
            )

        ref = problem.reference_solution or ""
        class_match = re.search(r"public\s+class\s+(\w+)", ref)
        class_name = class_match.group(1) if class_match else "Solution"
        method_match = re.search(
            r"(public\s+static\s+\w[\w<>, \[\]]*\s+\w+\s*\([^)]*\))\s*\{",
            ref,
        )
        if method_match:
            stub = (
                f"public class {class_name} {{\n"
                f"    {method_match.group(1)} {{\n"
                f"        // TODO: implement\n"
                f"    }}\n"
                f"}}"
            )
        else:
            stub = f"public class {class_name} {{ }}"

        prompt = (
            f"Write a complete Java class that solves the following problem.\n\n"
            f"Problem: {problem.title}\n"
            f"Description:\n{problem.description}\n\n"
            f"Starting code (fill in the method body):\n"
            f"```java\n{stub}\n```\n\n"
            f"Test examples:\n{test_examples}"
            f"Return ONLY the complete Java source code, no explanations.\n"
        )
        return prompt

    def verify_solution(
        self,
        problem_id: str,
        generated_code: str,
    ) -> list[VerificationResult]:
        problem = self.get_problem(problem_id)
        suite = self.get_test_suite(problem_id)
        results: list[VerificationResult] = []

        entry_point = problem.metadata.get("entry_point", "")
        has_method = entry_point in generated_code if entry_point else True
        has_class = "class " in generated_code

        for tc in suite.cases:
            if not has_class:
                results.append(self._make_verification_result(
                    test_id=tc.test_id, passed=False,
                    error_message="No class definition found",
                ))
            elif not has_method:
                results.append(self._make_verification_result(
                    test_id=tc.test_id, passed=False,
                    error_message=f"Method '{entry_point}' not found",
                ))
            else:
                results.append(self._make_verification_result(
                    test_id=tc.test_id, passed=False,
                    error_message="PENDING_SANDBOX_EXECUTION",
                ))

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_junit_code(self, problem_id: str) -> str:
        self._ensure_loaded()
        if problem_id not in self._junit_code:
            raise KeyError(f"No JUnit code for '{problem_id}'")
        return self._junit_code[problem_id]

    def get_problems_by_difficulty(self, difficulty: str) -> list[Problem]:
        """Filter loaded problems by difficulty tier."""
        self._ensure_loaded()
        return [
            p for p in self._problems.values()
            if p.difficulty == difficulty
        ]

    def _load_builtin_examples(self) -> list[Problem]:
        problems: list[Problem] = []
        for entry in _EXAMPLE_PROBLEMS:
            prob = Problem(
                problem_id=entry["problem_id"],
                title=entry["title"],
                description=entry["description"],
                difficulty=entry["difficulty"],
                language=entry["language"],
                reference_solution=entry["reference_solution"],
                tags=entry["tags"],
                metadata=entry["metadata"],
            )
            problems.append(prob)

            cases = [TestCase(**tc) for tc in entry["test_cases"]]
            suite = TestSuite(
                suite_id=f"suite_{entry['problem_id']}",
                problem_id=entry["problem_id"],
                cases=cases,
            )
            self._register_test_suite(suite)
            self._junit_code[entry["problem_id"]] = entry["junit_code"]

        return problems

    def _load_from_jsonl(self, path: Path) -> list[Problem]:
        problems: list[Problem] = []
        with open(path) as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping invalid JSON at line %d: %s", line_no, exc)
                    continue

                prob = Problem(
                    problem_id=entry["problem_id"],
                    title=entry.get("title", ""),
                    description=entry.get("description", ""),
                    difficulty=entry.get("difficulty", "medium"),
                    language="java",
                    reference_solution=entry.get("reference_solution", ""),
                    tags=entry.get("tags", []),
                    metadata=entry.get("metadata", {}),
                )
                problems.append(prob)

                if "test_cases" in entry:
                    cases = [TestCase(**tc) for tc in entry["test_cases"]]
                    suite = TestSuite(
                        suite_id=f"suite_{entry['problem_id']}",
                        problem_id=entry["problem_id"],
                        cases=cases,
                    )
                    self._register_test_suite(suite)

                if "junit_code" in entry:
                    self._junit_code[entry["problem_id"]] = entry["junit_code"]

        logger.info("Loaded %d MBPP problems from %s", len(problems), path)
        return problems
