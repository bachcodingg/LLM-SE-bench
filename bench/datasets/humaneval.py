"""
bench.datasets.humaneval — HumanEval-Java dataset adapter (164 problems).

Adapts the HumanEval benchmark (Chen et al., 2021) for Java.  Each problem
consists of a method signature, a docstring, and a set of JUnit test cases.
The LLM is asked to complete the method body.

This module ships with three built-in example problems for development and
testing.  The full 164-problem dataset is loaded from JSONL files placed in
``data_dir/humaneval/``.
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
# Built-in example problems (3 of 164)
# ======================================================================

_EXAMPLE_PROBLEMS: list[dict[str, Any]] = [
    {
        "problem_id": "HumanEval_0",
        "title": "Has Close Elements",
        "description": (
            "Check if in given list of numbers, are any two numbers closer "
            "to each other than given threshold.\n\n"
            ">>> hasCloseElements(Arrays.asList(1.0, 2.0, 3.9, 4.0, 5.0, 2.2), 0.3)\n"
            "true\n"
            ">>> hasCloseElements(Arrays.asList(1.0, 2.0, 3.9, 4.0, 5.0, 2.2), 0.05)\n"
            "false"
        ),
        "difficulty": "easy",
        "language": "java",
        "reference_solution": (
            "import java.util.List;\n\n"
            "public class HasCloseElements {\n"
            "    public static boolean hasCloseElements(List<Double> numbers, double threshold) {\n"
            "        for (int i = 0; i < numbers.size(); i++) {\n"
            "            for (int j = i + 1; j < numbers.size(); j++) {\n"
            "                if (Math.abs(numbers.get(i) - numbers.get(j)) < threshold) {\n"
            "                    return true;\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "        return false;\n"
            "    }\n"
            "}\n"
        ),
        "tags": ["list", "math", "easy"],
        "metadata": {"original_index": 0, "entry_point": "hasCloseElements"},
        "test_cases": [
            {
                "test_id": "HumanEval_0_test_1",
                "input_data": {"numbers": [1.0, 2.0, 3.9, 4.0, 5.0, 2.2], "threshold": 0.3},
                "expected_output": True,
                "is_hidden": False,
                "weight": 1.0,
                "timeout_seconds": 10.0,
            },
            {
                "test_id": "HumanEval_0_test_2",
                "input_data": {"numbers": [1.0, 2.0, 3.9, 4.0, 5.0, 2.2], "threshold": 0.05},
                "expected_output": False,
                "is_hidden": False,
                "weight": 1.0,
                "timeout_seconds": 10.0,
            },
            {
                "test_id": "HumanEval_0_test_3",
                "input_data": {"numbers": [1.0, 2.0, 5.9, 4.0, 5.0], "threshold": 0.95},
                "expected_output": True,
                "is_hidden": True,
                "weight": 1.0,
                "timeout_seconds": 10.0,
            },
            {
                "test_id": "HumanEval_0_test_4",
                "input_data": {"numbers": [1.0, 2.0, 5.9, 4.0, 5.0], "threshold": 0.8},
                "expected_output": False,
                "is_hidden": True,
                "weight": 1.0,
                "timeout_seconds": 10.0,
            },
        ],
        "junit_code": (
            "import org.junit.Test;\n"
            "import static org.junit.Assert.*;\n"
            "import java.util.Arrays;\n\n"
            "public class HasCloseElementsTest {\n"
            "    @Test\n"
            "    public void testCloseElements() {\n"
            "        assertTrue(HasCloseElements.hasCloseElements(\n"
            "            Arrays.asList(1.0, 2.0, 3.9, 4.0, 5.0, 2.2), 0.3));\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testNotClose() {\n"
            "        assertFalse(HasCloseElements.hasCloseElements(\n"
            "            Arrays.asList(1.0, 2.0, 3.9, 4.0, 5.0, 2.2), 0.05));\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testEdgeClose() {\n"
            "        assertTrue(HasCloseElements.hasCloseElements(\n"
            "            Arrays.asList(1.0, 2.0, 5.9, 4.0, 5.0), 0.95));\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testEdgeNotClose() {\n"
            "        assertFalse(HasCloseElements.hasCloseElements(\n"
            "            Arrays.asList(1.0, 2.0, 5.9, 4.0, 5.0), 0.8));\n"
            "    }\n"
            "}\n"
        ),
    },
    {
        "problem_id": "HumanEval_1",
        "title": "Separate Paren Groups",
        "description": (
            "Input to this function is a string containing multiple groups of "
            "nested parentheses. Your goal is to separate those groups into "
            "separate strings and return the list of those.\n"
            "Separate groups are balanced (each open brace is properly closed) "
            "and not nested within each other.\n"
            "Ignore any spaces in the input string.\n\n"
            ">>> separateParenGroups(\"( ) (( )) (( )( ))\")\n"
            "[\"()\", \"(())\", \"(()())\"]\n"
        ),
        "difficulty": "medium",
        "language": "java",
        "reference_solution": (
            "import java.util.ArrayList;\nimport java.util.List;\n\n"
            "public class SeparateParenGroups {\n"
            "    public static List<String> separateParenGroups(String parenString) {\n"
            "        List<String> result = new ArrayList<>();\n"
            "        StringBuilder current = new StringBuilder();\n"
            "        int depth = 0;\n"
            "        for (char c : parenString.toCharArray()) {\n"
            "            if (c == '(') {\n"
            "                depth++;\n"
            "                current.append(c);\n"
            "            } else if (c == ')') {\n"
            "                depth--;\n"
            "                current.append(c);\n"
            "                if (depth == 0) {\n"
            "                    result.add(current.toString());\n"
            "                    current = new StringBuilder();\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "        return result;\n"
            "    }\n"
            "}\n"
        ),
        "tags": ["string", "stack", "medium"],
        "metadata": {"original_index": 1, "entry_point": "separateParenGroups"},
        "test_cases": [
            {
                "test_id": "HumanEval_1_test_1",
                "input_data": {"parenString": "(()()) ((())) () ((())()())"},
                "expected_output": ["(()())", "((()))", "()", "((())()())"],
                "is_hidden": False,
                "weight": 1.0,
                "timeout_seconds": 10.0,
            },
            {
                "test_id": "HumanEval_1_test_2",
                "input_data": {"parenString": "() (()) ((())) (((())))"},
                "expected_output": ["()", "(())", "((()))", "(((())))"],
                "is_hidden": False,
                "weight": 1.0,
                "timeout_seconds": 10.0,
            },
            {
                "test_id": "HumanEval_1_test_3",
                "input_data": {"parenString": "(()(()))"},
                "expected_output": ["(()(()))"],
                "is_hidden": True,
                "weight": 1.0,
                "timeout_seconds": 10.0,
            },
        ],
        "junit_code": (
            "import org.junit.Test;\n"
            "import static org.junit.Assert.*;\n"
            "import java.util.Arrays;\n\n"
            "public class SeparateParenGroupsTest {\n"
            "    @Test\n"
            "    public void testMultipleGroups() {\n"
            "        assertEquals(\n"
            "            Arrays.asList(\"(()())\", \"((()))\", \"()\", \"((())()())\"),\n"
            "            SeparateParenGroups.separateParenGroups(\n"
            "                \"(()()) ((())) () ((())()())\"));\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testNested() {\n"
            "        assertEquals(\n"
            "            Arrays.asList(\"()\", \"(())\", \"((()))\", \"(((())))\"),\n"
            "            SeparateParenGroups.separateParenGroups(\n"
            "                \"() (()) ((())) (((())))\"));\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testSingleGroup() {\n"
            "        assertEquals(\n"
            "            Arrays.asList(\"(()(()))\"),\n"
            "            SeparateParenGroups.separateParenGroups(\"(()(()))\"));\n"
            "    }\n"
            "}\n"
        ),
    },
    {
        "problem_id": "HumanEval_4",
        "title": "Mean Absolute Deviation",
        "description": (
            "For a given list of input numbers, calculate Mean Absolute "
            "Deviation around the mean of this dataset.\n"
            "Mean Absolute Deviation is the average absolute difference "
            "between each element and a centerpoint (mean in this case):\n"
            "MAD = average | x - x_mean |\n\n"
            ">>> meanAbsoluteDeviation(Arrays.asList(1.0, 2.0, 3.0, 4.0))\n"
            "1.0\n"
        ),
        "difficulty": "easy",
        "language": "java",
        "reference_solution": (
            "import java.util.List;\n\n"
            "public class MeanAbsoluteDeviation {\n"
            "    public static double meanAbsoluteDeviation(List<Double> numbers) {\n"
            "        double mean = numbers.stream()\n"
            "            .mapToDouble(Double::doubleValue)\n"
            "            .average()\n"
            "            .orElse(0.0);\n"
            "        return numbers.stream()\n"
            "            .mapToDouble(x -> Math.abs(x - mean))\n"
            "            .average()\n"
            "            .orElse(0.0);\n"
            "    }\n"
            "}\n"
        ),
        "tags": ["math", "statistics", "easy"],
        "metadata": {"original_index": 4, "entry_point": "meanAbsoluteDeviation"},
        "test_cases": [
            {
                "test_id": "HumanEval_4_test_1",
                "input_data": {"numbers": [1.0, 2.0, 3.0, 4.0]},
                "expected_output": 1.0,
                "is_hidden": False,
                "weight": 1.0,
                "timeout_seconds": 10.0,
            },
            {
                "test_id": "HumanEval_4_test_2",
                "input_data": {"numbers": [1.0, 2.0, 3.0, 4.0, 5.0]},
                "expected_output": 1.2,
                "is_hidden": False,
                "weight": 1.0,
                "timeout_seconds": 10.0,
            },
            {
                "test_id": "HumanEval_4_test_3",
                "input_data": {"numbers": [5.0, 5.0, 5.0, 5.0]},
                "expected_output": 0.0,
                "is_hidden": True,
                "weight": 1.0,
                "timeout_seconds": 10.0,
            },
        ],
        "junit_code": (
            "import org.junit.Test;\n"
            "import static org.junit.Assert.*;\n"
            "import java.util.Arrays;\n\n"
            "public class MeanAbsoluteDeviationTest {\n"
            "    @Test\n"
            "    public void testBasic() {\n"
            "        assertEquals(1.0,\n"
            "            MeanAbsoluteDeviation.meanAbsoluteDeviation(\n"
            "                Arrays.asList(1.0, 2.0, 3.0, 4.0)), 1e-6);\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testOddCount() {\n"
            "        assertEquals(1.2,\n"
            "            MeanAbsoluteDeviation.meanAbsoluteDeviation(\n"
            "                Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0)), 1e-6);\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testAllSame() {\n"
            "        assertEquals(0.0,\n"
            "            MeanAbsoluteDeviation.meanAbsoluteDeviation(\n"
            "                Arrays.asList(5.0, 5.0, 5.0, 5.0)), 1e-6);\n"
            "    }\n"
            "}\n"
        ),
    },
]


class HumanEvalDataset(Dataset):
    """
    Adapter for the HumanEval-Java benchmark (164 code-generation problems).

    Parameters
    ----------
    data_dir : Path | str
        Directory containing ``humaneval/problems.jsonl``.  When the file
        is absent the adapter falls back to three built-in example problems.
    """

    DATASET_NAME = "humaneval-java"
    EXPECTED_SIZE = 164

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
        """Load built-in examples plus any additional problems from JSONL."""
        builtin = self._load_builtin_examples()
        jsonl_path = self.data_dir / "humaneval" / "problems.jsonl"
        if jsonl_path.exists():
            return builtin + self._load_from_jsonl(jsonl_path)
        return builtin

    def get_test_suite(self, problem_id: str) -> TestSuite:
        """Return the JUnit test suite for *problem_id*."""
        self._ensure_loaded()
        if problem_id not in self._test_suites:
            raise KeyError(
                f"No test suite for '{problem_id}' in {self.name}"
            )
        return self._test_suites[problem_id]

    def format_prompt(self, problem_id: str) -> str:
        """
        Build a zero-shot code-generation prompt for *problem_id*.

        The prompt includes the method signature, docstring, and visible
        test cases formatted as examples.
        """
        problem = self.get_problem(problem_id)
        suite = self.get_test_suite(problem_id)

        visible_tests = [tc for tc in suite.cases if not tc.is_hidden]
        test_examples = ""
        for tc in visible_tests[:3]:
            test_examples += (
                f"  Input:    {json.dumps(tc.input_data)}\n"
                f"  Expected: {json.dumps(tc.expected_output)}\n\n"
            )

        # Extract class name and method stub from reference solution.
        # The JUnit tests call static methods, so we show the exact stub.
        ref = problem.reference_solution or ""
        class_match = re.search(r"public\s+class\s+(\w+)", ref)
        class_name = class_match.group(1) if class_match else "Solution"

        # Build method stub: class header + method signature (without body)
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
            f"Complete the following Java method.\n\n"
            f"Problem: {problem.title}\n"
            f"Description:\n{problem.description}\n\n"
            f"Starting code (fill in the method body):\n"
            f"```java\n{stub}\n```\n\n"
            f"Examples:\n{test_examples}"
            f"Return ONLY the complete Java source code, no explanations.\n"
        )
        return prompt

    def verify_solution(
        self,
        problem_id: str,
        generated_code: str,
    ) -> list[VerificationResult]:
        """
        Verify generated code against the test suite.

        In production this delegates to ``DockerSandbox``.  For unit-level
        checking it performs a lightweight structural validation.
        """
        problem = self.get_problem(problem_id)
        suite = self.get_test_suite(problem_id)
        results: list[VerificationResult] = []

        # Lightweight structural checks (sandbox does the real work)
        entry_point = problem.metadata.get("entry_point", "")
        has_method = entry_point in generated_code if entry_point else True
        has_class = "class " in generated_code

        for tc in suite.cases:
            if not has_class:
                results.append(self._make_verification_result(
                    test_id=tc.test_id,
                    passed=False,
                    error_message="Generated code does not contain a class definition",
                ))
            elif not has_method:
                results.append(self._make_verification_result(
                    test_id=tc.test_id,
                    passed=False,
                    error_message=f"Method '{entry_point}' not found in generated code",
                ))
            else:
                # Mark as needing sandbox execution
                results.append(self._make_verification_result(
                    test_id=tc.test_id,
                    passed=False,
                    error_message="PENDING_SANDBOX_EXECUTION",
                ))

        return results

    # ------------------------------------------------------------------
    # JUnit helpers
    # ------------------------------------------------------------------

    def get_junit_code(self, problem_id: str) -> str:
        """Return the JUnit test class source for sandbox execution."""
        self._ensure_loaded()
        if problem_id not in self._junit_code:
            raise KeyError(f"No JUnit code for '{problem_id}'")
        return self._junit_code[problem_id]

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_builtin_examples(self) -> list[Problem]:
        """Construct Problem and TestSuite objects from built-in data."""
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
        """Parse the full 164-problem JSONL file."""
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

        logger.info("Loaded %d HumanEval problems from %s", len(problems), path)
        return problems
