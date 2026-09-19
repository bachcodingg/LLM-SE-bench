"""
bench.datasets.defects4j — Defects4J dataset adapter (50 real-world bugs).

Wraps 50 bugs from five Apache/open-source projects (10 each from
JFreeChart, Commons-Lang, Commons-Math, Mockito, Closure Compiler).
Each bug has a buggy source file, a failing test, and a developer-written
fix.  The LLM is asked to produce a patch that makes the failing test pass.

Ships with three built-in example bugs for development.
"""

from __future__ import annotations

import json
import logging
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
# Built-in example bugs (3 of 50)
# ======================================================================

_EXAMPLE_BUGS: list[dict[str, Any]] = [
    {
        "problem_id": "D4J_Lang_1",
        "title": "Commons-Lang: NumberUtils.createNumber NPE",
        "description": (
            "Bug in Apache Commons Lang ``NumberUtils.createNumber(String)``.\n"
            "The method throws a ``NullPointerException`` when the input "
            "string ends with 'l' or 'L' and the numeric portion is empty "
            "or invalid.\n\n"
            "Fix the bug so the method correctly throws a "
            "``NumberFormatException`` instead of NPE for invalid inputs."
        ),
        "difficulty": "medium",
        "language": "java",
        "reference_solution": (
            "// Fix: add null/empty check before Long.decode call\n"
            "public class NumberUtils {\n"
            "    public static Number createNumber(String str) {\n"
            "        if (str == null) {\n"
            "            return null;\n"
            "        }\n"
            "        if (str.isEmpty()) {\n"
            "            throw new NumberFormatException(\"Empty string\");\n"
            "        }\n"
            "        if (str.endsWith(\"l\") || str.endsWith(\"L\")) {\n"
            "            String numPart = str.substring(0, str.length() - 1);\n"
            "            if (numPart.isEmpty()) {\n"
            "                throw new NumberFormatException(\n"
            "                    \"Invalid number: \" + str);\n"
            "            }\n"
            "            return Long.decode(numPart);\n"
            "        }\n"
            "        return Double.valueOf(str);\n"
            "    }\n"
            "}\n"
        ),
        "tags": ["bug-fix", "commons-lang", "null-check"],
        "metadata": {
            "project": "Lang",
            "bug_id": 1,
            "buggy_file": "src/main/java/org/apache/commons/lang3/math/NumberUtils.java",
            "failing_test": "org.apache.commons.lang3.math.NumberUtilsTest#testCreateNumber",
            "entry_point": "createNumber",
        },
        "buggy_code": (
            "public class NumberUtils {\n"
            "    public static Number createNumber(String str) {\n"
            "        if (str == null) {\n"
            "            return null;\n"
            "        }\n"
            "        if (str.isEmpty()) {\n"
            "            throw new NumberFormatException(\"Empty string\");\n"
            "        }\n"
            "        if (str.endsWith(\"l\") || str.endsWith(\"L\")) {\n"
            "            // BUG: does not check if numPart is empty\n"
            "            return Long.decode(str.substring(0, str.length() - 1));\n"
            "        }\n"
            "        return Double.valueOf(str);\n"
            "    }\n"
            "}\n"
        ),
        "test_cases": [
            {
                "test_id": "D4J_Lang_1_test_1",
                "input_data": {"str": "l"},
                "expected_output": "NumberFormatException",
                "is_hidden": False,
                "weight": 2.0,
                "timeout_seconds": 15.0,
            },
            {
                "test_id": "D4J_Lang_1_test_2",
                "input_data": {"str": "123L"},
                "expected_output": 123,
                "is_hidden": False,
                "weight": 1.0,
                "timeout_seconds": 15.0,
            },
            {
                "test_id": "D4J_Lang_1_test_3",
                "input_data": {"str": "3.14"},
                "expected_output": 3.14,
                "is_hidden": True,
                "weight": 1.0,
                "timeout_seconds": 15.0,
            },
        ],
        "junit_code": (
            "import org.junit.Test;\n"
            "import static org.junit.Assert.*;\n\n"
            "public class NumberUtilsTest {\n"
            "    @Test(expected = NumberFormatException.class)\n"
            "    public void testCreateNumberJustL() {\n"
            "        NumberUtils.createNumber(\"l\");\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testCreateNumberValidLong() {\n"
            "        assertEquals(123L, NumberUtils.createNumber(\"123L\"));\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testCreateNumberDouble() {\n"
            "        assertEquals(3.14, NumberUtils.createNumber(\"3.14\"));\n"
            "    }\n"
            "}\n"
        ),
    },
    {
        "problem_id": "D4J_Math_2",
        "title": "Commons-Math: HypergeometricDistribution sample overflow",
        "description": (
            "Bug in Apache Commons Math ``HypergeometricDistribution``.\n"
            "The ``sample()`` method uses integer arithmetic for an "
            "intermediate multiplication that can overflow for large "
            "population sizes, producing incorrect probabilities.\n\n"
            "Fix: cast to long before the multiplication."
        ),
        "difficulty": "hard",
        "language": "java",
        "reference_solution": (
            "public class HypergeometricDistribution {\n"
            "    private int populationSize;\n"
            "    private int numberOfSuccesses;\n"
            "    private int sampleSize;\n\n"
            "    public HypergeometricDistribution(int popSize, int successes, int sample) {\n"
            "        this.populationSize = popSize;\n"
            "        this.numberOfSuccesses = successes;\n"
            "        this.sampleSize = sample;\n"
            "    }\n\n"
            "    public double upperCumulativeProbability(int x) {\n"
            "        // Fix: cast to long to prevent overflow\n"
            "        long numerator = (long) numberOfSuccesses * sampleSize;\n"
            "        double mean = (double) numerator / populationSize;\n"
            "        if (x < mean) return 1.0;\n"
            "        return 0.5;\n"
            "    }\n"
            "}\n"
        ),
        "tags": ["bug-fix", "commons-math", "overflow"],
        "metadata": {
            "project": "Math",
            "bug_id": 2,
            "buggy_file": "src/main/java/org/apache/commons/math3/distribution/HypergeometricDistribution.java",
            "failing_test": "org.apache.commons.math3.distribution.HypergeometricDistributionTest",
            "entry_point": "upperCumulativeProbability",
        },
        "buggy_code": (
            "public class HypergeometricDistribution {\n"
            "    private int populationSize;\n"
            "    private int numberOfSuccesses;\n"
            "    private int sampleSize;\n\n"
            "    public HypergeometricDistribution(int popSize, int successes, int sample) {\n"
            "        this.populationSize = popSize;\n"
            "        this.numberOfSuccesses = successes;\n"
            "        this.sampleSize = sample;\n"
            "    }\n\n"
            "    public double upperCumulativeProbability(int x) {\n"
            "        // BUG: integer overflow on multiplication\n"
            "        int numerator = numberOfSuccesses * sampleSize;\n"
            "        double mean = (double) numerator / populationSize;\n"
            "        if (x < mean) return 1.0;\n"
            "        return 0.5;\n"
            "    }\n"
            "}\n"
        ),
        "test_cases": [
            {
                "test_id": "D4J_Math_2_test_1",
                "input_data": {"popSize": 3456, "successes": 2400, "sample": 1500, "x": 900},
                "expected_output": 1.0,
                "is_hidden": False,
                "weight": 2.0,
                "timeout_seconds": 15.0,
            },
            {
                "test_id": "D4J_Math_2_test_2",
                "input_data": {"popSize": 100000, "successes": 60000, "sample": 50000, "x": 40000},
                "expected_output": 0.5,
                "is_hidden": False,
                "weight": 1.0,
                "timeout_seconds": 15.0,
            },
            {
                "test_id": "D4J_Math_2_test_3",
                "input_data": {"popSize": 10, "successes": 5, "sample": 5, "x": 1},
                "expected_output": 1.0,
                "is_hidden": True,
                "weight": 1.0,
                "timeout_seconds": 15.0,
            },
        ],
        "junit_code": (
            "import org.junit.Test;\n"
            "import static org.junit.Assert.*;\n\n"
            "public class HypergeometricDistributionTest {\n"
            "    @Test\n"
            "    public void testLargePopulation() {\n"
            "        HypergeometricDistribution d =\n"
            "            new HypergeometricDistribution(3456, 2400, 1500);\n"
            "        assertEquals(1.0, d.upperCumulativeProbability(900), 1e-6);\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testOverflowCase() {\n"
            "        HypergeometricDistribution d =\n"
            "            new HypergeometricDistribution(100000, 60000, 50000);\n"
            "        assertEquals(0.5, d.upperCumulativeProbability(40000), 1e-6);\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testSmallPopulation() {\n"
            "        HypergeometricDistribution d =\n"
            "            new HypergeometricDistribution(10, 5, 5);\n"
            "        assertEquals(1.0, d.upperCumulativeProbability(1), 1e-6);\n"
            "    }\n"
            "}\n"
        ),
    },
    {
        "problem_id": "D4J_Chart_3",
        "title": "JFreeChart: TimeSeries createCopy range error",
        "description": (
            "Bug in JFreeChart ``TimeSeries.createCopy(int, int)``.\n"
            "When creating a copy of a time series for a sub-range, the "
            "method fails to handle the case where ``start == end`` "
            "correctly, resulting in an ``IllegalArgumentException``.\n\n"
            "Fix: allow ``start == end`` to return a series with one item."
        ),
        "difficulty": "medium",
        "language": "java",
        "reference_solution": (
            "import java.util.ArrayList;\nimport java.util.List;\n\n"
            "public class TimeSeries {\n"
            "    private List<Double> data = new ArrayList<>();\n"
            "    private String name;\n\n"
            "    public TimeSeries(String name) { this.name = name; }\n\n"
            "    public void add(double value) { data.add(value); }\n\n"
            "    public int getItemCount() { return data.size(); }\n\n"
            "    public double getValue(int index) { return data.get(index); }\n\n"
            "    public TimeSeries createCopy(int start, int end) {\n"
            "        // Fix: use <= instead of <\n"
            "        if (start < 0 || end >= data.size() || start > end) {\n"
            "            throw new IllegalArgumentException(\"Invalid range\");\n"
            "        }\n"
            "        TimeSeries copy = new TimeSeries(this.name + \"_copy\");\n"
            "        for (int i = start; i <= end; i++) {\n"
            "            copy.add(data.get(i));\n"
            "        }\n"
            "        return copy;\n"
            "    }\n"
            "}\n"
        ),
        "tags": ["bug-fix", "jfreechart", "off-by-one"],
        "metadata": {
            "project": "Chart",
            "bug_id": 3,
            "buggy_file": "source/org/jfree/data/time/TimeSeries.java",
            "failing_test": "org.jfree.data.time.TimeSeriesTest#testCreateCopy",
            "entry_point": "createCopy",
        },
        "buggy_code": (
            "import java.util.ArrayList;\nimport java.util.List;\n\n"
            "public class TimeSeries {\n"
            "    private List<Double> data = new ArrayList<>();\n"
            "    private String name;\n\n"
            "    public TimeSeries(String name) { this.name = name; }\n\n"
            "    public void add(double value) { data.add(value); }\n\n"
            "    public int getItemCount() { return data.size(); }\n\n"
            "    public double getValue(int index) { return data.get(index); }\n\n"
            "    public TimeSeries createCopy(int start, int end) {\n"
            "        // BUG: start < end should be start > end; also uses < end instead of <= end\n"
            "        if (start < 0 || end >= data.size() || start < end) {\n"
            "            throw new IllegalArgumentException(\"Invalid range\");\n"
            "        }\n"
            "        TimeSeries copy = new TimeSeries(this.name + \"_copy\");\n"
            "        for (int i = start; i < end; i++) {\n"
            "            copy.add(data.get(i));\n"
            "        }\n"
            "        return copy;\n"
            "    }\n"
            "}\n"
        ),
        "test_cases": [
            {
                "test_id": "D4J_Chart_3_test_1",
                "input_data": {"series": [1.0, 2.0, 3.0, 4.0, 5.0], "start": 1, "end": 3},
                "expected_output": {"count": 3, "values": [2.0, 3.0, 4.0]},
                "is_hidden": False,
                "weight": 1.0,
                "timeout_seconds": 15.0,
            },
            {
                "test_id": "D4J_Chart_3_test_2",
                "input_data": {"series": [10.0, 20.0, 30.0], "start": 1, "end": 1},
                "expected_output": {"count": 1, "values": [20.0]},
                "is_hidden": False,
                "weight": 2.0,
                "timeout_seconds": 15.0,
            },
            {
                "test_id": "D4J_Chart_3_test_3",
                "input_data": {"series": [7.0], "start": 0, "end": 0},
                "expected_output": {"count": 1, "values": [7.0]},
                "is_hidden": True,
                "weight": 1.0,
                "timeout_seconds": 15.0,
            },
        ],
        "junit_code": (
            "import org.junit.Test;\n"
            "import static org.junit.Assert.*;\n\n"
            "public class TimeSeriesTest {\n"
            "    @Test\n"
            "    public void testCreateCopyRange() {\n"
            "        TimeSeries ts = new TimeSeries(\"test\");\n"
            "        ts.add(1.0); ts.add(2.0); ts.add(3.0);\n"
            "        ts.add(4.0); ts.add(5.0);\n"
            "        TimeSeries copy = ts.createCopy(1, 3);\n"
            "        assertEquals(3, copy.getItemCount());\n"
            "        assertEquals(2.0, copy.getValue(0), 1e-6);\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testCreateCopySingleItem() {\n"
            "        TimeSeries ts = new TimeSeries(\"test\");\n"
            "        ts.add(10.0); ts.add(20.0); ts.add(30.0);\n"
            "        TimeSeries copy = ts.createCopy(1, 1);\n"
            "        assertEquals(1, copy.getItemCount());\n"
            "        assertEquals(20.0, copy.getValue(0), 1e-6);\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testCreateCopySingleElement() {\n"
            "        TimeSeries ts = new TimeSeries(\"test\");\n"
            "        ts.add(7.0);\n"
            "        TimeSeries copy = ts.createCopy(0, 0);\n"
            "        assertEquals(1, copy.getItemCount());\n"
            "        assertEquals(7.0, copy.getValue(0), 1e-6);\n"
            "    }\n"
            "}\n"
        ),
    },
]


class Defects4JDataset(Dataset):
    """
    Adapter for the Defects4J v2.0 bug-fixing benchmark (50 bugs).

    10 bugs each from: JFreeChart, Commons-Lang, Commons-Math,
    Mockito, Closure Compiler.

    Parameters
    ----------
    data_dir : Path | str
        Directory containing ``defects4j/bugs.jsonl``.
    """

    DATASET_NAME = "defects4j"
    EXPECTED_SIZE = 50
    PROJECTS = ["Chart", "Lang", "Math", "Mockito", "Closure"]
    BUGS_PER_PROJECT = 10

    def __init__(self, data_dir: Path | str = "data") -> None:
        super().__init__(
            name=self.DATASET_NAME,
            data_dir=data_dir,
            language="java",
        )
        self._junit_code: dict[str, str] = {}
        self._buggy_code: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def load_problems(self) -> list[Problem]:
        builtin = self._load_builtin_examples()
        jsonl_path = self.data_dir / "defects4j" / "bugs.jsonl"
        if jsonl_path.exists():
            return builtin + self._load_from_jsonl(jsonl_path)
        return builtin

    def get_test_suite(self, problem_id: str) -> TestSuite:
        self._ensure_loaded()
        if problem_id not in self._test_suites:
            raise KeyError(f"No test suite for '{problem_id}' in {self.name}")
        return self._test_suites[problem_id]

    def format_prompt(self, problem_id: str) -> str:
        """
        Build a bug-fixing prompt containing buggy code + failing test info.
        """
        problem = self.get_problem(problem_id)
        buggy = self.get_buggy_code(problem_id)
        meta = problem.metadata

        prompt = (
            f"Fix the bug in the following Java code.\n\n"
            f"Project: {meta.get('project', 'Unknown')}\n"
            f"Bug description:\n{problem.description}\n\n"
            f"Buggy code:\n```java\n{buggy}\n```\n\n"
            f"Failing test: {meta.get('failing_test', 'N/A')}\n\n"
            f"Return the COMPLETE fixed Java class. Do NOT return a diff "
            f"or patch — return the full corrected source code.\n"
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

        for tc in suite.cases:
            if not has_method:
                results.append(self._make_verification_result(
                    test_id=tc.test_id, passed=False,
                    error_message=f"Method '{entry_point}' not found in fix",
                ))
            else:
                results.append(self._make_verification_result(
                    test_id=tc.test_id, passed=False,
                    error_message="PENDING_SANDBOX_EXECUTION",
                ))

        return results

    # ------------------------------------------------------------------
    # Defects4J-specific
    # ------------------------------------------------------------------

    def get_buggy_code(self, problem_id: str) -> str:
        """Return the original buggy source code."""
        self._ensure_loaded()
        if problem_id not in self._buggy_code:
            raise KeyError(f"No buggy code for '{problem_id}'")
        return self._buggy_code[problem_id]

    def get_junit_code(self, problem_id: str) -> str:
        self._ensure_loaded()
        if problem_id not in self._junit_code:
            raise KeyError(f"No JUnit code for '{problem_id}'")
        return self._junit_code[problem_id]

    def get_bugs_by_project(self, project: str) -> list[Problem]:
        """Filter loaded bugs by D4J project name."""
        self._ensure_loaded()
        return [
            p for p in self._problems.values()
            if p.metadata.get("project") == project
        ]

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_builtin_examples(self) -> list[Problem]:
        problems: list[Problem] = []
        for entry in _EXAMPLE_BUGS:
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
            self._buggy_code[entry["problem_id"]] = entry["buggy_code"]

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
                if "buggy_code" in entry:
                    self._buggy_code[entry["problem_id"]] = entry["buggy_code"]

        logger.info("Loaded %d Defects4J bugs from %s", len(problems), path)
        return problems
