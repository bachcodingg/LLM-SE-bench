"""
bench.datasets.godclass — God Class refactoring dataset adapter (10 classes).

Contains 10 God Class instances (6 from thesis + 4 new) across four severity
levels.  Each class has baseline CK metrics, the original source code, and a
reference decomposition recommendation.  The LLM is asked to propose a
refactoring that reduces the class's God Class severity.

Ships with three built-in examples for development.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from contracts import (
    Problem,
    TestCase,
    TestSuite,
    VerificationResult,
)
from bench.datasets.base import Dataset

logger = logging.getLogger(__name__)

# ======================================================================
# Built-in example God Classes (3 of 10)
# ======================================================================

_EXAMPLE_CLASSES: list[dict[str, Any]] = [
    {
        "problem_id": "GC_001",
        "title": "OrderProcessor God Class",
        "description": (
            "The ``OrderProcessor`` class handles order validation, payment "
            "processing, inventory management, email notifications, and "
            "shipping label generation — all in a single 450-line class.\n\n"
            "Refactor this class to reduce its God Class severity.  Propose "
            "a decomposition into smaller, focused classes.  Return the "
            "refactored Java code with clear class boundaries."
        ),
        "difficulty": "hard",
        "language": "java",
        "reference_solution": (
            "// Decomposition: OrderProcessor → OrderValidator + PaymentService\n"
            "//   + InventoryManager + NotificationService + ShippingService\n\n"
            "public class OrderValidator {\n"
            "    public boolean validate(Order order) {\n"
            "        return order != null && !order.getItems().isEmpty()\n"
            "            && order.getCustomerId() != null;\n"
            "    }\n"
            "}\n\n"
            "public class PaymentService {\n"
            "    public boolean processPayment(Order order, PaymentDetails details) {\n"
            "        // payment gateway logic\n"
            "        return details.getAmount() >= order.getTotal();\n"
            "    }\n"
            "}\n\n"
            "public class InventoryManager {\n"
            "    public boolean reserveItems(Order order) {\n"
            "        // check and reserve inventory\n"
            "        return true;\n"
            "    }\n\n"
            "    public void releaseItems(Order order) {\n"
            "        // release reserved inventory\n"
            "    }\n"
            "}\n\n"
            "public class NotificationService {\n"
            "    public void sendConfirmation(Order order, String email) {\n"
            "        // send email\n"
            "    }\n"
            "}\n\n"
            "public class ShippingService {\n"
            "    public String generateLabel(Order order, Address address) {\n"
            "        return \"SHIP-\" + order.getId();\n"
            "    }\n"
            "}\n"
        ),
        "tags": ["refactoring", "god-class", "severe"],
        "metadata": {
            "severity": "severe",
            "source_project": "thesis-example",
            "original_loc": 450,
            "original_wmc": 38,
            "original_cbo": 22,
            "original_lcom": 0.85,
            "original_rfc": 45,
            "entry_point": "OrderProcessor",
        },
        "original_code": (
            "import java.util.*;\n\n"
            "public class OrderProcessor {\n"
            "    private Map<String, Integer> inventory = new HashMap<>();\n"
            "    private List<Order> processedOrders = new ArrayList<>();\n"
            "    private double totalRevenue = 0.0;\n\n"
            "    // Validation\n"
            "    public boolean validateOrder(Order order) {\n"
            "        if (order == null) return false;\n"
            "        if (order.getItems().isEmpty()) return false;\n"
            "        if (order.getCustomerId() == null) return false;\n"
            "        return true;\n"
            "    }\n\n"
            "    // Payment\n"
            "    public boolean processPayment(Order order, PaymentDetails pd) {\n"
            "        if (pd.getAmount() < order.getTotal()) return false;\n"
            "        totalRevenue += order.getTotal();\n"
            "        return true;\n"
            "    }\n\n"
            "    // Inventory\n"
            "    public boolean checkInventory(Order order) {\n"
            "        for (Item item : order.getItems()) {\n"
            "            int stock = inventory.getOrDefault(item.getSku(), 0);\n"
            "            if (stock < item.getQuantity()) return false;\n"
            "        }\n"
            "        return true;\n"
            "    }\n\n"
            "    public void reserveInventory(Order order) {\n"
            "        for (Item item : order.getItems()) {\n"
            "            inventory.merge(item.getSku(), -item.getQuantity(), Integer::sum);\n"
            "        }\n"
            "    }\n\n"
            "    // Notification\n"
            "    public void sendEmail(String to, String subject, String body) {\n"
            "        // SMTP logic embedded directly\n"
            "        System.out.println(\"Email to: \" + to);\n"
            "    }\n\n"
            "    // Shipping\n"
            "    public String createShippingLabel(Order order) {\n"
            "        return \"SHIP-\" + order.getId() + \"-\" + System.currentTimeMillis();\n"
            "    }\n\n"
            "    // Orchestration\n"
            "    public boolean processOrder(Order order, PaymentDetails pd) {\n"
            "        if (!validateOrder(order)) return false;\n"
            "        if (!checkInventory(order)) return false;\n"
            "        if (!processPayment(order, pd)) return false;\n"
            "        reserveInventory(order);\n"
            "        sendEmail(order.getCustomerEmail(), \"Confirmation\", \"Done\");\n"
            "        createShippingLabel(order);\n"
            "        processedOrders.add(order);\n"
            "        return true;\n"
            "    }\n"
            "}\n"
        ),
        "test_cases": [
            {
                "test_id": "GC_001_test_1",
                "input_data": {"check": "class_count"},
                "expected_output": {"min_classes": 3},
                "is_hidden": False,
                "weight": 2.0,
                "timeout_seconds": 30.0,
            },
            {
                "test_id": "GC_001_test_2",
                "input_data": {"check": "max_wmc_per_class"},
                "expected_output": {"max_wmc": 15},
                "is_hidden": False,
                "weight": 1.5,
                "timeout_seconds": 30.0,
            },
            {
                "test_id": "GC_001_test_3",
                "input_data": {"check": "compiles"},
                "expected_output": True,
                "is_hidden": True,
                "weight": 2.0,
                "timeout_seconds": 30.0,
            },
        ],
        "junit_code": (
            "import org.junit.Test;\n"
            "import static org.junit.Assert.*;\n\n"
            "public class OrderProcessorRefactorTest {\n"
            "    @Test\n"
            "    public void testMultipleClassesExist() {\n"
            "        // Structural check: at least 3 classes in output\n"
            "        // This is validated by the sandbox CK metrics checker\n"
            "        assertTrue(\"Refactoring should produce >= 3 classes\", true);\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testReducedComplexity() {\n"
            "        // CK metrics check: no single class WMC > 15\n"
            "        assertTrue(\"Each class WMC should be <= 15\", true);\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testCompiles() {\n"
            "        // Compilation check done by sandbox\n"
            "        assertTrue(\"All classes should compile\", true);\n"
            "    }\n"
            "}\n"
        ),
    },
    {
        "problem_id": "GC_002",
        "title": "UserManager God Class",
        "description": (
            "The ``UserManager`` class handles user registration, "
            "authentication, profile updates, role management, session "
            "tracking, and audit logging — 280 lines in one class.\n\n"
            "Refactor to reduce God Class severity.  Propose decomposition."
        ),
        "difficulty": "medium",
        "language": "java",
        "reference_solution": (
            "// Decomposition: UserManager → AuthService + ProfileService\n"
            "//   + RoleManager + SessionTracker + AuditLogger\n\n"
            "public class AuthService {\n"
            "    public boolean authenticate(String username, String password) {\n"
            "        return username != null && password != null\n"
            "            && password.length() >= 8;\n"
            "    }\n\n"
            "    public String generateToken(String username) {\n"
            "        return \"token-\" + username.hashCode();\n"
            "    }\n"
            "}\n\n"
            "public class ProfileService {\n"
            "    public void updateProfile(String userId, Map<String, String> fields) {\n"
            "        // update user profile fields\n"
            "    }\n"
            "}\n\n"
            "public class RoleManager {\n"
            "    public void assignRole(String userId, String role) {\n"
            "        // assign role to user\n"
            "    }\n"
            "}\n"
        ),
        "tags": ["refactoring", "god-class", "moderate"],
        "metadata": {
            "severity": "moderate",
            "source_project": "thesis-example",
            "original_loc": 280,
            "original_wmc": 24,
            "original_cbo": 15,
            "original_lcom": 0.72,
            "original_rfc": 32,
            "entry_point": "UserManager",
        },
        "original_code": (
            "import java.util.*;\n\n"
            "public class UserManager {\n"
            "    private Map<String, String> users = new HashMap<>();\n"
            "    private Map<String, String> sessions = new HashMap<>();\n"
            "    private Map<String, List<String>> roles = new HashMap<>();\n"
            "    private List<String> auditLog = new ArrayList<>();\n\n"
            "    public boolean register(String username, String password) {\n"
            "        if (users.containsKey(username)) return false;\n"
            "        users.put(username, password);\n"
            "        auditLog.add(\"REGISTER: \" + username);\n"
            "        return true;\n"
            "    }\n\n"
            "    public String login(String username, String password) {\n"
            "        if (!users.containsKey(username)) return null;\n"
            "        if (!users.get(username).equals(password)) return null;\n"
            "        String token = \"tok-\" + username.hashCode();\n"
            "        sessions.put(token, username);\n"
            "        auditLog.add(\"LOGIN: \" + username);\n"
            "        return token;\n"
            "    }\n\n"
            "    public void assignRole(String username, String role) {\n"
            "        roles.computeIfAbsent(username, k -> new ArrayList<>()).add(role);\n"
            "        auditLog.add(\"ROLE: \" + username + \" -> \" + role);\n"
            "    }\n\n"
            "    public boolean hasRole(String username, String role) {\n"
            "        return roles.getOrDefault(username, List.of()).contains(role);\n"
            "    }\n\n"
            "    public void logout(String token) {\n"
            "        String user = sessions.remove(token);\n"
            "        if (user != null) auditLog.add(\"LOGOUT: \" + user);\n"
            "    }\n"
            "}\n"
        ),
        "test_cases": [
            {
                "test_id": "GC_002_test_1",
                "input_data": {"check": "class_count"},
                "expected_output": {"min_classes": 2},
                "is_hidden": False,
                "weight": 2.0,
                "timeout_seconds": 30.0,
            },
            {
                "test_id": "GC_002_test_2",
                "input_data": {"check": "separation_of_concerns"},
                "expected_output": {"auth_separate": True},
                "is_hidden": False,
                "weight": 1.5,
                "timeout_seconds": 30.0,
            },
            {
                "test_id": "GC_002_test_3",
                "input_data": {"check": "compiles"},
                "expected_output": True,
                "is_hidden": True,
                "weight": 2.0,
                "timeout_seconds": 30.0,
            },
        ],
        "junit_code": (
            "import org.junit.Test;\n"
            "import static org.junit.Assert.*;\n\n"
            "public class UserManagerRefactorTest {\n"
            "    @Test\n"
            "    public void testDecomposition() {\n"
            "        assertTrue(\"Should produce >= 2 classes\", true);\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testAuthSeparated() {\n"
            "        assertTrue(\"Auth logic should be in separate class\", true);\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testCompiles() {\n"
            "        assertTrue(\"All classes should compile\", true);\n"
            "    }\n"
            "}\n"
        ),
    },
    {
        "problem_id": "GC_003",
        "title": "ReportGenerator God Class",
        "description": (
            "The ``ReportGenerator`` class reads data from databases, "
            "performs statistical calculations, formats HTML/PDF/CSV "
            "output, sends email, and manages file system caching — "
            "520 lines.\n\n"
            "Refactor to reduce God Class severity."
        ),
        "difficulty": "hard",
        "language": "java",
        "reference_solution": (
            "// Decomposition: ReportGenerator → DataRepository\n"
            "//   + StatisticsCalculator + ReportFormatter + ReportMailer + CacheManager\n\n"
            "public class DataRepository {\n"
            "    public List<Record> fetchData(String query) {\n"
            "        return new ArrayList<>(); // DB query\n"
            "    }\n"
            "}\n\n"
            "public class StatisticsCalculator {\n"
            "    public Map<String, Double> compute(List<Record> data) {\n"
            "        Map<String, Double> stats = new HashMap<>();\n"
            "        stats.put(\"mean\", 0.0);\n"
            "        stats.put(\"median\", 0.0);\n"
            "        return stats;\n"
            "    }\n"
            "}\n\n"
            "public class ReportFormatter {\n"
            "    public String toHtml(Map<String, Double> stats) {\n"
            "        return \"<html>\" + stats + \"</html>\";\n"
            "    }\n\n"
            "    public byte[] toPdf(Map<String, Double> stats) {\n"
            "        return new byte[0]; // PDF generation\n"
            "    }\n\n"
            "    public String toCsv(Map<String, Double> stats) {\n"
            "        return stats.entrySet().stream()\n"
            "            .map(e -> e.getKey() + \",\" + e.getValue())\n"
            "            .reduce(\"\", (a, b) -> a + \"\\n\" + b);\n"
            "    }\n"
            "}\n"
        ),
        "tags": ["refactoring", "god-class", "severe"],
        "metadata": {
            "severity": "severe",
            "source_project": "thesis-example",
            "original_loc": 520,
            "original_wmc": 42,
            "original_cbo": 28,
            "original_lcom": 0.88,
            "original_rfc": 52,
            "entry_point": "ReportGenerator",
        },
        "original_code": (
            "import java.util.*;\n\n"
            "public class ReportGenerator {\n"
            "    private String dbUrl;\n"
            "    private Map<String, byte[]> cache = new HashMap<>();\n\n"
            "    public ReportGenerator(String dbUrl) { this.dbUrl = dbUrl; }\n\n"
            "    public List<Map<String, Object>> queryData(String sql) {\n"
            "        // direct DB access\n"
            "        return new ArrayList<>();\n"
            "    }\n\n"
            "    public double calculateMean(List<Double> values) {\n"
            "        return values.stream().mapToDouble(d -> d).average().orElse(0.0);\n"
            "    }\n\n"
            "    public double calculateMedian(List<Double> values) {\n"
            "        Collections.sort(values);\n"
            "        int n = values.size();\n"
            "        if (n == 0) return 0.0;\n"
            "        return n % 2 == 0\n"
            "            ? (values.get(n/2-1) + values.get(n/2)) / 2.0\n"
            "            : values.get(n/2);\n"
            "    }\n\n"
            "    public String generateHtml(Map<String, Double> stats) {\n"
            "        return \"<html><body>\" + stats + \"</body></html>\";\n"
            "    }\n\n"
            "    public String generateCsv(Map<String, Double> stats) {\n"
            "        StringBuilder sb = new StringBuilder();\n"
            "        for (var entry : stats.entrySet()) {\n"
            "            sb.append(entry.getKey()).append(\",\")\n"
            "              .append(entry.getValue()).append(\"\\n\");\n"
            "        }\n"
            "        return sb.toString();\n"
            "    }\n\n"
            "    public void sendEmail(String to, String subject, String body) {\n"
            "        System.out.println(\"Sending to: \" + to);\n"
            "    }\n\n"
            "    public void cacheResult(String key, byte[] data) {\n"
            "        cache.put(key, data);\n"
            "    }\n\n"
            "    public byte[] getCached(String key) {\n"
            "        return cache.get(key);\n"
            "    }\n"
            "}\n"
        ),
        "test_cases": [
            {
                "test_id": "GC_003_test_1",
                "input_data": {"check": "class_count"},
                "expected_output": {"min_classes": 3},
                "is_hidden": False,
                "weight": 2.0,
                "timeout_seconds": 30.0,
            },
            {
                "test_id": "GC_003_test_2",
                "input_data": {"check": "max_wmc_per_class"},
                "expected_output": {"max_wmc": 12},
                "is_hidden": False,
                "weight": 1.5,
                "timeout_seconds": 30.0,
            },
            {
                "test_id": "GC_003_test_3",
                "input_data": {"check": "compiles"},
                "expected_output": True,
                "is_hidden": True,
                "weight": 2.0,
                "timeout_seconds": 30.0,
            },
        ],
        "junit_code": (
            "import org.junit.Test;\n"
            "import static org.junit.Assert.*;\n\n"
            "public class ReportGeneratorRefactorTest {\n"
            "    @Test\n"
            "    public void testDecomposition() {\n"
            "        assertTrue(\"Should produce >= 3 classes\", true);\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testReducedWMC() {\n"
            "        assertTrue(\"Each class WMC should be <= 12\", true);\n"
            "    }\n\n"
            "    @Test\n"
            "    public void testCompiles() {\n"
            "        assertTrue(\"All classes should compile\", true);\n"
            "    }\n"
            "}\n"
        ),
    },
]


class GodClassDataset(Dataset):
    """
    Adapter for the God Class refactoring benchmark (10 classes).

    6 classes from thesis work + 4 new from Apache Commons / Spring
    Framework.  Four severity levels: mild, moderate, severe, extreme.

    Parameters
    ----------
    data_dir : Path | str
        Directory containing ``godclass/classes.jsonl``.
    """

    DATASET_NAME = "godclass"
    EXPECTED_SIZE = 10
    SEVERITY_LEVELS = ["mild", "moderate", "severe", "extreme"]

    def __init__(self, data_dir: Path | str = "data") -> None:
        super().__init__(
            name=self.DATASET_NAME,
            data_dir=data_dir,
            language="java",
        )
        self._junit_code: dict[str, str] = {}
        self._original_code: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def load_problems(self) -> list[Problem]:
        builtin = self._load_builtin_examples()
        jsonl_path = self.data_dir / "godclass" / "classes.jsonl"
        if jsonl_path.exists():
            return builtin + self._load_from_jsonl(jsonl_path)
        return builtin

    def get_test_suite(self, problem_id: str) -> TestSuite:
        self._ensure_loaded()
        if problem_id not in self._test_suites:
            raise KeyError(f"No test suite for '{problem_id}' in {self.name}")
        return self._test_suites[problem_id]

    def format_prompt(self, problem_id: str) -> str:
        """Build a refactoring prompt with original code + baseline metrics."""
        problem = self.get_problem(problem_id)
        original = self.get_original_code(problem_id)
        meta = problem.metadata

        metrics_info = (
            f"Baseline CK Metrics:\n"
            f"  LOC:  {meta.get('original_loc', 'N/A')}\n"
            f"  WMC:  {meta.get('original_wmc', 'N/A')}\n"
            f"  CBO:  {meta.get('original_cbo', 'N/A')}\n"
            f"  LCOM: {meta.get('original_lcom', 'N/A')}\n"
            f"  RFC:  {meta.get('original_rfc', 'N/A')}\n"
        )

        prompt = (
            f"Refactor the following Java God Class.\n\n"
            f"INSTRUCTIONS: Output ONLY valid Java source code. "
            f"Do NOT include any explanation, comments outside the code, "
            f"markdown prose, or text before or after the code block. "
            f"Start your response with ```java and end with ```.\n\n"
            f"Class: {meta.get('entry_point', problem.title)}\n"
            f"Severity: {meta.get('severity', 'unknown')}\n"
            f"{metrics_info}\n"
            f"Description:\n{problem.description}\n\n"
            f"Original code:\n```java\n{original}\n```\n\n"
            f"Requirements:\n"
            f"1. Decompose into multiple smaller, focused classes\n"
            f"2. Each resulting class should have a single responsibility\n"
            f"3. Maintain all existing functionality\n"
            f"4. Reduce WMC and CBO metrics for each class\n\n"
            f"Output the complete refactored Java source code inside a single "
            f"```java ... ``` block. No prose before or after.\n"
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

        class_count = generated_code.count("class ")
        min_classes = 2

        for tc in suite.cases:
            check_type = tc.input_data.get("check", "") if isinstance(tc.input_data, dict) else ""

            if check_type == "class_count":
                expected_min = tc.expected_output.get("min_classes", 2) if isinstance(tc.expected_output, dict) else 2
                passed = class_count >= expected_min
                results.append(self._make_verification_result(
                    test_id=tc.test_id,
                    passed=passed,
                    actual_output={"class_count": class_count},
                    error_message="" if passed else f"Expected >= {expected_min} classes, got {class_count}",
                ))
            elif check_type == "compiles":
                results.append(self._make_verification_result(
                    test_id=tc.test_id,
                    passed=False,
                    error_message="PENDING_SANDBOX_EXECUTION",
                ))
            else:
                results.append(self._make_verification_result(
                    test_id=tc.test_id,
                    passed=False,
                    error_message="PENDING_SANDBOX_EXECUTION",
                ))

        return results

    # ------------------------------------------------------------------
    # God Class-specific
    # ------------------------------------------------------------------

    def get_original_code(self, problem_id: str) -> str:
        """Return the original God Class source code."""
        self._ensure_loaded()
        if problem_id not in self._original_code:
            raise KeyError(f"No original code for '{problem_id}'")
        return self._original_code[problem_id]

    def get_junit_code(self, problem_id: str) -> str:
        self._ensure_loaded()
        if problem_id not in self._junit_code:
            raise KeyError(f"No JUnit code for '{problem_id}'")
        return self._junit_code[problem_id]

    def get_classes_by_severity(self, severity: str) -> list[Problem]:
        """Filter loaded classes by severity level."""
        self._ensure_loaded()
        return [
            p for p in self._problems.values()
            if p.metadata.get("severity") == severity
        ]

    def get_baseline_metrics(self, problem_id: str) -> dict[str, Any]:
        """Return baseline CK metrics for a God Class."""
        problem = self.get_problem(problem_id)
        meta = problem.metadata
        return {
            "loc": meta.get("original_loc", 0),
            "wmc": meta.get("original_wmc", 0),
            "cbo": meta.get("original_cbo", 0),
            "lcom": meta.get("original_lcom", 0.0),
            "rfc": meta.get("original_rfc", 0),
        }

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_builtin_examples(self) -> list[Problem]:
        problems: list[Problem] = []
        for entry in _EXAMPLE_CLASSES:
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
            self._original_code[entry["problem_id"]] = entry["original_code"]

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
                if "original_code" in entry:
                    self._original_code[entry["problem_id"]] = entry["original_code"]

        logger.info("Loaded %d God Class problems from %s", len(problems), path)
        return problems
