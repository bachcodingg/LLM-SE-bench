"""
quality.difftest — differential testing across a refactoring (M5).

The test suite proves a refactoring did not break the cases someone thought
to write down. Differential testing checks the cases nobody did: generate
inputs, run them against the original and the refactored version, and
compare outputs.

It is the only behaviour check here that can catch a refactoring which
passes every existing test and is still wrong, which is the failure mode a
test suite is structurally unable to see.

What this does and does not do
------------------------------
It generates inputs by *type*, from a method signature, and drives both
versions through a harness compiled in the sandbox. It is a boundary-value
and small-random generator, not a coverage-guided fuzzer: it will find an
off-by-one at zero, at the empty list, at ``Integer.MAX_VALUE``, and it will
not find a bug that needs a specific 12-character string.

Agreement is therefore evidence, not proof, and
:class:`DifferentialResult` reports how many inputs were tried so a reader
can weigh it. A real EvoSuite or JQF integration would be strictly better
and is not here.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "InputGenerator",
    "DifferentialResult",
    "generate_inputs",
    "build_harness",
    "run_differential",
]

#: Boundary values worth trying for each supported type. Boundaries first,
#: because that is where the bugs are.
_BOUNDARIES: dict[str, list[str]] = {
    "int": ["0", "1", "-1", "2", "-2", "Integer.MAX_VALUE", "Integer.MIN_VALUE", "100"],
    "long": ["0L", "1L", "-1L", "Long.MAX_VALUE", "Long.MIN_VALUE"],
    "double": ["0.0", "1.0", "-1.0", "0.5", "-0.5", "Double.MIN_VALUE", "1e10"],
    "float": ["0.0f", "1.0f", "-1.0f"],
    "boolean": ["true", "false"],
    "char": ["'a'", "'Z'", "'0'", "' '"],
    "String": ['""', '"a"', '"abc"', '"  "', '"ABC"', '"a b c"', '"123"'],
}

#: Collection types and the element type used to populate them.
_COLLECTIONS = {
    "List<Integer>": ("java.util.Arrays.asList({})", ["", "1", "1, 2, 3", "-1, 0, 1", "5, 5, 5"]),
    "List<String>": ('java.util.Arrays.asList({})', ["", '"a"', '"a", "b"', '"", "x"']),
    "List<Double>": ("java.util.Arrays.asList({})", ["", "1.0", "1.0, 2.0, 3.0"]),
    "int[]": ("new int[]{{{}}}", ["", "1", "1, 2, 3", "-1, 0, 1"]),
    "String[]": ("new String[]{{{}}}", ["", '"a"', '"a", "b"']),
    "double[]": ("new double[]{{{}}}", ["", "1.0", "1.0, 2.0"]),
}


@dataclass
class DifferentialResult:
    """Outcome of comparing two versions on generated inputs."""

    inputs_tried: int = 0
    agreements: int = 0
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    executed: bool = True
    method: str = ""

    @property
    def agreement_rate(self) -> float:
        """Fraction of inputs on which the two versions agreed, 0.0-1.0.

        Returns 0.0 when nothing was tried — not 1.0. An untested pair has
        shown no agreement, and defaulting to perfect agreement would let a
        generator that failed to produce inputs read as a clean bill of
        health.
        """
        return self.agreements / self.inputs_tried if self.inputs_tried else 0.0

    @property
    def behaviour_preserved(self) -> bool:
        """True when every input tried agreed, and something was tried."""
        return self.executed and self.inputs_tried > 0 and not self.disagreements

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "executed": self.executed,
            "inputs_tried": self.inputs_tried,
            "agreements": self.agreements,
            "agreement_rate": round(self.agreement_rate, 4),
            "behaviour_preserved": self.behaviour_preserved,
            "disagreements": self.disagreements[:10],
            "errors": self.errors[:5],
            "caveat": (
                "Boundary-value and small-random generation, not "
                "coverage-guided fuzzing. Agreement is evidence, not proof."
            ),
        }


class InputGenerator:
    """Generates Java literal expressions for a parameter type.

    Parameters
    ----------
    seed
        Seeds the random component, so a run is reproducible. Recorded in
        the run manifest alongside everything else that is pinned.
    """

    def __init__(self, seed: int = 0) -> None:
        self.random = random.Random(seed)
        self.seed = seed

    def for_type(self, java_type: str, count: int = 8) -> list[str]:
        """Return up to *count* literal expressions of *java_type*.

        Boundary values come first and are never displaced by random ones:
        a generator that reports "20 inputs tried" while missing zero has
        tried the wrong 20.
        """
        normalised = re.sub(r"\s+", "", java_type).replace("java.util.", "")

        if normalised in _BOUNDARIES:
            values = list(_BOUNDARIES[normalised])
            values.extend(self._random_scalars(normalised, count - len(values)))
            return values[:count]

        for collection, (template, elements) in _COLLECTIONS.items():
            if normalised == re.sub(r"\s+", "", collection):
                return [template.format(element) for element in elements][:count]

        # An unsupported type gets `null` only, which produces one input and
        # is reported as such rather than silently producing none.
        logger.debug("No generator for type %r; using null", java_type)
        return ["null"]

    def _random_scalars(self, java_type: str, count: int) -> list[str]:
        if count <= 0:
            return []
        if java_type == "int":
            return [str(self.random.randint(-1000, 1000)) for _ in range(count)]
        if java_type == "long":
            return [f"{self.random.randint(-10**9, 10**9)}L" for _ in range(count)]
        if java_type in ("double", "float"):
            suffix = "f" if java_type == "float" else ""
            return [
                f"{self.random.uniform(-1000, 1000):.4f}{suffix}" for _ in range(count)
            ]
        if java_type == "String":
            alphabet = "abcXYZ 019"
            return [
                '"' + "".join(
                    self.random.choice(alphabet)
                    for _ in range(self.random.randint(0, 8))
                ) + '"'
                for _ in range(count)
            ]
        return []


def generate_inputs(
    parameter_types: list[str],
    max_cases: int = 25,
    seed: int = 0,
) -> list[list[str]]:
    """Argument tuples for a method with *parameter_types*.

    Not the full cartesian product: for three parameters with eight values
    each that is 512 compilations. Instead every value of every parameter
    appears at least once, varied one parameter at a time from a boundary
    baseline — which is where the interesting cases are and is bounded by
    the sum rather than the product.
    """
    if not parameter_types:
        return [[]]

    generator = InputGenerator(seed=seed)
    per_parameter = [generator.for_type(java_type) for java_type in parameter_types]

    baseline = [values[0] for values in per_parameter]
    cases: list[list[str]] = [list(baseline)]

    for index, values in enumerate(per_parameter):
        for value in values[1:]:
            if len(cases) >= max_cases:
                return cases
            case = list(baseline)
            case[index] = value
            cases.append(case)
    return cases[:max_cases]


def build_harness(
    class_name_before: str,
    class_name_after: str,
    method_name: str,
    cases: list[list[str]],
    is_static: bool = True,
) -> str:
    """A Java class that runs both versions on every case and prints both.

    Each case is wrapped individually: an exception on case 3 must not stop
    cases 4 onward, and a *thrown exception is itself an output* — two
    versions that both throw ``NumberFormatException`` agree, and one that
    throws where the other returns does not.
    """
    lines = [
        "import java.util.Arrays;",
        "",
        "public class DiffHarness {",
        "    private static String render(Object value) {",
        "        if (value == null) return \"null\";",
        "        if (value instanceof Object[]) return Arrays.deepToString((Object[]) value);",
        "        if (value instanceof int[]) return Arrays.toString((int[]) value);",
        "        if (value instanceof double[]) return Arrays.toString((double[]) value);",
        "        return String.valueOf(value);",
        "    }",
        "",
        "    public static void main(String[] args) {",
    ]

    receiver_before = class_name_before if is_static else f"new {class_name_before}()"
    receiver_after = class_name_after if is_static else f"new {class_name_after}()"

    for index, case in enumerate(cases):
        arguments = ", ".join(case)
        lines.extend([
            f"        // case {index}",
            "        {",
            "            String a, b;",
            "            try {",
            f"                a = render({receiver_before}.{method_name}({arguments}));",
            "            } catch (Throwable t) {",
            "                a = \"THREW:\" + t.getClass().getSimpleName();",
            "            }",
            "            try {",
            f"                b = render({receiver_after}.{method_name}({arguments}));",
            "            } catch (Throwable t) {",
            "                b = \"THREW:\" + t.getClass().getSimpleName();",
            "            }",
            f'            System.out.println("CASE\\t{index}\\t" + a + "\\t" + b);',
            "        }",
        ])

    lines.extend(["    }", "}"])
    return "\n".join(lines)


def run_differential(
    before_source: str,
    after_sources: list[str],
    method_name: str,
    parameter_types: list[str],
    sandbox: Any = None,
    max_cases: int = 25,
    seed: int = 0,
    is_static: bool = True,
) -> DifferentialResult:
    """Compile both versions with a harness and compare their outputs.

    The original is renamed so the two versions can coexist in one
    compilation unit; the rename is textual and therefore approximate, which
    is why a compile failure is reported as an error rather than as a
    disagreement.

    Returns a result with ``executed=False`` when Docker is unavailable —
    never a fabricated agreement.
    """
    result = DifferentialResult(method=method_name)

    if sandbox is None:
        from bench.sandbox.docker_sandbox import DockerSandbox

        sandbox = DockerSandbox()
    if not sandbox.is_docker_available():
        result.executed = False
        result.errors.append(
            "Docker is unavailable, so no differential test ran. This is not "
            "evidence of agreement."
        )
        return result

    before_class = _class_name(before_source)
    if not before_class:
        result.errors.append("Could not find a class name in the original source.")
        return result

    renamed_class = f"{before_class}__Original"
    before_renamed = re.sub(
        rf"\b{re.escape(before_class)}\b", renamed_class, before_source
    )

    after_class = ""
    for source in after_sources:
        if re.search(rf"\b{re.escape(method_name)}\s*\(", source):
            after_class = _class_name(source)
            break
    if not after_class:
        result.errors.append(
            f"No decomposed class declares {method_name}(); it may have been "
            f"renamed or removed."
        )
        return result

    cases = generate_inputs(parameter_types, max_cases=max_cases, seed=seed)
    harness = build_harness(
        renamed_class, after_class, method_name, cases, is_static=is_static
    )

    files = {f"{renamed_class}.java": before_renamed, "DiffHarness.java": harness}
    for index, source in enumerate(after_sources):
        files[f"{_class_name(source) or f'After{index}'}.java"] = source

    sandbox_result = sandbox.run_files(
        files, test_class_name=None, problem_id="difftest"
    )
    if not sandbox_result.compiled:
        result.errors.append(
            f"Harness did not compile: {sandbox_result.error_message[:300]}"
        )
        return result

    output = sandbox_result.compile_stdout + sandbox_result.test_stdout
    for line in output.splitlines():
        if not line.startswith("CASE\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        index, before_value, after_value = parts[1], parts[2], parts[3]
        result.inputs_tried += 1
        if before_value == after_value:
            result.agreements += 1
        else:
            case_index = int(index) if index.isdigit() else -1
            result.disagreements.append({
                "case": case_index,
                "arguments": cases[case_index] if 0 <= case_index < len(cases) else [],
                "before": before_value,
                "after": after_value,
            })

    if result.inputs_tried == 0:
        result.errors.append(
            "The harness compiled but produced no output. Check that the "
            "method is reachable and that it is static as declared."
        )
    return result


def _class_name(source: str) -> str:
    """First public class declared in *source*, or any class."""
    match = re.search(r"public\s+(?:final\s+|abstract\s+)?class\s+(\w+)", source)
    if match:
        return match.group(1)
    match = re.search(r"\bclass\s+(\w+)", source)
    return match.group(1) if match else ""
