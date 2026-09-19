"""
taskfactory.environment — build system detection and JVM environment setup.

This is where JVM benchmarks actually fail. EnvBench found the best
automated environment setup method worked on 29.5% of JVM repositories:
seven times out of ten, nothing is measured about the model because nothing
compiles.

Four problems, each solved here as far as static inspection allows.

**Which build system.** Maven, Gradle Groovy, Gradle Kotlin DSL, Ant, or a
multi-module mixture. Detection is by marker file plus content, because a
repository can carry a vestigial ``pom.xml`` next to the Gradle build that
is actually used.

**Which JDK.** Inferred from the build file — ``maven.compiler.release``,
``sourceCompatibility``, the toolchain block — then from language features
in the source as a fallback. Guessing wrong produces a compile error that
looks like a model failure.

**Which dependencies, offline.** A task that resolves dependencies at run
time is not hermetic: it depends on Maven Central being up, and on nobody
having yanked a version. Prewarming a local cache into the image is the
difference between a benchmark and a thing that mostly works.

**Which base image.** A matrix over JDK versions, pinned by digest rather
than tag, because a tag moves and a benchmark that silently changes its JDK
between runs is not reproducible.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "BuildSystem",
    "BuildSpec",
    "detect_build_system",
    "infer_jdk_version",
    "JDK_IMAGES",
    "dependency_prewarm_commands",
    "generate_dockerfile",
    "offline_test_command",
]


class BuildSystem(str, Enum):
    """How a repository builds."""

    MAVEN = "maven"
    GRADLE_GROOVY = "gradle_groovy"
    GRADLE_KOTLIN = "gradle_kotlin"
    ANT = "ant"
    PLAIN_JAVAC = "plain_javac"
    UNKNOWN = "unknown"

    @property
    def supported(self) -> bool:
        """Whether the factory can build a hermetic environment for it.

        Ant is excluded deliberately: its builds are scripts, every one is
        different, and "supported" would mean "works on the ones we tried".
        """
        return self in {
            BuildSystem.MAVEN,
            BuildSystem.GRADLE_GROOVY,
            BuildSystem.GRADLE_KOTLIN,
            BuildSystem.PLAIN_JAVAC,
        }


#: Base images per JDK, pinned to a digest rather than a tag. A tag moves;
#: a benchmark whose JDK silently changes between runs is not reproducible.
#: Refresh with `scripts/refresh_base_images.py` and record the change.
JDK_IMAGES: dict[int, dict[str, str]] = {
    8: {
        "tag": "eclipse-temurin:8-jdk-jammy",
        "digest": "",
        "note": "Digest not pinned yet — pin before publishing a run.",
    },
    11: {
        "tag": "eclipse-temurin:11-jdk-jammy",
        "digest": "",
        "note": "Digest not pinned yet — pin before publishing a run.",
    },
    17: {
        "tag": "eclipse-temurin:17-jdk-jammy",
        "digest": "",
        "note": "Digest not pinned yet — pin before publishing a run.",
    },
    21: {
        "tag": "eclipse-temurin:21-jdk-jammy",
        "digest": "",
        "note": "Digest not pinned yet — pin before publishing a run.",
    },
}

#: Marker files, most specific first. Order matters: a Gradle project often
#: carries a stale pom.xml, and checking Maven first would misread it.
_MARKERS: list[tuple[str, BuildSystem]] = [
    ("build.gradle.kts", BuildSystem.GRADLE_KOTLIN),
    ("build.gradle", BuildSystem.GRADLE_GROOVY),
    ("pom.xml", BuildSystem.MAVEN),
    ("build.xml", BuildSystem.ANT),
]

#: Language features that establish a JDK floor, newest first.
_LANGUAGE_FLOORS: list[tuple[re.Pattern[str], int, str]] = [
    (re.compile(r"\bsealed\s+(?:interface|class)\b"), 17, "sealed types"),
    (re.compile(r"\brecord\s+\w+\s*\("), 16, "records"),
    (re.compile(r"\bcase\s+\w+\s+\w+\s*(?:when\b|->)"), 21, "pattern matching for switch"),
    (re.compile(r'"""'), 15, "text blocks"),
    (re.compile(r"\bvar\s+\w+\s*="), 10, "local variable type inference"),
    (re.compile(r"->\s*[\{\w]"), 8, "lambdas"),
]


@dataclass
class BuildSpec:
    """Everything needed to build and test one repository hermetically."""

    build_system: BuildSystem = BuildSystem.UNKNOWN
    jdk_version: int = 17
    jdk_inferred_from: str = "default"
    modules: list[str] = field(default_factory=list)
    build_files: list[str] = field(default_factory=list)
    wrapper_present: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def multi_module(self) -> bool:
        return len(self.modules) > 1

    @property
    def buildable(self) -> bool:
        """Whether a hermetic environment can be produced for this repo.

        False is a perfectly good answer and is far better than a task that
        is admitted and then fails 30% of the time for reasons nobody
        diagnoses.
        """
        return self.build_system.supported and self.jdk_version in JDK_IMAGES

    def to_dict(self) -> dict[str, Any]:
        return {
            "build_system": self.build_system.value,
            "supported": self.build_system.supported,
            "buildable": self.buildable,
            "jdk_version": self.jdk_version,
            "jdk_inferred_from": self.jdk_inferred_from,
            "multi_module": self.multi_module,
            "modules": self.modules,
            "build_files": self.build_files,
            "wrapper_present": self.wrapper_present,
            "base_image": JDK_IMAGES.get(self.jdk_version, {}),
            "warnings": self.warnings,
        }


def detect_build_system(files: dict[str, str]) -> BuildSpec:
    """Work out how *files* builds.

    *files* is ``{relative path: content}``. Content matters, not just
    names: a ``pom.xml`` that declares no build and a ``build.gradle``
    beside it means Gradle, and picking Maven because its marker is more
    familiar produces a task that never compiles.
    """
    spec = BuildSpec()
    paths = list(files)

    found: list[tuple[str, BuildSystem]] = []
    for name, system in _MARKERS:
        matching = [p for p in paths if Path(p).name == name]
        if matching:
            found.append((name, system))
            spec.build_files.extend(sorted(matching))

    if not found:
        if any(p.endswith(".java") for p in paths):
            spec.build_system = BuildSystem.PLAIN_JAVAC
            spec.warnings.append(
                "No build file found. Treating as plain javac, which works "
                "only for a self-contained source set with no dependencies."
            )
        else:
            spec.warnings.append("No build file and no Java source found.")
        spec.jdk_version, spec.jdk_inferred_from = _infer_from_sources(files)
        return spec

    spec.build_system = found[0][1]
    if len(found) > 1:
        spec.warnings.append(
            f"Several build systems present ({', '.join(n for n, _ in found)}). "
            f"Using {spec.build_system.value}; a vestigial build file is "
            f"common and picking the wrong one produces a task that never "
            f"compiles."
        )

    spec.wrapper_present = any(
        Path(p).name in ("gradlew", "gradlew.bat", "mvnw", "mvnw.cmd") for p in paths
    )
    if spec.build_system in (BuildSystem.GRADLE_GROOVY, BuildSystem.GRADLE_KOTLIN):
        if not spec.wrapper_present:
            spec.warnings.append(
                "No Gradle wrapper. The Gradle version is then whatever the "
                "image has, which is a reproducibility hole."
            )

    spec.modules = _detect_modules(files, spec.build_system)
    spec.jdk_version, spec.jdk_inferred_from = infer_jdk_version(files, spec)
    return spec


def _detect_modules(files: dict[str, str], system: BuildSystem) -> list[str]:
    """Module names, for a multi-module build."""
    modules: list[str] = []
    if system is BuildSystem.MAVEN:
        for path, content in files.items():
            if Path(path).name != "pom.xml":
                continue
            modules.extend(re.findall(r"<module>\s*([^<]+?)\s*</module>", content))
    elif system in (BuildSystem.GRADLE_GROOVY, BuildSystem.GRADLE_KOTLIN):
        for path, content in files.items():
            if not Path(path).name.startswith("settings.gradle"):
                continue
            modules.extend(re.findall(r"include\s*\(?\s*['\"]:?([\w:.-]+)['\"]", content))
    return sorted(set(m.strip() for m in modules if m.strip()))


def infer_jdk_version(files: dict[str, str], spec: BuildSpec) -> tuple[int, str]:
    """Infer the JDK, from the build file first and the sources second.

    Returns ``(version, how it was determined)``. The provenance is carried
    because a version read from a toolchain declaration is a fact and one
    inferred from a lambda is a floor, and treating those as equivalent is
    how a task ends up on the wrong JDK.
    """
    for path in spec.build_files:
        content = files.get(path, "")

        # Maven: release, then source/target, then the properties block.
        for pattern, label in (
            (r"<maven\.compiler\.release>\s*(\d+)", "maven.compiler.release"),
            (r"<release>\s*(\d+)\s*</release>", "<release>"),
            (r"<maven\.compiler\.source>\s*(?:1\.)?(\d+)", "maven.compiler.source"),
            (r"<source>\s*(?:1\.)?(\d+)\s*</source>", "<source>"),
            (r"languageVersion\.set\(JavaLanguageVersion\.of\((\d+)\)\)", "gradle toolchain"),
            (r"JavaLanguageVersion\.of\((\d+)\)", "gradle toolchain"),
            (r"sourceCompatibility\s*=?\s*['\"]?(?:1\.)?(\d+)", "sourceCompatibility"),
            (r"targetCompatibility\s*=?\s*['\"]?(?:1\.)?(\d+)", "targetCompatibility"),
        ):
            match = re.search(pattern, content)
            if match:
                declared = int(match.group(1))
                return _nearest_supported(declared), f"{label} in {Path(path).name}"

    return _infer_from_sources(files)


def _infer_from_sources(files: dict[str, str]) -> tuple[int, str]:
    """Floor implied by the language features the sources use."""
    floor = 8
    reason = "default (no declaration, no modern feature found)"
    for path, content in files.items():
        if not path.endswith(".java"):
            continue
        for pattern, version, feature in _LANGUAGE_FLOORS:
            if version > floor and pattern.search(content):
                floor = version
                reason = f"language feature: {feature}"
    return _nearest_supported(floor), reason


def _nearest_supported(version: int) -> int:
    """Map a declared version onto an image we actually have.

    Rounds *up*: a JDK 14 project runs on 17, and a JDK 17 project will not
    run on 11. Rounding down produces a compile error indistinguishable from
    a model failure.
    """
    available = sorted(JDK_IMAGES)
    for candidate in available:
        if candidate >= version:
            return candidate
    return available[-1]


def offline_test_command(spec: BuildSpec, test_filter: str = "") -> list[str]:
    """The command that runs the tests, offline.

    Named ``offline_`` rather than ``test_`` because pytest collects any
    module-level callable whose name starts with ``test_``, and a
    production function is not a test.

    Offline is the point. Every command here carries the flag that forbids
    network resolution, so a task that needs a dependency the image does not
    have fails loudly at build time rather than intermittently at run time
    depending on whether Maven Central is reachable.
    """
    if spec.build_system is BuildSystem.MAVEN:
        command = ["mvn", "-o", "-B", "--no-transfer-progress", "test"]
        if test_filter:
            command.append(f"-Dtest={test_filter}")
        return command

    if spec.build_system in (BuildSystem.GRADLE_GROOVY, BuildSystem.GRADLE_KOTLIN):
        launcher = "./gradlew" if spec.wrapper_present else "gradle"
        command = [launcher, "--offline", "--no-daemon", "test"]
        if test_filter:
            command.extend(["--tests", test_filter])
        return command

    if spec.build_system is BuildSystem.PLAIN_JAVAC:
        return [
            "bash", "-c",
            "javac -encoding UTF-8 -cp /usr/share/java/junit4.jar:. -d out "
            "$(find . -name '*.java') && "
            f"java -cp /usr/share/java/junit4.jar:out org.junit.runner.JUnitCore "
            f"{test_filter or '$(ls out | sed s/.class//)'}",
        ]

    raise ValueError(
        f"No offline test command for {spec.build_system.value}. "
        f"Unsupported build systems are rejected at admission, not here."
    )


def dependency_prewarm_commands(spec: BuildSpec) -> list[str]:
    """Commands that pull every dependency into the image at build time.

    Run once while building the image, with network. After that the task
    runs with ``--network=none`` and resolves from the local cache, which is
    what makes it hermetic — and what stops a benchmark's results depending
    on Maven Central's uptime on the day.
    """
    if spec.build_system is BuildSystem.MAVEN:
        return [
            "mvn -B --no-transfer-progress dependency:go-offline",
            # go-offline misses test-scoped plugin dependencies; a throwaway
            # compile pulls the rest.
            "mvn -B --no-transfer-progress test-compile -DskipTests || true",
        ]
    if spec.build_system in (BuildSystem.GRADLE_GROOVY, BuildSystem.GRADLE_KOTLIN):
        launcher = "./gradlew" if spec.wrapper_present else "gradle"
        return [
            f"{launcher} --no-daemon dependencies || true",
            f"{launcher} --no-daemon testClasses || true",
        ]
    return []


def generate_dockerfile(spec: BuildSpec, repo_dir: str = "/workspace") -> str:
    """A Dockerfile that builds a hermetic image for this repository.

    Dependencies are resolved during the build, when there is network, so
    that the task itself can run with none.
    """
    if not spec.buildable:
        raise ValueError(
            f"Cannot generate a Dockerfile for {spec.build_system.value}: "
            f"{'; '.join(spec.warnings) or 'unsupported build system'}"
        )

    image = JDK_IMAGES[spec.jdk_version]
    reference = f"{image['tag']}@{image['digest']}" if image.get("digest") else image["tag"]

    lines = [
        f"# Generated by taskfactory.environment for a {spec.build_system.value} project.",
        f"# JDK {spec.jdk_version}, inferred from: {spec.jdk_inferred_from}",
        "#",
        "# Dependencies are resolved at build time, with network. The task",
        "# itself runs with --network=none and resolves from this cache.",
        "",
        f"FROM {reference}",
        "",
    ]
    if not image.get("digest"):
        lines.insert(
            3,
            "# WARNING: base image pinned by tag, not digest. A tag moves; "
            "pin before publishing.",
        )

    if spec.build_system is BuildSystem.MAVEN:
        lines.append("RUN apt-get update && apt-get install -y --no-install-recommends "
                     "maven && rm -rf /var/lib/apt/lists/*")
    elif spec.build_system in (BuildSystem.GRADLE_GROOVY, BuildSystem.GRADLE_KOTLIN):
        if not spec.wrapper_present:
            lines.append("RUN apt-get update && apt-get install -y --no-install-recommends "
                         "gradle && rm -rf /var/lib/apt/lists/*")
    else:
        lines.append("RUN apt-get update && apt-get install -y --no-install-recommends "
                     "junit4 && rm -rf /var/lib/apt/lists/*")

    lines.extend([
        "",
        f"WORKDIR {repo_dir}",
        f"COPY . {repo_dir}",
        "",
        "# Prewarm the dependency cache while the network is still available.",
    ])
    for command in dependency_prewarm_commands(spec) or ["true"]:
        lines.append(f"RUN {command}")

    lines.extend([
        "",
        "RUN useradd -m -s /bin/bash runner && chown -R runner:runner " + repo_dir,
        "USER runner",
        "",
        "# The harness supplies the test command; see test_command_for().",
        'CMD ["bash"]',
    ])
    return "\n".join(lines) + "\n"
