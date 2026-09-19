"""
taskfactory.patches — reverse-patch construction and test-set derivation.

The trick that makes a benchmark out of a repository's history: take a
merged pull request that fixed something, apply the *inverse* of its code
changes, and you have a repository in its broken state with a test suite
that already proves what "fixed" means.

Two sets come out of it and the distinction is the whole game:

``FAIL_TO_PASS``
    Tests that fail before the fix and pass after. These are the target: a
    solution must make all of them pass.

``PASS_TO_PASS``
    Tests that pass before *and* after. These are the guard rail: a solution
    must not break any of them. Without this set, deleting the failing test
    is a valid solution.

The inverse is applied to **code only**. The tests stay at their
post-fix state, which is what makes them able to judge. Reverting the tests
too would produce a repository that passes its own suite and measures
nothing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "FileChange",
    "PullRequest",
    "TaskInstance",
    "is_test_path",
    "split_changes",
    "build_reverse_patch",
    "derive_test_sets",
    "TestOutcomeSet",
]

#: Path shapes that mean "this is a test", across the usual JVM layouts.
_TEST_PATH = re.compile(
    r"(^|/)(src/test/|test/|tests/)|"
    r"(^|/)[A-Z]\w*(Test|Tests|TestCase|IT|ITCase)\.java$|"
    r"(^|/)(Test|IT)[A-Z]\w*\.java$",
)


def is_test_path(path: str) -> bool:
    """Whether *path* is a test file.

    Path-based, because it has to work on a diff without compiling
    anything. It errs toward calling things tests: misclassifying a test as
    code puts it in the reverse patch and produces a task whose suite has
    been quietly reverted, which is much worse than the other way round.
    """
    return bool(_TEST_PATH.search(path.replace("\\", "/")))


@dataclass
class FileChange:
    """One file's before and after state in a pull request."""

    path: str
    before: str | None  # None when the file was added
    after: str | None   # None when the file was deleted

    @property
    def is_addition(self) -> bool:
        return self.before is None and self.after is not None

    @property
    def is_deletion(self) -> bool:
        return self.before is not None and self.after is None

    @property
    def is_test(self) -> bool:
        return is_test_path(self.path)

    @property
    def lines_changed(self) -> int:
        before_lines = (self.before or "").splitlines()
        after_lines = (self.after or "").splitlines()
        return abs(len(after_lines) - len(before_lines)) + sum(
            1 for a, b in zip(before_lines, after_lines) if a != b
        )


@dataclass
class PullRequest:
    """A merged pull request, as the factory needs it."""

    repo: str
    number: int
    title: str = ""
    body: str = ""
    merge_commit: str = ""
    base_commit: str = ""
    merged_at: str = ""
    changes: list[FileChange] = field(default_factory=list)
    linked_issue: str = ""

    @property
    def code_changes(self) -> list[FileChange]:
        return [change for change in self.changes if not change.is_test]

    @property
    def test_changes(self) -> list[FileChange]:
        return [change for change in self.changes if change.is_test]

    @property
    def touches_code_and_tests(self) -> bool:
        """The basic admission filter.

        A PR with no test change has nothing to judge a solution by. One
        with no code change has nothing to revert. Only the intersection is
        usable.
        """
        return bool(self.code_changes) and bool(self.test_changes)

    @property
    def total_lines_changed(self) -> int:
        return sum(change.lines_changed for change in self.changes)


@dataclass
class TestOutcomeSet:
    """The two test sets that define a task."""

    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    #: Tests that failed both before and after. Broken independently of this
    #: PR, so they are excluded from scoring rather than counted against a
    #: solution that had nothing to do with them.
    fail_to_fail: list[str] = field(default_factory=list)
    #: Tests that passed before and fail after. A red flag about the PR
    #: itself, not about any future solution.
    pass_to_fail: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """A task needs at least one target test to be a task at all."""
        return bool(self.fail_to_pass)

    @property
    def warnings(self) -> list[str]:
        notes: list[str] = []
        if not self.fail_to_pass:
            notes.append(
                "No FAIL_TO_PASS test: the PR's tests already pass against "
                "the broken code, so there is nothing to solve."
            )
        if self.pass_to_fail:
            notes.append(
                f"{len(self.pass_to_fail)} test(s) pass before the fix and "
                f"fail after it. The reference fix breaks them, so the task "
                f"is not well formed."
            )
        if not self.pass_to_pass:
            notes.append(
                "No PASS_TO_PASS test: nothing guards against a solution "
                "that passes the target by breaking everything else."
            )
        return notes

    def to_dict(self) -> dict[str, Any]:
        return {
            "fail_to_pass": self.fail_to_pass,
            "pass_to_pass": self.pass_to_pass,
            "fail_to_fail": self.fail_to_fail,
            "pass_to_fail": self.pass_to_fail,
            "usable": self.usable,
            "warnings": self.warnings,
        }


@dataclass
class TaskInstance:
    """A candidate task, before validation."""

    instance_id: str
    repo: str
    base_commit: str
    pull_request: int
    problem_statement: str = ""
    #: The repository in its broken state: the code reverted, tests kept.
    broken_files: dict[str, str] = field(default_factory=dict)
    #: The reference fix, as ``{path: content}``. Never shown to a model.
    golden_patch: dict[str, str] = field(default_factory=dict)
    test_sets: TestOutcomeSet = field(default_factory=TestOutcomeSet)
    environment: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def golden_files(self) -> list[str]:
        """Paths the reference fix touches. Used for localisation scoring."""
        return sorted(self.golden_patch)

    def to_dict(self, include_golden: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "pull_request": self.pull_request,
            "problem_statement": self.problem_statement,
            "test_sets": self.test_sets.to_dict(),
            "environment": self.environment,
            "metadata": self.metadata,
            "golden_files": self.golden_files,
        }
        if include_golden:
            payload["golden_patch"] = self.golden_patch
        return payload


def split_changes(
    pull_request: PullRequest,
) -> tuple[list[FileChange], list[FileChange]]:
    """Split a PR's changes into (code, tests).

    The split decides what gets reverted. Getting it wrong in the direction
    of "this test is code" reverts the test too, producing a repository that
    passes its own suite and measures nothing — so
    :func:`is_test_path` deliberately errs the other way.
    """
    return pull_request.code_changes, pull_request.test_changes


def build_reverse_patch(pull_request: PullRequest) -> dict[str, str]:
    """The repository in its broken state: code reverted, tests kept.

    Returns ``{path: content}`` for every file the PR touched. Code files
    take their pre-fix content; test files take their post-fix content,
    because the post-fix tests are what defines success.

    A code file the PR *added* is omitted entirely — before the fix it did
    not exist, and a solution is expected to create it.
    """
    state: dict[str, str] = {}

    for change in pull_request.code_changes:
        if change.is_addition:
            continue  # did not exist before the fix
        if change.before is not None:
            state[change.path] = change.before

    for change in pull_request.test_changes:
        if change.after is not None:
            state[change.path] = change.after
        # A test the PR deleted stays deleted: the fix removed it for a
        # reason and reinstating it would judge against a test its own
        # author retired.

    return state


def build_golden_patch(pull_request: PullRequest) -> dict[str, str]:
    """The reference fix: every code file at its post-fix content.

    Never shown to a model. Used to validate the task — the golden patch
    must make FAIL_TO_PASS pass — and to score localisation.
    """
    return {
        change.path: change.after
        for change in pull_request.code_changes
        if change.after is not None
    }


def derive_test_sets(
    before_results: dict[str, bool],
    after_results: dict[str, bool],
) -> TestOutcomeSet:
    """Classify every test by how it behaves before and after the fix.

    *before_results* and *after_results* map a test identifier to whether it
    passed. Tests present in only one of the two runs are skipped: a test
    that does not exist on both sides cannot be classified by its transition
    and silently assuming a value for the missing side is how a task ends up
    with a target test that was never failing.
    """
    sets = TestOutcomeSet()
    shared = set(before_results) & set(after_results)

    missing = set(before_results) ^ set(after_results)
    if missing:
        logger.debug(
            "%d test(s) appear in only one run and were not classified: %s",
            len(missing), sorted(missing)[:5],
        )

    for test in sorted(shared):
        before, after = before_results[test], after_results[test]
        if not before and after:
            sets.fail_to_pass.append(test)
        elif before and after:
            sets.pass_to_pass.append(test)
        elif not before and not after:
            sets.fail_to_fail.append(test)
        else:
            sets.pass_to_fail.append(test)

    return sets


def build_problem_statement(pull_request: PullRequest) -> str:
    """The task description a model sees.

    The linked issue if there is one, the PR body otherwise. Deliberately
    *not* the diff, and not the PR title alone: a statement that names the
    file and the line turns the task into transcription, and a statement
    that is only a title is usually unsolvable.
    """
    parts: list[str] = []
    if pull_request.title:
        parts.append(pull_request.title.strip())
    body = (pull_request.linked_issue or pull_request.body or "").strip()
    if body:
        parts.append(body)

    statement = "\n\n".join(parts)
    # Strip anything that gives the answer away.
    statement = re.sub(r"```diff.*?```", "", statement, flags=re.DOTALL)
    statement = re.sub(r"(?m)^\s*[-+]{3}\s+[ab]/.*$", "", statement)
    return statement.strip()
