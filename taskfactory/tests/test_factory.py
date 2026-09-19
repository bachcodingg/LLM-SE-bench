"""
Tests for the task factory (M3).

Everything here runs against fixtures: no GitHub token, no network, no
Docker. That is the point of keeping the network inside ``GitHubClient`` —
the pipeline's *judgement* is what determines dataset quality, and judgement
is testable without ever making a request.
"""

from __future__ import annotations

import pytest

from taskfactory.contamination import (
    HIGH_EXPOSURE_REPOS,
    assess_contamination,
    stratify_by_cutoff,
)
from taskfactory.environment import (
    BuildSystem,
    detect_build_system,
    generate_dockerfile,
    offline_test_command,
)
from taskfactory.mining import MiningCriteria, filter_pull_requests, rejection_reasons
from taskfactory.patches import (
    FileChange,
    PullRequest,
    TaskInstance,
    build_golden_patch,
    build_problem_statement,
    build_reverse_patch,
    derive_test_sets,
    is_test_path,
)
from taskfactory.pipeline import build_instance, run_pipeline
from taskfactory.validation import Gate, check_flakiness, check_solvability, validate_instance

BUGGY = "public class Calc { public int add(int a, int b) { return a - b; } }"
FIXED = "public class Calc { public int add(int a, int b) { return a + b; } }"
TEST = "public class CalcTest { @Test public void t() { assertEquals(3, add(1,2)); } }"


def _pull_request(**kwargs) -> PullRequest:
    defaults = {
        "repo": "acme/widget",
        "number": 42,
        "title": "Fix addition returning the difference",
        "body": (
            "Calc.add returns a - b instead of a + b, so every sum is wrong. "
            "Expected add(1, 2) to return 3 but it throws off every caller."
        ),
        "base_commit": "abc123",
        "merge_commit": "def456",
        "merged_at": "2024-06-01T12:00:00Z",
        "changes": [
            FileChange("src/main/java/Calc.java", BUGGY, FIXED),
            FileChange("src/test/java/CalcTest.java", "", TEST),
        ],
    }
    defaults.update(kwargs)
    return PullRequest(**defaults)


class TestTestPathDetection:
    @pytest.mark.parametrize("path", [
        "src/test/java/Foo.java",
        "test/Foo.java",
        "tests/Foo.java",
        "src/main/java/FooTest.java",
        "src/main/java/FooIT.java",
        "a/b/TestFoo.java",
    ])
    def test_recognises_test_paths(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize("path", [
        "src/main/java/Foo.java",
        "pom.xml",
        "src/main/java/Contest.java",
    ])
    def test_leaves_code_alone(self, path):
        assert is_test_path(path) is False

    def test_windows_separators(self):
        assert is_test_path("src\\test\\java\\Foo.java") is True


class TestReversePatch:
    def test_code_is_reverted_and_tests_are_kept(self):
        """Reverting the tests too would produce a repo that passes its own suite."""
        state = build_reverse_patch(_pull_request())
        assert state["src/main/java/Calc.java"] == BUGGY
        assert state["src/test/java/CalcTest.java"] == TEST

    def test_a_newly_added_code_file_is_omitted(self):
        pull_request = _pull_request(changes=[
            FileChange("src/main/java/New.java", None, "public class New {}"),
            FileChange("src/test/java/NewTest.java", None, TEST),
        ])
        state = build_reverse_patch(pull_request)
        assert "src/main/java/New.java" not in state
        assert "src/test/java/NewTest.java" in state

    def test_golden_patch_is_the_fixed_code(self):
        golden = build_golden_patch(_pull_request())
        assert golden == {"src/main/java/Calc.java": FIXED}
        assert "src/test/java/CalcTest.java" not in golden


class TestTestSetDerivation:
    def test_classifies_every_transition(self):
        sets = derive_test_sets(
            before_results={"a": False, "b": True, "c": False, "d": True},
            after_results={"a": True, "b": True, "c": False, "d": False},
        )
        assert sets.fail_to_pass == ["a"]
        assert sets.pass_to_pass == ["b"]
        assert sets.fail_to_fail == ["c"]
        assert sets.pass_to_fail == ["d"]

    def test_usable_requires_a_target(self):
        assert derive_test_sets({"a": True}, {"a": True}).usable is False
        assert derive_test_sets({"a": False}, {"a": True}).usable is True

    def test_tests_in_only_one_run_are_skipped(self):
        """Assuming a value for the missing side invents a target test."""
        sets = derive_test_sets({"a": False}, {"a": True, "new": True})
        assert sets.fail_to_pass == ["a"]
        assert "new" not in sets.pass_to_pass

    def test_warnings_name_the_problems(self):
        sets = derive_test_sets({"a": True, "b": True}, {"a": True, "b": False})
        assert any("nothing to solve" in w for w in sets.warnings)
        assert any("breaks them" in w for w in sets.warnings)


class TestProblemStatement:
    def test_uses_the_linked_issue_when_present(self):
        pull_request = _pull_request(linked_issue="The real issue text here.")
        assert "The real issue text" in build_problem_statement(pull_request)

    def test_strips_a_leaked_diff(self):
        """A statement containing the fix turns the task into transcription."""
        pull_request = _pull_request(
            body="It is broken.\n\n```diff\n- return a - b;\n+ return a + b;\n```"
        )
        statement = build_problem_statement(pull_request)
        assert "return a + b" not in statement


class TestMiningFilters:
    def test_a_good_pull_request_passes(self):
        assert rejection_reasons(_pull_request()) == []

    def test_no_test_change_is_rejected(self):
        pull_request = _pull_request(changes=[
            FileChange("src/main/java/Calc.java", BUGGY, FIXED)
        ])
        assert any("nothing to judge" in r for r in rejection_reasons(pull_request))

    def test_no_code_change_is_rejected(self):
        pull_request = _pull_request(changes=[
            FileChange("src/test/java/CalcTest.java", "", TEST)
        ])
        assert any("nothing to revert" in r for r in rejection_reasons(pull_request))

    @pytest.mark.parametrize("title", [
        "Revert \"Fix the thing\"",
        "Merge branch 'main'",
        "Bump jackson from 2.13 to 2.14",
        "chore: update changelog",
    ])
    def test_non_fixes_are_rejected_by_title(self, title):
        reasons = rejection_reasons(_pull_request(title=title))
        assert any("revert, merge or bump" in r for r in reasons)

    def test_a_sprawling_pull_request_is_rejected(self):
        changes = [
            FileChange(f"src/main/java/F{i}.java", "a", "b") for i in range(20)
        ] + [FileChange("src/test/java/T.java", "", TEST)]
        reasons = rejection_reasons(_pull_request(changes=changes))
        assert any("refactor, not a defect fix" in r for r in reasons)

    def test_a_thin_statement_is_rejected(self):
        reasons = rejection_reasons(_pull_request(body="fix"))
        assert any("statement is" in r for r in reasons)

    def test_an_old_pull_request_is_rejected(self):
        reasons = rejection_reasons(_pull_request(merged_at="2015-01-01T00:00:00Z"))
        assert any("toolchain of its era" in r for r in reasons)

    def test_documentation_changes_do_not_count_as_code(self):
        pull_request = _pull_request(changes=[
            FileChange("README.md", "a", "b"),
            FileChange("src/test/java/T.java", "", TEST),
        ])
        assert any("nothing to revert" in r for r in rejection_reasons(pull_request))

    def test_every_reason_is_returned_not_just_the_first(self):
        """The distribution over reasons is how you learn the filters are wrong."""
        pull_request = _pull_request(title="Revert x", body="no", changes=[])
        assert len(rejection_reasons(pull_request)) >= 3

    def test_excluded_repositories(self):
        criteria = MiningCriteria(excluded_repos=frozenset({"acme/widget"}))
        assert any("excluded" in r for r in rejection_reasons(_pull_request(), criteria))

    def test_filter_returns_accepted_and_rejected(self):
        accepted, rejected = filter_pull_requests([
            _pull_request(),
            _pull_request(number=43, title="Revert something"),
        ])
        assert len(accepted) == 1
        assert "acme/widget#43" in rejected


class TestEnvironmentDetection:
    def test_maven(self):
        spec = detect_build_system({
            "pom.xml": "<project><properties><maven.compiler.release>17"
                       "</maven.compiler.release></properties></project>",
            "src/main/java/A.java": "class A {}",
        })
        assert spec.build_system is BuildSystem.MAVEN
        assert spec.jdk_version == 17
        assert "maven.compiler.release" in spec.jdk_inferred_from

    def test_gradle_kotlin_beats_a_vestigial_pom(self):
        """A stale pom.xml beside a Gradle build is common and misleads."""
        spec = detect_build_system({
            "build.gradle.kts": "java { }",
            "pom.xml": "<project/>",
        })
        assert spec.build_system is BuildSystem.GRADLE_KOTLIN
        assert any("Several build systems" in w for w in spec.warnings)

    def test_gradle_toolchain_jdk(self):
        spec = detect_build_system({
            "build.gradle": "java { toolchain { languageVersion.set("
                            "JavaLanguageVersion.of(21)) } }",
        })
        assert spec.jdk_version == 21

    def test_jdk_inferred_from_language_features(self):
        spec = detect_build_system({"A.java": "public record Point(int x, int y) {}"})
        assert spec.jdk_version >= 16
        assert "language feature" in spec.jdk_inferred_from

    def test_version_rounds_up_to_an_available_image(self):
        """Rounding down produces a compile error that looks like a model failure."""
        spec = detect_build_system({
            "pom.xml": "<project><maven.compiler.release>14</maven.compiler.release></project>"
        })
        assert spec.jdk_version == 17

    def test_multi_module_maven(self):
        spec = detect_build_system({
            "pom.xml": "<project><modules><module>core</module>"
                       "<module>api</module></modules></project>",
        })
        assert spec.multi_module is True
        assert spec.modules == ["api", "core"]

    def test_no_build_file_falls_back_to_javac(self):
        spec = detect_build_system({"A.java": "class A {}"})
        assert spec.build_system is BuildSystem.PLAIN_JAVAC
        assert spec.warnings

    def test_ant_is_not_supported(self):
        """Every Ant build is a bespoke script; 'supported' would be a lie."""
        spec = detect_build_system({"build.xml": "<project/>"})
        assert spec.build_system is BuildSystem.ANT
        assert spec.buildable is False

    def test_missing_gradle_wrapper_is_a_warning(self):
        spec = detect_build_system({"build.gradle": "java { }"})
        assert any("wrapper" in w for w in spec.warnings)


class TestBuildCommands:
    def test_maven_runs_offline(self):
        spec = detect_build_system({"pom.xml": "<project/>"})
        assert "-o" in offline_test_command(spec)

    def test_gradle_runs_offline(self):
        spec = detect_build_system({"build.gradle": "java { }"})
        assert "--offline" in offline_test_command(spec)

    def test_gradle_wrapper_is_preferred(self):
        spec = detect_build_system({"build.gradle": "java { }", "gradlew": "#!/bin/sh"})
        assert offline_test_command(spec)[0] == "./gradlew"

    def test_test_filter_is_passed_through(self):
        spec = detect_build_system({"pom.xml": "<project/>"})
        assert "-Dtest=FooTest" in offline_test_command(spec, "FooTest")

    def test_unsupported_build_system_raises(self):
        spec = detect_build_system({"build.xml": "<project/>"})
        with pytest.raises(ValueError, match="No offline test command"):
            offline_test_command(spec)

    def test_dockerfile_prewarms_dependencies(self):
        """Prewarming is what lets the task itself run with --network=none."""
        spec = detect_build_system({"pom.xml": "<project/>"})
        dockerfile = generate_dockerfile(spec)
        assert "dependency:go-offline" in dockerfile
        assert "FROM eclipse-temurin" in dockerfile

    def test_dockerfile_warns_about_an_unpinned_base_image(self):
        spec = detect_build_system({"pom.xml": "<project/>"})
        assert "pinned by tag, not digest" in generate_dockerfile(spec)

    def test_dockerfile_refuses_an_unsupported_build(self):
        spec = detect_build_system({"build.xml": "<project/>"})
        with pytest.raises(ValueError):
            generate_dockerfile(spec)


class TestValidation:
    def _instance(self, **kwargs) -> TaskInstance:
        instance = build_instance(_pull_request())
        # build_instance runs no tests, so the sets are empty. Supply a
        # usable pair so the gates past TARGET_FAILS_BEFORE are reachable.
        instance.test_sets = derive_test_sets(
            before_results={"CalcTest.t": False, "CalcTest.other": True},
            after_results={"CalcTest.t": True, "CalcTest.other": True},
        )
        for key, value in kwargs.items():
            setattr(instance, key, value)
        return instance

    def test_a_stable_suite_passes_the_flake_gate(self):
        result = check_flakiness(lambda: {"a": True, "b": False}, runs=5)
        assert result.passed is True

    def test_an_unstable_test_fails_the_flake_gate(self):
        calls = {"n": 0}

        def flaky() -> dict[str, bool]:
            calls["n"] += 1
            return {"a": calls["n"] % 2 == 0}

        result = check_flakiness(flaky, runs=5)
        assert result.passed is False
        assert "a" in result.evidence["unstable_tests"]

    def test_the_flake_gate_needs_more_than_one_run(self):
        with pytest.raises(ValueError):
            check_flakiness(lambda: {}, runs=1)

    def test_a_raising_runner_fails_the_gate(self):
        def explode() -> dict[str, bool]:
            raise RuntimeError("container died")

        assert check_flakiness(explode, runs=3).passed is False

    def test_a_thin_statement_fails_solvability(self):
        result = check_solvability(self._instance(problem_statement="fix it"))
        assert result.passed is False
        assert "measure guessing" in result.detail

    def test_a_statement_with_no_failure_word_fails(self):
        result = check_solvability(self._instance(
            problem_statement="It would be nice if this module were faster "
                              "and generally more pleasant to work with overall."
        ))
        assert result.passed is False

    def test_a_good_statement_passes(self):
        assert check_solvability(self._instance()).passed is True

    def test_execution_gates_fail_without_a_runner(self):
        """An unrun gate is not a passed gate."""
        report = validate_instance(self._instance())
        assert report.admitted is False
        assert Gate.NOT_FLAKY in report.failed_gates
        flake = next(r for r in report.results if r.gate is Gate.NOT_FLAKY)
        assert "Not verified" in flake.detail

    def test_a_task_with_no_target_test_is_rejected_immediately(self):
        from taskfactory.patches import TestOutcomeSet

        report = validate_instance(self._instance(test_sets=TestOutcomeSet()))
        assert report.admitted is False
        assert Gate.TARGET_FAILS_BEFORE in report.failed_gates


class TestContamination:
    def test_pre_cutoff_is_flagged(self):
        risk = assess_contamination("i", "acme/widget", "2020-01-01", "gpt-4o")
        assert risk.stratum == "pre_cutoff"
        assert risk.memorisation_likely is True

    def test_post_cutoff_is_not_flagged(self):
        risk = assess_contamination("i", "acme/widget", "2026-01-01", "gpt-4o")
        assert risk.stratum == "post_cutoff"
        assert risk.memorisation_likely is False

    def test_near_cutoff_goes_in_neither_stratum(self):
        """Cutoffs are approximate; forcing a task into a stratum is dishonest."""
        risk = assess_contamination("i", "acme/widget", "2023-10-15", "gpt-4o")
        assert risk.stratum == "near_cutoff"

    def test_a_high_exposure_repo_is_flagged_even_post_cutoff(self):
        repo = sorted(HIGH_EXPOSURE_REPOS)[0]
        risk = assess_contamination("i", repo, "2026-06-01", "gpt-4o")
        assert risk.stratum == "post_cutoff"
        assert risk.memorisation_likely is True

    def test_missing_date_cannot_be_assessed(self):
        risk = assess_contamination("i", "acme/widget", None, "gpt-4o")
        assert risk.stratum == "unknown"
        assert any("cannot be recovered later" in n for n in risk.notes)

    def test_unknown_model_is_reported(self):
        risk = assess_contamination("i", "acme/widget", "2024-01-01", "nonexistent")
        assert any("No training cutoff known" in n for n in risk.notes)

    def test_stratify_reports_the_gap(self):
        outcomes = (
            [{"instance_id": f"pre{i}", "repo": "acme/w", "commit_date": "2020-01-01",
              "model_id": "gpt-4o", "solved": True} for i in range(10)]
            + [{"instance_id": f"post{i}", "repo": "acme/w", "commit_date": "2026-01-01",
                "model_id": "gpt-4o", "solved": i < 3} for i in range(10)]
        )
        result = stratify_by_cutoff(outcomes)
        assert result["contamination_gap"] == pytest.approx(0.7)
        assert "large contamination signal" in result["interpretation"]

    def test_no_gap_is_reported_honestly(self):
        """'Consistent with little contamination' is not 'no contamination'."""
        outcomes = (
            [{"instance_id": f"a{i}", "repo": "r", "commit_date": "2020-01-01",
              "model_id": "gpt-4o", "solved": i < 5} for i in range(10)]
            + [{"instance_id": f"b{i}", "repo": "r", "commit_date": "2026-01-01",
                "model_id": "gpt-4o", "solved": i < 5} for i in range(10)]
        )
        result = stratify_by_cutoff(outcomes)
        assert result["contamination_gap"] == pytest.approx(0.0)
        assert "equally consistent with too few tasks" in result["interpretation"]

    def test_one_sided_data_cannot_be_compared(self):
        outcomes = [{"instance_id": "a", "repo": "r", "commit_date": "2020-01-01",
                     "model_id": "gpt-4o", "solved": True}]
        assert stratify_by_cutoff(outcomes)["contamination_gap"] is None


class TestPipeline:
    def test_builds_an_instance(self):
        instance = build_instance(_pull_request())
        assert instance.instance_id == "acme__widget-42"
        assert instance.broken_files["src/main/java/Calc.java"] == BUGGY
        assert instance.golden_files == ["src/main/java/Calc.java"]

    def test_the_golden_patch_is_excluded_by_default(self):
        """A dataset anyone can point a model at must not contain the answers."""
        payload = build_instance(_pull_request()).to_dict()
        assert "golden_patch" not in payload
        assert "golden_files" in payload

    def test_nothing_is_admitted_without_runners(self):
        result = run_pipeline([_pull_request()])
        assert result.admitted == []
        assert result.reports

    def test_mining_rejections_are_reported(self):
        result = run_pipeline([_pull_request(number=7, title="Revert it")])
        assert "acme/widget#7" in result.mining_rejections
        assert result.summary()["rejected_at_mining"] == 1

    def test_summary_names_the_first_failure_gate(self):
        summary = run_pipeline([_pull_request()]).summary()
        assert summary["validation"]["first_failure_by_gate"]

    def test_writes_instances_and_answers_separately(self, tmp_path):
        result = run_pipeline([_pull_request()])
        result.admitted.append(build_instance(_pull_request()))
        out = result.write(tmp_path)

        instances = (out / "instances.jsonl").read_text(encoding="utf-8")
        assert "golden_patch" not in instances
        assert (out / "golden_patches.jsonl").exists()
        assert (out / "rejections.json").exists()
        assert (out / "summary.json").exists()
