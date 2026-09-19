"""
taskfactory.pipeline — mine, build, validate, admit.

The entry point that chains the other modules into one run.

::

    python -m taskfactory.pipeline --repos apache/commons-lang --limit 20
    python -m taskfactory.pipeline --from-cache candidates.json --validate

**Running this for real needs a GitHub token, network access, Docker, and
hours.** Mining 150-300 validated instances means thousands of API calls and
a Maven or Gradle build per candidate, five times over for the flake gate.
The machinery is here and tested; producing the dataset is a job you start
and come back to, not a function call.

What the pipeline guarantees: nothing reaches the output without passing
every gate in :mod:`taskfactory.validation`, and every rejection is written
to the rejection log with its reason. The rejection log is not a byproduct —
it is how you find out that the pipeline cannot parse a build convention
that 40% of your candidates use.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from taskfactory.contamination import assess_contamination
from taskfactory.environment import detect_build_system
from taskfactory.mining import GitHubClient, MiningCriteria, filter_pull_requests, hydrate_changes
from taskfactory.patches import (
    PullRequest,
    TaskInstance,
    build_golden_patch,
    build_problem_statement,
    build_reverse_patch,
)
from taskfactory.validation import ValidationReport, summarise_rejections, validate_instance

logger = logging.getLogger(__name__)

__all__ = ["PipelineResult", "build_instance", "run_pipeline", "main"]


@dataclass
class PipelineResult:
    """What one pipeline run produced."""

    admitted: list[TaskInstance] = field(default_factory=list)
    reports: list[ValidationReport] = field(default_factory=list)
    mining_rejections: dict[str, list[str]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "admitted": len(self.admitted),
            "validated": len(self.reports),
            "rejected_at_mining": len(self.mining_rejections),
            "errors": len(self.errors),
            "validation": summarise_rejections(self.reports),
            "mining_rejection_reasons": _count_reasons(self.mining_rejections),
        }

    def write(self, out_dir: Path | str) -> Path:
        """Write instances, reports and rejection logs to *out_dir*.

        The golden patches go in a separate file. A dataset directory that
        anyone can point a model at must not contain the answers, and
        keeping them in the same file as the task is how that happens by
        accident.
        """
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)

        with (root / "instances.jsonl").open("w", encoding="utf-8") as handle:
            for instance in self.admitted:
                handle.write(json.dumps(instance.to_dict(), default=str) + "\n")

        with (root / "golden_patches.jsonl").open("w", encoding="utf-8") as handle:
            for instance in self.admitted:
                handle.write(json.dumps(
                    {"instance_id": instance.instance_id,
                     "golden_patch": instance.golden_patch},
                    default=str,
                ) + "\n")

        (root / "validation.json").write_text(
            json.dumps([report.to_dict() for report in self.reports], indent=2),
            encoding="utf-8",
        )
        (root / "rejections.json").write_text(
            json.dumps(
                {"mining": self.mining_rejections, "errors": self.errors},
                indent=2,
            ),
            encoding="utf-8",
        )
        (root / "summary.json").write_text(
            json.dumps(self.summary(), indent=2), encoding="utf-8"
        )
        return root


def _count_reasons(rejections: dict[str, list[str]]) -> dict[str, int]:
    """How often each rejection reason fired, most common first."""
    from collections import Counter

    counter: Counter[str] = Counter()
    for reasons in rejections.values():
        for reason in reasons:
            # Collapse the numbers out so reasons group.
            counter[reason.split(";")[0].split("(")[0].strip()] += 1
    return dict(counter.most_common())


def build_instance(pull_request: PullRequest) -> TaskInstance:
    """Turn an accepted pull request into a candidate task.

    Not validated yet: this constructs the artefact, and
    :func:`~taskfactory.validation.validate_instance` decides whether it is
    admissible. Keeping those separate means a rejected candidate still
    leaves behind something a human can inspect.
    """
    broken = build_reverse_patch(pull_request)
    golden = build_golden_patch(pull_request)
    spec = detect_build_system(broken)

    return TaskInstance(
        instance_id=f"{pull_request.repo.replace('/', '__')}-{pull_request.number}",
        repo=pull_request.repo,
        base_commit=pull_request.base_commit,
        pull_request=pull_request.number,
        problem_statement=build_problem_statement(pull_request),
        broken_files=broken,
        golden_patch=golden,
        environment=spec.to_dict(),
        metadata={
            "merged_at": pull_request.merged_at,
            "merge_commit": pull_request.merge_commit,
            "title": pull_request.title,
            "files_changed": len(pull_request.changes),
            "lines_changed": pull_request.total_lines_changed,
        },
    )


def run_pipeline(
    pull_requests: Iterable[PullRequest],
    criteria: MiningCriteria | None = None,
    run_before: Callable[[TaskInstance], dict[str, bool]] | None = None,
    run_after: Callable[[TaskInstance], dict[str, bool]] | None = None,
    models_for_contamination: Iterable[str] = (),
) -> PipelineResult:
    """Filter, build and validate every candidate.

    *run_before* and *run_after* execute a task's suite against the broken
    and fixed repositories. Without them the execution gates record as
    failed — never as passed — and nothing is admitted, which is the
    correct behaviour: a task nobody ran is a task nobody has validated.
    """
    result = PipelineResult()
    accepted, result.mining_rejections = filter_pull_requests(pull_requests, criteria)

    for pull_request in accepted:
        key = f"{pull_request.repo}#{pull_request.number}"
        try:
            instance = build_instance(pull_request)
        except Exception as exc:
            logger.exception("Could not build an instance from %s", key)
            result.errors[key] = f"{type(exc).__name__}: {exc}"
            continue

        if models_for_contamination:
            instance.metadata["contamination"] = [
                assess_contamination(
                    instance.instance_id, instance.repo,
                    pull_request.merged_at, model,
                ).to_dict()
                for model in models_for_contamination
            ]

        report = validate_instance(
            instance,
            run_before=(lambda i=instance: run_before(i)) if run_before else None,
            run_after=(lambda i=instance: run_after(i)) if run_after else None,
        )
        result.reports.append(report)
        if report.admitted:
            result.admitted.append(instance)
        else:
            logger.info("Rejected %s: %s", instance.instance_id, report.rejection_reason)

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m taskfactory.pipeline",
        description="Mine, build and validate JVM repository tasks.",
    )
    parser.add_argument("--repos", nargs="*", default=[],
                        help="owner/name repositories to mine.")
    parser.add_argument("--limit", type=int, default=20,
                        help="Pull requests to consider per repository.")
    parser.add_argument("--out", default="data/mined",
                        help="Output directory.")
    parser.add_argument("--from-cache", type=Path, default=None,
                        help="A JSON file of previously fetched pull requests, "
                             "so filters can be re-run without the network.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Filter and build only. No network, no Docker, "
                             "nothing admitted — the execution gates cannot "
                             "pass without a runner.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr,
        format="%(levelname)-8s %(name)s — %(message)s",
    )

    pull_requests: list[PullRequest] = []
    if args.from_cache:
        payload = json.loads(args.from_cache.read_text(encoding="utf-8"))
        pull_requests = [_pull_request_from_dict(row) for row in payload]
        print(f"Loaded {len(pull_requests)} pull request(s) from cache.")
    elif args.repos:
        client = GitHubClient()
        if not client.authenticated:
            print(
                "No GITHUB_TOKEN. Unauthenticated search allows 10 requests a "
                "minute, which cannot mine a dataset. Set GITHUB_TOKEN and "
                "retry.",
                file=sys.stderr,
            )
            return 2
        for repo in args.repos:
            print(f"Mining {repo}...")
            items = client.search_pull_requests(
                f"repo:{repo} is:pr is:merged label:bug", per_page=min(args.limit, 100)
            )
            for item in items[: args.limit]:
                try:
                    pull_request = client.fetch_pull_request(repo, item["number"])
                    pull_requests.append(hydrate_changes(client, pull_request))
                except Exception as exc:
                    print(f"  skipped #{item.get('number')}: {exc}", file=sys.stderr)
    else:
        parser.print_help()
        return 2

    result = run_pipeline(pull_requests)
    summary = result.summary()

    print(json.dumps(summary, indent=2))
    if args.dry_run:
        print(
            "\nDry run: the execution gates cannot pass without a runner, so "
            "nothing was admitted. This shows which candidates survive the "
            "cheap filters.",
            file=sys.stderr,
        )
        return 0

    out = result.write(args.out)
    print(f"\nWrote {len(result.admitted)} admitted instance(s) to {out}")
    if not result.admitted:
        print(
            "Nothing was admitted. Read rejections.json: the first-failure "
            "distribution says where the pipeline gives up.",
            file=sys.stderr,
        )
    return 0


def _pull_request_from_dict(row: dict[str, Any]) -> PullRequest:
    from taskfactory.patches import FileChange

    return PullRequest(
        repo=row.get("repo", ""),
        number=int(row.get("number", 0)),
        title=row.get("title", ""),
        body=row.get("body", ""),
        merge_commit=row.get("merge_commit", ""),
        base_commit=row.get("base_commit", ""),
        merged_at=row.get("merged_at", ""),
        linked_issue=row.get("linked_issue", ""),
        changes=[
            FileChange(
                path=change["path"],
                before=change.get("before"),
                after=change.get("after"),
            )
            for change in row.get("changes", [])
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
