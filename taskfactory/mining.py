"""
taskfactory.mining — finding pull requests worth turning into tasks.

The search is the cheap part. The filtering is what determines whether the
dataset is any good, and every filter here exists because its absence
produces a specific kind of bad task:

``touches code and tests``
    Without a test change there is nothing to judge a solution by. Without a
    code change there is nothing to revert.

``bounded size``
    A PR touching 40 files is a refactor or a merge, not a defect fix. A
    task built from one is unsolvable and will sit in the dataset at a 0%
    resolve rate, dragging every aggregate down and telling you nothing.

``has a real statement``
    "Fix build" is not a task description.

``not a revert, not a merge, not a dependency bump``
    All three are common, all three look like fixes to a naive filter, and
    none of them is one.

``recent enough to build``
    A 2014 PR needs a 2014 toolchain. The environment gate would reject it
    later; rejecting it here is cheaper.

Network access lives in :class:`GitHubClient` and nowhere else, so every
filter is testable against fixtures without a token.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable

from taskfactory.patches import PullRequest

logger = logging.getLogger(__name__)

__all__ = [
    "MiningCriteria",
    "GitHubClient",
    "filter_pull_requests",
    "rejection_reasons",
    "DEFAULT_SEARCH_QUERY",
]

#: Titles that mean this is not a defect fix.
_NOT_A_FIX = re.compile(
    r"^\s*(revert|merge |bump |chore|release |prepare release|"
    r"update (dependency|dependencies|changelog|readme)|"
    r"\[maven-release-plugin\])",
    re.IGNORECASE,
)

#: Paths whose changes never constitute a code fix.
_IGNORABLE_PATHS = re.compile(
    r"(^|/)(\.github/|docs?/|\.mvn/|gradle/wrapper/)|"
    r"\.(md|txt|rst|adoc|png|jpg|svg|yml|yaml|properties|lock)$",
    re.IGNORECASE,
)

#: GitHub search that finds merged Java PRs referencing a closed bug.
DEFAULT_SEARCH_QUERY = (
    "language:Java is:pr is:merged "
    "label:bug "
    "sort:updated-desc"
)


@dataclass
class MiningCriteria:
    """What makes a pull request worth building a task from.

    Every bound has a reason and the reasons are in the class docstring of
    this module. They are exposed rather than hard-coded so a study can
    state the values it used — a dataset built with different bounds is a
    different dataset.
    """

    min_code_files: int = 1
    max_code_files: int = 5
    min_test_files: int = 1
    max_total_files: int = 12
    max_lines_changed: int = 400
    min_statement_chars: int = 80
    earliest_merge_date: date = date(2020, 1, 1)
    #: Repositories to skip entirely. Contamination-heavy ones can go here
    #: when a study wants a clean post-cutoff set.
    excluded_repos: frozenset[str] = field(default_factory=frozenset)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_code_files": self.min_code_files,
            "max_code_files": self.max_code_files,
            "min_test_files": self.min_test_files,
            "max_total_files": self.max_total_files,
            "max_lines_changed": self.max_lines_changed,
            "min_statement_chars": self.min_statement_chars,
            "earliest_merge_date": self.earliest_merge_date.isoformat(),
            "excluded_repos": sorted(self.excluded_repos),
        }


def _significant_changes(pull_request: PullRequest) -> list[Any]:
    """Changes that are neither documentation nor build scaffolding."""
    return [
        change for change in pull_request.changes
        if not _IGNORABLE_PATHS.search(change.path)
    ]


def rejection_reasons(
    pull_request: PullRequest,
    criteria: MiningCriteria | None = None,
) -> list[str]:
    """Every reason *pull_request* is unsuitable. Empty means suitable.

    Returns all reasons rather than the first, because the distribution over
    reasons is what tells you whether the criteria are too strict — and that
    question is unanswerable if every PR reports only its first failure.
    """
    criteria = criteria or MiningCriteria()
    reasons: list[str] = []

    if pull_request.repo in criteria.excluded_repos:
        reasons.append(f"repository {pull_request.repo} is excluded")

    if _NOT_A_FIX.match(pull_request.title or ""):
        reasons.append(f"title looks like a revert, merge or bump: {pull_request.title!r}")

    significant = _significant_changes(pull_request)
    code = [c for c in significant if not c.is_test]
    tests = [c for c in significant if c.is_test]

    if len(code) < criteria.min_code_files:
        reasons.append("no code file changed; nothing to revert")
    elif len(code) > criteria.max_code_files:
        reasons.append(
            f"{len(code)} code files changed (max {criteria.max_code_files}); "
            f"a PR this wide is a refactor, not a defect fix"
        )

    if len(tests) < criteria.min_test_files:
        reasons.append("no test file changed; nothing to judge a solution by")

    if len(significant) > criteria.max_total_files:
        reasons.append(
            f"{len(significant)} files changed (max {criteria.max_total_files})"
        )

    lines = sum(change.lines_changed for change in significant)
    if lines > criteria.max_lines_changed:
        reasons.append(
            f"{lines} lines changed (max {criteria.max_lines_changed}); the "
            f"resulting task would be unsolvable and would sit at 0%"
        )

    statement = (pull_request.linked_issue or pull_request.body or "").strip()
    if len(statement) < criteria.min_statement_chars:
        reasons.append(
            f"statement is {len(statement)} characters "
            f"(min {criteria.min_statement_chars})"
        )

    merged = _parse_date(pull_request.merged_at)
    if merged is None:
        reasons.append("no merge date; contamination cannot be assessed")
    elif merged < criteria.earliest_merge_date:
        reasons.append(
            f"merged {merged.isoformat()}, before "
            f"{criteria.earliest_merge_date.isoformat()}; needs a toolchain "
            f"of its era"
        )

    if not pull_request.base_commit:
        reasons.append("no base commit recorded; the task cannot be reconstructed")

    return reasons


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        return None


def filter_pull_requests(
    pull_requests: Iterable[PullRequest],
    criteria: MiningCriteria | None = None,
) -> tuple[list[PullRequest], dict[str, list[str]]]:
    """Split candidates into (accepted, {pr key: reasons rejected}).

    The rejection map is returned, not logged and discarded. It is what says
    whether the criteria are wrong: a pipeline rejecting 95% of candidates
    for "no test file changed" has found a repository convention it does not
    understand, not a repository without tests.
    """
    criteria = criteria or MiningCriteria()
    accepted: list[PullRequest] = []
    rejected: dict[str, list[str]] = {}

    for pull_request in pull_requests:
        reasons = rejection_reasons(pull_request, criteria)
        key = f"{pull_request.repo}#{pull_request.number}"
        if reasons:
            rejected[key] = reasons
        else:
            accepted.append(pull_request)

    logger.info(
        "Mining filter: %d accepted, %d rejected", len(accepted), len(rejected)
    )
    return accepted, rejected


class GitHubClient:
    """The only thing here that touches the network.

    Isolated deliberately: every filter above is a pure function over
    :class:`~taskfactory.patches.PullRequest` objects and is tested against
    fixtures, so the pipeline's judgement can be verified without a token
    and without hitting rate limits.

    Parameters
    ----------
    token
        A GitHub token. Falls back to ``GITHUB_TOKEN``. Unauthenticated
        search is limited to 10 requests a minute, which makes mining at
        any scale impossible rather than merely slow.
    """

    API_ROOT = "https://api.github.com"

    def __init__(self, token: str | None = None, session: Any = None) -> None:
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self._session = session

    @property
    def authenticated(self) -> bool:
        return bool(self.token)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """One GET against the API. Raises with a usable message."""
        if self._session is None:
            import urllib.error
            import urllib.parse
            import urllib.request

            url = f"{self.API_ROOT}{path}"
            if params:
                url = f"{url}?{urllib.parse.urlencode(params)}"
            request = urllib.request.Request(url, headers=self._headers())
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    import json

                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code == 403:
                    raise RuntimeError(
                        "GitHub rate limit or permission error. "
                        + ("Check the token's scopes." if self.token else
                           "Set GITHUB_TOKEN: unauthenticated search allows "
                           "10 requests a minute, which cannot mine a dataset.")
                    ) from exc
                raise RuntimeError(f"GitHub API {exc.code} for {path}") from exc

        response = self._session.get(
            f"{self.API_ROOT}{path}", headers=self._headers(), params=params, timeout=30
        )
        response.raise_for_status()
        return response.json()

    def search_pull_requests(
        self,
        query: str = DEFAULT_SEARCH_QUERY,
        per_page: int = 50,
        pages: int = 1,
    ) -> list[dict[str, Any]]:
        """Search for pull requests. Returns raw API items.

        GitHub's search API caps results at 1,000 per query regardless of
        paging, so mining at scale means many narrow queries — per
        repository, per date window — rather than one broad one.
        """
        items: list[dict[str, Any]] = []
        for page in range(1, pages + 1):
            payload = self._get(
                "/search/issues",
                {"q": query, "per_page": per_page, "page": page},
            )
            page_items = payload.get("items", [])
            items.extend(page_items)
            if len(page_items) < per_page:
                break
        return items

    def fetch_pull_request(self, repo: str, number: int) -> PullRequest:
        """Fetch one PR with its file changes.

        Note the cost: one call for the PR, one for its files, one per
        changed file for its content, and one more if it links an issue.
        Mining 300 instances is thousands of calls, which is why the
        pipeline caches aggressively and why a token is not optional.
        """
        from taskfactory.patches import FileChange

        data = self._get(f"/repos/{repo}/pulls/{number}")
        files = self._get(f"/repos/{repo}/pulls/{number}/files", {"per_page": 100})

        changes: list[FileChange] = []
        for entry in files:
            status = entry.get("status", "")
            changes.append(FileChange(
                path=entry.get("filename", ""),
                before=None if status == "added" else "",
                after=None if status == "removed" else "",
            ))

        body = data.get("body") or ""
        linked = ""
        match = re.search(r"(?:closes|fixes|resolves)\s+#(\d+)", body, re.IGNORECASE)
        if match:
            try:
                issue = self._get(f"/repos/{repo}/issues/{match.group(1)}")
                linked = f"{issue.get('title', '')}\n\n{issue.get('body') or ''}"
            except RuntimeError as exc:
                logger.debug("Could not fetch linked issue: %s", exc)

        return PullRequest(
            repo=repo,
            number=number,
            title=data.get("title") or "",
            body=body,
            merge_commit=data.get("merge_commit_sha") or "",
            base_commit=(data.get("base") or {}).get("sha") or "",
            merged_at=data.get("merged_at") or "",
            changes=changes,
            linked_issue=linked,
        )

    def fetch_file(self, repo: str, path: str, ref: str) -> str | None:
        """File content at a ref, or None when it does not exist there."""
        import base64

        try:
            payload = self._get(f"/repos/{repo}/contents/{path}", {"ref": ref})
        except RuntimeError:
            return None
        if isinstance(payload, list) or payload.get("encoding") != "base64":
            return None
        try:
            return base64.b64decode(payload["content"]).decode("utf-8")
        except (KeyError, ValueError, UnicodeDecodeError):
            return None


def hydrate_changes(
    client: GitHubClient,
    pull_request: PullRequest,
) -> PullRequest:
    """Fill in before/after content for every changed file.

    Split from :meth:`GitHubClient.fetch_pull_request` because it is the
    expensive half — two calls per changed file — and a PR rejected by the
    cheap filters should never reach it.
    """
    for change in pull_request.changes:
        if change.before is not None:
            change.before = client.fetch_file(
                pull_request.repo, change.path, pull_request.base_commit
            ) or ""
        if change.after is not None:
            change.after = client.fetch_file(
                pull_request.repo, change.path, pull_request.merge_commit
            ) or ""
    return pull_request
