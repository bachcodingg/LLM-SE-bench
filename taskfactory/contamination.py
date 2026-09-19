"""
taskfactory.contamination — training-data contamination checks.

Popular Apache and Spring repositories are almost certainly in every
model's training data. A task mined from a 2019 commit to Commons Lang does
not measure whether a model can fix the bug; it measures whether it
remembers the fix.

This cannot be solved, only measured and disclosed. What it does:

* record each task's commit date against each model's training cutoff;
* stratify results into pre-cutoff and post-cutoff;
* report the gap between the two strata, which is the contamination signal;
* flag repositories whose popularity makes memorisation near-certain.

The honest position is the one the guide states: record the dates, stratify
the results, and state the limitation loudly. A benchmark that quietly mixes
memorised and unmemorised tasks reports a capability number that is partly a
recall number, and nobody can tell which part.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable

logger = logging.getLogger(__name__)

__all__ = [
    "TRAINING_CUTOFFS",
    "HIGH_EXPOSURE_REPOS",
    "ContaminationRisk",
    "assess_contamination",
    "stratify_by_cutoff",
]

#: Stated or best-known training cutoffs, as dates. These are published
#: approximations and providers update models behind a stable id, so a
#: cutoff can move without the name changing. Treat a task near a cutoff as
#: unclassifiable rather than as safely post-cutoff.
TRAINING_CUTOFFS: dict[str, date] = {
    "claude-opus-4-7": date(2026, 5, 1),
    "claude-sonnet-4-6": date(2026, 5, 1),
    "claude-haiku-4-5-20251001": date(2025, 10, 1),
    "claude-3-5-sonnet-20241022": date(2024, 4, 1),
    "gpt-4o": date(2023, 10, 1),
    "gpt-4-turbo": date(2023, 12, 1),
    "gemini-2.5-pro": date(2025, 1, 1),
    "gemini-2.5-flash": date(2025, 1, 1),
    "gemini-2.0-flash": date(2024, 8, 1),
}

#: Margin around a cutoff within which a task is treated as unclassifiable.
#: A commit three weeks before a stated cutoff may or may not be in the
#: corpus, and forcing it into a stratum makes that stratum dishonest.
CUTOFF_MARGIN_DAYS = 90

#: Repositories whose code is so widely reproduced — in tutorials, in Stack
#: Overflow answers, in forks — that a model has likely seen the fix even
#: for a recent commit. Membership is a judgement call and is listed openly
#: so it can be argued with.
HIGH_EXPOSURE_REPOS: frozenset[str] = frozenset({
    "apache/commons-lang",
    "apache/commons-math",
    "apache/commons-collections",
    "apache/commons-io",
    "google/guava",
    "google/gson",
    "spring-projects/spring-framework",
    "spring-projects/spring-boot",
    "junit-team/junit4",
    "junit-team/junit5",
    "mockito/mockito",
    "JodaOrg/joda-time",
    "jfree/jfreechart",
    "google/closure-compiler",
    "FasterXML/jackson-databind",
    "square/retrofit",
    "square/okhttp",
    "elastic/elasticsearch",
    "netty/netty",
})


@dataclass
class ContaminationRisk:
    """How likely it is that a model has memorised this task."""

    instance_id: str
    repo: str
    commit_date: date | None
    model_id: str
    cutoff: date | None
    stratum: str = "unknown"  # "pre_cutoff" | "post_cutoff" | "near_cutoff" | "unknown"
    high_exposure_repo: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def memorisation_likely(self) -> bool:
        """True when the result should be read as an upper bound.

        High-exposure repositories count even post-cutoff: their code is
        reproduced in tutorials and forks faster than any cutoff moves.
        """
        return self.stratum == "pre_cutoff" or self.high_exposure_repo

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "model_id": self.model_id,
            "commit_date": self.commit_date.isoformat() if self.commit_date else None,
            "cutoff": self.cutoff.isoformat() if self.cutoff else None,
            "stratum": self.stratum,
            "high_exposure_repo": self.high_exposure_repo,
            "memorisation_likely": self.memorisation_likely,
            "notes": self.notes,
        }


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        return None


def assess_contamination(
    instance_id: str,
    repo: str,
    commit_date: Any,
    model_id: str,
) -> ContaminationRisk:
    """Classify one (task, model) pair against the model's training cutoff."""
    parsed = _as_date(commit_date)
    cutoff = TRAINING_CUTOFFS.get(model_id)

    risk = ContaminationRisk(
        instance_id=instance_id,
        repo=repo,
        commit_date=parsed,
        model_id=model_id,
        cutoff=cutoff,
        high_exposure_repo=repo in HIGH_EXPOSURE_REPOS,
    )

    if parsed is None:
        risk.notes.append(
            "No commit date recorded, so contamination cannot be assessed. "
            "Record the date at mining time; it cannot be recovered later "
            "without the repository."
        )
    elif cutoff is None:
        risk.notes.append(
            f"No training cutoff known for {model_id}. Add one to "
            f"TRAINING_CUTOFFS before publishing a stratified result."
        )
    else:
        delta_days = (parsed - cutoff).days
        if abs(delta_days) <= CUTOFF_MARGIN_DAYS:
            risk.stratum = "near_cutoff"
            risk.notes.append(
                f"Commit is {abs(delta_days)} day(s) from the stated cutoff, "
                f"inside the {CUTOFF_MARGIN_DAYS}-day margin. Cutoffs are "
                f"approximate and move silently; this task belongs in "
                f"neither stratum."
            )
        elif delta_days < 0:
            risk.stratum = "pre_cutoff"
            risk.notes.append(
                "Commit predates the training cutoff. Read any result on "
                "this task as an upper bound on capability."
            )
        else:
            risk.stratum = "post_cutoff"

    if risk.high_exposure_repo:
        risk.notes.append(
            f"{repo} is widely reproduced in tutorials, answers and forks. "
            f"Its code can reach a corpus faster than any cutoff moves, so "
            f"post-cutoff status is weak protection here."
        )

    return risk


def stratify_by_cutoff(
    outcomes: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Split results by contamination stratum and report the gap.

    Each outcome needs ``instance_id``, ``repo``, ``commit_date``,
    ``model_id`` and ``solved``.

    The gap between pre-cutoff and post-cutoff resolve rate is the
    contamination signal. A large positive gap means the model does
    noticeably better on code it has probably seen, which is the thing
    everyone suspects and almost nobody measures.
    """
    rows = list(outcomes)
    if not rows:
        return {"episodes": 0}

    strata: dict[str, list[bool]] = {
        "pre_cutoff": [], "post_cutoff": [], "near_cutoff": [], "unknown": [],
    }
    high_exposure: list[bool] = []
    risks: list[ContaminationRisk] = []

    for row in rows:
        risk = assess_contamination(
            instance_id=str(row.get("instance_id", "")),
            repo=str(row.get("repo", "")),
            commit_date=row.get("commit_date"),
            model_id=str(row.get("model_id", "")),
        )
        risks.append(risk)
        solved = bool(row.get("solved", False))
        strata[risk.stratum].append(solved)
        if risk.high_exposure_repo:
            high_exposure.append(solved)

    def rate(values: list[bool]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    pre = rate(strata["pre_cutoff"])
    post = rate(strata["post_cutoff"])
    gap = round(pre - post, 4) if pre is not None and post is not None else None

    interpretation: str
    if gap is None:
        interpretation = (
            "Not enough tasks in both strata to compare. A contamination "
            "claim needs tasks on both sides of the cutoff."
        )
    elif gap >= 0.15:
        interpretation = (
            f"Resolve rate is {gap:.0%} higher on pre-cutoff tasks. That is a "
            f"large contamination signal: the headline number is partly a "
            f"recall number and the post-cutoff figure is the more honest one."
        )
    elif gap >= 0.05:
        interpretation = (
            f"Resolve rate is {gap:.0%} higher on pre-cutoff tasks. Suggestive "
            f"but within what task-difficulty differences could explain. "
            f"Report both strata."
        )
    elif gap <= -0.05:
        interpretation = (
            f"Resolve rate is {abs(gap):.0%} *higher* on post-cutoff tasks, "
            f"which contamination does not explain. The strata probably differ "
            f"in difficulty; do not read this as evidence of no contamination."
        )
    else:
        interpretation = (
            "No meaningful gap between strata. Consistent with little "
            "contamination, and equally consistent with too few tasks to see "
            "it."
        )

    return {
        "episodes": len(rows),
        "strata": {
            name: {"episodes": len(values), "resolve_rate": rate(values)}
            for name, values in strata.items()
        },
        "high_exposure_repos": {
            "episodes": len(high_exposure),
            "resolve_rate": rate(high_exposure),
        },
        "contamination_gap": gap,
        "interpretation": interpretation,
        "risks": [risk.to_dict() for risk in risks[:50]],
        "disclosure": (
            "Training cutoffs are published approximations and providers "
            "update models behind a stable id. Tasks within "
            f"{CUTOFF_MARGIN_DAYS} days of a cutoff are placed in neither "
            "stratum rather than forced into one."
        ),
    }
