"""
expert_rating.py — Structured rubric and CSV collection framework for expert
code quality ratings.

Supports both human raters and automated rubric-based scoring.  Each rating
covers five dimensions on a 0–10 scale:

    correctness   – Does the code solve the problem correctly?
    readability   – Is the code easy to understand?
    efficiency    – Are algorithms and data structures appropriate?
    design        – Is the OO design sound (SOLID, low coupling, etc.)?
    overall       – Holistic assessment incorporating all dimensions.

CSV format:
    rating_id, response_id, problem_id, rater_id,
    correctness, readability, efficiency, design, overall, comments

The module also provides an automated rubric scorer that derives ratings
from CK metrics, complexity scores, and readability heuristics — useful as
a baseline or when human raters are unavailable.
"""

from __future__ import annotations

import csv
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rubric dimensions & scale
# ---------------------------------------------------------------------------

DIMENSIONS = ("correctness", "readability", "efficiency", "design", "overall")

RUBRIC_DESCRIPTORS = {
    "correctness": {
        (9, 10): "Fully correct: all tests pass, edge cases handled.",
        (7, 8): "Mostly correct: minor issues, most tests pass.",
        (5, 6): "Partially correct: some tests fail, logic errors.",
        (3, 4): "Largely incorrect: fundamental logic flaws.",
        (0, 2): "Non-functional: does not compile or crashes.",
    },
    "readability": {
        (9, 10): "Excellent: clean naming, well-documented, easy to follow.",
        (7, 8): "Good: minor naming issues, adequate comments.",
        (5, 6): "Fair: some unclear names, sparse documentation.",
        (3, 4): "Poor: confusing structure, misleading names.",
        (0, 2): "Very poor: unreadable, no documentation.",
    },
    "efficiency": {
        (9, 10): "Optimal: best-known complexity, efficient structures.",
        (7, 8): "Good: reasonable complexity, minor inefficiencies.",
        (5, 6): "Adequate: works but suboptimal for large inputs.",
        (3, 4): "Poor: unnecessary loops, wasteful allocations.",
        (0, 2): "Very poor: exponential where linear suffices.",
    },
    "design": {
        (9, 10): "Excellent: SOLID principles, clean separation.",
        (7, 8): "Good: mostly sound, minor coupling issues.",
        (5, 6): "Fair: works but violates key design principles.",
        (3, 4): "Poor: God Class, high coupling, poor cohesion.",
        (0, 2): "Very poor: no discernible design, everything in one method.",
    },
    "overall": {
        (9, 10): "Production-ready: would accept in code review.",
        (7, 8): "Good: needs minor revisions before merge.",
        (5, 6): "Acceptable: significant revisions needed.",
        (3, 4): "Below standard: major rework required.",
        (0, 2): "Unacceptable: needs complete rewrite.",
    },
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Rating:
    """A single expert or automated rating for a code submission."""
    rating_id: str
    response_id: str
    problem_id: str
    rater_id: str
    correctness: float
    readability: float
    efficiency: float
    design: float
    overall: float
    comments: str = ""
    rated_at: str = ""

    def __post_init__(self):
        if not self.rated_at:
            self.rated_at = datetime.utcnow().isoformat()

    @property
    def composite_score(self) -> float:
        """Weighted average (correctness counts double)."""
        weights = {"correctness": 2.0, "readability": 1.0,
                    "efficiency": 1.0, "design": 1.0}
        total = (self.correctness * weights["correctness"]
                 + self.readability * weights["readability"]
                 + self.efficiency * weights["efficiency"]
                 + self.design * weights["design"])
        return total / sum(weights.values())

    def to_dict(self) -> dict:
        return {
            "rating_id": self.rating_id,
            "response_id": self.response_id,
            "problem_id": self.problem_id,
            "rater_id": self.rater_id,
            "correctness": self.correctness,
            "readability": self.readability,
            "efficiency": self.efficiency,
            "design": self.design,
            "overall": self.overall,
            "comments": self.comments,
            "composite_score": round(self.composite_score, 2),
            "rated_at": self.rated_at,
        }

    def to_csv_row(self) -> list:
        return [
            self.rating_id, self.response_id, self.problem_id,
            self.rater_id, self.correctness, self.readability,
            self.efficiency, self.design, self.overall,
            self.comments, self.rated_at,
        ]


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

CSV_HEADER = [
    "rating_id", "response_id", "problem_id", "rater_id",
    "correctness", "readability", "efficiency", "design",
    "overall", "comments", "rated_at",
]


def write_ratings_csv(ratings: list[Rating], filepath: str | Path) -> None:
    """Write a list of ratings to a CSV file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for r in ratings:
            writer.writerow(r.to_csv_row())

    logger.info("Wrote %d ratings to %s", len(ratings), filepath)


def read_ratings_csv(filepath: str | Path) -> list[Rating]:
    """Read ratings from a CSV file."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning("Ratings file not found: %s", filepath)
        return []

    ratings = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rating = Rating(
                rating_id=row["rating_id"],
                response_id=row["response_id"],
                problem_id=row["problem_id"],
                rater_id=row["rater_id"],
                correctness=float(row["correctness"]),
                readability=float(row["readability"]),
                efficiency=float(row["efficiency"]),
                design=float(row["design"]),
                overall=float(row["overall"]),
                comments=row.get("comments", ""),
                rated_at=row.get("rated_at", ""),
            )
            ratings.append(rating)

    return ratings


def create_blank_rating_csv(
    filepath: str | Path,
    response_ids: list[str],
    problem_ids: list[str],
    rater_id: str = "human_01",
) -> None:
    """
    Generate a blank CSV template for human raters to fill in.

    Each (response_id, problem_id) pair gets one row with zeroed scores.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    ratings = []
    for resp_id, prob_id in zip(response_ids, problem_ids):
        ratings.append(Rating(
            rating_id=f"r-{uuid.uuid4().hex[:8]}",
            response_id=resp_id,
            problem_id=prob_id,
            rater_id=rater_id,
            correctness=0.0,
            readability=0.0,
            efficiency=0.0,
            design=0.0,
            overall=0.0,
            comments="",
        ))

    write_ratings_csv(ratings, filepath)
    logger.info("Created blank template with %d rows at %s",
                len(ratings), filepath)


# ---------------------------------------------------------------------------
# Automated rubric scorer
# ---------------------------------------------------------------------------

class AutomatedRubricScorer:
    """
    Derives expert-style ratings from quantitative metrics.

    Maps CK metrics, cyclomatic complexity, and readability scores onto
    the 0–10 rating scale to provide a reproducible baseline.
    """

    def score(
        self,
        response_id: str,
        problem_id: str,
        *,
        cyclomatic_complexity: float = 0.0,
        cognitive_complexity: float = 0.0,
        wmc: int = 0,
        cbo: int = 0,
        lcom: int = 0,
        rfc: int = 0,
        readability_score: float = 50.0,
        tests_passed: int = 0,
        tests_total: int = 0,
        loc: int = 0,
    ) -> Rating:
        """
        Generate an automated rating from quantitative metrics.

        Args:
            response_id: identifier for the code submission.
            problem_id: identifier for the problem.
            cyclomatic_complexity: average CC per method.
            cognitive_complexity: total cognitive complexity.
            wmc: Weighted Methods per Class.
            cbo: Coupling Between Objects.
            lcom: Lack of Cohesion of Methods.
            rfc: Response For a Class.
            readability_score: 0–100 readability composite.
            tests_passed: number of passing tests.
            tests_total: total tests.
            loc: lines of code.

        Returns:
            Rating with rater_id='auto'.
        """
        correctness = self._score_correctness(tests_passed, tests_total)
        readability = self._score_readability(readability_score)
        efficiency = self._score_efficiency(cyclomatic_complexity,
                                            cognitive_complexity, loc)
        design = self._score_design(wmc, cbo, lcom, rfc)
        overall = self._score_overall(correctness, readability,
                                      efficiency, design)

        return Rating(
            rating_id=f"auto-{uuid.uuid4().hex[:8]}",
            response_id=response_id,
            problem_id=problem_id,
            rater_id="auto",
            correctness=correctness,
            readability=readability,
            efficiency=efficiency,
            design=design,
            overall=overall,
            comments=f"Automated rubric: CC={cyclomatic_complexity:.1f}, "
                     f"WMC={wmc}, CBO={cbo}, LCOM={lcom}",
        )

    def _score_correctness(self, passed: int, total: int) -> float:
        if total == 0:
            return 5.0  # unknown
        ratio = passed / total
        return round(ratio * 10.0, 1)

    def _score_readability(self, readability_composite: float) -> float:
        return round(readability_composite / 10.0, 1)

    def _score_efficiency(self, cc: float, cog: float, loc: int) -> float:
        # Lower complexity → higher score
        cc_score = max(0, 10 - cc * 0.5)
        cog_score = max(0, 10 - cog * 0.1)
        # Reasonable LOC
        if loc <= 100:
            loc_score = 10
        elif loc <= 300:
            loc_score = 7
        elif loc <= 500:
            loc_score = 5
        else:
            loc_score = 3
        return round((cc_score + cog_score + loc_score) / 3, 1)

    def _score_design(self, wmc: int, cbo: int, lcom: int, rfc: int) -> float:
        scores = []
        # WMC
        if wmc <= 10:
            scores.append(10)
        elif wmc <= 20:
            scores.append(8)
        elif wmc <= 47:
            scores.append(5)
        else:
            scores.append(2)
        # CBO
        if cbo <= 5:
            scores.append(10)
        elif cbo <= 10:
            scores.append(7)
        elif cbo <= 14:
            scores.append(4)
        else:
            scores.append(2)
        # LCOM
        if lcom <= 2:
            scores.append(10)
        elif lcom <= 5:
            scores.append(7)
        elif lcom <= 10:
            scores.append(4)
        else:
            scores.append(2)
        # RFC
        if rfc <= 20:
            scores.append(10)
        elif rfc <= 35:
            scores.append(7)
        elif rfc <= 50:
            scores.append(4)
        else:
            scores.append(2)

        return round(sum(scores) / len(scores), 1)

    def _score_overall(self, correctness: float, readability: float,
                       efficiency: float, design: float) -> float:
        # Weighted: correctness 2x, others 1x
        total = correctness * 2 + readability + efficiency + design
        return round(total / 5, 1)


def get_rubric_text() -> str:
    """Return a formatted rubric description for human raters."""
    lines = ["=" * 60, "EXPERT RATING RUBRIC", "=" * 60, ""]
    lines.append("Rate each dimension from 0 to 10.\n")

    for dim, levels in RUBRIC_DESCRIPTORS.items():
        lines.append(f"--- {dim.upper()} ---")
        for (lo, hi), desc in sorted(levels.items(), reverse=True):
            lines.append(f"  {lo}–{hi}: {desc}")
        lines.append("")

    lines.append("Composite score = (correctness×2 + readability + "
                 "efficiency + design) / 5")
    return "\n".join(lines)
