"""
irr.py — Inter-Rater Reliability computation.

Implements:
    - Cohen's kappa (unweighted) for nominal agreement between two raters.
    - Weighted Cohen's kappa (linear and quadratic) for ordinal scales.
    - Fleiss' kappa for multiple raters on nominal data.
    - Percent agreement as a simple baseline.

Interpretation guidelines (Landis & Koch, 1977):
    κ < 0.00   Poor
    0.00–0.20  Slight
    0.21–0.40  Fair
    0.41–0.60  Moderate
    0.61–0.80  Substantial
    0.81–1.00  Almost perfect
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class KappaResult:
    """Result of a kappa computation."""
    kappa: float
    interpretation: str
    observed_agreement: float
    expected_agreement: float
    n_items: int
    n_categories: int
    method: str  # "cohen", "weighted_linear", "weighted_quadratic", "fleiss"
    weights: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "kappa": round(self.kappa, 4),
            "interpretation": self.interpretation,
            "observed_agreement": round(self.observed_agreement, 4),
            "expected_agreement": round(self.expected_agreement, 4),
            "n_items": self.n_items,
            "n_categories": self.n_categories,
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def interpret_kappa(kappa: float) -> str:
    """Interpret kappa value using Landis & Koch (1977) guidelines."""
    if kappa < 0.0:
        return "poor"
    elif kappa <= 0.20:
        return "slight"
    elif kappa <= 0.40:
        return "fair"
    elif kappa <= 0.60:
        return "moderate"
    elif kappa <= 0.80:
        return "substantial"
    else:
        return "almost_perfect"


# ---------------------------------------------------------------------------
# Cohen's kappa (unweighted)
# ---------------------------------------------------------------------------

def cohens_kappa(rater1: list, rater2: list) -> KappaResult:
    """
    Compute Cohen's kappa for two raters on the same set of items.

    Args:
        rater1: list of ratings/labels from rater 1.
        rater2: list of ratings/labels from rater 2.

    Returns:
        KappaResult with kappa value and interpretation.

    Raises:
        ValueError: if input lists have different lengths or are empty.
    """
    if len(rater1) != len(rater2):
        raise ValueError(
            f"Rater lists must have equal length: {len(rater1)} vs {len(rater2)}")
    if len(rater1) == 0:
        raise ValueError("Rater lists must not be empty")

    n = len(rater1)
    categories = sorted(set(rater1) | set(rater2))
    k = len(categories)
    cat_index = {c: i for i, c in enumerate(categories)}

    # Build confusion matrix
    matrix = [[0] * k for _ in range(k)]
    for r1, r2 in zip(rater1, rater2):
        matrix[cat_index[r1]][cat_index[r2]] += 1

    # Observed agreement
    observed = sum(matrix[i][i] for i in range(k)) / n

    # Expected agreement
    expected = 0.0
    for i in range(k):
        row_sum = sum(matrix[i][j] for j in range(k))
        col_sum = sum(matrix[j][i] for j in range(k))
        expected += (row_sum * col_sum)
    expected /= (n * n)

    # Kappa
    if expected == 1.0:
        kappa = 1.0
    else:
        kappa = (observed - expected) / (1.0 - expected)

    return KappaResult(
        kappa=kappa,
        interpretation=interpret_kappa(kappa),
        observed_agreement=observed,
        expected_agreement=expected,
        n_items=n,
        n_categories=k,
        method="cohen",
    )


# ---------------------------------------------------------------------------
# Weighted Cohen's kappa
# ---------------------------------------------------------------------------

def weighted_kappa(
    rater1: list[float | int],
    rater2: list[float | int],
    weight_type: str = "quadratic",
) -> KappaResult:
    """
    Compute weighted Cohen's kappa for ordinal ratings.

    Args:
        rater1: numeric ratings from rater 1.
        rater2: numeric ratings from rater 2.
        weight_type: 'linear' or 'quadratic'.

    Returns:
        KappaResult with weighted kappa.
    """
    if len(rater1) != len(rater2):
        raise ValueError(
            f"Rater lists must have equal length: {len(rater1)} vs {len(rater2)}")
    if len(rater1) == 0:
        raise ValueError("Rater lists must not be empty")

    n = len(rater1)
    categories = sorted(set(rater1) | set(rater2))
    k = len(categories)
    cat_index = {c: i for i, c in enumerate(categories)}

    # Build confusion matrix
    matrix = [[0] * k for _ in range(k)]
    for r1, r2 in zip(rater1, rater2):
        matrix[cat_index[r1]][cat_index[r2]] += 1

    # Weight matrix
    weights = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            if weight_type == "linear":
                weights[i][j] = abs(i - j) / (k - 1) if k > 1 else 0
            else:  # quadratic
                weights[i][j] = ((i - j) ** 2) / ((k - 1) ** 2) if k > 1 else 0

    # Row and column marginals
    row_sums = [sum(matrix[i]) for i in range(k)]
    col_sums = [sum(matrix[j][i] for j in range(k)) for i in range(k)]

    # Observed and expected weighted disagreement
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for i in range(k):
        for j in range(k):
            observed_disagreement += weights[i][j] * matrix[i][j] / n
            expected_disagreement += weights[i][j] * row_sums[i] * col_sums[j] / (n * n)

    # Weighted kappa
    if expected_disagreement == 0:
        kappa = 1.0
    else:
        kappa = 1.0 - observed_disagreement / expected_disagreement

    # For reporting, convert to agreement
    observed_agreement = 1.0 - observed_disagreement
    expected_agreement = 1.0 - expected_disagreement

    return KappaResult(
        kappa=kappa,
        interpretation=interpret_kappa(kappa),
        observed_agreement=observed_agreement,
        expected_agreement=expected_agreement,
        n_items=n,
        n_categories=k,
        method=f"weighted_{weight_type}",
        weights=weight_type,
    )


# ---------------------------------------------------------------------------
# Fleiss' kappa
# ---------------------------------------------------------------------------

def fleiss_kappa(ratings_matrix: list[list[int]]) -> KappaResult:
    """
    Compute Fleiss' kappa for multiple raters.

    Args:
        ratings_matrix: N×k matrix where N = number of items,
            k = number of categories, and each entry is the count of
            raters who assigned that category to that item.

    Returns:
        KappaResult with Fleiss' kappa value.
    """
    if not ratings_matrix:
        raise ValueError("Ratings matrix must not be empty")

    n = len(ratings_matrix)
    k = len(ratings_matrix[0])

    # Total number of raters per item (should be constant)
    raters_per_item = sum(ratings_matrix[0])
    if raters_per_item < 2:
        raise ValueError("Need at least 2 raters")

    r = raters_per_item

    # Column proportions
    p_j = []
    total_ratings = n * r
    for j in range(k):
        col_sum = sum(ratings_matrix[i][j] for i in range(n))
        p_j.append(col_sum / total_ratings)

    # Per-item agreement
    p_i_list = []
    for i in range(n):
        row = ratings_matrix[i]
        sum_sq = sum(x * x for x in row)
        p_i = (sum_sq - r) / (r * (r - 1)) if r > 1 else 0
        p_i_list.append(p_i)

    # Overall observed agreement
    p_bar = sum(p_i_list) / n if n > 0 else 0

    # Expected agreement
    p_e = sum(pj * pj for pj in p_j)

    # Fleiss' kappa
    if p_e == 1.0:
        kappa = 1.0
    else:
        kappa = (p_bar - p_e) / (1.0 - p_e)

    return KappaResult(
        kappa=kappa,
        interpretation=interpret_kappa(kappa),
        observed_agreement=p_bar,
        expected_agreement=p_e,
        n_items=n,
        n_categories=k,
        method="fleiss",
    )


# ---------------------------------------------------------------------------
# Percent agreement (simple baseline)
# ---------------------------------------------------------------------------

def percent_agreement(rater1: list, rater2: list) -> float:
    """
    Compute raw percent agreement between two raters.

    This is a naive measure that does not account for chance agreement.
    """
    if len(rater1) != len(rater2):
        raise ValueError("Rater lists must have equal length")
    if len(rater1) == 0:
        return 0.0
    agreements = sum(1 for a, b in zip(rater1, rater2) if a == b)
    return agreements / len(rater1)


# ---------------------------------------------------------------------------
# Convenience for expert ratings
# ---------------------------------------------------------------------------

def compute_irr_for_dimension(
    ratings_by_rater: dict[str, list[float]],
    dimension: str = "overall",
    weight_type: str = "quadratic",
) -> dict[str, KappaResult]:
    """
    Compute pairwise weighted kappa for a single rating dimension.

    Args:
        ratings_by_rater: dict mapping rater_id → list of scores for the
            dimension (all lists must be same length, items aligned).
        dimension: name of the dimension (for labelling only).
        weight_type: 'linear' or 'quadratic'.

    Returns:
        Dict mapping "(raterA, raterB)" → KappaResult for each pair.
    """
    rater_ids = sorted(ratings_by_rater.keys())
    results: dict[str, KappaResult] = {}

    for i in range(len(rater_ids)):
        for j in range(i + 1, len(rater_ids)):
            r1_id = rater_ids[i]
            r2_id = rater_ids[j]
            r1 = ratings_by_rater[r1_id]
            r2 = ratings_by_rater[r2_id]

            key = f"({r1_id}, {r2_id})"
            try:
                result = weighted_kappa(r1, r2, weight_type=weight_type)
                results[key] = result
            except ValueError as exc:
                logger.warning("Could not compute kappa for %s on %s: %s",
                               key, dimension, exc)

    return results
