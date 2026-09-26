"""Scoring, validation and error attribution.

The metric is macro-averaged F0.5 over Source 1 entities.  An entity that has
no true match and receives no prediction scores 1.0; one that has no true match
but receives a prediction scores 0.0.  That asymmetry is what makes false
positives so expensive and drives every precision-oriented decision downstream.
"""

from .error_analysis import ErrorAnalysis, analyse_errors, count_by_strategy, summarise_entity_scores
from .metrics import (
    BETA_SQ,
    F05Evaluator,
    ThresholdPoint,
    best_threshold,
    f05_from_counts,
    f05_per_entity,
    f05_score,
    summarize_confusion,
)
from .validation import (
    CandidateRecall,
    ModelScore,
    ValidationReport,
    cross_validate,
    make_entity_folds,
    measure_candidate_recall,
)

__all__ = [
    "BETA_SQ",
    "CandidateRecall",
    "ErrorAnalysis",
    "F05Evaluator",
    "ModelScore",
    "ThresholdPoint",
    "ValidationReport",
    "analyse_errors",
    "best_threshold",
    "count_by_strategy",
    "cross_validate",
    "f05_from_counts",
    "f05_per_entity",
    "f05_score",
    "make_entity_folds",
    "measure_candidate_recall",
    "summarise_entity_scores",
    "summarize_confusion",
]
