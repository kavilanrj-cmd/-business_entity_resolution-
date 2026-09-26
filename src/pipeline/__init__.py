"""End-to-end orchestration.

The same three stages run in training and inference:

1. **preprocess** -- normalise names, addresses and countries
2. **retrieve**   -- build the candidate set (the Source 1 -> Source 2/3 union)
3. **score**      -- featurise the candidates and apply the learned model

The only difference is where the labels come from.  Training reads the
ground-truth file; inference has no labels at all and simply never asks for
them.  Keeping the stages in one place means the inference path cannot drift
away from the path that was validated.
"""

from .stages import (
    RetrievalResult,
    ScoredCandidates,
    build_retrieval,
    generate_candidates_for,
    load_and_preprocess,
    score_candidates,
)
from .train_pipeline import TrainingOutcome, run_training
from .inference_pipeline import InferenceOutcome, run_inference

__all__ = [
    "InferenceOutcome",
    "RetrievalResult",
    "ScoredCandidates",
    "TrainingOutcome",
    "build_retrieval",
    "generate_candidates_for",
    "load_and_preprocess",
    "run_inference",
    "run_training",
    "score_candidates",
]
