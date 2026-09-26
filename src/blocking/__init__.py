"""Blocking / candidate generation.

The union of all strategies is the *only* pair set handed to the model, so
candidate recall is a hard upper bound on achievable F0.5.
"""

from __future__ import annotations

from .candidate_generator import (
    STRATEGY_NAMES,
    CandidateGenerator,
    CandidateSet,
    build_pool,
    strategies_from_mask,
)
from .character_blocking import CharTfidfRetriever
from .exact_blocking import exact_block, exact_name_candidates
from .tfidf_blocking import SparseRetriever, build_tfidf_matrix
from .token_blocking import TokenBlockingIndex, address_token_block, name_token_block, tokenize

__all__ = [
    "STRATEGY_NAMES",
    "CandidateGenerator",
    "CandidateSet",
    "CharTfidfRetriever",
    "SparseRetriever",
    "TokenBlockingIndex",
    "address_token_block",
    "build_pool",
    "build_tfidf_matrix",
    "exact_block",
    "exact_name_candidates",
    "name_token_block",
    "strategies_from_mask",
    "tokenize",
]
