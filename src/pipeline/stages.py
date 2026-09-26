"""The stages shared by the training and inference pipelines."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..blocking.candidate_generator import CandidateGenerator, CandidateSet, build_pool
from ..config import ID_COLUMN, Config
from ..data.ground_truth import GroundTruth
from ..data.loader import RawDataset
from ..features.feature_builder import FeatureMatrix, build_features
from ..preprocessing.preprocess import preprocess_table

LOGGER = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Preprocessed tables plus the candidate set built from them."""

    source1: pd.DataFrame
    pool: pd.DataFrame
    candidates: CandidateSet
    seconds: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def source1_ids(self) -> list[str]:
        return [str(x) for x in self.source1[ID_COLUMN].tolist()]

    def candidates_by_entity(self) -> dict[str, list[str]]:
        """``source1_id -> sorted candidate ids`` for the candidate-pairs file."""
        out: dict[str, list[str]] = {s1: [] for s1 in self.source1_ids}
        for q, p in zip(self.candidates.query_pos, self.candidates.pool_pos):
            out[str(self.candidates.query_ids[q])].append(str(self.candidates.pool_ids[p]))
        return {k: sorted(set(v)) for k, v in out.items()}


@dataclass
class ScoredCandidates:
    """Feature matrix plus the per-pair model scores."""

    features: FeatureMatrix
    scores: np.ndarray
    seconds: float = 0.0


def load_and_preprocess(raw: RawDataset, config: Config) -> dict[str, pd.DataFrame]:
    """Normalise every raw table with the same configuration."""
    started = time.perf_counter()
    preprocessed = {
        "source1": preprocess_table(raw.source1, config.preprocess),
        "source2": preprocess_table(raw.source2, config.preprocess),
        "source3": preprocess_table(raw.source3, config.preprocess),
    }
    LOGGER.info(
        "Preprocessed %d / %d / %d rows in %.1fs",
        len(preprocessed["source1"]), len(preprocessed["source2"]), len(preprocessed["source3"]),
        time.perf_counter() - started,
    )
    return preprocessed


def generate_candidates_for(
    source1: pd.DataFrame,
    pool: pd.DataFrame,
    config: Config,
) -> CandidateSet:
    """Run the blocking strategies and return the deduplicated union."""
    return CandidateGenerator(pool, config.blocking).generate(source1)


def build_retrieval(raw: RawDataset, config: Config) -> RetrievalResult:
    """Preprocess, then retrieve candidates for every Source 1 entity."""
    pre = load_and_preprocess(raw, config)
    pool = build_pool(pre["source2"], pre["source3"])
    t0 = time.perf_counter()
    candidates = generate_candidates_for(pre["source1"], pool, config)
    elapsed = time.perf_counter() - t0
    LOGGER.info(
        "Retrieval finished in %.1fs: %d pairs for %d Source 1 entities (%.1f pairs/entity)",
        elapsed, len(candidates.query_pos), len(candidates.query_ids),
        len(candidates.query_pos) / max(1, len(candidates.query_ids)),
    )
    return RetrievalResult(
        source1=pre["source1"], pool=pool, candidates=candidates, seconds=elapsed,
        timings=dict(candidates.stats.get("timings", {})),
    )


def score_candidates(
    retrieval: RetrievalResult,
    config: Config,
    model: object | None,
) -> ScoredCandidates:
    """Featurise the candidate set and, if a model is given, score it."""
    t0 = time.perf_counter()
    features = build_features(retrieval.source1, retrieval.pool, retrieval.candidates, config.features)
    elapsed = time.perf_counter() - t0
    if not np.isfinite(features.X).all():
        raise ValueError("Feature matrix contains NaN or infinite values; check the feature builder")
    LOGGER.info("Built %d x %d features in %.1fs", features.X.shape[0], features.X.shape[1], elapsed)
    if model is None:
        return ScoredCandidates(features=features, scores=np.zeros(len(features.X)), seconds=elapsed)
    scores = model.predict_proba(features.X)
    return ScoredCandidates(features=features, scores=np.asarray(scores, dtype=float), seconds=elapsed)


def candidate_map(retrieval: RetrievalResult) -> dict[str, list[str]]:
    return retrieval.candidates_by_entity()
