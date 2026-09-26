"""Strategy C - character n-gram TF-IDF blocking.

Handles names that are *noisy* rather than merely abbreviated: OCR-style
character errors, transpositions and truncated strings.  Character n-grams are
robust to all three, which is why this is the main recall-oriented retriever.

The analyzer default is ``char_wb`` (n-grams padded within word boundaries),
which is markedly less noisy than a raw ``char`` analyzer for business names
containing punctuation and separators.
"""

from __future__ import annotations

import logging

import pandas as pd
from scipy import sparse

from ..config import BlockingConfig
from ..preprocessing.preprocess import COLS
from .tfidf_blocking import SparseRetriever, build_tfidf_matrix

LOGGER = logging.getLogger(__name__)


class CharTfidfRetriever:
    """Reusable character n-gram TF-IDF retriever for a fixed pool.

    Fitting happens once per pool; the same object serves name retrieval,
    address retrieval and the country-restricted passes, so the (comparatively
    expensive) vectorizer fit is paid only once.
    """

    def __init__(self, pool: pd.DataFrame, config: BlockingConfig, column: str = COLS.name_core, n_jobs: int = -1) -> None:
        self.config = config
        self.column = column
        if column == COLS.address_core:
            ngram = (config.tfidf_name_ngram_min, min(3, config.tfidf_name_ngram_max))
        else:
            ngram = (config.tfidf_name_ngram_min, config.tfidf_name_ngram_max)
        self.matrix, self.vectorizer = build_tfidf_matrix(
            pool[column],
            analyzer=config.tfidf_name_analyzer,
            ngram_range=ngram,
        )
        self.retriever = SparseRetriever(self.matrix, n_jobs=n_jobs)
        LOGGER.info(
            "Char TF-IDF on '%s': analyzer=%s ngram_range=%s vocab=%d corpus=%d",
            column, config.tfidf_name_analyzer, ngram, self.matrix.shape[1], self.matrix.shape[0],
        )

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        values = frame[self.column].fillna("").astype(str).tolist()
        return self.vectorizer.transform(values)

    def topk(
        self,
        queries: pd.DataFrame,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[tuple[int, int, float]]:
        """Top-K pool matches for every query row (as ``(q, pool, score)``)."""
        top_k = self.config.tfidf_name_top_k if top_k is None else top_k
        min_score = self.config.tfidf_name_min_score if min_score is None else min_score
        result = self.retriever.query(self.transform(queries), top_k, min_score)
        return list(result.iter_topk())


def char_block(
    queries: pd.DataFrame,
    pool: pd.DataFrame,
    retriever: CharTfidfRetriever,
    *,
    top_k: int | None = None,
    min_score: float | None = None,
    strategy_name: str = "tfidf_name",
) -> list[tuple[int, int, float, str]]:
    """Strategy C: character n-gram TF-IDF top-K over the full pool."""
    result = retriever.retriever.query(
        retriever.transform(queries),
        retriever.config.tfidf_name_top_k if top_k is None else top_k,
        retriever.config.tfidf_name_min_score if min_score is None else min_score,
    )
    return [(q, p, s, strategy_name) for q, p, s in result.iter_topk()]
