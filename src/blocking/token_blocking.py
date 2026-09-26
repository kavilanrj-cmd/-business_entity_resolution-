"""Strategies B and D - IDF-weighted token blocking.

* **Strategy B** blocks on the informative tokens of the business *name*.
* **Strategy D** blocks on the informative tokens of the business *address*.

Both use the same machinery: tokens are weighted by inverse document frequency
estimated on the pool, the resulting rows are L2-normalized so that a dot
product is a cosine, and top-K neighbours are retrieved with a sparse product.
IDF weighting is what makes this work -- "pvt ltd" contributes almost nothing,
while a rare house number or street name dominates the score.

Very common tokens (document frequency above ``max_document_frequency``) and
very short tokens are dropped, which keeps the blocks small and precise.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd

from ..preprocessing.preprocess import COLS

LOGGER = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Tokens that are structurally part of an address and identify nothing.
GENERIC_ADDRESS_TOKENS: frozenset[str] = frozenset(
    {
        "near", "opposite", "behind", "beside", "next", "nearby", "road", "street", "avenue",
        "lane", "drive", "boulevard", "court", "square", "place", "main", "cross", "junction",
        "nagar", "colony", "sector", "block", "plot", "house", "building", "floor", "flat",
        "apartment", "shop", "office", "premises", "new", "old", "station", "bus", "stand",
        "market", "temple", "church", "park", "lake", "view", "hill", "garden", "towers",
        "apart", "and", "the", "of", "at", "in", "to", "postoffice", "pincode", "zipcode",
    }
)


def tokenize(text: str, *, drop_generic_address: bool = False, min_length: int = 1) -> list[str]:
    """Split a normalized string into blocking tokens."""
    tokens = _TOKEN_RE.findall(str(text).lower())
    tokens = [t for t in tokens if len(t) >= min_length]
    if drop_generic_address:
        tokens = [t for t in tokens if t not in GENERIC_ADDRESS_TOKENS]
    return tokens


class TokenBlockingIndex:
    """An inverted index over IDF-weighted tokens, with cosine top-K retrieval."""

    def __init__(
        self,
        pool_tokens: Sequence[Sequence[str]],
        *,
        max_document_frequency: float = 0.7,
        min_document_frequency: int = 1,
        n_jobs: int = -1,
        label: str = "tokens",
    ) -> None:
        self.label = label
        df: Counter[str] = Counter()
        for tokens in pool_tokens:
            df.update(set(tokens))
        n_docs = max(1, len(pool_tokens))
        max_df_count = max_document_frequency * n_docs
        keep = {t: c for t, c in df.items() if min_document_frequency <= c <= max_df_count}
        if not keep:
            LOGGER.warning("Token index '%s' dropped every token; falling back to min_df only", label)
            keep = {t: c for t, c in df.items() if c >= min_document_frequency}
        self.vocabulary: dict[str, int] = {t: i for i, t in enumerate(sorted(keep))}
        self.idf: dict[str, float] = {t: float(np.log((1.0 + n_docs) / (1.0 + c)) + 1.0) for t, c in keep.items()}
        LOGGER.info(
            "Token index '%s': %d/%d tokens kept (df<=%.0f%%), %d pool rows",
            label, len(self.vocabulary), len(df), max_document_frequency * 100, len(pool_tokens),
        )
        from .tfidf_blocking import SparseRetriever, token_series_to_matrix

        matrix = token_series_to_matrix(list(pool_tokens), self.vocabulary, self.idf)
        self._retriever = SparseRetriever(matrix, n_jobs=n_jobs)

    @property
    def n_corpus(self) -> int:
        return self._retriever.n_corpus

    def query(
        self,
        query_tokens: Sequence[Sequence[str]],
        *,
        top_k: int,
        min_score: float,
        restrict_rows: np.ndarray | None = None,
    ) -> list[tuple[int, int, float]]:
        """Top-K pool matches per query row, optionally within ``restrict_rows``."""
        from .tfidf_blocking import token_series_to_matrix

        if restrict_rows is None:
            matrix = token_series_to_matrix(list(query_tokens), self.vocabulary, self.idf)
            result = self._retriever.query(matrix, top_k, min_score)
            return list(result.iter_topk())
        # Country-restricted pass: slice the fitted corpus, fit a light
        # retriever on the slice, then map positions back to the global pool.
        rows = np.asarray(restrict_rows, dtype=np.int64)
        sub = self._retriever.matrix[rows]
        from .tfidf_blocking import SparseRetriever

        sub_retriever = SparseRetriever(sub, n_jobs=-1)
        q = token_series_to_matrix(list(query_tokens), self.vocabulary, self.idf)
        result = sub_retriever.query(q, top_k, min_score)
        out: list[tuple[int, int, float]] = []
        for q_pos, p_pos, score in result.iter_topk():
            out.append((q_pos, int(rows[p_pos]), score))
        return out


def name_token_block(
    queries: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    max_per_query: int = 150,
    min_score: float = 0.30,
    min_token_length: int = 2,
    n_jobs: int = -1,
) -> list[tuple[int, int, float, str]]:
    """Strategy B: IDF-weighted name-token overlap."""
    pool_tokens = [tokenize(t, min_length=min_token_length) for t in pool[COLS.name_core].fillna("").astype(str)]
    query_tokens = [tokenize(t, min_length=min_token_length) for t in queries[COLS.name_core].fillna("").astype(str)]
    index = TokenBlockingIndex(pool_tokens, label="name", n_jobs=n_jobs)
    triples = index.query(query_tokens, top_k=max_per_query, min_score=min_score)
    return [(q, p, s, "token_name") for q, p, s in triples]


def address_token_block(
    queries: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    max_per_query: int = 100,
    min_score: float = 0.35,
    min_token_length: int = 2,
    n_jobs: int = -1,
) -> list[tuple[int, int, float, str]]:
    """Strategy D: IDF-weighted address-token overlap (generic words removed)."""
    pool_tokens = [
        tokenize(t, drop_generic_address=True, min_length=min_token_length)
        for t in pool[COLS.address_core].fillna("").astype(str)
    ]
    query_tokens = [
        tokenize(t, drop_generic_address=True, min_length=min_token_length)
        for t in queries[COLS.address_core].fillna("").astype(str)
    ]
    index = TokenBlockingIndex(pool_tokens, label="address", n_jobs=n_jobs)
    triples = index.query(query_tokens, top_k=max_per_query, min_score=min_score)
    return [(q, p, s, "address_token") for q, p, s in triples]
