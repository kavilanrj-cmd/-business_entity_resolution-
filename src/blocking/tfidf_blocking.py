"""Sparse retrieval engine shared by all approximate blocking strategies.

The engine wraps scikit-learn's brute-force cosine search over sparse
matrices, which is what makes multi-strategy blocking affordable: retrieval is
a sparse matrix product instead of an O(N^2) Python-level comparison, and the
per-entity candidate cap is applied by ``kneighbors`` before anything large is
materialised.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from ..config import ID_COLUMN, setup_logging  # noqa: F401  (re-exported convenience)

LOGGER = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Top-K neighbours for a set of query rows.

    Attributes
    ----------
    indices:
        ``(n_queries, k)`` int32 array of positions into the fitted corpus.
        ``-1`` marks a slot that was not filled.
    similarity:
        ``(n_queries, k)`` float32 cosine similarities in ``[0, 1]``.
    """

    indices: np.ndarray
    similarity: np.ndarray

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def iter_topk(self):
        """Yield ``(query_pos, corpus_pos, score)`` for every valid slot."""
        rows, cols = np.nonzero(self.indices >= 0)
        for q, c in zip(rows.tolist(), cols.tolist()):
            pos = int(self.indices[q, c])
            yield int(q), pos, float(self.similarity[q, c])


class SparseRetriever:
    """Cosine top-K retrieval over a sparse TF-IDF / IDF-weighted matrix."""

    def __init__(self, matrix: sparse.csr_matrix, *, n_jobs: int = -1, metric: str = "cosine") -> None:
        self.matrix = matrix.tocsr()
        self.n_corpus = matrix.shape[0]
        self._nn = NearestNeighbors(n_neighbors=1, metric=metric, algorithm="brute", n_jobs=n_jobs)
        self._nn.fit(self.matrix)

    def query(
        self,
        queries: sparse.csr_matrix,
        top_k: int,
        min_score: float = 0.0,
        exclude_self: np.ndarray | None = None,
    ) -> RetrievalResult:
        """Retrieve the top-``k`` corpus rows for every query row.

        ``exclude_self`` optionally provides, per query, a corpus position that
        must be dropped (used when queries and corpus are the same frame).
        """
        n_queries = queries.shape[0]
        if n_queries == 0 or self.n_corpus == 0:
            empty_i = np.full((n_queries, 0), -1, dtype=np.int32)
            return RetrievalResult(empty_i, np.zeros((n_queries, 0), dtype=np.float32))
        k = int(max(1, min(top_k, self.n_corpus)))
        # Ask for one extra neighbour so that a self-exclusion does not shrink k.
        k_query = int(min(self.n_corpus, k + 1)) if exclude_self is not None else k
        distances, indices = self._nn.kneighbors(queries, n_neighbors=k_query, return_distance=True)
        similarity = np.clip(1.0 - distances, 0.0, 1.0).astype(np.float32)

        if exclude_self is not None:
            keep = np.ones_like(indices, dtype=bool)
            drop = indices == exclude_self[:, None]
            # Drop the self hit, and if that leaves a gap, shift the tail left.
            if drop.any():
                for q in np.nonzero(drop.any(axis=1))[0]:
                    row = indices[q]
                    valid = row[row >= 0]
                    valid = valid[valid != exclude_self[q]][:k]
                    indices[q, : len(valid)] = valid
                    if len(valid) < k_query:
                        indices[q, len(valid) :] = -1
                    similarity[q, : len(valid)] = similarity[q, : len(valid)]
                    keep[q, : len(valid)] = True
                    keep[q, len(valid) :] = False
        keep = (indices >= 0) & (similarity >= min_score)
        indices = np.where(keep, indices, -1).astype(np.int32)
        return RetrievalResult(indices, similarity.astype(np.float32))

    def pairwise_cosine(self, query_rows: np.ndarray, corpus_rows: np.ndarray) -> np.ndarray:
        """Cosine similarity for explicit index pairs (used by feature code)."""
        if len(query_rows) == 0:
            return np.zeros(0, dtype=np.float32)
        q = self.matrix[query_rows]
        c = self.matrix[corpus_rows]
        # Element-wise row dot product, computed in small chunks to bound memory.
        out = np.zeros(len(query_rows), dtype=np.float32)
        chunk = 50_000
        for start in range(0, len(query_rows), chunk):
            stop = min(start + chunk, len(query_rows))
            prod = q[start:stop].multiply(c[start:stop])
            out[start:stop] = np.asarray(prod.sum(axis=1)).ravel()
        return out


def build_tfidf_matrix(
    documents: pd.Series,
    *,
    analyzer: str = "char_wb",
    ngram_range: tuple[int, int] = (2, 5),
    min_df: int = 1,
    max_df: float = 0.995,
    sublinear_tf: bool = True,
) -> tuple[sparse.csr_matrix, object]:
    """Fit a TF-IDF vectorizer on ``documents`` and transform them.

    Returns the L2-normalized CSR matrix (so a dot product is a cosine) and the
    fitted vectorizer.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = documents.fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df if max_df < 1.0 else 1.0,
        sublinear_tf=sublinear_tf,
        lowercase=False,  # inputs are already normalized
        norm="l2",
        dtype=np.float32,
    )
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # Empty vocabulary (e.g. every field blank) -> an all-zero matrix.
        LOGGER.warning("TF-IDF vocabulary is empty for analyzer=%s; producing a zero matrix", analyzer)
        matrix = sparse.csr_matrix((len(corpus), 1), dtype=np.float32)
    return matrix.tocsr(), vectorizer


def token_series_to_matrix(
    token_lists: list[list[str]],
    vocabulary: dict[str, int],
    weights: dict[str, float],
) -> sparse.csr_matrix:
    """Build an IDF-weighted, L2-normalized sparse token matrix."""
    indptr = np.zeros(len(token_lists) + 1, dtype=np.int64)
    indices: list[int] = []
    data: list[float] = []
    for row, tokens in enumerate(token_lists):
        acc: dict[int, float] = {}
        for tok in tokens:
            col = vocabulary.get(tok)
            if col is None:
                continue
            acc[col] = acc.get(col, 0.0) + weights.get(tok, 1.0)
        for col, val in acc.items():
            norm = np.sqrt(val * val)
            indices.append(col)
            data.append(val if norm == 0 else val)
        indptr[row + 1] = len(indices)
    matrix = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int32), indptr),
        shape=(len(token_lists), max(1, len(vocabulary))),
    )
    return normalize_rows(matrix)


def normalize_rows(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """L2-normalize each row of a sparse matrix in place-safe fashion."""
    matrix = matrix.tocsr(copy=True)
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    inv = sparse.diags(1.0 / norms)
    return (inv @ matrix).tocsr()


def idf_from_documents(token_lists: list[list[str]], min_df: int = 1) -> tuple[dict[str, float], dict[str, int]]:
    """Smoothed IDF weights and a token vocabulary from the pool documents.

    The IDF is estimated **only from the retrieval pool** (Source 2 + Source 3),
    never from the ground truth, so it can be reused unchanged at inference.
    """
    from collections import Counter

    df: Counter[str] = Counter()
    for tokens in token_lists:
        df.update(set(tokens))
    n_docs = max(1, len(token_lists))
    weights = {tok: float(np.log((1.0 + n_docs) / (1.0 + c)) + 1.0) for tok, c in df.items() if c >= min_df}
    vocab = {tok: i for i, tok in enumerate(sorted(weights))}
    return weights, vocab


def entity_ids(frame: pd.DataFrame) -> np.ndarray:
    """Entity ids as a plain numpy string array."""
    return frame[ID_COLUMN].fillna("").astype(str).to_numpy()
