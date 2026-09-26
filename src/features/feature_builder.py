"""Assembles the full feature matrix for every Source 1 -> candidate pair.

Design
------
* Per-entity grouping lets the expensive string similarities (rapidfuzz) and
  the sparse TF-IDF cosines be computed in vectorised batches, so the cost is
  O(number of candidate pairs) and never O(N^2) in the corpus.
* No hand-tuned weights are introduced.  The "combined" features are
  parameter-free summaries of evidence the model already sees (geometric mean,
  minimum, maximum, difference), so any weighting decision is still made by the
  learned model and validated on held-out Source 1 entities.
* Every feature is defined even when a field is missing, so a row never
  contains ``NaN``; missingness is expressed explicitly instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from ..blocking.candidate_generator import CandidateSet
from ..blocking.tfidf_blocking import build_tfidf_matrix, token_series_to_matrix
from ..config import FeatureConfig, ID_COLUMN
from ..preprocessing.preprocess import COLS
from .address_features import ADDRESS_FEATURES, AddressFeatureBuilder
from .country_features import COUNTRY_FEATURES, country_block
from .name_features import NAME_FEATURES, NameFeatureBuilder

LOGGER = logging.getLogger(__name__)

COMBINED_FEATURES: tuple[str, ...] = (
    "name_address_geometric_score",
    "name_address_arithmetic_score",
    "name_address_min_score",
    "name_address_max_score",
    "name_address_score_gap",
    "exact_name_with_address_conflict",
    "exact_name_with_pincode_conflict",
    "exact_name_with_house_number_conflict",
)


def feature_names(config: FeatureConfig | None = None) -> list[str]:
    """Ordered list of every feature column produced by :class:`FeatureBuilder`."""
    names = list(NAME_FEATURES) + list(ADDRESS_FEATURES) + list(COUNTRY_FEATURES)
    if (config or FeatureConfig()).use_combined_features:
        names += list(COMBINED_FEATURES)
    return names


@dataclass
class FeatureMatrix:
    """Feature matrix plus the identifiers needed to join it back."""

    source1_ids: np.ndarray
    candidate_ids: np.ndarray
    candidate_sources: np.ndarray
    X: np.ndarray
    columns: list[str]
    candidate_scores: np.ndarray
    strategy_mask: np.ndarray

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def as_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.X, columns=self.columns)
        frame.insert(0, "candidate_entity_id", self.candidate_ids)
        frame.insert(0, "source1_entity_id", self.source1_ids)
        return frame

    def candidate_pairs(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "source1_entity_id": self.source1_ids,
                "candidate_entity_id": self.candidate_ids,
                "candidate_source": self.candidate_sources,
                "candidate_score": self.candidate_scores,
                "strategy_mask": self.strategy_mask,
            }
        )


class FeatureBuilder:
    """Computes the feature matrix for a candidate set.

    Parameters
    ----------
    queries, pool:
        Preprocessed (normalized) record frames.  ``pool`` is the *same* frame
        the candidate generator used, so pool positions line up.
    config:
        Feature switches; defaults are the validated ones.
    """

    def __init__(
        self,
        queries: pd.DataFrame,
        pool: pd.DataFrame,
        config: FeatureConfig | None = None,
    ) -> None:
        self.config = config or FeatureConfig()
        self.queries = queries.reset_index(drop=True)
        self.pool = pool.reset_index(drop=True)
        self.name_builder = NameFeatureBuilder(
            prefix_chars=self.config.prefix_chars,
            jw_prefix_weight=self.config.jaro_winkler_prefix_weight,
        )
        self.address_builder = AddressFeatureBuilder(prefix_chars=self.config.prefix_chars)
        self._prepare_caches()
        self._fit_tfidf()

    # -- setup ------------------------------------------------------------
    def _col(self, frame: pd.DataFrame, name: str) -> np.ndarray:
        if name in frame.columns:
            return frame[name].fillna("").astype(str).to_numpy()
        return np.array([""] * len(frame), dtype=object)

    def _prepare_caches(self) -> None:
        self.q_name = self._col(self.queries, COLS.name)
        self.q_name_core = self._col(self.queries, COLS.name_core)
        self.q_name_raw = self._col(self.queries, "business_name")
        self.q_address = self._col(self.queries, COLS.address)
        self.q_country = self._col(self.queries, COLS.country)
        self.p_name = self._col(self.pool, COLS.name)
        self.p_name_core = self._col(self.pool, COLS.name_core)
        self.p_name_raw = self._col(self.pool, "business_name")
        self.p_address = self._col(self.pool, COLS.address)
        self.p_country = self._col(self.pool, COLS.country)
        self.name_builder.cache(self.q_name)
        self.name_builder.cache(self.q_name_core)
        self.name_builder.cache(self.q_name_raw)
        self.name_builder.cache(self.p_name)
        self.name_builder.cache(self.p_name_core)
        self.name_builder.cache(self.p_name_raw)
        self.address_builder.cache(self.q_address)
        self.address_builder.cache(self.p_address)
        if COLS.pincode in self.pool.columns:
            # Keyed by the *short* component names that the feature builder
            # looks up, not by the raw column names.
            self._components = [
                {
                    "pincode": str(row.get(COLS.pincode, "") or ""),
                    "house_number": str(row.get(COLS.house_number, "") or ""),
                    "city": str(row.get(COLS.city, "") or ""),
                    "state": str(row.get(COLS.state, "") or ""),
                }
                for row in self.pool[[COLS.pincode, COLS.house_number, COLS.city, COLS.state]]
                .fillna("")
                .to_dict("records")
            ]
        else:
            self._components = [{} for _ in range(len(self.pool))]

    def _fit_tfidf(self) -> None:
        """Fit the TF-IDF models on pool + queries so both live in one space."""
        self._name_tfidf = self._fit_pair_tfidf(
            self.q_name_core, self.p_name_core,
            self.config.name_tfidf_ngram_min, self.config.name_tfidf_ngram_max,
        )
        self._address_tfidf = self._fit_pair_tfidf(
            self.q_address, self.p_address,
            self.config.address_tfidf_ngram_min, self.config.address_tfidf_ngram_max,
        )
        self._name_token_matrix = self._fit_pair_tokens(self.q_name_core, self.p_name_core)
        self._address_token_matrix = self._fit_pair_tokens(self.q_address, self.p_address)

    def _fit_pair_tfidf(self, q_values: np.ndarray, p_values: np.ndarray, n_min: int, n_max: int):
        combined = np.concatenate([p_values, q_values]) if len(q_values) else p_values
        matrix, vectorizer = build_tfidf_matrix(pd.Series(combined), analyzer="char_wb", ngram_range=(n_min, n_max))
        n_pool = len(p_values)
        return matrix[:n_pool], matrix[n_pool:]

    def _fit_pair_tokens(self, q_values: np.ndarray, p_values: np.ndarray):
        """IDF-weighted token matrices for the combined Jaccard features."""
        from collections import Counter

        import re

        token_re = re.compile(r"[a-z0-9]+")
        pool_tokens = [token_re.findall(str(v).lower()) for v in p_values]
        query_tokens = [token_re.findall(str(v).lower()) for v in q_values]
        df: Counter[str] = Counter()
        for toks in pool_tokens:
            df.update(set(toks))
        n_docs = max(1, len(pool_tokens))
        vocab = {t: i for i, t in enumerate(sorted(df))}
        idf = {t: float(np.log((1.0 + n_docs) / (1.0 + c)) + 1.0) for t, c in df.items()}
        p_matrix = token_series_to_matrix(pool_tokens, vocab, idf)
        q_matrix = token_series_to_matrix(query_tokens, vocab, idf) if len(query_tokens) else None
        return q_matrix, p_matrix

    def _pairwise_cosines(
        self, q_matrix: sparse.csr_matrix | None, p_matrix: sparse.csr_matrix, q_rows: np.ndarray, p_rows: np.ndarray
    ) -> np.ndarray:
        if q_matrix is None or len(q_rows) == 0:
            return np.zeros(len(p_rows), dtype=np.float32)
        q_block = q_matrix[q_rows]
        p_block = p_matrix[p_rows]
        return np.asarray(q_block.multiply(p_block).sum(axis=1)).ravel().astype(np.float32)

    # -- feature construction --------------------------------------------
    def build(self, candidates: CandidateSet, *, chunk_log: bool = True) -> FeatureMatrix:
        """Compute the full feature matrix for a candidate set."""
        columns = feature_names(self.config)
        n = len(candidates)
        X = np.zeros((n, len(columns)), dtype=np.float32)
        if n == 0:
            return FeatureMatrix(
                source1_ids=np.zeros(0, dtype=object), candidate_ids=np.zeros(0, dtype=object),
                candidate_sources=np.zeros(0, dtype=object), X=X, columns=columns,
                candidate_scores=np.zeros(0, dtype=np.float32), strategy_mask=np.zeros(0, dtype=np.int64),
            )
        q_pos = candidates.query_pos
        p_pos = candidates.pool_pos
        groups = candidates.iter_query_groups()
        n_query_entities = len(self.queries)
        n_name = len(NAME_FEATURES)
        n_address = len(ADDRESS_FEATURES)
        n_country = len(COUNTRY_FEATURES)
        for i, (q, p_rows, row_indices) in enumerate(groups):
            if len(p_rows) == 0:
                continue
            q_rows = np.full(len(p_rows), q, dtype=np.int64)
            name_tfidf = self._pairwise_cosines(self._name_tfidf[1], self._name_tfidf[0], q_rows, p_rows)
            addr_tfidf = self._pairwise_cosines(self._address_tfidf[1], self._address_tfidf[0], q_rows, p_rows)
            name_block = self.name_builder.block(
                [self.q_name[q]], self.p_name[p_rows],
                query_core=[self.q_name_core[q]], candidate_core=self.p_name_core[p_rows],
                query_raw=[self.q_name_raw[q]], candidate_raw=self.p_name_raw[p_rows],
                tfidf_cosine=name_tfidf,
            )
            address_block = self.address_builder.block(
                self.q_address[q], self.p_address[p_rows],
                query_components=self._query_components(q),
                candidate_components=[self._components[p] for p in p_rows],
                tfidf_cosine=addr_tfidf,
            )
            country_matrix = country_block(self.q_country[q], self.p_country[p_rows])
            base = np.hstack([name_block, address_block, country_matrix])
            if self.config.use_combined_features:
                base = np.hstack([base, self._combined_block(base)])
            X[row_indices] = base
            if chunk_log and (i + 1) % 500 == 0:
                LOGGER.info("  features: %d/%d query entities", i + 1, n_query_entities)
        return FeatureMatrix(
            source1_ids=candidates.query_ids[q_pos],
            candidate_ids=candidates.pool_ids[p_pos],
            candidate_sources=candidates.pool_source[p_pos],
            X=X,
            columns=columns,
            candidate_scores=candidates.scores,
            strategy_mask=candidates.strategy_mask,
        )

    def _query_components(self, q: int) -> dict[str, str]:
        if COLS.pincode not in self.queries.columns:
            return {}
        row = self.queries.iloc[q]
        return {
            "pincode": str(row.get(COLS.pincode, "") or ""),
            "house_number": str(row.get(COLS.house_number, "") or ""),
            "city": str(row.get(COLS.city, "") or ""),
            "state": str(row.get(COLS.state, "") or ""),
        }

    def _combined_block(self, base: np.ndarray) -> np.ndarray:
        """Parameter-free summaries of the name / address evidence.

        The ``exact_name_with_*`` indicators encode genuine *conflicts* -- an
        identical name combined with contradicting address evidence.  They are
        not weights; they simply expose the interaction in a form a linear
        model can use and a tree model would otherwise have to discover.
        """
        n = base.shape[0]
        out = np.zeros((n, len(COMBINED_FEATURES)), dtype=np.float32)
        name_cos = np.clip(base[:, NAME_FEATURES.index("name_tfidf_cosine")], 0, 1)
        addr_cos = np.clip(base[:, ADDRESS_FEATURES.index("address_tfidf_cosine")], 0, 1)
        out[:, 0] = np.sqrt(name_cos * addr_cos)          # geometric mean
        out[:, 1] = (name_cos + addr_cos) / 2.0           # arithmetic mean
        out[:, 2] = np.minimum(name_cos, addr_cos)
        out[:, 3] = np.maximum(name_cos, addr_cos)
        out[:, 4] = np.abs(name_cos - addr_cos)           # contradictory evidence
        exact_name = (
            base[:, NAME_FEATURES.index("name_exact_match")]
            + base[:, NAME_FEATURES.index("name_core_exact_match")]
        ) > 0
        out[:, 5] = (exact_name & (base[:, ADDRESS_FEATURES.index("address_exact_match")] == 0)).astype(np.float32)
        out[:, 6] = (exact_name & (base[:, ADDRESS_FEATURES.index("pincode_conflict")] > 0)).astype(np.float32)
        out[:, 7] = (exact_name & (base[:, ADDRESS_FEATURES.index("house_number_conflict")] > 0)).astype(np.float32)
        return out


def build_features(
    queries: pd.DataFrame,
    pool: pd.DataFrame,
    candidates: CandidateSet,
    config: FeatureConfig | None = None,
) -> FeatureMatrix:
    """Convenience wrapper: fit the caches and build the matrix in one call."""
    return FeatureBuilder(queries, pool, config).build(candidates)


def label_from_ground_truth(
    source1_ids: Sequence[str], candidate_ids: Sequence[str], truth: dict[str, frozenset[str]]
) -> np.ndarray:
    """Binary labels for candidate pairs; known-true pairs can never be 0."""
    lookup: dict[str, frozenset[str]] = truth
    return np.fromiter(
        (1 if str(c) in lookup.get(str(s), frozenset()) else 0 for s, c in zip(source1_ids, candidate_ids)),
        dtype=np.int8,
        count=len(candidate_ids),
    )


def ensure_source1_ids(frame: pd.DataFrame) -> np.ndarray:
    return frame[ID_COLUMN].fillna("").astype(str).to_numpy()
