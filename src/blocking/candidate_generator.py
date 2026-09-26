"""Multi-strategy candidate generation (blocking) and the candidate union.

The generator is the only place that decides *which* Source 2/3 records a
Source 1 entity is even allowed to be compared against.  The set it produces
is passed verbatim to the feature/model stage, so candidate recall is an upper
bound on achievable F0.5 and is measured explicitly (see
:mod:`src.evaluation.validation`).

Strategies
----------
A ``exact_normalized`` / ``exact_core``
    Dictionary lookup on the normalized / core name.
B ``token_name``
    IDF-weighted name-token cosine top-K.
C ``tfidf_name``
    Character n-gram TF-IDF cosine top-K (noisy names).
D ``address_token``
    IDF-weighted address-token cosine top-K.
E ``country_aware``
    Character n-gram TF-IDF top-K restricted to same-country pool records,
    plus a global fallback for Source 1 rows with a missing/unknown country.

Country is used as a *retrieval restriction derived from the data*; the code
contains no country list, so an unseen test country is simply its own group.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from ..config import BlockingConfig, ID_COLUMN
from ..preprocessing.preprocess import COLS
from .character_blocking import CharTfidfRetriever
from .exact_blocking import exact_block
from .token_blocking import TokenBlockingIndex, address_token_block, name_token_block, tokenize

LOGGER = logging.getLogger(__name__)

#: Ordered strategy names, used for per-strategy reporting.
STRATEGY_NAMES: tuple[str, ...] = (
    "exact_normalized",
    "exact_core",
    "token_name",
    "tfidf_name",
    "address_token",
    "country_aware",
)

_STRATEGY_BIT: dict[str, int] = {name: 1 << i for i, name in enumerate(STRATEGY_NAMES)}


def strategies_from_mask(mask: int) -> list[str]:
    return [name for name in STRATEGY_NAMES if mask & _STRATEGY_BIT[name]]


@dataclass
class CandidateSet:
    """Deduplicated candidate pairs for a set of Source 1 entities.

    ``pool_pos`` indexes the pool frame that was handed to the generator;
    ``strategy_mask`` records which strategies proposed each pair.
    """

    query_ids: np.ndarray
    pool_ids: np.ndarray
    pool_source: np.ndarray
    query_pos: np.ndarray
    pool_pos: np.ndarray
    scores: np.ndarray
    strategy_mask: np.ndarray
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.query_pos.size)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                ID_COLUMN: self.query_ids[self.query_pos],
                "candidate_entity_id": self.pool_ids[self.pool_pos],
                "candidate_source": self.pool_source[self.pool_pos],
                "candidate_score": self.scores,
                "strategy_mask": self.strategy_mask,
            }
        )

    def iter_query_groups(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Yield ``(query_pos, pool_positions, row_indices)`` grouped by query.

        ``pool_positions`` index the pool frame and ``row_indices`` index this
        candidate set, so callers can use both without re-deriving either.
        """
        order = np.argsort(self.query_pos, kind="stable")
        q_sorted = self.query_pos[order]
        if q_sorted.size == 0:
            return
        boundaries = np.flatnonzero(np.diff(q_sorted)) + 1
        for chunk in np.split(order, boundaries):
            yield int(self.query_pos[chunk[0]]), self.pool_pos[chunk], chunk

    def candidates_by_query(self) -> dict[str, list[str]]:
        """``source1_id -> [candidate ids]`` (used for the output file)."""
        out: dict[str, list[str]] = defaultdict(list)
        for q, p in zip(self.query_pos, self.pool_pos):
            out[str(self.query_ids[q])].append(str(self.pool_ids[p]))
        return {k: sorted(set(v)) for k, v in out.items()}

    def strategy_names(self) -> np.ndarray:
        return np.array([",".join(strategies_from_mask(int(m))) for m in self.strategy_mask], dtype=object)


class CandidateGenerator:
    """Builds the candidate union for Source 1 queries against a fixed pool."""

    def __init__(self, pool: pd.DataFrame, config: BlockingConfig | None = None, n_jobs: int = -1) -> None:
        self.config = config or BlockingConfig()
        self.pool = pool.reset_index(drop=True)
        self.n_jobs = n_jobs
        self.enabled = set(self.config.enabled)
        self.pool_ids = self.pool[ID_COLUMN].fillna("").astype(str).to_numpy()
        self.pool_country = (
            self.pool[COLS.country].fillna("").astype(str).to_numpy() if COLS.country in self.pool.columns
            else np.array([""] * len(self.pool), dtype=object)
        )
        self._name_retriever: CharTfidfRetriever | None = None
        self._address_retriever: CharTfidfRetriever | None = None
        self._name_token_index: TokenBlockingIndex | None = None
        self._address_token_index: TokenBlockingIndex | None = None
        self._country_rows: dict[str, np.ndarray] | None = None

    # -- lazily-built shared structures ----------------------------------
    @property
    def name_retriever(self) -> CharTfidfRetriever:
        if self._name_retriever is None:
            self._name_retriever = CharTfidfRetriever(self.pool, self.config, COLS.name_core, self.n_jobs)
        return self._name_retriever

    @property
    def address_retriever(self) -> CharTfidfRetriever:
        if self._address_retriever is None:
            self._address_retriever = CharTfidfRetriever(self.pool, self.config, COLS.address_core, self.n_jobs)
        return self._address_retriever

    @property
    def name_token_index(self) -> TokenBlockingIndex:
        if self._name_token_index is None:
            tokens = [tokenize(t, min_length=2) for t in self.pool[COLS.name_core].fillna("").astype(str)]
            self._name_token_index = TokenBlockingIndex(tokens, label="name", n_jobs=self.n_jobs)
        return self._name_token_index

    @property
    def address_token_index(self) -> TokenBlockingIndex:
        if self._address_token_index is None:
            tokens = [
                tokenize(t, drop_generic_address=True, min_length=2)
                for t in self.pool[COLS.address_core].fillna("").astype(str)
            ]
            self._address_token_index = TokenBlockingIndex(tokens, label="address", n_jobs=self.n_jobs)
        return self._address_token_index

    @property
    def country_rows(self) -> dict[str, np.ndarray]:
        """``country -> pool positions``; groups are derived purely from data."""
        if self._country_rows is None:
            mapping: dict[str, list[int]] = defaultdict(list)
            for pos, c in enumerate(self.pool_country):
                mapping[str(c)].append(pos)
            self._country_rows = {k: np.asarray(v, dtype=np.int64) for k, v in mapping.items()}
            LOGGER.info("Country partition of the pool: %d groups, sizes top-5=%s",
                        len(self._country_rows),
                        sorted((len(v) for v in self._country_rows.values()), reverse=True)[:5])
        return self._country_rows

    # -- individual strategies --------------------------------------------
    def _run_exact(self, queries: pd.DataFrame) -> list[tuple[int, int, float, str]]:
        return exact_block(queries, self.pool, max_per_query=self.config.exact_name_max_per_s1)

    def _run_token_name(self, queries: pd.DataFrame) -> list[tuple[int, int, float, str]]:
        pool_tokens = [tokenize(t, min_length=2) for t in self.pool[COLS.name_core].fillna("").astype(str)]
        query_tokens = [tokenize(t, min_length=2) for t in queries[COLS.name_core].fillna("").astype(str)]
        triples = self.name_token_index.query(
            query_tokens,
            top_k=self.config.token_name_max_per_s1,
            min_score=self.config.token_name_min_score,
        )
        return [(q, p, s, "token_name") for q, p, s in triples]

    def _run_tfidf_name(self, queries: pd.DataFrame) -> list[tuple[int, int, float, str]]:
        result = self.name_retriever.retriever.query(
            self.name_retriever.transform(queries),
            self.config.tfidf_name_top_k,
            self.config.tfidf_name_min_score,
        )
        return [(q, p, s, "tfidf_name") for q, p, s in result.iter_topk()]

    def _run_address_token(self, queries: pd.DataFrame) -> list[tuple[int, int, float, str]]:
        pool_tokens = [
            tokenize(t, drop_generic_address=True, min_length=2)
            for t in self.pool[COLS.address_core].fillna("").astype(str)
        ]
        query_tokens = [
            tokenize(t, drop_generic_address=True, min_length=2)
            for t in queries[COLS.address_core].fillna("").astype(str)
        ]
        triples = self.address_token_index.query(
            query_tokens,
            top_k=self.config.address_token_max_per_s1,
            min_score=self.config.address_token_min_score,
        )
        return [(q, p, s, "address_token") for q, p, s in triples]

    def _run_country_aware(self, queries: pd.DataFrame) -> list[tuple[int, int, float, str]]:
        """Retrieve within same-country partitions, with a global fallback.

        A Source 1 row whose country is missing, or whose country has too few
        pool records, falls back to a global character-TFIDF pass so that an
        open-set or absent country can never silently lose all candidates.
        """
        out: list[tuple[int, int, float, str]] = []
        query_country = (
            queries[COLS.country].fillna("").astype(str).to_numpy() if COLS.country in queries.columns
            else np.array([""] * len(queries), dtype=object)
        )
        query_tokens = [tokenize(t, min_length=2) for t in queries[COLS.name_core].fillna("").astype(str)]
        rows = self.country_rows
        for country, group_rows in rows.items():
            if len(group_rows) < self.config.country_aware_min_pool_size:
                continue
            q_pos = np.flatnonzero(query_country == country)
            if q_pos.size == 0:
                continue
            sub_queries = [query_tokens[i] for i in q_pos]
            triples = self.name_token_index.query(
                sub_queries,
                top_k=self.config.country_aware_top_k,
                min_score=self.config.country_aware_min_score,
                restrict_rows=group_rows,
            )
            out.extend((int(q_pos[q]), p, s, "country_aware") for q, p, s in triples)
        # Fallback for queries with no usable country partition.
        fallback_q = [
            i
            for i, c in enumerate(query_country)
            if not c or len(rows.get(c, ())) < self.config.country_aware_min_pool_size
        ]
        if fallback_q:
            sub_queries = [query_tokens[i] for i in fallback_q]
            triples = self.name_token_index.query(
                sub_queries,
                top_k=self.config.country_aware_fallback_top_k,
                min_score=self.config.country_aware_min_score,
            )
            out.extend((int(fallback_q[q]), p, s, "country_aware") for q, p, s in triples)
        return out

    # -- public API -------------------------------------------------------
    def generate(self, queries: pd.DataFrame) -> CandidateSet:
        """Run all enabled strategies and return their deduplicated union."""
        queries = queries.reset_index(drop=True)
        n_queries = len(queries)
        per_strategy: dict[str, list[tuple[int, int, float, str]]] = {}
        runners = {
            "exact_name": self._run_exact,
            "token_name": self._run_token_name,
            "tfidf_name": self._run_tfidf_name,
            "address_token": self._run_address_token,
            "country_aware": self._run_country_aware,
        }
        for key in STRATEGY_ENABLED_ORDER:
            if key not in self.enabled:
                LOGGER.info("Strategy '%s' disabled by config", key)
                continue
            t0 = pd.Timestamp.utcnow()
            triples = runners[key](queries)
            per_strategy[key] = triples
            LOGGER.info("Strategy %-16s produced %9d raw pairs in %.1fs", key, len(triples),
                        (pd.Timestamp.utcnow() - t0).total_seconds())
        return self._union(queries, per_strategy, n_queries)

    def _union(
        self,
        queries: pd.DataFrame,
        per_strategy: dict[str, list[tuple[int, int, float, str]]],
        n_queries: int,
    ) -> CandidateSet:
        """Deduplicate into ``(query, pool)`` keys, keeping the max score."""
        npool = len(self.pool)
        best: dict[int, float] = {}
        mask: dict[int, int] = {}
        origin: dict[int, int] = {}
        for key, triples in per_strategy.items():
            for q, p, score, name in triples:
                if not (0 <= q < n_queries) or not (0 <= p < npool):
                    continue
                composite = q * npool + p
                bit = _STRATEGY_BIT.get(name, 0)
                mask[composite] = mask.get(composite, 0) | bit
                if score > best.get(composite, -1.0):
                    best[composite] = score
                    origin[composite] = q
        total_raw = sum(len(v) for v in per_strategy.values())
        LOGGER.info("Candidate union: %d raw pairs -> %d unique pairs (%.1fx dedup)",
                    total_raw, len(best), (total_raw / len(best)) if best else 0.0)

        if not best:
            empty_i = np.zeros(0, dtype=np.int64)
            return CandidateSet(
                query_ids=queries[ID_COLUMN].fillna("").astype(str).to_numpy(),
                pool_ids=self.pool_ids,
                pool_source=self._pool_source(),
                query_pos=empty_i, pool_pos=empty_i, scores=np.zeros(0), strategy_mask=np.zeros(0, dtype=np.int64),
                stats={"raw_pairs": total_raw, "unique_pairs": 0, "capped_pairs": 0},
            )

        keys = np.fromiter(best.keys(), dtype=np.int64, count=len(best))
        q_pos = (keys // npool).astype(np.int64)
        p_pos = (keys % npool).astype(np.int64)
        scores = np.fromiter((best[k] for k in best), dtype=np.float32, count=len(best))
        masks = np.fromiter((mask[k] for k in best), dtype=np.int64, count=len(best))

        q_pos, p_pos, scores, masks, n_capped = self._apply_caps(q_pos, p_pos, scores, masks, n_queries)
        per_s1 = np.bincount(q_pos, minlength=n_queries) if len(q_pos) else np.zeros(n_queries, dtype=np.int64)
        stats = {
            "n_queries": int(n_queries),
            "n_pool": int(npool),
            "raw_pairs": int(total_raw),
            "unique_pairs": int(len(q_pos)),
            "capped_pairs": int(n_capped),
            "pairs_per_query_mean": float(per_s1.mean()) if n_queries else 0.0,
            "pairs_per_query_median": float(np.median(per_s1)) if n_queries else 0.0,
            "pairs_per_query_p95": float(np.percentile(per_s1, 95)) if n_queries else 0.0,
            "pairs_per_query_max": int(per_s1.max()) if n_queries else 0,
            "queries_with_zero_candidates": int((per_s1 == 0).sum()) if n_queries else 0,
            "per_strategy_raw_pairs": {k: len(v) for k, v in per_strategy.items()},
        }
        LOGGER.info(
            "Candidates per Source 1 entity: mean=%.1f median=%.0f p95=%.0f max=%d; zero-candidate entities=%d",
            stats["pairs_per_query_mean"], stats["pairs_per_query_median"],
            stats["pairs_per_query_p95"], stats["pairs_per_query_max"], stats["queries_with_zero_candidates"],
        )
        return CandidateSet(
            query_ids=queries[ID_COLUMN].fillna("").astype(str).to_numpy(),
            pool_ids=self.pool_ids,
            pool_source=self._pool_source(),
            query_pos=q_pos, pool_pos=p_pos, scores=scores, strategy_mask=masks,
            stats=stats,
        )

    def _apply_caps(
        self,
        q_pos: np.ndarray,
        p_pos: np.ndarray,
        scores: np.ndarray,
        masks: np.ndarray,
        n_queries: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Enforce the per-entity cap, always keeping the strongest candidates."""
        cap = self.config.max_candidates_per_s1
        floor = self.config.min_candidates_per_s1
        order = np.lexsort((-scores, q_pos))
        q_sorted = q_pos[order]
        if q_sorted.size == 0:
            return q_pos, p_pos, scores, masks, 0
        boundaries = np.flatnonzero(np.diff(q_sorted)) + 1
        keep_chunks: list[np.ndarray] = []
        n_capped = 0
        for chunk in np.split(order, boundaries):
            if cap is not None and len(chunk) > max(cap, floor):
                chunk = chunk[:cap]
                n_capped += 1
            keep_chunks.append(chunk)
        keep = np.sort(np.concatenate(keep_chunks)) if keep_chunks else np.zeros(0, dtype=np.int64)
        return q_pos[keep], p_pos[keep], scores[keep], masks[keep], n_capped

    def _pool_source(self) -> np.ndarray:
        col = "source"
        if col in self.pool.columns:
            return self.pool[col].fillna("").astype(str).to_numpy()
        return np.array([""] * len(self.pool), dtype=object)


#: Order in which strategies are executed (cheap / precise first).
STRATEGY_ENABLED_ORDER: tuple[str, ...] = ("exact_name", "token_name", "tfidf_name", "address_token", "country_aware")


def build_pool(source2: pd.DataFrame, source3: pd.DataFrame) -> pd.DataFrame:
    """Concatenate Source 2 and Source 3 into one retrieval pool.

    A ``source`` column records provenance, and ids are verified to be unique
    across the two tables.
    """
    parts = []
    for label, frame in (("source2", source2), ("source3", source3)):
        part = frame.copy()
        part["source"] = label
        parts.append(part)
    pool = pd.concat(parts, ignore_index=True)
    ids = pool[ID_COLUMN].fillna("").astype(str)
    if ids.duplicated().any():
        dupes = sorted(set(ids[ids.duplicated()].tolist()))[:5]
        raise ValueError(
            "Source 2 and Source 3 share entity ids, so a merged pool would be ambiguous "
            f"(e.g. {dupes}). Disambiguate the input files before running the pipeline."
        )
    pool[ID_COLUMN] = ids
    return pool.reset_index(drop=True)


def iter_candidate_counts(cand: CandidateSet) -> Iterable[tuple[str, int]]:
    """Yield ``(source1_id, n_candidates)`` for every query entity."""
    counts: dict[str, int] = defaultdict(int)
    for q in cand.query_pos:
        counts[str(cand.query_ids[q])] += 1
    for s1 in cand.query_ids:
        yield str(s1), int(counts.get(str(s1), 0))


def group_by_query(q_pos: Sequence[int], n_queries: int) -> list[np.ndarray]:
    """Group an array of query positions into per-query index arrays."""
    order = np.argsort(np.asarray(q_pos), kind="stable")
    sorted_q = np.asarray(q_pos)[order]
    if sorted_q.size == 0:
        return [np.zeros(0, dtype=np.int64) for _ in range(n_queries)]
    boundaries = np.flatnonzero(np.diff(sorted_q)) + 1
    groups = np.split(order, boundaries)
    out: list[np.ndarray] = [np.zeros(0, dtype=np.int64)] * n_queries
    for g in groups:
        out[int(np.asarray(q_pos)[g[0]])] = g
    return out
