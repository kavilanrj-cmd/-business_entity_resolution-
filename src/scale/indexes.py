"""Disk-backed retrieval indexes for the scalable entity-resolution layer.

Three index types, all built by sequential passes over the normalised Parquet
store and all queryable through ``searchsorted`` / fancy indexing on mmap-able
NumPy arrays:

``TokenIndex``
    An inverted index over name (or address) tokens.  Postings are a single
    ``int64`` per (term, rowid) pair, ``(term_id << 32) | pool_rowid``, sorted, so
    term ``t`` occupies the half-open slice ``[offsets[t], offsets[t + 1])``.  One
    flat array instead of millions of Python lists: ~330 MB for a 10.3M-row pool
    versus tens of GB for ``defaultdict(list)``.

``FixedKeyIndex``
    A sorted fixed-width key array plus the originating row order.  The *same*
    array serves exact-key lookup (``searchsorted`` left/right around a
    NUL-padded key) and sorted-neighbourhood lookup (expand a window around the
    insertion point of a prefix, keep only rows still sharing the prefix).  The
    common-prefix comparison is vectorised over the whole window, so a batch of
    queries costs one ``searchsorted`` plus one broadcast compare.

``CountryIndex``
    ``country -> sorted pool_rowid``; a couple of hundred groups, negligible
    memory, but it removes the single largest source of bad candidates for
    international suppliers.

Why a fixed-width byte array rather than a hash
-----------------------------------------------
An exact-key block is found by binary search on the raw truncated key.  A 64-bit
hash would be marginally faster but still needs the same sort to build, and
truncation at 32 characters is exact for every business name in practice, so the
hash would buy nothing except a collision argument to defend.  See
``reports/scalable_architecture.md``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import ScaleConfig
from .idmap import IdResolver, _as_fixed_bytes, measure_id_width
from .progress import MemoryGuard, format_bytes
from .store import PoolStore, RecordStore, read_store_manifest, write_store_manifest

LOGGER = logging.getLogger(__name__)

#: ``pool_rowid`` lives in the low half of a packed posting and ``term_id`` in
#: the high half.  32 bits addresses 4.3e9 pool rows, ~400x this dataset.
ROWID_BITS = 32
ROWID_MASK = (1 << ROWID_BITS) - 1

#: Upper bound on expanded postings per sub-block inside :meth:`TokenIndex.gather`.
#: The caller sets the *semantic* budget; this bounds the transient allocation so
#: a 10k-query batch at 3000 postings each (120 MB) never becomes peak RSS.
DEFAULT_EXPAND_CAP = 4_000_000

#: Byte-block width for prefix comparisons.  8 keeps the temporary boolean array
#: at ~82 MB for a 10.3M-row pool while still vectorising the compare.
PREFIX_BLOCK = 8

#: Queries per sub-block in :meth:`FixedKeyIndex.neighbourhood`.  Each query owns a
#: ``(2 * window + 1)``-row window, so 2048 x 101 x 32 bytes is 6.6 MB.
NEIGHBOURHOOD_BLOCK = 2048


def pack_postings(term_ids: np.ndarray, rowids: np.ndarray) -> np.ndarray:
    """``(term_id << 32) | rowid`` as int64, ready to be sorted."""
    return (term_ids.astype(np.int64) << ROWID_BITS) | rowids.astype(np.int64)


def unpack_postings(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_postings`."""
    return (packed >> ROWID_BITS).astype(np.int32), (packed & ROWID_MASK).astype(np.int32)


def _dedupe_sorted(packed: np.ndarray) -> np.ndarray:
    """Drop duplicate ``(term, rowid)`` pairs from a sorted packed array.

    A name that repeats a token ("star star imports") contributes two identical
    postings; counting it twice would inflate the document norm and the overlap.
    """
    if packed.size < 2:
        return packed
    keep = np.empty(packed.size, dtype=bool)
    keep[0] = True
    np.not_equal(packed[1:], packed[:-1], out=keep[1:])
    if keep.all():
        return packed
    return packed[keep]


# --------------------------------------------------------------------------
# tokenisation
# --------------------------------------------------------------------------
def as_arrow(values) -> pa.Array:
    """Coerce a pandas Series / numpy array / list of strings to ``pa.string()``."""
    if isinstance(values, pa.Array):
        return values
    if hasattr(values, "tolist"):
        values = values.tolist()
    return pa.array(list(values), type=pa.string())


def _drop_stop(tokens: pa.Array, stop: pa.Array | None) -> pa.Array:
    if stop is None or len(stop) == 0 or len(tokens) == 0:
        return tokens
    hit = pc.index_in(tokens, value_set=stop).fill_null(-1)
    return pc.filter(tokens, pc.less(hit, 0).fill_null(True))


def _split_tokens(
    values: pa.Array, rowid0: int, min_length: int, stop: pa.Array | None = None
) -> tuple[pa.Array, np.ndarray]:
    """Split on spaces, drop empties / short / stop tokens, keep each token's row.

    Returns ``(tokens, rowid_per_token)`` of equal length.  Every per-token filter
    happens *here*, while the row ids are still positional, so the two arrays can
    never drift apart.  The split is a single Arrow C++ kernel, so tokenising
    10.3M names never materialises a Python list of lists, and the row ids come
    from ``np.repeat`` rather than a Python loop.
    """
    if len(values) == 0:
        return pa.array([], type=pa.string()), np.zeros(0, dtype=np.int64)
    lists = pc.split_pattern(values, " ")
    counts = np.diff(np.asarray(lists.offsets)).astype(np.int64)
    flat = lists.values
    rowids = np.repeat(
        np.arange(int(rowid0), int(rowid0) + counts.size, dtype=np.int64), counts
    )
    keep = np.asarray(
        pc.and_(pc.is_valid(flat), pc.greater_equal(pc.utf8_length(flat), min_length)),
        dtype=bool,
    )
    if stop is not None and len(stop) and len(flat):
        is_stop = pc.index_in(flat, value_set=stop).fill_null(-1)
        keep &= np.asarray(pc.less(is_stop, 0).fill_null(True), dtype=bool)
    if not keep.all():
        flat = pc.filter(flat, pa.array(keep))
        rowids = rowids[keep]
    return flat, rowids


def encode_queries(
    index: TokenIndex,
    values,
    *,
    min_length: int,
    stop_tokens: frozenset[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a batch of queries as ``(term_ids, tf)``, padded with ``-1``.

    ``term_ids`` is ``(n_queries, max_tokens)`` int32 and ``tf`` the matching raw
    token count, so a query's vector is the idf-weighted bag of its tokens.
    """
    values = as_arrow(values)
    stop = pa.array(sorted(stop_tokens), type=pa.string()) if stop_tokens else None
    tokens, rowids = _split_tokens(values, 0, max(1, int(min_length)), stop)
    if len(tokens) == 0:
        return np.full((len(values), 0), -1, dtype=np.int32), np.zeros(
            (len(values), 0), dtype=np.float32
        )
    codes = (
        pc.index_in(tokens, value_set=index.vocabulary)
        .fill_null(-1)
        .to_numpy(zero_copy_only=False)
    )
    return (
        _densify(codes, rowids, len(values), -1, np.int32),
        _densify(np.ones(codes.size, dtype=np.float32), rowids, len(values), 0.0, np.float32),
    )


def _densify(
    flat: np.ndarray, rowids: np.ndarray, n_rows: int, fill, dtype
) -> np.ndarray:
    """Scatter a token-ordered array into a dense ``(n_rows, max_per_row)`` array."""
    if rowids.size == 0 or n_rows == 0:
        return np.full((n_rows, 0), fill, dtype=dtype)
    rows = rowids.astype(np.int64)
    order = np.argsort(rows, kind="stable")
    rows_s = rows[order]
    counts = np.bincount(rows_s, minlength=n_rows)
    width = int(counts.max())
    starts = np.zeros(n_rows, dtype=np.int64)
    if n_rows > 1:
        np.cumsum(counts[:-1], out=starts[1:])
    within = np.arange(rows_s.size, dtype=np.int64) - starts[rows_s]
    dense = np.full((n_rows, width), fill, dtype=dtype)
    dense[rows_s, within] = flat[order]
    return dense


# --------------------------------------------------------------------------
# token index
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TokenIndex:
    """Inverted index over one whitespace-tokenised field."""

    field: str
    vocabulary: pa.Array
    idf: np.ndarray
    idf_sq: np.ndarray
    df: np.ndarray
    postings: np.ndarray
    offsets: np.ndarray
    doc_norm: np.ndarray
    doc_terms: np.ndarray
    n_pool: int

    def __post_init__(self) -> None:
        if self.offsets.shape[0] != len(self.vocabulary) + 1:
            raise ValueError("offsets must have exactly one more entry than the vocabulary")
        if int(self.offsets[-1]) != self.postings.size:
            raise ValueError("offsets[-1] must equal the number of postings")

    # -- introspection ---------------------------------------------------
    @property
    def n_terms(self) -> int:
        return int(len(self.vocabulary))

    @property
    def n_postings(self) -> int:
        return int(self.postings.size)

    def nbytes(self) -> int:
        return int(
            self.idf.nbytes
            + self.idf_sq.nbytes
            + self.df.nbytes
            + self.postings.nbytes
            + self.offsets.nbytes
            + self.doc_norm.nbytes
            + self.doc_terms.nbytes
        )

    def posting_len(self, term_id: int) -> int:
        return int(self.offsets[term_id + 1] - self.offsets[term_id])

    def term_frequency(self, term_id: int) -> int:
        return int(self.df[term_id])

    def term(self, term_id: int) -> str:
        return str(self.vocabulary[term_id].as_py())

    def describe(self) -> str:
        return (
            f"{self.field}: {self.n_terms:,} terms, {self.n_postings:,} postings, "
            f"{format_bytes(self.nbytes())}"
        )

    # -- query side ------------------------------------------------------
    def query_norm(self, term_ids: np.ndarray, tf: np.ndarray) -> np.ndarray:
        """L2 norm of the idf-weighted query vector, per row.

        The norm covers *every* in-vocabulary term of the query, not only the
        ``max_terms`` that get expanded, otherwise a long name would score higher
        than a short one for exactly the same overlap.
        """
        safe = np.where(term_ids >= 0, term_ids, 0)
        idf = np.where(term_ids >= 0, self.idf[safe], 0.0)
        w = (idf * tf).astype(np.float64)
        return np.sqrt((w * w).sum(axis=1))

    def gather(
        self,
        term_ids: np.ndarray,
        tf: np.ndarray,
        *,
        budget: int,
        max_terms: int,
        expand_cap: int = DEFAULT_EXPAND_CAP,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score every pool row sharing an indexed term with a batch of queries.

        Returns ``(query_index, pool_rowid, cosine)`` sorted by query index and
        then by descending cosine within each query.

        Terms are visited rarest-first and expansion stops once the cumulative
        posting count would exceed ``budget``.  Because the vocabulary excludes
        any term with ``df > max_df``, no single posting list can dominate a
        query: that is what makes this ``O(queries * budget)`` rather than
        quadratic in the pool size.
        """
        n_queries = int(term_ids.shape[0]) if term_ids.ndim == 2 else 0
        if n_queries == 0 or term_ids.size == 0 or max_terms < 1 or budget < 1:
            return _empty_hits()
        block = max(1, int(expand_cap) // int(budget))
        parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for lo in range(0, n_queries, block):
            hi = min(lo + block, n_queries)
            hit = self._gather_block(
                term_ids[lo:hi], tf[lo:hi], budget=int(budget), max_terms=int(max_terms)
            )
            if hit[0].size:
                parts.append((hit[0] + np.int32(lo), hit[1], hit[2]))
        if not parts:
            return _empty_hits()
        if len(parts) == 1:
            return parts[0]
        return (
            np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]),
        )

    def _gather_block(
        self, term_ids: np.ndarray, tf: np.ndarray, *, budget: int, max_terms: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_queries = int(term_ids.shape[0])
        if n_queries == 0 or term_ids.size == 0:
            return _empty_hits()

        # 1. rarest terms first, then keep at most ``max_terms`` of them.
        safe = np.where(term_ids >= 0, term_ids, 0)
        df_view = self.df[safe]
        df_ranked = np.where(term_ids >= 0, df_view, np.iinfo(np.int64).max)
        order = np.argsort(df_ranked, axis=1, kind="stable")
        terms = np.take_along_axis(term_ids, order, axis=1)[:, :max_terms]
        freqs = np.take_along_axis(tf, order, axis=1)[:, :max_terms]
        lengths = np.where(terms >= 0, self.df[np.where(terms >= 0, terms, 0)], 0).astype(np.int64)

        # 2. budget: cheapest terms first, but always allow the rarest one so a
        #    hard entity is never left with no candidates at all.
        keep = np.cumsum(lengths, axis=1) <= int(budget)
        keep[:, 0] = lengths[:, 0] > 0
        keep &= terms >= 0
        if not keep.any():
            return _empty_hits()

        # 3. expand the kept (query, term) slots into concrete postings.
        slot = np.flatnonzero(keep.ravel())
        slot_len = lengths.ravel()[slot]
        total = int(slot_len.sum())
        if total == 0:
            return _empty_hits()
        slot_query = (slot // max_terms).astype(np.int64)
        slot_term = terms.ravel()[slot].astype(np.int64)
        within = np.arange(total, dtype=np.int64) - np.repeat(
            np.cumsum(slot_len) - slot_len, slot_len
        )
        flat_pos = np.repeat(self.offsets[slot_term], slot_len) + within
        term_of, row_of = unpack_postings(self.postings[flat_pos])
        # A term contributes ``idf[t] * qtf[t]`` to the query vector and
        # ``idf[t] * dtf[t]`` to the document vector, so the dot product carries
        # ``idf[t] ** 2``.  Dropping the square would score a name that shares all
        # of its terms at ~1/df instead of 1.0.
        weight = self.idf_sq[term_of] * np.repeat(freqs.ravel()[slot], slot_len)
        query_of = np.repeat(slot_query, slot_len)

        # 4. one idf-weighted overlap per (query, row), then a cosine.
        pair = query_of * np.int64(self.n_pool) + row_of
        rank = np.argsort(pair, kind="stable")
        pair_sorted = pair[rank]
        new_group = np.empty(pair_sorted.size, dtype=bool)
        new_group[0] = True
        np.not_equal(pair_sorted[1:], pair_sorted[:-1], out=new_group[1:])
        starts = np.flatnonzero(new_group)
        unique_pair = pair_sorted[starts]
        overlap = np.add.reduceat(weight[rank].astype(np.float64), starts)
        q_index = (unique_pair // self.n_pool).astype(np.int32)
        r_index = (unique_pair % self.n_pool).astype(np.int32)
        denom = self.query_norm(term_ids, tf)[q_index] * self.doc_norm[r_index]
        scores = np.divide(overlap, denom, out=np.zeros_like(overlap), where=denom > 0)
        # ``lexsort`` uses the last key as primary: group by query, best first.
        order2 = np.lexsort((-scores, q_index))
        return q_index[order2], r_index[order2], scores[order2].astype(np.float32)


def _empty_hits() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.float32),
    )


def build_token_index(
    store: RecordStore,
    field: str,
    *,
    n_pool: int,
    config: ScaleConfig,
    stop_tokens: frozenset[str],
    guard: MemoryGuard | None = None,
) -> TokenIndex:
    """Two streaming passes: document frequency, then postings and norms.

    Pass 1 needs only the *distinct* terms of each shard, so its memory is bounded
    by the vocabulary (a few million strings) rather than by the number of tokens
    (tens of millions).  Pass 2 never holds more than one shard's postings plus
    the growing output array.
    """
    started = time.perf_counter()
    min_length = max(1, int(config.token_min_length))
    min_df = max(1, int(config.min_df))
    max_df = int(config.max_df)
    stop = pa.array(sorted(stop_tokens), type=pa.string())

    df: dict[str, int] = {}
    raw_terms = 0
    for rowid0, _, frame in store.iter_shards([field]):
        tokens, rowids = _split_tokens(as_arrow(frame[field]), int(rowid0), min_length, stop)
        if len(tokens) == 0:
            continue
        encoded = pc.dictionary_encode(tokens)
        local = encoded.dictionary.to_pylist()
        raw_terms += len(local)
        # Document frequency, not term frequency: a name that repeats a token must
        # still count once, or ``df`` would disagree with the posting counts and
        # the ``max_df`` ceiling would reject the wrong terms.
        packed = pack_postings(
            encoded.indices.to_numpy(zero_copy_only=False).astype(np.int64), rowids
        )
        packed.sort(kind="stable")
        term_of, _ = unpack_postings(_dedupe_sorted(packed))
        counts = np.bincount(term_of, minlength=len(local))
        for term, count in zip(local, counts):
            if count:
                df[term] = df.get(term, 0) + int(count)
        if guard is not None:
            guard.check()

    kept = sorted(t for t, count in df.items() if min_df <= count <= max_df)
    df_final = np.array([df[t] for t in kept], dtype=np.int64)
    del df
    if not kept:
        raise ValueError(
            f"no {field} term survived df filtering (min_df={min_df}, max_df={max_df})"
        )
    vocabulary = pa.array(kept, type=pa.string())
    n_terms = len(kept)
    idf = np.log(1.0 + float(n_pool) / np.maximum(df_final, 1)).astype(np.float32)
    idf_sq = (idf * idf).astype(np.float32)
    LOGGER.info(
        "%s: %d distinct terms seen, %d indexed of %d surviving "
        "(min_df=%d, max_df=%d) in %.1fs",
        field, raw_terms, n_terms, len(df_final), min_df, max_df,
        time.perf_counter() - started,
    )

    idf_sq_weights = idf_sq.astype(np.float64)
    doc_norm_sq = np.zeros(int(n_pool), dtype=np.float64)
    doc_terms = np.zeros(int(n_pool), dtype=np.int32)
    parts: list[np.ndarray] = []
    # NOTE: ``iter_shards`` yields ``(start_rowid, n_rows, frame)``.
    for rowid0, _, frame in store.iter_shards([field]):
        tokens, rowids = _split_tokens(as_arrow(frame[field]), int(rowid0), min_length, stop)
        if len(tokens) == 0:
            continue
        codes = (
            pc.index_in(tokens, value_set=vocabulary)
            .fill_null(-1)
            .to_numpy(zero_copy_only=False)
        )
        hit = codes >= 0
        if not hit.all():
            codes = codes[hit]
            rowids = rowids[hit]
        if codes.size == 0:
            continue
        parts.append(pack_postings(codes, rowids))
        doc_norm_sq += np.bincount(rowids, weights=idf_sq_weights[codes], minlength=int(n_pool))
        doc_terms += np.bincount(rowids, minlength=int(n_pool))
        if guard is not None:
            guard.check()

    postings = np.concatenate(parts) if len(parts) > 1 else (parts[0] if parts else np.zeros(0, np.int64))
    del parts
    postings.sort(kind="stable")
    postings = _dedupe_sorted(postings)
    term_of, _ = unpack_postings(postings)
    counts = np.bincount(term_of, minlength=n_terms).astype(np.int64)
    offsets = np.zeros(n_terms + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    if not np.array_equal(counts, df_final):
        raise AssertionError(
            f"{field}: pass-2 posting counts disagree with pass-1 df "
            f"(max |diff| {int(np.abs(counts - df_final).max())})"
        )
    index = TokenIndex(
        field=field,
        vocabulary=vocabulary,
        idf=idf,
        idf_sq=idf_sq,
        df=df_final,
        postings=postings,
        offsets=offsets,
        doc_norm=np.sqrt(doc_norm_sq).astype(np.float32),
        doc_terms=doc_terms,
        n_pool=int(n_pool),
    )
    LOGGER.info("%s in %.1fs", index.describe(), time.perf_counter() - started)
    if guard is not None:
        guard.check()
    return index


# --------------------------------------------------------------------------
# fixed-width key index
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FixedKeyIndex:
    """Sorted fixed-width keys, the pool row each came from, and block starts."""

    field: str
    width: int
    keys: np.ndarray
    rows: np.ndarray
    starts: np.ndarray
    n_pool: int
    sample_max_chars: int = 0

    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.rows.nbytes + self.starts.nbytes)

    @property
    def n_blocks(self) -> int:
        return int(self.starts.size)

    def describe(self) -> str:
        return (
            f"{self.field}: {self.keys.size:,} keys x {self.width} B, "
            f"{self.n_blocks:,} blocks, {format_bytes(self.nbytes())}"
        )

    # -- exact-key lookup -------------------------------------------------
    def probe(self, key: str) -> np.ndarray:
        """Encode one key as a NUL-padded fixed-width probe."""
        return _fixed_probe([key], self.width, self.width).view(f"S{self.width}").reshape(1)

    def block_of(self, probe: np.ndarray) -> tuple[int, int]:
        """Half-open ``[lo, hi)`` slice of rows whose key equals ``probe``."""
        probe = np.asarray(probe).reshape(-1)
        if probe.size != 1:
            raise ValueError("block_of takes exactly one probe key")
        lo = int(np.searchsorted(self.keys, probe, side="left"))
        hi = int(np.searchsorted(self.keys, probe, side="right"))
        return lo, hi

    def lookup(self, key: str) -> np.ndarray:
        lo, hi = self.block_of(self.probe(key))
        return self.rows[lo:hi]

    # -- sorted neighbourhood --------------------------------------------
    def neighbourhood(
        self,
        keys: Sequence[str],
        *,
        prefix_chars: int,
        min_chars: int,
        window: int,
        cap: int,
    ) -> list[np.ndarray]:
        """Sorted neighbours of each key that still share its leading prefix.

        Two pool rows sort adjacently exactly when they share a prefix, so
        expanding ``window`` rows either side of a prefix's insertion point and
        comparing the shared prefix finds spelling variants that no token index
        can see ("Infosys" / "Info Systems").  The whole window is compared as one
        broadcast operation, so cost is ``O(queries * window * width)`` with no
        Python-level loop over rows.
        """
        total = len(keys)
        if total == 0 or window < 1:
            return [np.zeros(0, dtype=np.int32) for _ in range(total)]
        plen = max(1, min(int(prefix_chars), self.width))
        out: list[np.ndarray] = []
        for lo in range(0, total, NEIGHBOURHOOD_BLOCK):
            hi = min(lo + NEIGHBOURHOOD_BLOCK, total)
            out.extend(
                self._neighbourhood_block(
                    keys[lo:hi], plen=plen, min_chars=int(min_chars), window=int(window), cap=int(cap)
                )
            )
        return out

    def _neighbourhood_block(
        self, keys: Sequence[str], *, plen: int, min_chars: int, window: int, cap: int
    ) -> list[np.ndarray]:
        n = len(keys)
        if n == 0:
            return []
        probes = _fixed_probe(keys, self.width, plen)
        probe_keys = probes.view(f"S{self.width}").reshape(n)
        lengths = np.frombuffer(probes.tobytes(), dtype=np.uint8).reshape(n, self.width)
        # A key shorter than the prefix can only match itself, so only queries
        # with real content take part in the window search.
        active = lengths[:, : min_chars].any(axis=1) if min_chars <= self.width else np.ones(n, bool)
        anchor = np.searchsorted(self.keys, probe_keys, side="left")
        offsets = np.arange(-window, window + 1, dtype=np.int64)
        index = anchor[:, None] + offsets[None, :]
        valid = (index >= 0) & (index < self.keys.size) & active[:, None]
        clipped = np.clip(index, 0, max(self.keys.size - 1, 0))
        window_bytes = (
            self.keys[clipped].view(np.uint8).reshape(n, -1, self.width)[:, :, :plen]
        )
        match = valid & (window_bytes == probes[:, :plen].reshape(n, 1, plen)).all(axis=2)
        if cap > 0:
            # Keep the first ``cap`` hits of each row, scanning outwards from the
            # insertion point, which is the order ``cumsum`` over the window gives.
            rank = np.cumsum(match, axis=1)
            match = match & (rank <= cap)
        counts = match.sum(axis=1)
        rows_q, rows_m = np.nonzero(match)
        hit_rows = self.rows[clipped[rows_q, rows_m]]
        splits = np.cumsum(counts)[:-1]
        groups = np.split(hit_rows, splits) if splits.size else [hit_rows]
        return [np.ascontiguousarray(g, dtype=np.int32) for g in groups]


def _fixed_probe(keys: Sequence[str], width: int, truncate_to: int) -> np.ndarray:
    """``(n, width)`` uint8 array of NUL-padded, truncated key bytes."""
    limit = max(1, min(int(width), int(truncate_to)))
    n = len(keys)
    out = np.zeros((n, width), dtype=np.uint8)
    for i, key in enumerate(keys):
        raw = key.strip().encode("utf-8", "replace")[:limit]
        out[i, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return out


def build_fixed_key_index(
    store: RecordStore,
    field: str,
    *,
    n_pool: int,
    config: ScaleConfig,
    guard: MemoryGuard | None = None,
) -> FixedKeyIndex:
    """Read the key column once, sort it with its row order, mark block starts."""
    started = time.perf_counter()
    values = store.column(field)
    sample_max = measure_id_width(values)
    width = max(4, int(config.exact_key_chars))
    if sample_max > width:
        # Round up so the sampled maximum still fits exactly.
        width = int(np.ceil(sample_max / PREFIX_BLOCK) * PREFIX_BLOCK)
        LOGGER.warning(
            "%s: sampled key length %d exceeds exact_key_chars=%d, widening to %d",
            field, sample_max, config.exact_key_chars, width,
        )
    keys = _as_fixed_bytes(values, width)
    del values
    order = np.argsort(keys, kind="stable").astype(np.int32)
    sorted_keys = keys[order]
    boundary = np.ones(int(n_pool), dtype=bool)
    if n_pool > 1:
        np.not_equal(sorted_keys[1:], sorted_keys[:-1], out=boundary[1:])
    starts = np.flatnonzero(boundary).astype(np.int64)
    if guard is not None:
        guard.check()
    index = FixedKeyIndex(
        field=field,
        width=width,
        keys=sorted_keys,
        rows=order,
        starts=starts,
        n_pool=int(n_pool),
        sample_max_chars=int(sample_max),
    )
    LOGGER.info("%s in %.1fs", index.describe(), time.perf_counter() - started)
    return index


# --------------------------------------------------------------------------
# country index
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CountryIndex:
    """``country -> sorted pool_rowid``."""

    values: np.ndarray
    rows: np.ndarray
    offsets: np.ndarray
    n_pool: int

    def nbytes(self) -> int:
        return int(self.values.nbytes + self.rows.nbytes + self.offsets.nbytes)

    def describe(self) -> str:
        return f"country: {self.values.size} codes, {format_bytes(self.nbytes())}"

    def lookup(self, country: str) -> np.ndarray:
        key = country.strip().lower()
        if not key:
            return np.zeros(0, dtype=np.int32)
        pos = int(np.searchsorted(self.values, key))
        if pos >= self.values.size or self.values[pos] != key:
            return np.zeros(0, dtype=np.int32)
        return self.rows[self.offsets[pos] : self.offsets[pos + 1]]

    def lookup_many(self, countries: Sequence[str]) -> list[np.ndarray]:
        return [self.lookup(c) for c in countries]


def build_country_index(
    store: RecordStore, *, n_pool: int, config: ScaleConfig, guard: MemoryGuard | None = None
) -> CountryIndex:
    started = time.perf_counter()
    values = store.column("country")
    keys = np.char.lower(np.char.strip(np.asarray(values, dtype=str)))
    uniq, inverse = np.unique(keys, return_inverse=True)
    order = np.argsort(inverse, kind="stable").astype(np.int32)
    counts = np.bincount(inverse, minlength=uniq.size).astype(np.int64)
    offsets = np.zeros(uniq.size + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    if guard is not None:
        guard.check()
    index = CountryIndex(values=uniq, rows=order, offsets=offsets, n_pool=int(n_pool))
    LOGGER.info("%s in %.1fs", index.describe(), time.perf_counter() - started)
    return index


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------
def _save(directory: Path, arrays: dict[str, np.ndarray], payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(directory / f"{name}.npy", array)
    return write_store_manifest(directory, payload)


def save_token_index(index: TokenIndex, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"term": index.vocabulary}), directory / "vocabulary.parquet")
    return _save(
        directory,
        {
            "idf": index.idf,
            "idf_sq": index.idf_sq,
            "df": index.df,
            "postings": index.postings,
            "offsets": index.offsets,
            "doc_norm": index.doc_norm,
            "doc_terms": index.doc_terms,
        },
        {
            "kind": "token",
            "field": index.field,
            "n_pool": index.n_pool,
            "n_terms": index.n_terms,
            "n_postings": index.n_postings,
            "nbytes": index.nbytes(),
        },
    )


def load_token_index(directory: str | Path, *, mmap: bool = True) -> TokenIndex:
    directory = Path(directory)
    meta = read_store_manifest(directory) or {}
    mode = "r" if mmap else None
    vocabulary = pq.read_table(directory / "vocabulary.parquet").column("term").combine_chunks()
    return TokenIndex(
        field=str(meta.get("field", "?")),
        vocabulary=vocabulary,
        idf=np.load(directory / "idf.npy", mmap_mode=mode),
        idf_sq=np.load(directory / "idf_sq.npy", mmap_mode=mode),
        df=np.load(directory / "df.npy", mmap_mode=mode),
        postings=np.load(directory / "postings.npy", mmap_mode=mode),
        offsets=np.load(directory / "offsets.npy", mmap_mode=mode),
        doc_norm=np.load(directory / "doc_norm.npy", mmap_mode=mode),
        doc_terms=np.load(directory / "doc_terms.npy", mmap_mode=mode),
        n_pool=int(meta.get("n_pool", 0)),
    )


def save_fixed_key_index(index: FixedKeyIndex, directory: str | Path) -> Path:
    return _save(
        Path(directory),
        {"keys": index.keys, "rows": index.rows, "starts": index.starts},
        {
            "kind": "fixed_key",
            "field": index.field,
            "width": index.width,
            "n_pool": index.n_pool,
            "n_blocks": index.n_blocks,
            "sample_max_chars": index.sample_max_chars,
            "nbytes": index.nbytes(),
        },
    )


def load_fixed_key_index(directory: str | Path, *, mmap: bool = True) -> FixedKeyIndex:
    directory = Path(directory)
    meta = read_store_manifest(directory) or {}
    mode = "r" if mmap else None
    return FixedKeyIndex(
        field=str(meta.get("field", "?")),
        width=int(meta.get("width", 0)),
        keys=np.load(directory / "keys.npy", mmap_mode=mode),
        rows=np.load(directory / "rows.npy", mmap_mode=mode),
        starts=np.load(directory / "starts.npy", mmap_mode=mode),
        n_pool=int(meta.get("n_pool", 0)),
        sample_max_chars=int(meta.get("sample_max_chars", 0)),
    )


def save_country_index(index: CountryIndex, directory: str | Path) -> Path:
    return _save(
        Path(directory),
        {"values": index.values, "rows": index.rows, "offsets": index.offsets},
        {"kind": "country", "n_pool": index.n_pool, "n_codes": int(index.values.size), "nbytes": index.nbytes()},
    )


def load_country_index(directory: str | Path, *, mmap: bool = True) -> CountryIndex:
    directory = Path(directory)
    meta = read_store_manifest(directory) or {}
    mode = "r" if mmap else None
    return CountryIndex(
        values=np.load(directory / "values.npy", mmap_mode=mode),
        rows=np.load(directory / "rows.npy", mmap_mode=mode),
        offsets=np.load(directory / "offsets.npy", mmap_mode=mode),
        n_pool=int(meta.get("n_pool", 0)),
    )


# --------------------------------------------------------------------------
# bundle
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class IndexBundle:
    """The five indexes a split's candidate generator needs.

    ``pool_ids`` is optional: when present it maps ``entity_id <-> pool rowid``,
    which is what turns a scored candidate list back into a comparable id and
    what makes blocking recall measurable against the ground truth.
    """

    name: TokenIndex
    addr: TokenIndex
    name_norm: FixedKeyIndex
    name_core: FixedKeyIndex
    country: CountryIndex
    n_pool: int
    s2_rows: int
    pool_ids: "IdResolver | None" = None

    def describe(self) -> str:
        lines = [
            f"pool: {self.n_pool:,} rows "
            f"({self.s2_rows:,} source2 + {self.n_pool - self.s2_rows:,} source3)",
            self.name.describe(),
            self.addr.describe(),
            self.name_norm.describe(),
            self.name_core.describe(),
            self.country.describe(),
        ]
        if self.pool_ids is not None:
            lines.append(f"pool ids: {self.pool_ids.describe()}")
        return "\n".join(lines)

    def nbytes(self) -> int:
        return int(
            self.name.nbytes()
            + self.addr.nbytes()
            + self.name_norm.nbytes()
            + self.name_core.nbytes()
            + self.country.nbytes()
        )

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        save_token_index(self.name, directory / "name_tokens")
        save_token_index(self.addr, directory / "addr_tokens")
        save_fixed_key_index(self.name_norm, directory / "name_norm_keys")
        save_fixed_key_index(self.name_core, directory / "name_core_keys")
        save_country_index(self.country, directory / "country")
        if self.pool_ids is not None:
            self.pool_ids.save(directory / "pool_ids")
        return write_store_manifest(
            directory,
            {
                "kind": "bundle",
                "n_pool": self.n_pool,
                "s2_rows": self.s2_rows,
                "n_terms": self.name.n_terms,
                "n_postings": self.name.n_postings,
                "n_addr_terms": self.addr.n_terms,
                "n_addr_postings": self.addr.n_postings,
                "has_pool_ids": self.pool_ids is not None,
                "nbytes": self.nbytes(),
            },
        )

    @classmethod
    def load(cls, directory: str | Path, *, mmap: bool = True) -> "IndexBundle":
        directory = Path(directory)
        meta = read_store_manifest(directory) or {}
        ids_dir = directory / "pool_ids"
        pool_ids = IdResolver.load(ids_dir) if (ids_dir / "ids_manifest.json").exists() else None
        if meta.get("has_pool_ids") and pool_ids is None:
            LOGGER.warning("bundle manifest claims pool ids but %s is missing", ids_dir)
        return cls(
            name=load_token_index(directory / "name_tokens", mmap=mmap),
            addr=load_token_index(directory / "addr_tokens", mmap=mmap),
            name_norm=load_fixed_key_index(directory / "name_norm_keys", mmap=mmap),
            name_core=load_fixed_key_index(directory / "name_core_keys", mmap=mmap),
            country=load_country_index(directory / "country", mmap=mmap),
            n_pool=int(meta.get("n_pool", 0)),
            s2_rows=int(meta.get("s2_rows", 0)),
            pool_ids=pool_ids,
        )


def build_index_bundle(
    store: PoolStore,
    *,
    n_pool: int,
    s2_rows: int,
    config: ScaleConfig,
    guard: MemoryGuard | None = None,
    with_pool_ids: bool = True,
) -> IndexBundle:
    """Build every index the candidate generator reads.

    The name token index excludes legal-form tokens and terms above ``max_df``;
    the exact and sorted-neighbourhood indexes are kept on both the normalised
    and the legal-form-stripped name, because the first is the strongest possible
    signal and the second is what survives a legal-form disagreement.  The address
    index is separate because it needs a different stop list and a different
    vocabulary: querying name postings with address tokens would match nothing.

    ``store`` is a :class:`PoolStore`: Source 2 then Source 3, so one pass covers
    both blocking sources and pool rowids line up with the postings.
    """
    # The resolver is built first because ``IndexBundle`` is frozen, so the
    # pool-id map has to be in hand at construction rather than assigned after.
    pool_ids = (
        IdResolver.build_from_shards(store.iter_id_shards, total=int(n_pool), guard=guard)
        if with_pool_ids
        else None
    )
    bundle = IndexBundle(
        name=build_token_index(
            store,
            "name_core",
            n_pool=n_pool,
            config=config,
            stop_tokens=config.name_stop_tokens,
            guard=guard,
        ),
        addr=build_token_index(
            store,
            "addr_core",
            n_pool=n_pool,
            config=config,
            stop_tokens=config.address_stop_tokens,
            guard=guard,
        ),
        name_norm=build_fixed_key_index(store, "name_norm", n_pool=n_pool, config=config, guard=guard),
        name_core=build_fixed_key_index(store, "name_core", n_pool=n_pool, config=config, guard=guard),
        country=build_country_index(store, n_pool=n_pool, config=config, guard=guard),
        n_pool=int(n_pool),
        s2_rows=int(s2_rows),
        pool_ids=pool_ids,
    )
    LOGGER.info("index bundle built: %s", format_bytes(bundle.nbytes()))
    return bundle
