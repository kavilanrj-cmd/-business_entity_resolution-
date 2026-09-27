"""Batched candidate generation over the disk-backed indexes.

One Source 1 row at a time would mean 2.2M separate trips into the index arrays;
the whole pool at once would mean 2.3e13 pairs.  This module works in batches of
``config.batch_rows`` queries and, per batch, produces the union of six retrieval
strategies, deduplicated and capped, as five flat NumPy columns that Parquet can
write without a single Python object per pair.

Strategies and why each is here
-------------------------------
``exact_normalized`` / ``exact_core``
    Binary search for a byte-identical normalised name.  Near-zero cost, and the
    strongest possible single signal.
``name_token``
    Inverted index on legal-form-stripped name tokens, IDF-weighted cosine.  This
    carries the bulk of the recall and is the only name strategy that reaches
    across word order ("Star Imports" / "Imports Star").
``sorted_neighbourhood``
    Window around a name's insertion point in sort order, then a character-similar
    floor.  The only strategy that matches "Infosys" with "Info Systems", because
    those share a prefix but no whole token.
``address_token``
    A separate inverted index over address components.  Disambiguates chains of
    same-named businesses at different addresses.
``country``
    A same-country prior, kept as its own strategy bit so a later stage can treat
    it as a tie-breaker rather than as evidence of identity.

Every strategy sets a bit in the candidate's ``mask`` and contributes to
``rule_score``; ``content_score`` stays reserved for character/token similarity so
the two signals stay separable downstream.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from .config import STRATEGY_BITS, STRATEGY_ORDER, ScaleConfig
from .indexes import IndexBundle, encode_queries
from .progress import MemoryGuard, Progress, format_bytes
from .store import PoolStore, RecordStore

LOGGER = logging.getLogger(__name__)

#: Contribution of each strategy to the rule score, so an exact name match always
#: outranks a merely similar one no matter how high that one's cosine is.
STRATEGY_WEIGHT: dict[str, float] = {
    "exact_normalized": 100.0,
    "exact_core": 90.0,
    "name_token": 10.0,
    "sorted_neighbourhood": 6.0,
    "address_token": 4.0,
    "country_token": 1.0,
}

#: Queries per ``rapidfuzz`` block is not a knob: rapidfuzz has no pairwise
#: matrix primitive, so ratios are computed per blocked candidate directly.  The
#: only bound that matters is ``2 * sn_window + 1`` comparisons per query.
RATIO_BLOCK = 0

#: Strategies whose hits are exact and near-certain, so a quota should never
#: ration them.  They are also the rarest, which is why they survive a low cap
#: even without this -- but a quota configuration must not be able to starve them.
EXACT_STRATEGIES: tuple[str, ...] = ("exact_normalized", "exact_core")

#: The Source 1 columns every strategy needs.
QUERY_COLUMNS = ("rowid", "entity_id", "name_norm", "name_core", "addr_core", "country")

CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("s1_rowid", pa.int32()),
        pa.field("pool_rowid", pa.int32()),
        pa.field("mask", pa.int32()),
        pa.field("rule_score", pa.float32()),
        pa.field("content_score", pa.float32()),
    ]
)


@dataclass
class BatchResult:
    """Flattened candidates for one batch of Source 1 rows.

    ``query`` is the batch-local slot and is what the cap and the statistics group
    by; ``s1_rowid`` is the same value offset into the Source 1 table, which is
    what gets written.
    """

    query: np.ndarray
    pool_rowid: np.ndarray
    mask: np.ndarray
    rule_score: np.ndarray
    content_score: np.ndarray
    s1_rowid: np.ndarray
    n_queries: int = 0

    def __len__(self) -> int:
        return int(self.query.size)

    @property
    def per_query(self) -> np.ndarray:
        """Candidate count for each query of the batch, in batch order."""
        return np.bincount(self.query, minlength=self.n_queries)

    def as_table(self) -> pa.Table:
        return pa.table(
            {
                "s1_rowid": pa.array(self.s1_rowid, type=pa.int32()),
                "pool_rowid": pa.array(self.pool_rowid, type=pa.int32()),
                "mask": pa.array(self.mask, type=pa.int32()),
                "rule_score": pa.array(self.rule_score, type=pa.float32()),
                "content_score": pa.array(self.content_score, type=pa.float32()),
            },
            schema=CANDIDATE_SCHEMA,
        )

    def counts_by_strategy(self) -> dict[str, int]:
        return {
            name: int(np.count_nonzero(self.mask & STRATEGY_BITS[name]))
            for name in STRATEGY_ORDER
        }

    def named_strategies(self, i: int) -> list[str]:
        return [name for name in STRATEGY_ORDER if self.mask[i] & STRATEGY_BITS[name]]


def empty_batch(n_queries: int = 0) -> BatchResult:
    return BatchResult(
        query=np.zeros(0, dtype=np.int32),
        pool_rowid=np.zeros(0, dtype=np.int32),
        mask=np.zeros(0, dtype=np.int32),
        rule_score=np.zeros(0, dtype=np.float32),
        content_score=np.zeros(0, dtype=np.float32),
        s1_rowid=np.zeros(0, dtype=np.int32),
        n_queries=n_queries,
    )


class CandidateGenerator:
    """Union of the retrieval strategies, batched and memory-bounded."""

    def __init__(
        self,
        s1: RecordStore,
        pool: PoolStore,
        indexes: IndexBundle,
        config: ScaleConfig,
        *,
        guard: MemoryGuard | None = None,
    ) -> None:
        if indexes.n_pool != pool.n_pool:
            raise ValueError(
                f"index pool size {indexes.n_pool} != store pool size {pool.n_pool}"
            )
        self.s1 = s1
        self.pool = pool
        self.indexes = indexes
        self.config = config
        self.guard = guard

    # ------------------------------------------------------------------
    def batches(self, start: int = 0, stop: int | None = None) -> Iterator[tuple[int, BatchResult]]:
        """Yield ``(s1_start_rowid, candidates)`` for consecutive S1 batches."""
        cfg = self.config
        stop = len(self.s1) if stop is None else min(int(stop), len(self.s1))
        for rowid0, frame in self.s1.read_batches(cfg.batch_rows, list(QUERY_COLUMNS)):
            if rowid0 >= stop:
                break
            if rowid0 + len(frame) > stop:
                frame = frame.iloc[: stop - rowid0].reset_index(drop=True)
            yield rowid0, self.generate(rowid0, frame)

    def generate(self, rowid0: int, frame) -> BatchResult:
        """Candidates for one in-memory batch of Source 1 rows."""
        cfg = self.config
        n = len(frame)
        if n == 0:
            return empty_batch()
        names_norm = _strings(frame, "name_norm")
        names_core = _strings(frame, "name_core")
        addrs_core = _strings(frame, "addr_core")
        countries = _strings(frame, "country")

        queries: list[np.ndarray] = []
        pools: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        contents: list[np.ndarray] = []

        def add(slot: np.ndarray, pool_rowid: np.ndarray, strategy: str, content: np.ndarray) -> None:
            if slot.size == 0:
                return
            queries.append(slot)
            pools.append(pool_rowid)
            masks.append(np.full(slot.size, STRATEGY_BITS[strategy], dtype=np.int32))
            contents.append(content.astype(np.float32, copy=False))

        # -- inverted-index strategies ---------------------------------
        q, row, score = self._token_hits(
            self.indexes.name, names_core, cfg.name_stop_tokens,
            cfg.posting_budget_name, cfg.max_terms_per_query,
        )
        add(q, row, "name_token", score)
        q, row, score = self._token_hits(
            self.indexes.addr, addrs_core, cfg.address_stop_tokens,
            cfg.posting_budget_address, cfg.max_terms_per_query,
        )
        add(q, row, "address_token", score)

        # -- key strategies --------------------------------------------
        for values, strategy, key_index in (
            (names_norm, "exact_normalized", self.indexes.name_norm),
            (names_core, "exact_core", self.indexes.name_core),
        ):
            slot, row = _exact_hits(values, key_index, cfg.exact_block_cap)
            add(slot, row, strategy, np.zeros(slot.size, dtype=np.float32))

        slot, row = self._neighbourhood_hits(names_core)
        add(slot, row, "sorted_neighbourhood", np.zeros(slot.size, dtype=np.float32))

        slot, row = _country_hits(countries, self.indexes.country, cfg.country_top_k)
        add(slot, row, "country_token", np.zeros(slot.size, dtype=np.float32))

        if not queries:
            return empty_batch(n)
        return self._merge(
            rowid0,
            n,
            np.concatenate(queries),
            np.concatenate(pools),
            np.concatenate(masks),
            np.concatenate(contents),
        )

    # ------------------------------------------------------------------
    def _token_hits(
        self, index, values: np.ndarray, stop: frozenset[str], budget: int, max_terms: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        term_ids, tf = encode_queries(
            index, values, min_length=self.config.token_min_length, stop_tokens=stop
        )
        return index.gather(term_ids, tf, budget=budget, max_terms=max_terms)

    def _neighbourhood_hits(self, names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sorted-neighbourhood candidates filtered by a character-similarity floor.

        Sort order alone returns every neighbour sharing the prefix, which for
        "STAR INTERNATIONAL TRADING" is a large block of unrelated firms.  A
        ``rapidfuzz`` ratio floor removes them, and it is evaluated *only* on the
        blocked window -- at most ``2 * sn_window + 1`` rows per query, never on the
        pool.  All pool names are fetched in one ``read_where`` so a batch costs
        one sequential Parquet read, not one per candidate.

        Note on cost: rapidfuzz exposes no pairwise-matrix primitive (``cdist``
        compares whole lists and ``cpdist`` reduces to the best match), so the
        ratios come from a Python-level loop over the blocked pairs.  At the
        default window that is ~100 comparisons per query, which is the right
        trade against any whole-pool approach; if it ever dominates, the
        replacement is a small native kernel, not a wider window.
        """
        cfg = self.config
        keys = names.tolist()
        groups = self.indexes.name_core.neighbourhood(
            keys,
            prefix_chars=cfg.sn_prefix_chars,
            min_chars=cfg.sn_min_key_chars,
            window=cfg.sn_window,
            cap=cfg.exact_block_cap,
        )
        live = [(slot, group) for slot, group in enumerate(groups) if group.size]
        if not live:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

        wanted = np.unique(np.concatenate([group for _, group in live]))
        pool_names = self.pool.read_where(wanted, ["name_core"])["name_core"].to_numpy(dtype=object)
        floor = float(cfg.sn_min_ratio) * 100.0
        ratio = fuzz.ratio

        slots: list[np.ndarray] = []
        rows: list[np.ndarray] = []
        for slot, group in live:
            names_here = pool_names[np.searchsorted(wanted, group)]
            if floor <= 0:
                slots.append(np.full(group.size, slot, dtype=np.int32))
                rows.append(group)
                continue
            scores = np.fromiter(
                (ratio(keys[slot], choice) for choice in names_here.tolist()),
                dtype=np.int32,
                count=group.size,
            )
            keep = scores >= floor
            if keep.any():
                slots.append(np.full(int(keep.sum()), slot, dtype=np.int32))
                rows.append(group[keep])
        if not rows:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
        return np.concatenate(slots), np.concatenate(rows)

    # ------------------------------------------------------------------
    def _merge(
        self,
        rowid0: int,
        n_queries: int,
        query: np.ndarray,
        pool_rowid: np.ndarray,
        mask: np.ndarray,
        content: np.ndarray,
    ) -> BatchResult:
        """Deduplicate, score, cap, and return five flat columns.

        Deduplication is by ``(query, pool_rowid)`` with a bitwise-OR of the
        strategy masks and a max of the content scores, so a pair found by three
        strategies is kept once and remembers all three.
        """
        n_pool = self.indexes.n_pool
        cfg = self.config
        pair = query.astype(np.int64) * np.int64(n_pool) + pool_rowid.astype(np.int64)
        order = np.argsort(pair, kind="stable")
        pair_sorted = pair[order]
        new = np.empty(pair_sorted.size, dtype=bool)
        new[0] = True
        np.not_equal(pair_sorted[1:], pair_sorted[:-1], out=new[1:])
        starts = np.flatnonzero(new)
        unique_pair = pair_sorted[starts]

        merged_mask = np.bitwise_or.reduceat(mask[order], starts)
        merged_content = np.maximum.reduceat(content[order], starts)
        q_index = (unique_pair // n_pool).astype(np.int32)
        p_index = (unique_pair % n_pool).astype(np.int32)

        rule = np.zeros(unique_pair.size, dtype=np.float32)
        for name in STRATEGY_ORDER:
            hit = (merged_mask & STRATEGY_BITS[name]) != 0
            if hit.any():
                rule[hit] += np.float32(STRATEGY_WEIGHT[name])

        # Best-first within each query: rule score, then token cosine, then rowid
        # so the output is byte-identical across runs.
        rank = np.lexsort((p_index, -merged_content, -rule, q_index))
        q_index, p_index = q_index[rank], p_index[rank]
        merged_mask, merged_content, rule = merged_mask[rank], merged_content[rank], rule[rank]

        keep = _cap_per_query(
            q_index, cfg.max_candidates_per_s1, cfg.min_candidates_per_s1, n_queries
        )
        q_index, p_index = q_index[keep], p_index[keep]
        return BatchResult(
            query=q_index,
            pool_rowid=p_index,
            mask=merged_mask[keep],
            rule_score=rule[keep],
            content_score=merged_content[keep],
            s1_rowid=q_index + np.int32(rowid0),
            n_queries=n_queries,
        )


def _exact_hits(
    values: np.ndarray, index, cap: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(query slot, pool rowid)`` for byte-identical key hits.

    The per-query ``searchsorted`` is a C call and cheap; what must be avoided is
    materialising a Python list per hit, so hits are preallocated to ``cap``.
    """
    slots: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    for slot, key in enumerate(values.tolist()):
        if not key:
            continue
        found = index.lookup(key)
        if found.size == 0:
            continue
        if cap and found.size > cap:
            found = found[:cap]
        rows.append(found)
        slots.append(np.full(found.size, slot, dtype=np.int32))
    if not rows:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    return np.concatenate(slots), np.concatenate(rows)


def _country_hits(countries: np.ndarray, index, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    slots: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    for slot, country in enumerate(countries.tolist()):
        if not country:
            continue
        found = index.lookup(country)
        if found.size == 0:
            continue
        if top_k and found.size > top_k:
            found = found[:top_k]
        rows.append(found)
        slots.append(np.full(found.size, slot, dtype=np.int32))
    if not rows:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    return np.concatenate(slots), np.concatenate(rows)


def _cap_per_query(query: np.ndarray, cap: int, floor: int, n_queries: int) -> np.ndarray:
    """Boolean keep-mask taking the first ``cap`` candidates of each query.

    ``floor`` guarantees a minimum so a hard entity is never left with nothing;
    because candidates arrive sorted best-first, "the first ``floor``" is also the
    best ``floor``.  Implemented as a running rank rather than a per-query Python
    loop, which is what keeps 2.2M queries tractable.
    """
    total = int(query.size)
    if total == 0 or cap <= 0:
        return np.ones(total, dtype=bool)
    limit = max(int(cap), int(floor))
    return _rank_within_group(query) < limit


def _rank_within_group(query: np.ndarray, subset: np.ndarray | None = None) -> np.ndarray:
    """0-based position of each row within its query group; ``query`` is sorted.

    With ``subset`` given, only rows where it is true are counted, so the rank is
    "position among the rows that survived so far" rather than "position in the
    batch".  That is what lets a quota union be trimmed to the cap by rank
    without the trim happening before the union is formed.
    """
    total = int(query.size)
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    if subset is None:
        starts = np.flatnonzero(np.diff(query)) + 1
        starts = np.concatenate(([0], starts))
        sizes = np.diff(np.append(starts, total))
        return np.arange(total, dtype=np.int64) - np.repeat(starts, sizes)

    # Count the surviving rows before each row, per query.  A cumulative count
    # over the whole array is wrong across query boundaries, so subtract the
    # count that had accumulated by the time each query's first row was reached.
    surviving = subset.astype(np.int64)
    cumulative = np.cumsum(surviving) - surviving
    starts = np.flatnonzero(np.diff(query)) + 1
    starts = np.concatenate(([0], starts))
    sizes = np.diff(np.append(starts, total))
    return cumulative - np.repeat(cumulative[starts], sizes)


def select_with_quotas(
    query: np.ndarray,
    mask: np.ndarray,
    *,
    n_queries: int,
    cap: int,
    quotas: dict[str, int] | None = None,
    always: tuple[str, ...] = EXACT_STRATEGIES,
    exclude_only: tuple[str, ...] = (),
    cap_mode: str = "rank",
) -> np.ndarray:
    """Keep-mask for a per-strategy quota configuration.

    A single global rank cannot express "the name signal gets 50 slots and the
    address signal 30".  ``rule_score`` puts every ``name_token`` hit above every
    ``address_token`` hit regardless of strength, and then one cosine axis decides
    within each, so a strong address hit competes with a strong name hit as if
    they were the same kind of evidence.  Quotas fix the number of slots per
    strategy first -- so no strategy can be crowded out by a larger one -- and
    only then apply the total cap by rank.

    ``always`` names strategies that are never limited.  The exact ones belong
    there: they are few and near-certain, and they would otherwise be squeezed
    out of their own strategy's quota by look-alikes.

    ``exclude_only`` drops candidates whose *only* evidence came from the named
    strategies.  This is deliberately narrower than banning a strategy: a
    candidate that country matched and the name index also matched keeps the
    name evidence, it just stops being a country-only filler.  It matters
    because under a global rank, filler competes for the same cap slots as
    signal, and a strategy that never finds a true match should not hold any.

    ``cap_mode`` decides when the total cap is applied, and the difference
    decides whether quotas can do anything at all:

    ``"rank"``
        Take the global top ``cap`` first, then intersect with the quota union.
        Because ``rule_score`` puts every ``name_token`` hit above every
        ``address_token`` hit, the cap is already full of name candidates before
        any address quota is consulted, so the quotas are inert.
    ``"union"``
        Form the quota union first, then trim *it* to ``cap`` by rank.  Each
        strategy's quota is a floor that survives the trim, which is the only
        way a per-strategy budget can redirect slots.

    ``query`` must already be sorted best-first within each query, which is what
    :meth:`CandidateGenerator._merge` produces, so "the first ``quota``" means
    "the highest ranked ``quota``".  Everything is a vectorised running rank --
    no per-query Python loop -- so this stays usable at 2.2M queries.
    """
    total = int(query.size)
    if total == 0:
        return np.zeros(0, dtype=bool)
    if not quotas and not exclude_only:
        # No quotas is the plain cap: pure rank truncation, every strategy
        # competing on one axis.  The ``always`` list must not apply here or a
        # cap would silently mean "only exact hits".
        if cap <= 0:
            return np.ones(total, dtype=bool)
        return _rank_within_group(query) < cap

    # Exclusion is a filter applied first, and it composes with quotas: the two
    # are independent levers (drop useless strategies, then reallocate the
    # surviving slots).  Returning early on ``exclude_only`` would silently
    # discard the quotas, which is the kind of bug that makes a configuration
    # look like it "does nothing".
    base = np.ones(total, dtype=bool)
    if exclude_only:
        bits = 0
        for strategy in exclude_only:
            bits |= int(STRATEGY_BITS.get(strategy, 0))
        if bits:
            is_only = (mask & bits) != 0
            for strategy in STRATEGY_ORDER:
                if strategy in exclude_only:
                    continue
                is_only &= (mask & int(STRATEGY_BITS[strategy])) == 0
            base = ~is_only
            if not base.any():
                # Excluding everything would empty the batch; keep it all.
                return np.ones(total, dtype=bool)

    if not quotas:
        if cap <= 0:
            return base
        # Trimming by *survivor* rank is what makes exclusion worth doing: the
        # budget the excluded rows occupied is handed to the next best
        # candidates.  Trimming by global rank instead would just delete the
        # excluded rows and never backfill, so the cap would silently shrink.
        if cap_mode == "union" or not base.all():
            return base & (_rank_within_group(query, base) < cap)
        return _rank_within_group(query) < cap

    allowed = np.zeros(total, dtype=bool)
    for strategy in always:
        bit = STRATEGY_BITS.get(strategy)
        if bit is not None:
            allowed |= (mask & bit) != 0
    for strategy, quota in quotas.items():
        bit = STRATEGY_BITS.get(strategy)
        if bit is None or quota <= 0:
            continue
        selected = np.flatnonzero((mask & bit) != 0)
        if selected.size == 0:
            continue
        # Rank among this strategy's own candidates, per query, in global rank
        # order -- so the first ``quota`` are the best ``quota``.
        within = _rank_within_group(query[selected])
        allowed[selected[within < quota]] = True
    if not allowed.any():
        # No quota matched anything (e.g. every strategy was over-quota'd away);
        # fall back to plain rank so a query is never emptied by a quota.
        allowed[:] = True
    allowed &= base
    if cap > 0:
        if cap_mode == "union":
            allowed &= _rank_within_group(query, allowed) < cap
        else:
            allowed &= _rank_within_group(query) < cap
    return allowed


def _strings(frame, name: str) -> np.ndarray:
    if name not in frame.columns:
        return np.zeros(len(frame), dtype=object)
    return frame[name].to_numpy(dtype=object)


# ----------------------------------------------------------------------
# output
# ----------------------------------------------------------------------
class CandidateWriter:
    """Rotating Parquet writer for candidate shards."""

    def __init__(self, directory: str | Path, config: ScaleConfig) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        for stale in self.directory.glob("part-*.parquet"):
            stale.unlink()
        self.config = config
        self.shards: list[str] = []
        self.rows = 0
        self._buffer = empty_batch()

    def write(self, batch: BatchResult) -> None:
        self._buffer = concat_batches([self._buffer, batch]) if len(self._buffer) else batch
        while len(self._buffer) >= self.config.candidate_shard_rows:
            self._spill(self.config.candidate_shard_rows)

    def close(self) -> None:
        if len(self._buffer):
            self._spill(len(self._buffer))
        self._buffer = empty_batch()

    def _spill(self, rows: int) -> None:
        table = self._buffer.as_table()
        if rows < table.num_rows:
            head = table.slice(0, rows)
            self._buffer = batch_from_table(table.slice(rows))
        else:
            head = table
            self._buffer = empty_batch()
        path = self.directory / f"part-{len(self.shards):05d}.parquet"
        pq.write_table(
            head, path, compression=self.config.parquet_compression, row_group_size=262_144
        )
        self.shards.append(path.name)
        self.rows += head.num_rows

    def describe(self) -> str:
        size = sum((self.directory / s).stat().st_size for s in self.shards)
        return f"{self.rows:,} candidates in {len(self.shards)} shards ({format_bytes(size)})"


def concat_batches(batches: Sequence[BatchResult]) -> BatchResult:
    live = [b for b in batches if len(b)]
    if not live:
        return empty_batch()
    if len(live) == 1:
        return live[0]
    return BatchResult(
        query=np.concatenate([b.query for b in live]),
        pool_rowid=np.concatenate([b.pool_rowid for b in live]),
        mask=np.concatenate([b.mask for b in live]),
        rule_score=np.concatenate([b.rule_score for b in live]),
        content_score=np.concatenate([b.content_score for b in live]),
        s1_rowid=np.concatenate([b.s1_rowid for b in live]),
        n_queries=sum(b.n_queries for b in live),
    )


def batch_from_table(table: pa.Table) -> BatchResult:
    """Rebuild a batch from a written shard, for the leftover buffer only.

    ``n_queries`` is left at 0: a partially written buffer is never grouped or
    counted, only carried forward into the next shard.
    """
    if table.num_rows == 0:
        return empty_batch()

    def col(name: str) -> np.ndarray:
        return table.column(name).to_numpy(zero_copy_only=False)

    s1_rowid = col("s1_rowid")
    return BatchResult(
        query=(s1_rowid - int(s1_rowid.min())).astype(np.int32),
        pool_rowid=col("pool_rowid"),
        mask=col("mask"),
        rule_score=col("rule_score"),
        content_score=col("content_score"),
        s1_rowid=s1_rowid,
    )


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
@dataclass
class CandidateStats:
    """Running totals for a candidate-generation run."""

    queries: int = 0
    candidates: int = 0
    empty_queries: int = 0
    per_query_sum: int = 0
    per_query_max: int = 0
    seconds: float = 0.0
    peak_bytes: int = 0
    by_strategy: dict[str, int] = field(default_factory=dict)
    per_query_histogram: dict[str, int] = field(default_factory=dict)

    def update(self, batch: BatchResult, seconds: float, peak: int) -> None:
        counts = batch.per_query
        self.queries += batch.n_queries
        self.candidates += len(batch)
        self.empty_queries += int((counts == 0).sum())
        self.per_query_sum += int(counts.sum())
        self.per_query_max = max(self.per_query_max, int(counts.max()) if counts.size else 0)
        self.seconds += seconds
        self.peak_bytes = max(self.peak_bytes, peak)
        for name, count in batch.counts_by_strategy().items():
            self.by_strategy[name] = self.by_strategy.get(name, 0) + count
        for count in counts.tolist():
            self.per_query_histogram[count] = self.per_query_histogram.get(count, 0) + 1

    def as_dict(self) -> dict:
        return {
            "queries": self.queries,
            "candidates": self.candidates,
            "candidates_per_query_avg": round(self.per_query_sum / self.queries, 2)
            if self.queries
            else 0.0,
            "candidates_per_query_max": self.per_query_max,
            "empty_queries": self.empty_queries,
            "empty_query_pct": round(100.0 * self.empty_queries / self.queries, 3)
            if self.queries
            else 0.0,
            "seconds": round(self.seconds, 2),
            "queries_per_second": round(self.queries / self.seconds, 1) if self.seconds else 0.0,
            "candidates_per_second": int(self.candidates / self.seconds)
            if self.seconds
            else 0,
            "peak_rss_bytes": self.peak_bytes,
            "peak_rss": format_bytes(self.peak_bytes),
            "by_strategy": self.by_strategy,
        }

    def __str__(self) -> str:
        return "\n".join(f"  {key}: {value}" for key, value in self.as_dict().items())


def generate_candidates(
    s1: RecordStore,
    pool: PoolStore,
    indexes: IndexBundle,
    config: ScaleConfig,
    *,
    start: int = 0,
    stop: int | None = None,
    guard: MemoryGuard | None = None,
    writer: CandidateWriter | None = None,
    on_batch: Callable[[int, "BatchResult"], None] | None = None,
    label: str = "candidates",
) -> CandidateStats:
    """Generate candidates for a Source 1 rowid range, streaming and reporting.

    ``on_batch`` receives ``(s1_start_rowid, batch)`` for every batch.  Recall
    measurement uses it so the verified-pair check sees exactly the candidates
    that were written, without a second generation pass.
    """
    generator = CandidateGenerator(s1, pool, indexes, config, guard=guard)
    stats = CandidateStats()
    stop = len(s1) if stop is None else min(int(stop), len(s1))
    progress = Progress(
        f"{label} [{start}:{stop}]",
        total=stop - start,
        interval_s=config.progress_interval_s,
        guard=guard,
    )
    with progress:
        began = time.perf_counter()
        for rowid0, batch in generator.batches(start=start, stop=stop):
            # The batch is built during ``next()``, so the gap between yields is
            # the time this batch actually cost.  ``observe()`` rather than
            # ``peak()``: a fast batch may never trip a guard check, and a
            # reported peak of zero would be worse than useless.
            stats.update(batch, time.perf_counter() - began, guard.observe() if guard else 0)
            began = time.perf_counter()
            if on_batch is not None:
                on_batch(rowid0, batch)
            if writer is not None:
                writer.write(batch)
            progress.add(batch.n_queries, candidates=stats.candidates)
    return stats
