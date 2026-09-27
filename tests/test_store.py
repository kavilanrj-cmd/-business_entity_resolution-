"""Tests for the sharded store: correctness of random-access reads.

These exist because of a bug that only appeared on the real 10.3 M-row train
pool.  ``RecordStore.read_where`` grouped its rowids by shard and then looked up
the shard with ``sorted_shards[group[0]]`` -- but ``group`` holds indices into
the *original* arrays, not into the sorted one.  For any gather that spanned two
or more shards, the wrong shard was opened, and the resulting local offset could
go negative or past the end of the shard.  The 4,513-row fixture never triggered
it because its single shard made the group/position distinction invisible.

So the store tests deliberately build a store with several *uneven* shards and
compare every random-access read against the sequential path, which is
independent of the shard-grouping logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.scale.config import ScaleConfig
from src.scale.store import PoolStore, RecordStore, build_table

# Small enough to stay fast, large enough that shard_rows divides unevenly and
# the final shard is a partial one -- the real store's last shard always is.
N_ROWS = 250
SHARD_ROWS = 60
COLUMNS = ("rowid", "entity_id", "name_norm", "name_core", "addr_norm", "addr_core", "country")


def _write_tsv(path, n: int, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "entity_id": [f"{prefix}-{i:06d}" for i in range(n)],
            "business_name": [f"firm number {i} holdings" for i in range(n)],
            "business_address": [f"{i} Example Street, Testville" for i in range(n)],
            "country": ["US" if i % 2 else "India" for i in range(n)],
        }
    )
    frame.to_csv(path, sep="\t", index=False)


def _config() -> ScaleConfig:
    config = ScaleConfig()
    config.shard_rows = SHARD_ROWS
    config.read_chunk_rows = 37  # deliberately not a divisor of SHARD_ROWS
    return config


@pytest.fixture()
def store(tmp_path):
    config = _config()
    counts = {"s1": N_ROWS, "s2": N_ROWS, "s3": N_ROWS}
    for source, n in counts.items():
        _write_tsv(tmp_path / f"{source}.tsv", n, source.upper())
        build_table(tmp_path / f"{source}.tsv", tmp_path / source, source, config)
    return tmp_path, config


def test_shards_are_uneven_and_exact(store):
    tmp_path, config = store
    s2 = RecordStore(tmp_path, "s2", config)
    assert len(s2) == N_ROWS
    assert len(s2.shards) == -(-N_ROWS // SHARD_ROWS)  # ceiling division
    # The last shard must be a partial one, otherwise the offset arithmetic that
    # broke is never exercised.
    counts = [pq.ParquetFile(path).metadata.num_rows for path in s2.shards]
    assert counts[-1] == N_ROWS % SHARD_ROWS
    assert s2._offsets.tolist() == [0, *np.cumsum(counts).tolist()]


@pytest.mark.parametrize("source", ["s1", "s2", "s3"])
def test_read_where_matches_read_range(store, source):
    tmp_path, config = store
    record = RecordStore(tmp_path, source, config)
    sequential = record.read_range(0, len(record), ["name_core", "country"])

    rng = np.random.default_rng(0)
    rowids = rng.choice(len(record), size=min(200, len(record)), replace=False).astype(np.int64)
    gathered = record.read_where(rowids, ["name_core", "country"])
    assert gathered["name_core"].tolist() == sequential["name_core"].to_numpy()[rowids].tolist()
    assert gathered["country"].tolist() == sequential["country"].to_numpy()[rowids].tolist()


def test_read_where_spans_shard_boundaries(store):
    """Unsorted rowids that straddle every shard boundary in one call."""
    tmp_path, config = store
    record = RecordStore(tmp_path, "s2", config)
    boundaries = [SHARD_ROWS * i for i in range(1, len(record.shards))]
    picked: list[int] = []
    for edge in boundaries:
        picked.extend([edge - 1, edge, edge + 1])
    picked.extend([0, len(record) - 1])
    rowids = np.array(sorted(set(picked)), dtype=np.int64)
    # Reverse order so the first group is not shard 0.
    rowids = rowids[::-1].copy()

    sequential = record.read_range(0, len(record), ["name_core"])
    gathered = record.read_where(rowids, ["name_core"])
    assert gathered["name_core"].tolist() == sequential["name_core"].to_numpy()[rowids].tolist()


def test_read_where_preserves_request_order_and_duplicates(store):
    tmp_path, config = store
    record = RecordStore(tmp_path, "s3", config)
    rowids = np.array([5, 5, 200, 0, 61, 61, 12], dtype=np.int64)
    sequential = record.read_range(0, len(record), ["name_core"])
    gathered = record.read_where(rowids, ["name_core"])
    assert gathered["name_core"].tolist() == sequential["name_core"].to_numpy()[rowids].tolist()


def test_read_where_empty(store):
    tmp_path, config = store
    record = RecordStore(tmp_path, "s2", config)
    assert len(record.read_where(np.zeros(0, dtype=np.int64), ["name_core"])) == 0


def test_pool_rowids_are_global_and_contiguous(store):
    tmp_path, config = store
    pool = PoolStore(tmp_path, config)
    assert pool.n_pool == 2 * N_ROWS
    assert pool.s2_rows == N_ROWS

    rng = np.random.default_rng(3)
    rowids = rng.choice(pool.n_pool, size=150, replace=False).astype(np.int64)
    rowids = rowids[::-1].copy()
    gathered = pool.read_where(rowids, ["name_core", "rowid"])
    # Source 3 shards store a source-local rowid; the pool view must hand back
    # the pool-wide rowid the caller passed in.
    assert gathered["rowid"].astype(np.int64).tolist() == rowids.tolist()
    assert (pool.source_of(rowids) == (rowids >= N_ROWS).astype(np.int8)).all()

    expected = np.concatenate(
        [
            RecordStore(tmp_path, "s2", config).read_range(0, N_ROWS, ["name_core"])["name_core"].to_numpy(),
            RecordStore(tmp_path, "s3", config).read_range(0, N_ROWS, ["name_core"])["name_core"].to_numpy(),
        ]
    )
    assert gathered["name_core"].tolist() == expected[rowids].tolist()


def test_id_resolver_round_trips_through_pool(store):
    from src.scale.idmap import IdResolver

    tmp_path, config = store
    pool = PoolStore(tmp_path, config)
    resolver = IdResolver.build_from_shards(pool.iter_id_shards, total=pool.n_pool)
    rng = np.random.default_rng(5)
    rowids = rng.choice(pool.n_pool, size=120, replace=False).astype(np.int64)
    ids = pool.read_where(rowids, ["entity_id"])["entity_id"].to_numpy(dtype=object)
    assert resolver.resolve(ids).astype(np.int64).tolist() == rowids.tolist()
    assert resolver.ids_for(rowids).tolist() == list(ids)
    # Unknown ids must report -1 rather than silently matching something.
    assert resolver.resolve(np.array(["nope"], dtype=object)).tolist() == [-1]
