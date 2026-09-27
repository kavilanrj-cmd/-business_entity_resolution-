"""Resolve entity ids to ``int32`` row ids without a giant Python dict.

A ``dict`` of 12.5 M id strings costs roughly 1.8 GB in CPython (each key is a
separate ``str`` object plus a hash-table slot), and it is needed for two hot
paths: labelling the ground truth and decoding row ids back to ids for the
submission file.

Instead, ids are held as a **sorted fixed-width byte array** with a parallel
``int32`` row-id array, and lookups are a vectorised ``np.searchsorted``::

    keys : (M,) dtype 'S{width}', lexicographically sorted (numpy compares
           fixed-width bytes with memcmp semantics, padding with NULs)
    rows : (M,) int32, aligned with keys

Cost: ``M * (width + 4)`` bytes -- 212 MB for 12.5 M ids of width 13 -- and
``O(k log M)`` for ``k`` lookups with no per-lookup Python object.

Ids wider than the configured width, or so numerous that a fixed-width array
would be unsafe, fall back to a ``dict`` with a warning; correctness is never
traded away for memory.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import write_manifest
from .progress import Progress, format_bytes

LOGGER = logging.getLogger(__name__)

#: Above this many ids the dict fallback would cost more than the whole memory
#: budget, so refuse loudly rather than thrash.
DICT_FALLBACK_LIMIT = 2_000_000

_IDS_PARQUET = "ids.parquet"
_IDS_MANIFEST = "ids_manifest.json"


def measure_id_width(values: np.ndarray, probe: int = 20_000) -> int:
    """Longest id in ``values``, from an evenly spaced sample."""
    if values.size == 0:
        return 1
    if values.size <= probe:
        sample = values
    else:
        step = values.size / probe
        sample = values[(np.arange(probe) * step).astype(np.int64)]
    return int(max((len(str(v)) for v in sample.tolist()), default=1))


def _fill_fixed_bytes(dest: np.ndarray, values: np.ndarray) -> None:
    """Write ``values`` into the leading rows of the ``(n, width)`` uint8 ``dest``.

    Padding is done with chunked numpy gather/scatter off the Arrow buffers, so
    a 12.5 M-row column costs ``n * width`` bytes plus one ``chunk`` scratch
    buffer, never a list of 12.5 M Python ``bytes`` objects.
    """
    width = dest.shape[1]
    chunk = 1_000_000
    n = int(values.size)
    cols = np.arange(width, dtype=np.int32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        part = np.asarray(values[start:stop], dtype=object)
        arr = pa.array([str(v) for v in part.tolist()], type=pa.string())
        offsets = np.frombuffer(arr.buffers()[1], dtype=np.int32, count=stop - start + 1)
        data = np.frombuffer(arr.buffers()[2], dtype=np.uint8, count=int(offsets[-1]))
        lengths = np.minimum(offsets[1:] - offsets[:-1], width).astype(np.int32)
        mask = cols[None, :] < lengths[:, None]
        source = (offsets[:-1][:, None] + cols[None, :])[mask]
        target = (np.arange(stop - start, dtype=np.int32)[:, None] * width + cols[None, :])[mask]
        dest[start:stop].reshape(-1)[target] = data[source]


def _as_fixed_bytes(values: np.ndarray, width: int, chunk: int = 1_000_000) -> np.ndarray:
    """``(n,)`` str/object array -> ``(n,)`` numpy ``S{width}`` array."""
    n = int(values.size)
    width = max(1, int(width))
    out = np.zeros((n, width), dtype=np.uint8)
    if n == 0:
        return out.reshape(0).view(f"S{width}")
    _fill_fixed_bytes(out, np.asarray(values))
    return out.reshape(-1).view(f"S{width}")


class IdResolver:
    """Bidirectional ``entity_id <-> rowid`` map over sorted fixed-width keys."""

    def __init__(
        self,
        keys: np.ndarray,
        rows: np.ndarray,
        *,
        dict_fallback: dict[str, int] | None = None,
    ) -> None:
        self.keys = keys
        self.rows = np.asarray(rows, dtype=np.int32)
        self.width = int(keys.dtype.itemsize) if keys.size else 0
        self._dict = dict_fallback

    # -- construction ----------------------------------------------------
    @classmethod
    def build(cls, ids: np.ndarray, *, width: int = 0) -> "IdResolver":
        """Build a resolver from an id column given in rowid order."""
        ids = np.asarray(ids)
        if ids.size == 0:
            return cls(np.zeros(0, dtype="S1"), np.zeros(0, dtype=np.int32))

        needed = measure_id_width(ids)
        if width and width < needed:
            LOGGER.warning(
                "Configured id width %d is shorter than the observed maximum %d; "
                "falling back to a dict resolver.", width, needed,
            )
            return cls._build_dict(ids)
        w = width or needed
        if w > 64:
            LOGGER.warning("Entity ids reach %d characters; using a dict resolver.", w)
            return cls._build_dict(ids)
        if ids.size > DICT_FALLBACK_LIMIT and w * ids.size > 2 * 1024**3:
            LOGGER.warning(
                "A %d-byte id table for %d ids exceeds 2 GB; using a dict resolver.",
                w * ids.size, ids.size,
            )
            return cls._build_dict(ids)

        keys = _as_fixed_bytes(ids, w)
        order = np.argsort(keys, kind="stable")
        return cls(np.ascontiguousarray(keys[order]), np.arange(ids.size, dtype=np.int32)[order])

    @classmethod
    def build_from_shards(
        cls,
        shards_fn: Callable[[], Iterable[tuple[int, pa.Array]]],
        *,
        total: int | None = None,
        width: int = 0,
        guard: object | None = None,
    ) -> "IdResolver":
        """Build a resolver from Arrow string arrays yielded in rowid order.

        ``build`` wants the whole id column as one object array, which for a
        10.3 M-row pool means ~600 MB of Python ``str`` before any work starts.
        This version streams the shards twice -- once to measure the key width,
        once to fill -- so it only ever holds ``n * width`` bytes plus a single
        shard, i.e. ~124 MB for the train pool.
        """
        # Pass 1: widest key, and the row count if the caller did not know it.
        widest = 0
        seen = 0
        for _, array in shards_fn():
            offsets = np.frombuffer(array.buffers()[1], dtype=np.int32, count=len(array) + 1)
            if offsets.size > 1:
                widest = max(widest, int((offsets[1:] - offsets[:-1]).max()))
            seen += len(array)
        n = seen if total is None else int(total)
        if n != seen:
            raise ValueError(f"shards held {seen} rows but total={n}")

        def _as_dict_resolver() -> "IdResolver":
            LOGGER.warning("Falling back to a dict id resolver for %d ids.", n)
            return cls._build_dict(
                np.concatenate(
                    [np.asarray(a.to_pylist(), dtype=object) for _, a in shards_fn()]
                )
            )

        needed = max(1, widest)
        if width and width < needed:
            LOGGER.warning(
                "Configured id width %d is shorter than the observed maximum %d; "
                "falling back to a dict resolver.", width, needed,
            )
            return _as_dict_resolver()
        w = width or needed
        if w > 64:
            LOGGER.warning("Entity ids reach %d characters; using a dict resolver.", w)
            return _as_dict_resolver()
        if w * n > 2 * 1024**3:
            LOGGER.warning(
                "A %d-byte id table for %d ids exceeds 2 GB; using a dict resolver.",
                w * n, n,
            )
            return _as_dict_resolver()

        # Pass 2: fill the fixed-width table shard by shard.
        progress = Progress("fill id table", total=n, guard=guard)  # type: ignore[arg-type]
        out = np.zeros((n, w), dtype=np.uint8)
        with progress:
            for rowid0, array in shards_fn():
                _fill_fixed_bytes(
                    out[rowid0 : rowid0 + len(array)],
                    np.asarray(array.to_pylist(), dtype=object),
                )
                progress.add(len(array))
                if guard is not None:
                    guard.check()  # type: ignore[attr-defined]
        keys = out.reshape(-1).view(f"S{w}")
        del out
        order = np.argsort(keys, kind="stable")
        return cls(np.ascontiguousarray(keys[order]), np.arange(n, dtype=np.int32)[order])

    @classmethod
    def _build_dict(cls, ids: np.ndarray) -> "IdResolver":
        table: dict[str, int] = {}
        for i, value in enumerate(ids.tolist()):
            table.setdefault(str(value), int(i))
        return cls(
            np.zeros(0, dtype="S1"),
            np.zeros(0, dtype=np.int32),
            dict_fallback=table,
        )

    # -- lookup ----------------------------------------------------------
    def resolve(self, ids: np.ndarray) -> np.ndarray:
        """Rowid for each id; ``-1`` when the id is unknown.

        Duplicate ids in the index resolve to their lowest rowid, because the
        sort is stable and ``searchsorted(side="left")`` lands on the first.
        """
        ids = np.asarray(ids)
        if ids.size == 0:
            return np.zeros(0, dtype=np.int32)
        if self._dict is not None:
            table = self._dict
            return np.fromiter(
                (table.get(str(v), -1) for v in ids.tolist()), dtype=np.int32, count=ids.size
            )
        probe = _as_fixed_bytes(ids, self.width)
        lo = np.searchsorted(self.keys, probe, side="left")
        found = (lo < self.keys.size) & (self.keys[np.minimum(lo, self.keys.size - 1)] == probe)
        out = np.full(ids.size, -1, dtype=np.int32)
        if found.any():
            out[found] = self.rows[lo[found]]
        return out

    def ids_for(self, rowids: np.ndarray) -> np.ndarray:
        """Entity id for each rowid; ``''`` when out of range."""
        rowids = np.asarray(rowids, dtype=np.int64)
        if rowids.size == 0:
            return np.zeros(0, dtype=object)
        if self._dict is not None:
            inverse = {v: k for k, v in self._dict.items()}
            return np.array([inverse.get(int(r), "") for r in rowids.tolist()], dtype=object)
        order = np.argsort(self.rows, kind="stable")
        sorted_rows = self.rows[order]
        pos = np.clip(np.searchsorted(sorted_rows, rowids), 0, sorted_rows.size - 1)
        picked = order[pos]
        return np.array(
            [self.keys[i].decode("utf-8", errors="replace") for i in picked.tolist()], dtype=object
        )

    # -- persistence -----------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._dict is not None:
            items = list(self._dict.items())
            table = pa.table(
                {
                    "id": pa.array([k for k, _ in items], type=pa.string()),
                    "rowid": pa.array([v for _, v in items], type=pa.int32()),
                }
            )
            mode = "dict"
        else:
            keys = np.ascontiguousarray(self.keys)
            table = pa.table(
                {
                    "key": pa.FixedSizeBinaryArray.from_buffers(
                        pa.binary(self.width),
                        int(keys.size),
                        [None, pa.py_buffer(keys.view(np.uint8).tobytes())],
                    ),
                    "rowid": pa.array(self.rows, type=pa.int32()),
                }
            )
            mode = "fixed"
        pq.write_table(table, directory / _IDS_PARQUET, compression="zstd")
        return write_manifest(
            directory / _IDS_MANIFEST,
            {
                "mode": mode,
                "width": self.width,
                "n_ids": int(self.rows.size) if mode == "fixed" else len(self._dict or {}),
                "bytes": format_bytes(self.nbytes()),
            },
        )

    @classmethod
    def load(cls, directory: str | Path) -> "IdResolver":
        directory = Path(directory)
        manifest_path = directory / _IDS_MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No id resolver in {directory}. Run scripts/build_store.py first."
            )
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        table = pq.read_table(directory / _IDS_PARQUET)
        if meta["mode"] == "dict":
            ids = table.column("id").to_pylist()
            rows = table.column("rowid").to_numpy(zero_copy_only=False).astype(np.int32)
            mapping = {str(k): int(v) for k, v in zip(ids, rows.tolist())}
            return cls(np.zeros(0, dtype="S1"), np.zeros(0, dtype=np.int32), dict_fallback=mapping)
        column = table.column("key").combine_chunks()
        n = table.num_rows
        raw = column.buffers()[1]
        keys = np.frombuffer(raw, dtype=np.uint8, count=n * meta["width"]).view(
            f"S{meta['width']}"
        )
        rows = table.column("rowid").to_numpy(zero_copy_only=False).astype(np.int32)
        return cls(keys, rows)

    def nbytes(self) -> int:
        if self._dict is not None:
            return len(self._dict) * 96  # rough CPython str + hash-slot estimate
        return int(self.keys.nbytes + self.rows.nbytes)

    def describe(self) -> str:
        if self._dict is not None:
            return f"dict resolver, {len(self._dict):,} ids, ~{format_bytes(self.nbytes())}"
        return (
            f"sorted fixed-width resolver, {self.rows.size:,} ids, "
            f"width {self.width}, {format_bytes(self.nbytes())}"
        )


def build_resolver_from_columns(
    id_arrays: list[np.ndarray], *, width: int = 0, guard: object | None = None
) -> IdResolver:
    """Build a resolver from already-materialised id columns in rowid order."""
    progress = Progress("build id resolver", total=sum(len(a) for a in id_arrays), guard=guard)  # type: ignore[arg-type]
    parts: list[np.ndarray] = []
    with progress:
        for array in id_arrays:
            parts.append(np.asarray(array, dtype=object))
            progress.add(len(array))
    ids = np.concatenate(parts) if parts else np.zeros(0, dtype=object)
    del parts
    resolver = IdResolver.build(ids, width=width)
    LOGGER.info("Built %s", resolver.describe())
    return resolver
