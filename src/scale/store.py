"""The normalised record store: Parquet shards on disk, ``int32`` row ids.

Replaces ``pd.read_csv`` of a whole table with a streamed, shard-wise writer.
The original TSVs are opened read-only and never modified.

Layout::

    work/<split>/store/
      s1/part-00000.parquet ...     normalised Source 1 rows
      s2/part-00000.parquet ...     normalised Source 2 rows
      s3/part-00000.parquet ...     normalised Source 3 rows
      store_manifest.json

Schema (identical for all three tables, so one reader serves all of them)::

    rowid       int32   global row index within the split; the only join key
    entity_id   string  decoded to text only at the output boundary
    name_norm   string  business_name_normalized
    name_core   string  business_name_core        <- the primary blocking key
    addr_norm   string  business_address_normalized
    addr_core   string  business_address_core
    country     string  country_normalized
    pincode     string  extracted, "" when not confidently present
    house_number string
    city        string
    state       string

``rowid`` is what makes the rest of the pipeline cheap: candidates,
ground-truth pairs, features and predictions are all integer arrays indexed by
it, so nothing downstream ever joins on strings or holds an id list.

Pool row ids are assigned Source 2 first, then Source 3, so ``rowid`` alone
decodes the source of a candidate.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import ADDRESS_COLUMN, COUNTRY_COLUMN, ID_COLUMN, NAME_COLUMN
from .artifacts import STORE_MANIFEST, read_manifest, write_manifest
from .config import ScaleConfig
from .progress import MemoryGuard, Progress, format_bytes

LOGGER = logging.getLogger(__name__)

#: Canonical column order of a store shard.
STORE_COLUMNS: tuple[str, ...] = (
    "rowid",
    "entity_id",
    "name_norm",
    "name_core",
    "addr_norm",
    "addr_core",
    "country",
    "pincode",
    "house_number",
    "city",
    "state",
)

#: Columns read by a bare "give me the ids / names" query.  Reading a subset is
#: what keeps a feature batch from pulling the whole text of the corpus.
TEXT_COLUMNS: tuple[str, ...] = ("name_norm", "name_core", "addr_norm", "addr_core")

#: Source 1 / Source 2 / Source 3 are stored under these sub-directory names.
SOURCES: tuple[str, ...] = ("s1", "s2", "s3")

#: Sub-directory name -> TSV file name, for the raw input of one split.
SOURCE_FILES: dict[str, str] = {
    "s1": "train_source1.tsv",
    "s2": "train_source2.tsv",
    "s3": "train_source3.tsv",
}

#: Input-file column aliases, copied from the legacy loader so the same files
#: keep working without duplicating the mapping in two places.
_COLUMN_ALIASES: dict[str, str] = {
    "id": ID_COLUMN, "entityid": ID_COLUMN, "entity_id": ID_COLUMN,
    "businessid": ID_COLUMN, "business_id": ID_COLUMN,
    "name": NAME_COLUMN, "businessname": NAME_COLUMN, "business_name": NAME_COLUMN,
    "company_name": NAME_COLUMN, "companyname": NAME_COLUMN,
    "address": ADDRESS_COLUMN, "businessaddress": ADDRESS_COLUMN,
    "business_address": ADDRESS_COLUMN, "addr": ADDRESS_COLUMN,
    "country_code": COUNTRY_COLUMN, "countrycode": COUNTRY_COLUMN,
    "country": COUNTRY_COLUMN, "nation": COUNTRY_COLUMN,
}

_ARROW_SCHEMA = pa.schema(
    [
        ("rowid", pa.int32()),
        ("entity_id", pa.string()),
        ("name_norm", pa.string()),
        ("name_core", pa.string()),
        ("addr_norm", pa.string()),
        ("addr_core", pa.string()),
        ("country", pa.string()),
        ("pincode", pa.string()),
        ("house_number", pa.string()),
        ("city", pa.string()),
        ("state", pa.string()),
    ]
)

_TSV_DTYPES = {ID_COLUMN: str, NAME_COLUMN: str, ADDRESS_COLUMN: str, COUNTRY_COLUMN: str}


def _shard_name(index: int) -> str:
    return f"part-{index:05d}.parquet"


def _truncate(values: list[str], width: int, chunk: int) -> tuple[list[str], int]:
    """Right-truncate values to ``width`` characters, returning the count lost.

    Normalised names are short, so this is a safety valve against a pathological
    input row; the count is reported rather than silently ignored.
    """
    if width <= 0:
        return values, 0
    out: list[str] = []
    lost = 0
    for start in range(0, len(values), chunk):
        part = values[start : start + chunk]
        too_long = [v for v in part if len(v) > width]
        if too_long:
            lost += len(too_long)
            out.extend([v if len(v) <= width else v[:width] for v in part])
        else:
            out.extend(part)
    return out, lost


@dataclass
class StoreStats:
    """What one table's build produced."""

    source: str
    rows: int
    shards: list[str]
    seconds: float
    input_path: str
    input_bytes: int
    output_bytes: int
    file_bytes: int = 0
    empty_names: int = 0
    empty_addresses: int = 0
    empty_countries: int = 0

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "rows": self.rows,
            "shards": self.shards,
            "seconds": round(self.seconds, 2),
            "input_path": self.input_path,
            "input_mb": round(self.input_bytes / 1024**2, 1),
            "tsv_file_mb": round((self.file_bytes or self.input_bytes) / 1024**2, 1),
            "output_mb": round(self.output_bytes / 1024**2, 1),
            "empty_names": self.empty_names,
            "empty_addresses": self.empty_addresses,
            "empty_countries": self.empty_countries,
        }


def count_lines(path: Path) -> int:
    """Count newlines with a coarse binary scan.

    Used only to report how much of a TSV a ``--limit-*`` run actually consumed.
    ``TextFileReader`` exposes no byte offset, and the compression ratio is
    worthless (and quietly wrong) if it is measured against the whole file.
    """
    total = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 22)
            if not chunk:
                break
            total += chunk.count(b"\n")
    return total


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------
def normalize_chunk(frame: pd.DataFrame, *, name_width: int = 512, address_width: int = 512) -> dict:
    """Normalise one raw TSV chunk into the store schema (as python lists).

    Reuses ``src.preprocessing`` so the store is the single definition of what
    "normalised" means for the whole project.  ``functools.lru_cache`` inside
    those helpers is per-value, which is what we want: a business name repeated
    across 50 records is normalised once.
    """
    from ..preprocessing.address_normalizer import extract_components, normalize_address
    from ..preprocessing.country_normalizer import normalize_country
    from ..preprocessing.name_normalizer import normalize_name

    def column(name: str) -> list[str]:
        if name not in frame.columns:
            return [""] * len(frame)
        return ["" if v is None else str(v) for v in frame[name].tolist()]

    raw_ids = column(ID_COLUMN)
    raw_names = column(NAME_COLUMN)
    raw_addr = column(ADDRESS_COLUMN)
    raw_country = column(COUNTRY_COLUMN)

    name_norm = [normalize_name(v, drop_legal_forms=False) for v in raw_names]
    name_core = [normalize_name(v, drop_legal_forms=True) for v in raw_names]
    addr_norm = [normalize_address(v) for v in raw_addr]
    addr_core = [
        " ".join(t for t in normalize_address(v).split() if t not in _ADDRESS_NOISE) for v in raw_addr
    ]
    country = [normalize_country(v) for v in raw_country]
    components = [extract_components(v) for v in raw_addr]

    name_norm, lost_n = _truncate(name_norm, name_width, 1_000_000)
    name_core, lost_nc = _truncate(name_core, name_width, 1_000_000)
    addr_norm, lost_a = _truncate(addr_norm, address_width, 1_000_000)
    addr_core, lost_ac = _truncate(addr_core, address_width, 1_000_000)
    if lost_n + lost_nc + lost_a + lost_ac:
        LOGGER.warning(
            "Truncated %d over-long normalised values (names %d/%d, addresses %d/%d)",
            lost_n + lost_nc + lost_a + lost_ac, lost_n, lost_nc, lost_a, lost_ac,
        )

    return {
        "entity_id": [v.strip() for v in raw_ids],
        "name_norm": name_norm,
        "name_core": name_core,
        "addr_norm": addr_norm,
        "addr_core": addr_core,
        "country": country,
        "pincode": [c.pincode for c in components],
        "house_number": [c.house_number for c in components],
        "city": [c.city for c in components],
        "state": [c.state for c in components],
    }


#: Mirrors ``src.preprocessing.address_normalizer.NOISE_TOKENS``; imported lazily
#: at module scope would create an import cycle through ``..config``.
_ADDRESS_NOISE = frozenset(
    {
        "and", "the", "of", "at", "in", "near", "nr", "opposite", "opp", "behind",
        "beside", "next", "to", "from", "via", "by", "nearby",
    }
)


def build_table(
    tsv_path: str | Path,
    out_dir: str | Path,
    source: str,
    config: ScaleConfig,
    *,
    guard: MemoryGuard | None = None,
    start_rowid: int = 0,
    limit_rows: int | None = None,
) -> StoreStats:
    """Stream one TSV into normalised Parquet shards.  Bounded memory."""
    tsv_path = Path(tsv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("part-*.parquet"):
        stale.unlink()

    started = time.perf_counter()
    header = _read_header(tsv_path)
    column_map = _resolve_columns(header)
    if ID_COLUMN not in column_map.values():
        raise ValueError(
            f"{tsv_path}: no entity-id column. Header={header}. "
            f"Accepted aliases: {sorted(set(_COLUMN_ALIASES))}"
        )
    usecols = list(column_map)

    shards: list[str] = []
    rowid = start_rowid
    total = 0
    empty_names = empty_addr = empty_country = 0
    shard_rows: list[int] = []
    buffer: dict[str, list] = {}
    buffered = 0

    def write_shard(rows: int) -> None:
        """Write exactly ``rows`` buffered rows as one self-contained Parquet file.

        A fresh ``ParquetWriter`` per shard: the writer is bound to a single file,
        so reusing one across shard boundaries would keep appending to the first
        file while the manifest claimed every shard existed.  Slicing to an exact
        row count keeps shards at ``shard_rows`` even when a TSV read chunk is
        larger than that.
        """
        nonlocal buffered, buffer
        if rows <= 0:
            return
        path = out_dir / _shard_name(len(shards))
        table = pa.table(
            {
                "rowid": pa.array(
                    np.arange(rowid - buffered, rowid - buffered + rows, dtype=np.int32),
                    type=pa.int32(),
                ),
                **{
                    name: pa.array(values[:rows], type=pa.string())
                    for name, values in buffer.items()
                },
            },
            schema=_ARROW_SCHEMA,
        )
        with pq.ParquetWriter(
            path, _ARROW_SCHEMA, compression=config.parquet_compression
        ) as writer:
            writer.write_table(table, row_group_size=config.row_group_rows)
        shards.append(path.name)
        shard_rows.append(rows)
        if rows >= buffered:
            buffer = {}
            buffered = 0
        else:
            buffer = {name: values[rows:] for name, values in buffer.items()}
            buffered -= rows

    def flush(final: bool = False) -> None:
        while buffered >= config.shard_rows:
            write_shard(config.shard_rows)
        if final and buffered:
            write_shard(buffered)

    progress = Progress(
        f"normalise {source} <- {tsv_path.name}",
        total=limit_rows,
        interval_s=config.progress_interval_s,
        guard=guard,
    )
    with progress:
        reader = pd.read_csv(
            tsv_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            na_values=[],
            encoding="utf-8",
            usecols=usecols,
            chunksize=config.read_chunk_rows,
            nrows=limit_rows,
        )
        for chunk in reader:
            if column_map:
                chunk = chunk.rename(columns=column_map)
            normalized = normalize_chunk(chunk)
            empty_names += sum(1 for v in normalized["name_norm"] if not v)
            empty_addr += sum(1 for v in normalized["addr_norm"] if not v)
            empty_country += sum(1 for v in normalized["country"] if not v)
            n = len(next(iter(normalized.values())))
            for name, values in normalized.items():
                buffer.setdefault(name, []).extend(values)
            buffered += n
            rowid += n
            total += n
            progress.add(n, rows_out=total)
            if buffered >= config.shard_rows:
                flush()
            guard and guard.check(f"normalise {source}")
        flush(final=True)

    output_bytes = sum((out_dir / name).stat().st_size for name in shards)
    file_bytes = tsv_path.stat().st_size
    file_rows = max(1, count_lines(tsv_path)) if total else 1
    # Partial run: scale the file size by the fraction of rows consumed, so the
    # manifest's compression ratio describes the rows actually written.
    input_bytes = int(file_bytes * total / file_rows) if total < file_rows else file_bytes
    stats = StoreStats(
        source=source,
        rows=total,
        shards=shards,
        seconds=time.perf_counter() - started,
        input_path=str(tsv_path),
        input_bytes=input_bytes,
        file_bytes=file_bytes,
        output_bytes=output_bytes,
        empty_names=empty_names,
        empty_addresses=empty_addr,
        empty_countries=empty_country,
    )
    LOGGER.info(
        "%s: %s rows in %d shards (%.1f MB in -> %.1f MB out, %.1f%% of name empty)",
        source,
        f"{total:,}",
        len(shards),
        stats.input_bytes / 1024**2,
        output_bytes / 1024**2,
        100.0 * empty_names / max(1, total),
    )
    return stats


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return handle.readline().rstrip("\n").split("\t")


def _resolve_columns(header: Sequence[str]) -> dict[str, str]:
    """``original -> canonical`` for the header, keeping only known columns.

    ``pandas.read_csv(usecols=...)`` needs the *file's* names, so the rename has
    to happen after the read, not before.
    """
    mapping: dict[str, str] = {}
    for column in header:
        name = str(column).strip()
        canonical = _COLUMN_ALIASES.get(name.lower(), name)
        if canonical in _TSV_DTYPES and canonical not in mapping.values():
            mapping[name] = canonical
    return mapping


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
class RecordStore:
    """Read-only, column-selective view over a split's normalised shards."""

    def __init__(self, root: str | Path, source: str, config: ScaleConfig | None = None) -> None:
        self.root = Path(root)
        self.source = source
        self.dir = self.root / source
        self.config = config or ScaleConfig()
        self.shards = sorted(self.dir.glob("part-*.parquet"))
        if not self.shards:
            raise FileNotFoundError(
                f"No shards in {self.dir}. Run scripts/build_store.py for this split/source first."
            )
        self._metadata = [pq.ParquetFile(path).metadata for path in self.shards]
        self._offsets = np.zeros(len(self.shards) + 1, dtype=np.int64)
        for i, meta in enumerate(self._metadata):
            self._offsets[i + 1] = self._offsets[i] + meta.num_rows
        self.n_rows = int(self._offsets[-1])

    # -- geometry --------------------------------------------------------
    def __len__(self) -> int:
        return self.n_rows

    def shard_of(self, rowid: int) -> int:
        """Index of the shard containing a global row id."""
        return int(np.searchsorted(self._offsets, rowid, side="right") - 1)

    def range_of_shard(self, index: int) -> tuple[int, int]:
        return int(self._offsets[index]), int(self._offsets[index + 1])

    # -- reading ---------------------------------------------------------
    def iter_shards(self, columns: Sequence[str] = STORE_COLUMNS) -> Iterator[tuple[int, int, pd.DataFrame]]:
        """Yield ``(start_rowid, n_rows, frame)`` per shard."""
        for index, path in enumerate(self.shards):
            table = pq.read_table(path, columns=list(columns))
            frame = table.to_pandas(split_blocks=True, self_destruct=True)
            yield int(self._offsets[index]), len(frame), frame

    def read_range(self, start: int, stop: int, columns: Sequence[str] = STORE_COLUMNS) -> pd.DataFrame:
        """Read a rowid range, touching only the shards that overlap it."""
        start = max(0, int(start))
        stop = min(self.n_rows, int(stop))
        if stop <= start:
            empty = {c: np.zeros(0, dtype=object) for c in columns}
            return pd.DataFrame(empty)
        first = self.shard_of(start)
        last = self.shard_of(stop - 1)
        if first == last:
            table = pq.read_table(
                self.shards[first], columns=list(columns)
            ).slice(start - self._offsets[first], stop - start)
        else:
            parts = []
            for index in range(first, last + 1):
                lo = max(start, self._offsets[index]) - self._offsets[index]
                hi = min(stop, self._offsets[index + 1]) - self._offsets[index]
                if hi > lo:
                    parts.append(
                        pq.read_table(self.shards[index], columns=list(columns)).slice(lo, hi - lo)
                    )
            table = pa.concat_tables(parts)
        return table.to_pandas(split_blocks=True, self_destruct=True)

    def read_batches(
        self, batch_rows: int, columns: Sequence[str] = STORE_COLUMNS
    ) -> Iterator[tuple[int, pd.DataFrame]]:
        """Yield ``(start_rowid, frame)`` batches of at most ``batch_rows`` rows."""
        for start, _, frame in self.iter_shards(columns):
            for offset in range(0, len(frame), batch_rows):
                chunk = frame.iloc[offset : offset + batch_rows]
                if len(chunk):
                    yield start + offset, chunk.reset_index(drop=True)

    def column(self, name: str) -> np.ndarray:
        """The whole column as one numpy object array (use only when it fits)."""
        parts = [
            pq.read_table(path, columns=[name]).column(name).to_numpy(zero_copy_only=False)
            for path in self.shards
        ]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=object)

    def read_where(self, rowids: np.ndarray, columns: Sequence[str]) -> pd.DataFrame:
        """Gather arbitrary rows by global row id.

        Grouped by shard so each Parquet file is read once and in increasing
        row order, which keeps the read sequential instead of random.
        """
        rowids = np.asarray(rowids, dtype=np.int64)
        out = {c: np.zeros(rowids.size, dtype=object) for c in columns}
        if rowids.size == 0:
            return pd.DataFrame(out)
        shard_ids = np.searchsorted(self._offsets, rowids, side="right") - 1
        order = np.argsort(shard_ids, kind="stable")
        sorted_shards = shard_ids[order]
        boundaries = np.flatnonzero(np.diff(sorted_shards)) + 1
        # Walk the groups by *position*, not by member: ``group`` holds indices
        # into the original arrays, so ``sorted_shards[group[0]]`` would read the
        # shard of some other group whenever a gather spans two or more shards.
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [order.size]))
        for first, last in zip(starts, ends):
            if last <= first:
                continue
            index = int(sorted_shards[first])
            group = order[first:last]
            local = rowids[group] - self._offsets[index]
            table = pq.read_table(self.shards[index], columns=list(columns)).take(
                pa.array(local, type=pa.int64())
            )
            frame = table.to_pandas(split_blocks=True, self_destruct=True)
            for column in columns:
                out[column][group] = frame[column].to_numpy(dtype=object)
        return pd.DataFrame(out)

    def describe(self) -> str:
        size = sum(path.stat().st_size for path in self.shards)
        return (
            f"{self.source}: {self.n_rows:,} rows in {len(self.shards)} shards, "
            f"{format_bytes(size)}"
        )


class PoolStore:
    """The blocking pool: Source 2 rows followed by Source 3 rows.

    Pool rowids are contiguous across the two sources, so a rowid encodes its
    own source (``rowid < s2_rows``) and every index can be built with one
    streaming pass instead of one pass per source plus a merge.
    """

    def __init__(self, root: str | Path, config: ScaleConfig | None = None) -> None:
        self.root = Path(root)
        self.config = config or ScaleConfig()
        self.s2 = RecordStore(self.root, "s2", self.config)
        self.s3 = RecordStore(self.root, "s3", self.config)
        self.s2_rows = len(self.s2)
        self.n_pool = self.s2_rows + len(self.s3)

    def __len__(self) -> int:
        return self.n_pool

    def source_of(self, pool_rowid: np.ndarray) -> np.ndarray:
        return (np.asarray(pool_rowid, dtype=np.int64) >= self.s2_rows).astype(np.int8)

    def iter_shards(self, columns: Sequence[str] = STORE_COLUMNS) -> Iterator[tuple[int, int, pd.DataFrame]]:
        for rowid0, n_rows, frame in self.s2.iter_shards(columns):
            yield rowid0, n_rows, frame
        offset = self.s2_rows
        for rowid0, n_rows, frame in self.s3.iter_shards(columns):
            yield rowid0 + offset, n_rows, frame

    def read_batches(
        self, batch_rows: int, columns: Sequence[str] = STORE_COLUMNS
    ) -> Iterator[tuple[int, pd.DataFrame]]:
        for start, _, frame in self.iter_shards(columns):
            for offset in range(0, len(frame), batch_rows):
                chunk = frame.iloc[offset : offset + batch_rows]
                if len(chunk):
                    yield start + offset, chunk.reset_index(drop=True)

    def column(self, name: str) -> np.ndarray:
        return np.concatenate([self.s2.column(name), self.s3.column(name)])

    def read_where(self, rowids: np.ndarray, columns: Sequence[str]) -> pd.DataFrame:
        rowids = np.asarray(rowids, dtype=np.int64)
        in_s2 = rowids < self.s2_rows
        frames = []
        for mask, store in ((in_s2, self.s2), (~in_s2, self.s3)):
            pool_rowids = rowids[mask]
            if pool_rowids.size == 0:
                continue
            selected = pool_rowids - self.s2_rows if store is self.s3 else pool_rowids
            part = store.read_where(selected, columns)
            if "rowid" in part.columns:
                # The caller passed pool rowids and expects them back.  Source 3
                # shards store a source-local rowid, so restore the pool-wide
                # value or the documented "pool rowid in, pool rowid out"
                # contract quietly breaks for the second half of the pool.  It
                # has to be the pre-decrement ``pool_rowids``: reusing
                # ``selected`` here would hand back Source 3 local rowids.
                part["rowid"] = pool_rowids
            part.index = np.flatnonzero(mask)
            frames.append(part)
        if not frames:
            return pd.DataFrame({c: np.zeros(rowids.size, dtype=object) for c in columns})
        return pd.concat(frames).sort_index().reset_index(drop=True)

    def iter_id_shards(self, column: str = "entity_id"):
        """Yield ``(pool_rowid0, arrow_array)`` for one column, in pool order."""
        for rowid0, _, frame in self.iter_shards([column]):
            yield rowid0, pa.array(frame[column], type=pa.string())

    def describe(self) -> str:
        return (
            f"pool: {self.n_pool:,} rows "
            f"({self.s2_rows:,} source2 + {self.n_pool - self.s2_rows:,} source3), "
            f"{format_bytes(sum(p.stat().st_size for p in self.s2.shards + self.s3.shards))}"
        )


class SplitStore:
    """The three tables of one split, plus the pool view over Source 2 + 3."""

    def __init__(self, root: str | Path, config: ScaleConfig | None = None) -> None:
        self.root = Path(root)
        self.config = config or ScaleConfig()
        self.s1 = RecordStore(self.root, "s1", self.config)
        self.pool = PoolStore(self.root, self.config)
        self.s2 = self.pool.s2
        self.s3 = self.pool.s3

    @property
    def n_pool(self) -> int:
        """Pool rows; Source 2 first, so a pool rowid decodes its own source."""
        return self.pool.n_pool

    @property
    def s2_rows(self) -> int:
        return self.pool.s2_rows

    def source_of(self, pool_rowid: np.ndarray) -> np.ndarray:
        """``0`` for a Source 2 row, ``1`` for a Source 3 row."""
        return self.pool.source_of(pool_rowid)

    def describe(self) -> str:
        return "\n".join([self.s1.describe(), self.s2.describe(), self.s3.describe(),
                          f"pool: {self.n_pool:,} rows"])


# --------------------------------------------------------------------------
# manifests
# --------------------------------------------------------------------------
def write_store_manifest(directory: str | Path, payload: dict) -> Path:
    return write_manifest(Path(directory) / STORE_MANIFEST, payload)


def read_store_manifest(directory: str | Path) -> dict | None:
    return read_manifest(Path(directory) / STORE_MANIFEST)


#: Re-exported so callers do not need to import pandas/pyarrow themselves.
ARROW_SCHEMA = _ARROW_SCHEMA
