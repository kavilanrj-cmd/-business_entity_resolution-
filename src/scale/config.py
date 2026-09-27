"""Configuration for the scalable, disk-backed pipeline.

Every number that bounds memory or runtime lives here.  Nothing in
``src/scale`` reads a corpus size at import time and nothing scales itself
from the data: the budgets are explicit so that a run is reproducible and so
that peak memory is *knowable before it is spent*.

Sizing model
------------
Let ``Q`` be the number of Source 1 rows and ``N`` the pool size
(``|source2| + |source3|``).  For this dataset ``Q = 2,206,821`` and
``N = 10,320,219``, so the brute-force pair space is 2.3e13.

Candidate generation here is ``O(Q * posting_budget)`` rather than
``O(Q * N)``: a query only ever expands its own rarest term postings and stops
at ``posting_budget_name``.  Peak memory is therefore::

    ~ chunk_rows * 100 B                      (one TSV read chunk, pandas)
  + posting_budget_name * batch_rows * 4 B   (one candidate batch, int32)
  + index arrays (mmap'd, so they live in the OS page cache)

With the defaults below: 500k * 100 B = 50 MB, 3000 * 10k * 4 B = 120 MB.
Total peak RSS is dominated by Parquet write buffers and stays under ~1.5 GB
for the build, which is why the 6 GB budget has headroom.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
@dataclass
class ScalePaths:
    """Where the derived artefacts live.  The original TSVs are never written to."""

    work_dir: str = "work"

    def split_dir(self, split: str) -> Path:
        return Path(self.work_dir) / split

    def store_dir(self, split: str) -> Path:
        return self.split_dir(split) / "store"

    def index_dir(self, split: str) -> Path:
        return self.split_dir(split) / "index"

    def candidate_dir(self, split: str) -> Path:
        return self.split_dir(split) / "candidates"

    def feature_dir(self, split: str) -> Path:
        return self.split_dir(split) / "features"

    def ensure(self, split: str) -> None:
        for path in (
            self.store_dir(split),
            self.index_dir(split),
            self.candidate_dir(split),
            self.feature_dir(split),
        ):
            path.mkdir(parents=True, exist_ok=True)


def default_work_dir(base: str | Path | None = None) -> str:
    """Default scratch directory, inside the project so it is easy to delete."""
    root = Path(base) if base is not None else PROJECT_ROOT
    return str(root / "work")


# --------------------------------------------------------------------------
# the config
# --------------------------------------------------------------------------
@dataclass
class ScaleConfig:
    """All budgets and switches for the scalable layer."""

    paths: ScalePaths = field(default_factory=ScalePaths)

    # -- I/O budgets ----------------------------------------------------
    #: Rows per TSV read chunk.  500k x ~100 B of text ~ 50 MB; the pandas
    #: C parser needs roughly 3x that transiently.
    read_chunk_rows: int = 500_000
    #: Rows per normalised Parquet shard.  250k keeps shards at 15-25 MB so a
    #: later stage can read one without touching the rest of the file.
    shard_rows: int = 250_000
    #: Parquet row-group size, aligned to ``shard_rows`` so a shard is one group.
    row_group_rows: int = 250_000
    parquet_compression: str = "zstd"

    # -- query batching -------------------------------------------------
    #: Source 1 rows per candidate batch.  The single most important memory
    #: knob: peak candidate memory is this x ``posting_budget_name``.
    batch_rows: int = 10_000
    #: Candidate pairs written per Parquet shard before rotating.
    candidate_shard_rows: int = 4_000_000

    # -- inverted index -------------------------------------------------
    #: Minimum term length to index (single letters are noise).
    token_min_length: int = 2
    #: Terms occurring in fewer than this many pool records are dropped: they
    #: cannot contribute to recall and only inflate the vocabulary.
    min_df: int = 2
    #: Terms occurring in more than this many pool records are NOT indexed.
    #: This is the guarantee that a posting list can never blow up a query;
    #: such terms are handled by sorted neighbourhood instead.
    max_df: int = 5_000
    #: Tokens dropped from the *name* index because they are legal forms.
    name_stop_tokens: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "and", "the", "of", "for", "a", "an", "at", "in", "on", "to", "by",
                "corp", "inc", "co", "ltd", "pvt", "gmbh", "ag", "bv", "nv", "sa",
                "sas", "sarl", "spa", "pty", "sdn", "bhd", "kft", "sp", "zoo", "kg",
                "ohg", "ek", "pc", "psc", "lp", "lllp", "llp", "llc", "plc", "trust",
                "trustee", "company", "limited", "private", "corporation",
                "incorporated",
            }
        )
    )
    #: Tokens dropped from the *address* index: street furniture and joiners
    #: that appear in millions of records and identify nothing.
    address_stop_tokens: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "and", "the", "of", "at", "in", "to", "near", "nr", "opp", "opposite",
                "behind", "beside", "next", "from", "via", "by", "nearby", "1st", "2nd",
                "3rd", "rd", "st", "av", "blvd", "dr", "ln", "ct", "sq", "plz", "pl",
                "apt", "fl", "bldg", "dist", "soc", "hsg", "grp", "jn", "x", "mkt",
                "po", "no", "nos", "nro", "num", "shop", "unit", "sec", "bl", "col",
            }
        )
    )

    # -- exact-key index ------------------------------------------------
    #: Keys shorter than this are not indexed, to avoid one degenerate block.
    exact_key_min_length: int = 2
    #: Maximum number of pool rows taken from one exact-key block.  A generic
    #: name ("Sharma Traders") can have a block in the thousands; truncating
    #: bounds the query and the recall cost is measured, not assumed.
    exact_block_cap: int = 1_000

    # -- fixed-width key index (exact + sorted neighbourhood) -------------
    #: Width in bytes of the fixed-width sort key.  32 characters covers
    #: essentially every business name, so exact-key lookup is exact for all
    #: practical purposes and the sorted-neighbourhood prefix search reuses the
    #: same array instead of paying for a second index.
    exact_key_chars: int = 32
    #: Minimum shared prefix for two sorted neighbours to be considered.
    sn_prefix_chars: int = 6
    #: Window of sorted neighbours examined on each side of the insertion point.
    sn_window: int = 50
    #: Minimum sort-key length worth searching with.
    sn_min_key_chars: int = 4

    # -- per-query candidate budgets -------------------------------------
    #: Postings a single query may expand from the name index.  THE knob that
    #: makes candidate generation sub-quadratic.  Default 3000 x 10k batch
    #: rows = 3e7 int32 = 120 MB.
    posting_budget_name: int = 3_000
    #: Same, for the address index (addresses are noisier, so a smaller budget).
    posting_budget_address: int = 1_500
    #: Terms expanded per query, rarest first.  3 is enough to be selective
    #: without ever reading a generic term's postings.
    max_terms_per_query: int = 3
    #: Candidates retained per query after the union, strongest first.
    max_candidates_per_s1: int = 100
    #: Minimum retained per query, so a hard entity is never left with nothing.
    min_candidates_per_s1: int = 3
    #: rapidfuzz ratio floor for a sorted-neighbourhood hit to be kept.
    sn_min_ratio: float = 0.55
    #: Same-country filter on the name postings, top-K kept.
    country_top_k: int = 50

    # -- ids ------------------------------------------------------------
    #: Fixed-width id array width.  0 means "measure from the data at build
    #: time", which is the safe default; ids longer than this fall back to a
    #: Python dict (and log a warning).
    id_width: int = 0

    # -- observability ---------------------------------------------------
    #: Seconds between progress log lines.
    progress_interval_s: float = 10.0
    #: Hard ceiling on process RSS.  Exceeding it raises, so a runaway is loud
    #: rather than an OOM kill at 90% of physical memory.
    memory_budget_bytes: int = 6 * 1024**3
    #: Warn above this fraction of ``memory_budget_bytes``.
    memory_warn_fraction: float = 0.80

    # -- labels (training only) ------------------------------------------
    #: Append verified true pairs that blocking missed, so the classifier
    #: always sees every positive.  Recall is still measured, not assumed.
    inject_positives: bool = True

    # ------------------------------------------------------------------
    def validate(self) -> "ScaleConfig":
        if self.batch_rows < 1:
            raise ValueError("batch_rows must be >= 1")
        if self.posting_budget_name < 1 or self.posting_budget_address < 1:
            raise ValueError("posting budgets must be >= 1")
        if self.max_candidates_per_s1 < self.min_candidates_per_s1:
            raise ValueError("max_candidates_per_s1 must be >= min_candidates_per_s1")
        if self.sn_window < 1:
            raise ValueError("sn_window must be >= 1")
        if not 1 <= self.sn_prefix_chars <= self.exact_key_chars:
            raise ValueError("require 1 <= sn_prefix_chars <= exact_key_chars")
        return self

    def peak_batch_bytes(self) -> int:
        """Worst-case candidate memory for one batch, in bytes."""
        per_query = (
            self.posting_budget_name
            + self.posting_budget_address
            + 2 * self.sn_window
            + 2 * self.exact_block_cap
        )
        # int32 pool_rowid + float32 score + uint8 mask, doubled to allow for
        # the intermediate copies made while deduping.
        return int(per_query * self.batch_rows) * 14

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["paths"] = asdict(self.paths)
        return out

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["name_stop_tokens"] = sorted(self.name_stop_tokens)
        payload["address_stop_tokens"] = sorted(self.address_stop_tokens)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ScaleConfig":
        cfg = cls()
        if path is None:
            return cfg
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        paths = data.pop("paths", None) or {}
        for key, value in data.items():
            if not hasattr(cfg, key):
                LOGGER.debug("Ignoring unknown config key %r", key)
                continue
            if key in ("name_stop_tokens", "address_stop_tokens"):
                setattr(cfg, key, frozenset(value))
            else:
                setattr(cfg, key, value)
        if paths:
            cfg.paths = ScalePaths(**{k: v for k, v in paths.items() if hasattr(ScalePaths, k)})
        return cfg.validate()


#: Strategy bit flags stored in the candidate table's ``mask`` column.
STRATEGY_BITS: dict[str, int] = {
    "exact_normalized": 1 << 0,
    "exact_core": 1 << 1,
    "name_token": 1 << 2,
    "address_token": 1 << 3,
    "sorted_neighbourhood": 1 << 4,
    "country_token": 1 << 5,
}
STRATEGY_ORDER: tuple[str, ...] = (
    "exact_normalized",
    "exact_core",
    "name_token",
    "sorted_neighbourhood",
    "address_token",
    "country_token",
)

#: Source 2 rows come first in the pool, then Source 3, so a pool rowid
#: decodes to a source without a lookup.
POOL_SOURCE2 = 0
POOL_SOURCE3 = 1


def strategy_names(mask: int) -> list[str]:
    """Names of the strategies encoded in a candidate's ``mask``."""
    return [name for name in STRATEGY_ORDER if mask & STRATEGY_BITS[name]]
