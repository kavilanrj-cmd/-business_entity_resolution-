"""Disk-backed, index-based entity resolution for 2M-10M row datasets.

This package is the replacement for the in-memory pipeline in ``src/pipeline``
and ``src/blocking``.  It never materialises a whole table, never enumerates
the query x pool pair space, and never writes a single large pickle.  See
``reports/scalable_architecture.md`` for the full design and the complexity
analysis.

Pipeline shape::

    TSV (read-only)
      -> RecordStore      normalised Parquet shards, one int32 ``rowid`` per record
      -> IndexSet         inverted token / exact-key / sorted-neighbourhood / country
      -> CandidateWriter  per-batch candidate pairs, capped, sharded Parquet

Every stage is bounded by :class:`~src.scale.config.ScaleConfig` budgets, logs
rows / elapsed / candidates / RSS, and is resumable from a manifest.
"""

from __future__ import annotations

from .config import ScaleConfig, default_work_dir
from .progress import MemoryGuard, Progress, format_bytes, log_stage, make_guard

__all__ = [
    "ScaleConfig",
    "default_work_dir",
    "MemoryGuard",
    "Progress",
    "format_bytes",
    "log_stage",
    "make_guard",
]
