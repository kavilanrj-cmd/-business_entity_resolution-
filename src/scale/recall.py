"""Blocking recall: did the candidate generator retrieve the verified pairs?

Throughput alone does not prove blocking works -- 100 candidates per query is
worthless if the true match is not among them, and it is trivial to hit that
target on a pool this size.  This module measures the thing that actually
matters:

    recall = |verified pairs retrieved| / |verified pairs|

Reported three ways, because they answer different questions:

``query_recall``
    Fraction of Source 1 rows that retrieved *at least one* verified pool row.
    This is what the classifier ultimately cares about: a Source 1 row with no
    candidate cannot be predicted.
``pair_recall``
    Fraction of individual verified pairs that appear in some query's candidate
    list.  Lower than ``query_recall``, because multi-match Source 1 rows need
    several of their true partners found.
``all_pairs_recall``
    Fraction of Source 1 rows where *every* verified pool row was retrieved.
    A stricter variant that punishes recall lost to the per-query cap.

The ground truth is read in bounded chunks and filtered to the queried rowids
immediately, so measuring recall on a 121 MB / 2.2 M-row labels file costs about
as much as the candidate generation it is checking.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .progress import Progress

LOGGER = logging.getLogger("scale.recall")

#: Column names in the published labels file.
GT_SOURCE_COLUMN = "source1_entity_id"
GT_MATCHED_COLUMN = "matched_entity_ids"


@dataclass
class RecallResult:
    """Blocking recall over a set of Source 1 rowids."""

    queries: int = 0
    queries_labelled: int = 0
    queries_unlabelled: int = 0
    verified_pairs: int = 0
    pairs_retrieved: int = 0
    queries_with_any: int = 0
    queries_with_all: int = 0
    missing_pool_ids: int = 0
    seconds: float = 0.0
    unmatched_examples: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        labelled = self.queries_labelled
        pairs = self.verified_pairs
        return {
            "recall_queries": self.queries,
            "recall_queries_labelled": labelled,
            "recall_queries_unlabelled": self.queries_unlabelled,
            "recall_verified_pairs": pairs,
            "recall_pairs_retrieved": self.pairs_retrieved,
            "recall_query_recall": round(self.queries_with_any / labelled, 4) if labelled else None,
            "recall_pair_recall": round(self.pairs_retrieved / pairs, 4) if pairs else None,
            "recall_all_pairs_recall": round(self.queries_with_all / labelled, 4) if labelled else None,
            "recall_pool_ids_missing": self.missing_pool_ids,
            "recall_seconds": round(self.seconds, 2),
        }

    def describe(self) -> str:
        return (
            f"query_recall={self.as_dict()['recall_query_recall']} "
            f"pair_recall={self.as_dict()['recall_pair_recall']} "
            f"all_pairs_recall={self.as_dict()['recall_all_pairs_recall']} "
            f"({self.queries_with_any}/{self.queries_labelled} rows, "
            f"{self.pairs_retrieved}/{self.verified_pairs} pairs)"
        )


def load_verified_pairs(
    path: str | Path,
    wanted_s1_ids: np.ndarray,
    *,
    chunk_rows: int = 500_000,
) -> dict[str, list[str]]:
    """Return ``{s1_entity_id: [pool_entity_id, ...]}`` for the wanted ids only.

    The labels file is one Source 1 row per line with a comma-separated match
    list, so it cannot be loaded into a DataFrame and sliced -- the split has to
    happen here.  Stopping early once every wanted id is found keeps the common
    "labels sit in file order" case cheap.
    """
    wanted = {str(v) for v in np.asarray(wanted_s1_ids).tolist()}
    if not wanted:
        return {}
    out: dict[str, list[str]] = {}
    header: list[str] | None = None
    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        chunksize=chunk_rows,
    ):
        if header is None:
            header = list(chunk.columns)
            if GT_SOURCE_COLUMN not in header or GT_MATCHED_COLUMN not in header:
                raise KeyError(
                    f"{path} has columns {header}, expected {GT_SOURCE_COLUMN!r} and {GT_MATCHED_COLUMN!r}"
                )
        for key, value in zip(chunk[GT_SOURCE_COLUMN].tolist(), chunk[GT_MATCHED_COLUMN].tolist()):
            if key in wanted:
                out[key] = [v for v in value.split(",") if v]
        if len(out) == len(wanted):
            break
    return out


def measure_recall(
    truth_path: str | Path,
    s1_ids: np.ndarray,
    s1_rowids: np.ndarray,
    retrieved: dict[int, set[int]],
    pool_ids,
    *,
    config,
    examples: int = 5,
    progress: Progress | None = None,
) -> RecallResult:
    """Compare verified pairs with the candidates already retrieved.

    ``retrieved`` maps a Source 1 rowid to the pool rowids its candidate list
    contained.  ``s1_ids``/``s1_rowids`` are parallel arrays of Source 1 entity
    ids and their rowids, so the caller can supply only the rows it examined.
    """
    began = time.perf_counter()
    result = RecallResult(queries=int(np.asarray(s1_rowids).size))
    if result.queries == 0:
        return result
    s1_ids = np.asarray(s1_ids)
    s1_rowids = np.asarray(s1_rowids, dtype=np.int64)
    labels = load_verified_pairs(truth_path, s1_ids)
    if not labels:
        LOGGER.warning("no verified pairs found for the examined Source 1 rows")
        return result

    id_to_row = {str(v): int(r) for v, r in zip(s1_ids.tolist(), s1_rowids.tolist())}
    for key, matches in labels.items():
        rowid = id_to_row.get(key)
        if rowid is None:
            continue
        if not matches:
            # The labels file lists this Source 1 row with no verified partner.
            # That is not a recall failure -- there is nothing to retrieve -- so
            # it is counted and kept out of every denominator.
            result.queries_unlabelled += 1
            continue
        result.queries_labelled += 1
        found = retrieved.get(int(rowid), set())
        # Batch-resolve the whole match list: one resolver call per Source 1 row
        # instead of one per verified pair.
        pool_rowids = np.asarray(pool_ids.resolve(np.array(matches, dtype=object)), dtype=np.int64)
        known = pool_rowids >= 0
        result.missing_pool_ids += int((~known).sum())
        if not known.any():
            if len(result.unmatched_examples) < examples:
                result.unmatched_examples.append(
                    {"s1_id": key, "retrieved": 0, "of": 0, "missing_ids": matches[:5]}
                )
            continue
        total = int(known.sum())
        hits = int(np.isin(pool_rowids[known], np.fromiter(found, dtype=np.int64, count=len(found))).sum())
        result.verified_pairs += total
        result.pairs_retrieved += hits
        if hits:
            result.queries_with_any += 1
        if hits == total:
            result.queries_with_all += 1
        if len(result.unmatched_examples) < examples and hits < total:
            result.unmatched_examples.append(
                {
                    "s1_id": key,
                    "retrieved": hits,
                    "of": total,
                    "missing_ids": [m for m, ok in zip(matches, known.tolist()) if not ok][:5],
                }
            )
        if progress is not None:
            progress.add(1)
    result.seconds = time.perf_counter() - began
    return result
