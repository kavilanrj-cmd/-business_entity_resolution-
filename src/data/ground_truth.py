"""Ground-truth parsing and pair labelling.

The challenge ships ``train_ground_truth.tsv``.  Depending on the release it
is either *wide* (one row per Source 1 entity, comma-separated match ids) or
*long* (one row per true pair).  Both layouts -- and a labelled variant -- are
supported here, because the parser must not be tied to one file revision.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from ..config import ID_COLUMN, SOURCE2, SOURCE3
from .loader import GT_ID_ALIASES, find_file, read_tsv

LOGGER = logging.getLogger(__name__)

#: Characters historically used to separate multiple matched ids in one cell.
_ID_SPLIT_RE = r"[;,|\s]+"

#: Column names that may carry a target column in a labelled variant.
_LABEL_COLUMNS = ("is_match", "label", "match", "target", "y")


class GroundTruth:
    """Set-valued view of the ground truth, keyed by Source 1 entity id."""

    def __init__(self, matches: Mapping[str, frozenset[str]], source1_ids: Sequence[str] | None = None):
        self._matches: dict[str, frozenset[str]] = {k: frozenset(v) for k, v in matches.items()}
        self._source1_ids: list[str] = list(source1_ids) if source1_ids is not None else sorted(self._matches)

    # -- container protocol ------------------------------------------------
    def __len__(self) -> int:
        return len(self._source1_ids)

    def __iter__(self):
        return iter(self._source1_ids)

    def __contains__(self, s1_id: object) -> bool:
        return str(s1_id) in self._source1_ids

    # -- lookups -----------------------------------------------------------
    def get(self, s1_id: str, default: frozenset[str] = frozenset()) -> frozenset[str]:
        return self._matches.get(str(s1_id), default)

    @property
    def source1_ids(self) -> list[str]:
        return list(self._source1_ids)

    @property
    def matches(self) -> dict[str, frozenset[str]]:
        return dict(self._matches)

    def as_frame(self) -> pd.DataFrame:
        rows = [
            {"source1_id": s1, "matched_entity_ids": ",".join(sorted(self._matches.get(s1, ())))}
            for s1 in self._source1_ids
        ]
        return pd.DataFrame(rows, columns=["source1_id", "matched_entity_ids"])

    def restrict_to(self, s1_ids: Iterable[str]) -> "GroundTruth":
        """Sub-view limited to a set of Source 1 entities (e.g. a fold)."""
        keep = list(dict.fromkeys(str(x) for x in s1_ids))
        return GroundTruth({k: v for k, v in self._matches.items() if k in set(keep)}, keep)

    def counts_by_cardinality(self) -> dict[str, int]:
        """Histogram of #matches per Source 1 entity (0, 1, 2, ...)."""
        hist: dict[str, int] = defaultdict(int)
        for s1 in self._source1_ids:
            hist[str(len(self._matches.get(s1, ())))] += 1
        return dict(sorted(hist.items(), key=lambda kv: int(kv[0])))

    def entity_summary(self) -> pd.DataFrame:
        """One row per Source 1 entity with its match count."""
        return pd.DataFrame(
            {"source1_id": self._source1_ids, "n_matches": [len(self.get(s)) for s in self._source1_ids]}
        )

    def positive_pairs(self) -> set[tuple[str, str]]:
        """All ``(source1_id, matched_id)`` true pairs."""
        return {(s1, m) for s1, ms in self._matches.items() for m in ms}


def _split_ids(cell: object) -> list[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    text = str(cell).strip()
    if not text or text.lower() in {"na", "nan", "none", "[]", "null"}:
        return []
    import re

    return [part for part in re.split(_ID_SPLIT_RE, text) if part]


def parse_ground_truth_frame(df: pd.DataFrame, source1_ids: Sequence[str] | None = None) -> GroundTruth:
    """Parse a raw ground-truth dataframe into a :class:`GroundTruth`."""
    df = df.rename(columns=lambda c: GT_ID_ALIASES.get(str(c).strip().lower(), str(c).strip()))
    cols = {str(c).strip() for c in df.columns}
    if "source1_id" not in cols:
        raise ValueError(
            f"Ground truth must contain a Source 1 id column. Found columns: {sorted(cols)}"
        )
    label_col = next((c for c in _LABEL_COLUMNS if c in cols), None)
    s1_col = "source1_id"
    if label_col is not None and "matched_ids" in cols:
        df = df[df[label_col].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "match", "matched"})]
    elif label_col is not None:
        # Labelled format with one candidate id per row but no explicit
        # "matched_ids" column -> every row *is* a positive example.
        LOGGER.info("Ground truth has a '%s' column but no match list; treating all rows as positive", label_col)
    if "matched_ids" in cols:
        pairs: dict[str, set[str]] = defaultdict(set)
        for s1, cell in zip(df[s1_col], df["matched_ids"]):
            for mid in _split_ids(cell):
                pairs[str(s1)].add(mid)
        gt = GroundTruth(pairs, source1_ids)
    else:
        # Long format: a single matched id per row.
        other = next(
            (c for c in ("matched_entity_id", "source2_id", "source3_id", "target_id", "match_id") if c in cols),
            None,
        )
        if other is None:
            raise ValueError(
                "Ground truth must contain either a 'matched_entity_ids' list column or a single "
                f"matched-id column. Found: {sorted(cols)}"
            )
        pairs = defaultdict(set)
        for s1, mid in zip(df[s1_col], df[other]):
            if pd.isna(mid) or not str(mid).strip():
                continue
            pairs[str(s1)].add(str(mid).strip())
        gt = GroundTruth(pairs, source1_ids)
    if source1_ids is None:
        LOGGER.info(
            "Ground truth: %d Source 1 entities annotated, %d true pairs",
            len(gt),
            sum(len(v) for v in gt.matches.values()),
        )
    return gt


def load_ground_truth(train_dir: str | Path, source1_ids: Sequence[str] | None = None) -> GroundTruth:
    """Locate and parse ``*ground_truth*.tsv`` inside ``train_dir``."""
    path = find_file(
        train_dir,
        "train_ground_truth.tsv",
        "ground_truth.tsv",
        "train_groundtruth.tsv",
        "groundtruth.tsv",
    )
    df = read_tsv(path, name="ground_truth")
    df = df.rename(columns={c: c for c in df.columns})
    return parse_ground_truth_frame(df, source1_ids=source1_ids)


def build_label_lookup(
    gt: GroundTruth,
    source1_ids: Sequence[str],
    valid_match_ids: frozenset[str],
) -> dict[tuple[str, str], int]:
    """Pre-compute ``(s1, match) -> 1`` for every true pair, dropping invalid ids.

    ``valid_match_ids`` is the union of Source 2 and Source 3 ids present in the
    data, so that a ground-truth row pointing at a non-existent record cannot
    silently create an unmatchable label.
    """
    lookup: dict[tuple[str, str], int] = {}
    n_dropped = 0
    for s1 in source1_ids:
        for mid in gt.get(s1):
            if mid in valid_match_ids:
                lookup[(str(s1), mid)] = 1
            else:
                n_dropped += 1
    if n_dropped:
        LOGGER.warning("Dropped %d ground-truth pairs whose target id is absent from Source 2/3", n_dropped)
    return lookup


def match_source_of(match_id: str, s2_ids: frozenset[str], s3_ids: frozenset[str]) -> str:
    """Return :data:`SOURCE2` or :data:`SOURCE3` for a given match id."""
    if match_id in s2_ids:
        return SOURCE2
    if match_id in s3_ids:
        return SOURCE3
    return "unknown"
