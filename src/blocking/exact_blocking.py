"""Strategy A - exact normalized-name blocking.

Candidates are retrieved by exact equality of a normalized name key.  Two keys
are used, and they are reported separately so their individual contribution is
visible:

``exact_normalized``
    equality on the *full* normalized name (legal forms canonicalised);
``exact_core``
    equality on the aggressive "core" name (legal forms removed), which is the
    key that actually links e.g. ``Sharma Sons Pvt Ltd`` to ``Sharma Sons``.

Both are pure dictionary lookups: O(#query tokens), no pairwise comparison.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable

import pandas as pd

from ..config import ID_COLUMN
from ..preprocessing.preprocess import COLS

LOGGER = logging.getLogger(__name__)

#: Minimum key length worth indexing -- avoids a degenerate mega-block of "".
MIN_KEY_LENGTH = 2


def build_name_index(pool: pd.DataFrame, key_column: str) -> dict[str, list[int]]:
    """Invert ``key_column`` of the pool into ``key -> pool positions``."""
    index: dict[str, list[int]] = defaultdict(list)
    values = pool[key_column].fillna("").astype(str).to_numpy()
    for pos, value in enumerate(values):
        if len(value) < MIN_KEY_LENGTH:
            continue
        index[value].append(pos)
    return dict(index)


def exact_name_candidates(
    queries: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    key_column: str = COLS.name,
    max_per_query: int = 200,
    strategy_name: str = "exact_normalized",
) -> list[tuple[int, int, float]]:
    """Return ``(query_pos, pool_pos, score)`` triples from exact key matches.

    The score is a constant ``1.0`` because the key matched exactly; the real
    discrimination between these pairs is left to the feature-based model.
    """
    index = build_name_index(pool, key_column)
    if not index:
        LOGGER.warning("Exact blocking on '%s' produced an empty index", key_column)
        return []
    out: list[tuple[int, int, float]] = []
    keys = queries[key_column].fillna("").astype(str).to_numpy()
    for q_pos, key in enumerate(keys):
        if len(key) < MIN_KEY_LENGTH:
            continue
        positions = index.get(key)
        if not positions:
            continue
        if len(positions) > max_per_query:
            # A pathological block (a very generic name); keep it bounded but
            # log loudly because this directly costs candidate recall.
            LOGGER.warning(
                "Exact block '%s' has %d pool records; truncating to %d", key, len(positions), max_per_query
            )
            positions = positions[:max_per_query]
        for p in positions:
            out.append((q_pos, p, 1.0))
    return out


def exact_block(
    queries: pd.DataFrame, pool: pd.DataFrame, *, max_per_query: int = 200
) -> list[tuple[int, int, float, str]]:
    """Strategy A: exact match on the full and the core normalized name."""
    triples: list[tuple[int, int, float, str]] = []
    triples.extend(
        (q, p, s, "exact_normalized")
        for q, p, s in exact_name_candidates(
            queries, pool, key_column=COLS.name, max_per_query=max_per_query, strategy_name="exact_normalized"
        )
    )
    triples.extend(
        (q, p, s, "exact_core")
        for q, p, s in exact_name_candidates(
            queries, pool, key_column=COLS.name_core, max_per_query=max_per_query, strategy_name="exact_core"
        )
    )
    return triples


def id_column(frame: pd.DataFrame) -> Iterable[str]:
    return frame[ID_COLUMN].fillna("").astype(str)
