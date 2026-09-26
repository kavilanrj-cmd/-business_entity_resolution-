"""Country features.

Country is an **open-set** string: no value is special-cased, and no list of
countries is hard-coded.  The features therefore only describe the *relation*
between the two values, never their identity:

``country_exact_match``   both present and equal
``country_conflict``      both present and different
``country_missing``       at least one side is missing
``country_query_missing`` only the Source 1 side is missing
``country_candidate_missing`` only the candidate side is missing

Keeping "missing" separate from "conflicting" matters: a record with no
country at all is weak evidence, whereas two *different* countries are strong
evidence against a match.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

COUNTRY_FEATURES: tuple[str, ...] = (
    "country_exact_match",
    "country_conflict",
    "country_missing",
    "country_query_missing",
    "country_candidate_missing",
)


def country_block(
    query_country: str,
    candidate_countries: Sequence[str],
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Return the country block for one query group.

    ``out`` may be supplied to write into a preallocated matrix.
    """
    n = len(candidate_countries)
    if out is None:
        out = np.zeros((n, len(COUNTRY_FEATURES)), dtype=np.float32)
    else:
        out[:] = 0.0
    if n == 0:
        return out
    q = str(query_country or "")
    q_missing = q == ""
    for i, value in enumerate(candidate_countries):
        c = str(value or "")
        c_missing = c == ""
        both = not q_missing and not c_missing
        out[i] = (
            1.0 if (both and q == c) else 0.0,
            1.0 if (both and q != c) else 0.0,
            1.0 if (q_missing or c_missing) else 0.0,
            1.0 if q_missing else 0.0,
            1.0 if c_missing else 0.0,
        )
    return out
