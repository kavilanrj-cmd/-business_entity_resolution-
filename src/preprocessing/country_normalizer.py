"""Country normalization for an **open-set** vocabulary.

The test split may contain countries never seen during training, therefore no
country whitelist is hard-coded anywhere in this project.  Normalization is
purely mechanical:

* Unicode NFKC + accent stripping + case folding
* punctuation / decoration removal (``U.S.A.`` -> ``usa``)
* whitespace and separator collapsing (``"  in "`` -> ``"in"``)
* rejection of well-known missing-value sentinels

An optional user-supplied alias file may map *observed* values onto a canonical
form; it is empty by default, which keeps the field fully open-set and
data-driven.
"""

from __future__ import annotations

import functools
import json
import logging
import re
from pathlib import Path

import pandas as pd

from .name_normalizer import strip_accents

LOGGER = logging.getLogger(__name__)

_MISSING_SENTINELS: frozenset[str] = frozenset(
    {"", "na", "n/a", "nan", "none", "null", "-", "--", "?", "unknown", "unspecified", "not available"}
)

_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Loaded from disk when provided; maps a normalized variant -> canonical form.
#: Deliberately empty by default so the system stays open-set.
ALIASES: dict[str, str] = {}


def load_aliases(path: str | Path | None) -> None:
    """Load an optional ``{"raw_value": "canonical"}`` JSON alias map.

    The map is keyed by the *normalized* form of the raw value.  Only values
    that actually occur in the data should be listed by the user; nothing is
    downloaded or inferred.
    """
    global ALIASES
    if path is None:
        ALIASES = {}
        return
    p = Path(path)
    if not p.exists():
        LOGGER.warning("Country alias file not found: %s (continuing without aliases)", p)
        ALIASES = {}
        return
    raw = json.loads(p.read_text(encoding="utf-8"))
    ALIASES = {normalize_country(k): str(v) for k, v in raw.items()}
    LOGGER.info("Loaded %d country aliases from %s", len(ALIASES), p)


@functools.lru_cache(maxsize=1 << 18)
def normalize_country(value: object) -> str:
    """Normalize a country value; returns ``""`` for missing/sentinel values."""
    if value is None:
        return ""
    text = str(value)
    if not text.strip():
        return ""
    import unicodedata

    text = unicodedata.normalize("NFKD", text)
    text = strip_accents(text).lower()
    text = _PUNCT_RE.sub(" ", text)
    tokens = [t for t in _WS_RE.split(text.strip()) if t]
    if not tokens:
        return ""
    joined = " ".join(tokens)
    if joined in _MISSING_SENTINELS:
        return ""
    return ALIASES.get(joined, joined)


def is_missing_country(value: str) -> bool:
    return value == ""


def add_normalized_country_column(df: pd.DataFrame, prefix: str = "country") -> pd.DataFrame:
    """Add ``<prefix>_normalized`` while preserving the original column."""
    src = df[prefix] if prefix in df.columns else pd.Series([""] * len(df), index=df.index)
    out = df.copy()
    out[f"{prefix}_normalized"] = src.map(normalize_country).astype(str)
    return out
