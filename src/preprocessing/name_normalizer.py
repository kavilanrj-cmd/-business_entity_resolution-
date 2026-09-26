"""Business-name normalization.

Two normalized views are produced, and the raw value is always preserved:

``business_name_normalized``
    Canonical, comparable form.  Case folded, transliterated, punctuation
    removed, ``&`` folded to ``and``, whitespace collapsed and legal-form
    words *canonicalised* (e.g. ``Corporation`` -> ``corp``) but **not
    deleted**, so that no potentially distinguishing signal is lost.

``business_name_core``
    Aggressive view used only for blocking/retrieval: legal forms and a small
    generic business filler vocabulary are dropped.  Anything that may carry
    identity (proper nouns, numbers, street/industry words) is retained.
"""

from __future__ import annotations

import functools
import logging
import re
import unicodedata

import pandas as pd

LOGGER = logging.getLogger(__name__)

#: Legal-form words folded onto a single canonical token.  Mapping (not
#: deletion) keeps "Pvt Ltd" and "Private Limited" comparable while retaining
#: the fact that the record carries a legal-form token.
LEGAL_FORMS: dict[str, str] = {
    "corporation": "corp",
    "corp": "corp",
    "incorporated": "inc",
    "inc": "inc",
    "incorporated.": "inc",
    "company": "co",
    "co": "co",
    "ltd": "ltd",
    "limited": "ltd",
    "private": "pvt",
    "pvt": "pvt",
    "pvt.": "pvt",
    "llp": "llp",
    "llc": "llc",
    "plc": "plc",
    "gmbh": "gmbh",
    "ag": "ag",
    "bv": "bv",
    "nv": "nv",
    "sa": "sa",
    "sas": "sas",
    "sarl": "sarl",
    "spa": "spa",
    "pty": "pty",
    "sdn": "sdn",
    "bhd": "bhd",
    "kft": "kft",
    "sp": "sp",
    "sp.": "sp",
    "z.o.o": "zoo",
    "zoo": "zoo",
    "kg": "kg",
    "ohg": "ohg",
    "e.k": "ek",
    "pc": "pc",
    "psc": "psc",
    "lp": "lp",
    "lllp": "lllp",
    "trust": "trust",
    "trustee": "trust",
}

#: Tokens dropped from the ``*_core`` view only.
CORE_STOP_TOKENS: frozenset[str] = frozenset(
    set(LEGAL_FORMS.values()) | {"and", "the", "of", "for", "a", "an", "at", "in", "on"}
)

#: Non-alphanumeric characters that act as token separators.
_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Common transliterations that NFKD alone does not resolve.
_TRANSLIT: dict[str, str] = {
    "&": " and ",
    "＆": " and ",
    "+": " and ",
    "@": " at ",
    "ß": "ss",
    "æ": "ae",
    "œ": "oe",
    "ø": "o",
    "đ": "d",
    "ð": "d",
    "þ": "th",
    "ł": "l",
    "'": "",
    "\u2019": "",
    "\u02bc": "",
}

_TRANSLIT_TABLE = str.maketrans({k: v for k, v in _TRANSLIT.items()})


def strip_accents(text: str) -> str:
    """Remove diacritics via NFKD decomposition."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


@functools.lru_cache(maxsize=1 << 20)
def normalize_name(value: object, *, drop_legal_forms: bool = True, canon_legal_forms: bool = True) -> str:
    """Normalize a single business name.

    Parameters
    ----------
    value:
        Raw value; ``None``/NaN are tolerated.
    drop_legal_forms:
        When ``False`` the ``*_core`` view equals the normalized view.
    canon_legal_forms:
        Fold legal-form spellings onto canonical tokens.
    """
    if value is None:
        return ""
    text = str(value)
    if not text.strip() or text.strip().lower() in {"na", "nan", "none", "null"}:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = strip_accents(text)
    text = text.translate(_TRANSLIT_TABLE)
    text = _PUNCT_RE.sub(" ", text)
    tokens = [t for t in _WS_RE.split(text.strip()) if t]
    if not tokens:
        return ""
    out: list[str] = []
    for tok in tokens:
        if canon_legal_forms and tok in LEGAL_FORMS:
            out.append(LEGAL_FORMS[tok])
        else:
            out.append(tok)
    if not drop_legal_forms:
        return " ".join(out)
    core = [t for t in out if t not in CORE_STOP_TOKENS]
    # Never let the core view become empty when a name is purely a legal form.
    return " ".join(core) if core else " ".join(out)


def name_tokens(value: object) -> list[str]:
    """Token list of the *core* normalized name (used for blocking)."""
    return _TOKEN_RE.findall(normalize_name(value))


def content_tokens(value: object) -> list[str]:
    """Token list of the core view, keeping legal-form tokens (blocking)."""
    return _TOKEN_RE.findall(normalize_name(value, drop_legal_forms=False))


def add_normalized_name_columns(df: pd.DataFrame, prefix: str = "business_name") -> pd.DataFrame:
    """Add ``<prefix>_normalized`` and ``<prefix>_core`` columns in place-safe fashion."""
    src = df[prefix] if prefix in df.columns else pd.Series([""] * len(df), index=df.index)
    normalized = src.map(lambda v: normalize_name(v, drop_legal_forms=False)).astype(str)
    core = src.map(lambda v: normalize_name(v, drop_legal_forms=True)).astype(str)
    out = df.copy()
    out[f"{prefix}_normalized"] = normalized
    out[f"{prefix}_core"] = core
    out[f"{prefix}_tokens"] = core.map(lambda s: " ".join(s.split()))
    return out
