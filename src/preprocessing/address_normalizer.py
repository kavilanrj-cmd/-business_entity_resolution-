"""Business-address normalization plus component extraction.

Produces ``business_address_normalized`` / ``business_address_core`` and,
where they can be extracted with reasonable confidence, the components
``pincode``, ``house_number``, ``city`` and ``state``.

Everything here is heuristic and *geography agnostic*: no country-specific
address grammar is assumed, and no external gazetteer is consulted.  Each
component carries a presence flag so downstream features are only used when
**both** sides of a pair actually provide the component -- a missing or
mis-extracted component can then never manufacture evidence of a match.
"""

from __future__ import annotations

import functools
import logging
import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

from .name_normalizer import strip_accents

LOGGER = logging.getLogger(__name__)

#: Address words are canonicalised to their **short** form, never expanded.
#:
#: Canonicalising to the abbreviation (rather than the expansion) is what makes
#: this safe: a 2-letter token is left untouched, so a state code such as
#: ``FL`` or ``IN`` survives as ``fl``/``in`` instead of being mangled into
#: ``floor``/``number``.  No geography-specific list is involved.
ADDRESS_ABBREVIATIONS: dict[str, str] = {
    # long form -> canonical short form
    "road": "rd",
    "street": "st",
    "avenue": "av",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "square": "sq",
    "place": "pl",
    "plaza": "plz",
    "apartment": "apt",
    "floor": "fl",
    "building": "bldg",
    "district": "dist",
    "section": "sec",
    "society": "soc",
    "housing": "hsg",
    "group": "grp",
    "opposite": "opp",
    "number": "no",
    "numbers": "nos",
    "junction": "jn",
    "cross": "x",
    "market": "mkt",
    "postoffice": "po",
    "apart": "apt",
    "first": "1st",
    "second": "2nd",
    "third": "3rd",
    "fourth": "4th",
    "fifth": "5th",
}

#: Markers that introduce a house / building number.  Single letters are
#: deliberately excluded: "a" / "c" appear inside "Tower A" / "Block C" far
#: too often to be treated as a numbering marker.
_NUMBER_MARKERS: tuple[str, ...] = (
    "no", "nos", "nro", "num", "shop", "shopno", "unit", "flat", "apt", "fl", "floor",
    "sco", "plot", "house", "hs", "bldg", "block", "blk", "door", "stall", "off",
    "office", "premises", "site", "grd", "ground", "1st", "2nd", "3rd", "4th", "5th", "#",
)

#: Punctuation is replaced by a space, but **commas are preserved** because the
#: comma-delimited structure of an address carries the city/state/postcode
#: ordering that the component extractor relies on.
_PUNCT_RE = re.compile(r"[^\w\s,#]+", flags=re.UNICODE)
_COMMA_RE = re.compile(r"\s*,\s*")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Generic address noise: tokens that never identify a location.
NOISE_TOKENS: frozenset[str] = frozenset(
    {"and", "the", "of", "at", "in", "near", "nr", "opposite", "opp", "behind", "beside",
     "next", "to", "from", "via", "by", "nearby"}
)

#: Digit runs long enough to plausibly be a postal code.
_PIN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(\d{3}-\d{3})\b"),          # 123-456 (ISO style)
    re.compile(r"\b(\d{2,3}\s?\d{3}\s?\d{3})\b"),  # SWIFT-ish grouped codes
    re.compile(r"\b(\d{4,10})\b"),              # plain 4..10 digit code
)

#: ``... City, ST 12345`` / ``... City, ST, 12345`` (US style, generic 2-3 letter code)
_US_STATE_RE = re.compile(r"(?:^|[,]|\s)([A-Za-z]{2,3})\s+(\d{4,10})\s*$")

#: A segment that is nothing but a postal code.
_PURE_CODE_RE = re.compile(r"\d{3,10}")

#: Postal code at the very end of the final segment.
_TAIL_CODE_RE = re.compile(r"(\d{3,10})\s*$")
_TAIL_CODE_DASHED_RE = re.compile(r"(\d{3}-\d{3})\s*$")


@dataclass(frozen=True)
class AddressComponents:
    """Extracted address parts, each independently optional."""

    pincode: str = ""
    house_number: str = ""
    city: str = ""
    state: str = ""

    @property
    def has_pincode(self) -> bool:
        return bool(self.pincode)

    @property
    def has_house_number(self) -> bool:
        return bool(self.house_number)

    @property
    def has_city(self) -> bool:
        return bool(self.city)

    @property
    def has_state(self) -> bool:
        return bool(self.state)

    def as_dict(self) -> dict[str, str]:
        return {
            "pincode": self.pincode,
            "house_number": self.house_number,
            "city": self.city,
            "state": self.state,
        }


@functools.lru_cache(maxsize=1 << 20)
def normalize_address(value: object, *, expand_abbreviations: bool = True) -> str:
    """Canonical text form of an address.

    Commas survive as standalone ``","`` tokens so that the component extractor
    can reason about the ``..., City, State, Postcode`` layout; text
    comparisons ignore them because :func:`address_tokens` filters them out.
    """
    if value is None:
        return ""
    text = str(value)
    if not text.strip() or text.strip().lower() in {"na", "nan", "none", "null"}:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = strip_accents(text).lower()
    text = text.replace("#", " # ")
    text = _PUNCT_RE.sub(" ", text)
    text = _COMMA_RE.sub(" , ", text)
    tokens: list[str] = []
    for tok in _WS_RE.split(text.strip()):
        if not tok:
            continue
        if tok == ",":
            if tokens and tokens[-1] != ",":
                tokens.append(",")
            continue
        tokens.append(ADDRESS_ABBREVIATIONS.get(tok, tok) if expand_abbreviations else tok)
    while tokens and tokens[0] == ",":
        tokens.pop(0)
    while tokens and tokens[-1] == ",":
        tokens.pop()
    return " ".join(tokens)


def address_tokens(value: object) -> list[str]:
    """Token list of the normalized address."""
    return _TOKEN_RE.findall(normalize_address(value))


@functools.lru_cache(maxsize=1 << 20)
def extract_components(value: object) -> AddressComponents:
    """Best-effort extraction of pincode / house number / city / state.

    The heuristic follows the dominant ``..., City, State, Postcode`` layout and
    the ``City, ST 12345`` layout, but it degrades honestly: anything that
    cannot be located with confidence is returned empty, and the corresponding
    feature is then simply not used.  No country-specific grammar is assumed.
    """
    raw = "" if value is None else str(value)
    if not raw.strip():
        return AddressComponents()
    norm = normalize_address(raw)
    if not norm:
        return AddressComponents()

    segments = [seg.strip() for seg in norm.split(" , ")]
    segments = [seg for seg in segments if seg]

    # --- pincode ---------------------------------------------------------
    # The postal code is the trailing digit run of the **last** segment.  Only
    # looking at the tail is what keeps a 4-digit house number ("9462 Temple
    # Street") from being mistaken for a postal code.
    pincode = ""
    if segments:
        last = segments[-1]
        m = _TAIL_CODE_DASHED_RE.search(last) or _TAIL_CODE_RE.search(last)
        if m:
            pincode = re.sub(r"\s+", "", m.group(1))
            segments[-1] = last[: m.start()].strip()
    if not pincode:
        m = _US_STATE_RE.search(norm)
        if m:
            pincode = m.group(2)
    if not pincode:
        # Only trust a code found mid-string when the whole address is a single
        # free-form segment; otherwise a bare house number would be misread.
        single_segment = len([s for s in segments if s]) <= 1
        if single_segment:
            for pattern in _PIN_PATTERNS:
                matches = pattern.findall(norm)
                if matches:
                    pincode = re.sub(r"\s+", "", matches[-1])
                    break

    # Segments that consist only of the postal code carry no other information.
    content_segments = [
        seg for seg in segments
        if seg and not (pincode and re.sub(r"\s+", "", seg) == pincode)
    ]

    # --- state / city ----------------------------------------------------
    state = city = ""
    if len(content_segments) >= 3:
        tail = content_segments[-1]
        tail_tokens = tail.split()
        # A short, digit-free trailing segment is treated as a region when
        # there is a segment before it that can serve as the city.
        if 1 <= len(tail_tokens) <= 3 and len(tail) <= 32 and not any(ch.isdigit() for ch in tail):
            state = tail
            city = content_segments[-2]
        else:
            city = content_segments[-1]
    elif len(content_segments) == 2:
        city = content_segments[0]
    elif len(content_segments) == 1:
        # A single segment is a free-form address: only a trailing "ST 12345"
        # pattern is trustworthy, so the city is left empty on purpose.
        m = _US_STATE_RE.search(content_segments[0])
        if m:
            state = m.group(1)

    # --- house number ----------------------------------------------------
    house_number = ""
    if content_segments:
        first_tokens = content_segments[0].split()
        for i, tok in enumerate(first_tokens):
            if tok in _NUMBER_MARKERS:
                for nxt in first_tokens[i + 1 : i + 3]:
                    if any(ch.isdigit() for ch in nxt) and re.sub(r"\s+", "", nxt) != pincode:
                        house_number = nxt
                        break
                if house_number:
                    break
        if not house_number and first_tokens:
            tok = first_tokens[0]
            if any(ch.isdigit() for ch in tok) and len(tok) <= 8 and re.sub(r"\s+", "", tok) != pincode:
                house_number = tok

    def _clean(part: str) -> str:
        part = " ".join(t for t in part.split() if t not in NOISE_TOKENS)
        return "" if part == pincode else part

    city, state, house_number = _clean(city), _clean(state), _clean(house_number)
    if city == state:
        city = ""
    if house_number and house_number in {city, state}:
        house_number = ""
    return AddressComponents(
        pincode=pincode, house_number=house_number, city=city, state=state
    )


def add_normalized_address_columns(df: pd.DataFrame, prefix: str = "business_address") -> pd.DataFrame:
    """Add normalized text plus the four extracted components."""
    src = df[prefix] if prefix in df.columns else pd.Series([""] * len(df), index=df.index)
    out = df.copy()
    out[f"{prefix}_normalized"] = src.map(normalize_address).astype(str)
    out[f"{prefix}_core"] = src.map(lambda v: " ".join(t for t in address_tokens(v) if t not in NOISE_TOKENS)).astype(str)
    comps = src.map(extract_components)
    for field_name in ("pincode", "house_number", "city", "state"):
        out[f"{prefix}_{field_name}"] = [getattr(c, field_name) for c in comps]
    return out
