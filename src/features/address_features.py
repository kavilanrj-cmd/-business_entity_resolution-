"""Address-similarity features.

The address block mirrors the name block (text similarity) and then adds the
*component* features the challenge asks for: ``pincode``, ``house_number``,
``city`` and ``state``.

Component features are always emitted as a triple:

``<component>_match``
    Both sides provide the component and the values are equal.
``<component>_both_present``
    Both sides provide it (regardless of agreement) -- the model needs this
    as a separate signal because a *conflict* (same component, different value)
    is strong evidence **against** a match, and it can only be detected when
    both sides are populated.

Because the components come from a geography-agnostic heuristic extractor, the
``_both_present`` gate is what makes them safe: a component that fails to
extract simply drops out instead of manufacturing evidence.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from rapidfuzz import fuzz, process

from .name_features import StringFeatures, _jaccard, _scaled, value_features

LOGGER = logging.getLogger(__name__)

ADDRESS_FEATURES: tuple[str, ...] = (
    "address_exact_match",
    "address_prefix_match",
    "address_jaccard",
    "address_token_overlap",
    "address_levenshtein_similarity",
    "address_token_sort_ratio",
    "address_tfidf_cosine",
    "address_length_difference",
    "address_length_ratio",
    "address_first_token_match",
    "address_last_token_match",
    "address_first_token_jaccard",
    "address_digit_overlap",
    "address_digit_both_present",
    "address_both_missing",
    "pincode_match",
    "pincode_both_present",
    "pincode_conflict",
    "house_number_match",
    "house_number_both_present",
    "house_number_conflict",
    "city_match",
    "city_both_present",
    "city_conflict",
    "state_match",
    "state_both_present",
    "state_conflict",
)

#: Component triples in the order they are appended after the text features.
COMPONENTS: tuple[str, ...] = ("pincode", "house_number", "city", "state")


class AddressFeatureBuilder:
    """Builds the address block of the feature matrix for a group of pairs."""

    def __init__(self, prefix_chars: int = 4) -> None:
        self.prefix_chars = prefix_chars
        self._cache: dict[str, StringFeatures] = {}

    def cache(self, values: Sequence[str]) -> None:
        for value in values:
            if value and value not in self._cache:
                self._cache[value] = value_features(value)

    def get(self, value: str) -> StringFeatures:
        if not value:
            return value_features("")
        cached = self._cache.get(value)
        if cached is None:
            cached = value_features(value)
            self._cache[value] = cached
        return cached

    def block(
        self,
        query_value: str,
        candidate_values: Sequence[str],
        query_components: dict[str, str] | None = None,
        candidate_components: Sequence[dict[str, str]] | None = None,
        tfidf_cosine: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return an ``(n, len(ADDRESS_FEATURES))`` block for one query group."""
        n = len(candidate_values)
        out = np.zeros((n, len(ADDRESS_FEATURES)), dtype=np.float32)
        if n == 0:
            return out
        q = self.get(query_value)
        cands = [self.get(v) for v in candidate_values]
        texts_c = [c.text or "" for c in cands]
        ratio = _scaled([q.text] if q.text else [""], texts_c, fuzz.ratio)
        tsort = _scaled([q.text] if q.text else [""], texts_c, fuzz.token_sort_ratio)
        q_comp = query_components or {c: "" for c in COMPONENTS}
        c_comps = candidate_components or [{} for _ in range(n)]

        for i, c in enumerate(cands):
            inter = q.tokens & c.tokens
            union = q.tokens | c.tokens
            min_len = min(len(q.tokens), len(c.tokens)) or 1
            max_len = max(len(q.tokens), len(c.tokens)) or 1
            first_inter = (
                _jaccard(frozenset([q.first_token]) if q.first_token else frozenset(),
                         frozenset([c.first_token]) if c.first_token else frozenset())
            )
            digit_inter = q.numbers & c.numbers
            digit_union = q.numbers | c.numbers
            row = [
                1.0 if (q.text and q.text == c.text) else 0.0,
                1.0 if (q.text[: self.prefix_chars] and q.text[: self.prefix_chars] == c.text[: self.prefix_chars]) else 0.0,
                len(inter) / len(union) if union else 0.0,
                len(inter) / min_len,
                float(ratio[0, i]),
                float(tsort[0, i]),
                float(tfidf_cosine[i]) if tfidf_cosine is not None else 0.0,
                abs(q.length - c.length),
                1.0 - abs(q.length - c.length) / max(q.length, c.length, 1),
                1.0 if (q.first_token and q.first_token == c.first_token) else 0.0,
                1.0 if (q.last_token and q.last_token == c.last_token) else 0.0,
                first_inter,
                len(digit_inter) / len(digit_union) if digit_union else 0.0,
                1.0 if (q.numbers and c.numbers) else 0.0,
                1.0 if (q.is_empty and c.is_empty) else 0.0,
            ]
            # --- component triples ---------------------------------------
            cc = c_comps[i] or {}
            for comp in COMPONENTS:
                a = str(q_comp.get(comp, "") or "")
                b = str(cc.get(comp, "") or "")
                both = bool(a) and bool(b)
                agree = both and a == b
                row.extend((1.0 if agree else 0.0, 1.0 if both else 0.0, 1.0 if (both and not agree) else 0.0))
            if len(row) != len(ADDRESS_FEATURES):  # pragma: no cover - guard
                raise AssertionError(
                    f"address feature row has {len(row)} values but {len(ADDRESS_FEATURES)} are declared"
                )
            out[i] = row
        return out
