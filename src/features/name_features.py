"""Name-similarity features.

A per-value cache is built once for the *unique* strings in the dataset, so a
value shared by many records is tokenised and n-grammed only once.  Pairwise
work is then done group-by-group (one Source 1 entity at a time) with
``rapidfuzz``'s vectorised ``process.cdist``, which keeps the cost linear in
the number of candidate pairs instead of quadratic in the corpus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from ..preprocessing.preprocess import COLS

LOGGER = logging.getLogger(__name__)

#: Name features, in output order.  Order defines the feature-matrix columns.
NAME_FEATURES: tuple[str, ...] = (
    "name_exact_match",
    "name_core_exact_match",
    "name_raw_exact_match",
    "name_prefix_match",
    "name_common_prefix_ratio",
    "name_jaccard",
    "name_token_overlap",
    "name_token_containment_query",
    "name_token_containment_candidate",
    "name_levenshtein_similarity",
    "name_jaro_winkler",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_tfidf_cosine",
    "name_char_ngram_jaccard",
    "name_length_difference",
    "name_length_ratio",
    "name_first_token_match",
    "name_last_token_match",
    "name_numeric_match",
    "name_numeric_both_present",
    "name_both_missing",
)

NGRAM_N = 3


@dataclass
class StringFeatures:
    """Precomputed, reusable view of one normalized string."""

    text: str
    tokens: frozenset[str]
    token_order: tuple[str, ...]
    length: int
    first_token: str
    last_token: str
    ngrams: frozenset[str]
    numbers: frozenset[str]

    @property
    def is_empty(self) -> bool:
        return self.text == ""


_EMPTY = StringFeatures("", frozenset(), (), 0, "", "", frozenset(), frozenset())


def _numbers(text: str) -> frozenset[str]:
    """Digit runs in the text, used to compare e.g. ``Services 2019``."""
    out: set[str] = set()
    current: list[str] = []
    for ch in text:
        if ch.isdigit():
            current.append(ch)
        elif current:
            out.add("".join(current))
            current = []
    if current:
        out.add("".join(current))
    return frozenset(out)


def value_features(text: str) -> StringFeatures:
    """Tokenise, n-gram and dissect a normalized string (cached by caller)."""
    if not text:
        return _EMPTY
    tokens = tuple(text.split())
    ngrams = frozenset(text[i : i + NGRAM_N] for i in range(max(0, len(text) - NGRAM_N + 1)))
    return StringFeatures(
        text=text,
        tokens=frozenset(tokens),
        token_order=tokens,
        length=len(text),
        first_token=tokens[0] if tokens else "",
        last_token=tokens[-1] if tokens else "",
        ngrams=ngrams,
        numbers=_numbers(text),
    )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _common_prefix(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


class NameFeatureBuilder:
    """Builds the name block of the feature matrix for a group of pairs."""

    def __init__(self, prefix_chars: int = 4, jw_prefix_weight: float = 0.1) -> None:
        self.prefix_chars = prefix_chars
        self.jw_prefix_weight = jw_prefix_weight
        self._cache: dict[str, StringFeatures] = {}

    def cache(self, values: Iterable[str]) -> None:
        """Pre-compute the feature cache for a set of unique values."""
        for value in values:
            if value and value not in self._cache:
                self._cache[value] = value_features(value)

    def get(self, value: str) -> StringFeatures:
        if not value:
            return _EMPTY
        cached = self._cache.get(value)
        if cached is None:
            cached = value_features(value)
            self._cache[value] = cached
        return cached

    def block(
        self,
        query_values: Sequence[str],
        candidate_values: Sequence[str],
        query_core: Sequence[str] | None = None,
        candidate_core: Sequence[str] | None = None,
        query_raw: Sequence[str] | None = None,
        candidate_raw: Sequence[str] | None = None,
        tfidf_cosine: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return an ``(n, len(NAME_FEATURES))`` block for one query group.

        ``*_values`` are already restricted to this group's candidates.
        """
        n = len(candidate_values)
        out = np.zeros((n, len(NAME_FEATURES)), dtype=np.float32)
        if n == 0:
            return out
        q = self.get(query_values[0])
        q_core = self.get(query_core[0]) if query_core is not None else q
        q_raw = self.get(query_raw[0]) if query_raw is not None else q
        cands = [self.get(v) for v in candidate_values]
        cand_cores = [self.get(v) for v in candidate_core] if candidate_core is not None else cands
        cand_raws = [self.get(v) for v in candidate_raw] if candidate_raw is not None else cands

        texts_q = [q.text] if q.text else []
        texts_c = [c.text if c.text else "" for c in cands]
        # rapidfuzz ratio-style scorers return 0..100; everything in this module
        # is expressed on a 0..1 scale so the columns are mutually comparable.
        ratio = _scaled(texts_q, texts_c, fuzz.ratio)
        jw = _cdist(texts_q, texts_c, JaroWinkler.normalized_similarity,
                    scorer_kwargs={"prefix_weight": self.jw_prefix_weight})
        tsort = _scaled(texts_q, texts_c, fuzz.token_sort_ratio)
        tset = _scaled(texts_q, texts_c, fuzz.token_set_ratio)

        for i, c in enumerate(cands):
            cc = cand_cores[i]
            cr = cand_raws[i]
            both_missing = 1.0 if (q.is_empty and c.is_empty) else 0.0
            inter = q.tokens & c.tokens
            union = q.tokens | c.tokens
            min_len = min(len(q.tokens), len(c.tokens)) or 1
            max_len = max(len(q.tokens), len(c.tokens)) or 1
            prefix_len = _common_prefix(q.text, c.text)
            max_char_len = max(q.length, c.length, 1)
            nums_match = 1.0 if (q.numbers and c.numbers and (q.numbers & c.numbers)) else 0.0
            nums_both = 1.0 if (q.numbers and c.numbers) else 0.0
            out[i] = (
                1.0 if (q.text and q.text == c.text) else 0.0,
                1.0 if (q_core.text and q_core.text == cc.text) else 0.0,
                1.0 if (q_raw.text and q_raw.text == cr.text) else 0.0,
                1.0 if (q.text[: self.prefix_chars] and q.text[: self.prefix_chars] == c.text[: self.prefix_chars]) else 0.0,
                prefix_len / max_char_len,
                len(inter) / len(union) if union else 0.0,
                len(inter) / min_len,
                len(inter) / (len(q.tokens) or 1),
                len(inter) / (len(c.tokens) or 1),
                float(ratio[0, i]),
                float(jw[0, i]),
                float(tsort[0, i]),
                float(tset[0, i]),
                float(tfidf_cosine[i]) if tfidf_cosine is not None else 0.0,
                _jaccard(q.ngrams, c.ngrams),
                abs(q.length - c.length),
                1.0 - abs(q.length - c.length) / max(q.length, c.length, 1),
                1.0 if (q.first_token and q.first_token == c.first_token) else 0.0,
                1.0 if (q.last_token and q.last_token == c.last_token) else 0.0,
                nums_match,
                nums_both,
                both_missing,
            )
        if out.shape[1] != len(NAME_FEATURES):  # pragma: no cover - guard
            raise AssertionError(
                f"name feature row has {out.shape[1]} values but {len(NAME_FEATURES)} are declared"
            )
        return out


def _cdist(queries: list[str], choices: list[str], scorer, **kwargs) -> np.ndarray:
    """``rapidfuzz`` vectorised pairwise scoring, tolerant of empty inputs.

    ``scorer_kwargs`` is forwarded separately because that is how rapidfuzz
    passes extra arguments (e.g. ``prefix_weight``) to a distance scorer while
    still using the compiled C++ implementation.
    """
    if not queries or not choices:
        return np.zeros((len(queries), len(choices)), dtype=np.float32)
    if not any(queries) or not any(choices):
        return np.zeros((len(queries), len(choices)), dtype=np.float32)
    scorer_kwargs = kwargs.pop("scorer_kwargs", None)
    return process.cdist(
        queries, choices, scorer=scorer, workers=1, dtype=np.float32, scorer_kwargs=scorer_kwargs, **kwargs
    )


def _scaled(queries: list[str], choices: list[str], scorer) -> np.ndarray:
    """Pairwise scoring rescaled from rapidfuzz's 0..100 to 0..1."""
    return _cdist(queries, choices, scorer) / 100.0
