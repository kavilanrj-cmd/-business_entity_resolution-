"""Text normalization for names, addresses and countries.

Original columns are always preserved; normalized values are added alongside.
"""

from __future__ import annotations

from .address_normalizer import AddressComponents, extract_components, normalize_address
from .country_normalizer import load_aliases, normalize_country
from .name_normalizer import normalize_name
from .preprocess import COLS, coverage_report, preprocess_dataset, preprocess_table

__all__ = [
    "AddressComponents",
    "COLS",
    "coverage_report",
    "extract_components",
    "load_aliases",
    "normalize_address",
    "normalize_country",
    "normalize_name",
    "preprocess_dataset",
    "preprocess_table",
]
