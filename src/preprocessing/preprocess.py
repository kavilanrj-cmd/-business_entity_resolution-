"""Preprocessing entry point: build the normalized record table.

``preprocess_table`` is the single place where normalized columns are created.
Original columns are never modified or dropped, so every downstream feature
can be computed from either view.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from ..config import ADDRESS_COLUMN, COUNTRY_COLUMN, ID_COLUMN, NAME_COLUMN, PreprocessConfig
from .address_normalizer import add_normalized_address_columns
from .country_normalizer import add_normalized_country_column
from .name_normalizer import add_normalized_name_columns

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalizedColumns:
    """Names of the columns produced by :func:`preprocess_table`."""

    entity_id: str = ID_COLUMN
    name: str = f"{NAME_COLUMN}_normalized"
    name_core: str = f"{NAME_COLUMN}_core"
    address: str = f"{ADDRESS_COLUMN}_normalized"
    address_core: str = f"{ADDRESS_COLUMN}_core"
    country: str = f"{COUNTRY_COLUMN}_normalized"
    pincode: str = f"{ADDRESS_COLUMN}_pincode"
    house_number: str = f"{ADDRESS_COLUMN}_house_number"
    city: str = f"{ADDRESS_COLUMN}_city"
    state: str = f"{ADDRESS_COLUMN}_state"


COLS = NormalizedColumns()


def preprocess_table(df: pd.DataFrame, config: PreprocessConfig | None = None) -> pd.DataFrame:
    """Return a copy of ``df`` with the normalized feature columns added."""
    config = config or PreprocessConfig()
    if ID_COLUMN not in df.columns:
        raise ValueError(f"Cannot preprocess: column '{ID_COLUMN}' is missing. Columns={list(df.columns)}")
    out = df.copy()
    out[NAME_COLUMN] = out[NAME_COLUMN].fillna("").astype(str)
    out = add_normalized_name_columns(out, NAME_COLUMN)
    if ADDRESS_COLUMN in out.columns:
        out[ADDRESS_COLUMN] = out[ADDRESS_COLUMN].fillna("").astype(str)
        out = add_normalized_address_columns(out, ADDRESS_COLUMN)
    else:  # address is optional
        out[COLS.address] = ""
        out[COLS.address_core] = ""
        for col in (COLS.pincode, COLS.house_number, COLS.city, COLS.state):
            out[col] = ""
    if COUNTRY_COLUMN in out.columns:
        out[COUNTRY_COLUMN] = out[COUNTRY_COLUMN].fillna("").astype(str)
        out = add_normalized_country_column(out, COUNTRY_COLUMN)
    else:
        out[COLS.country] = ""
    if not config.extract_address_components:
        for col in (COLS.pincode, COLS.house_number, COLS.city, COLS.state):
            out[col] = ""
    return out


def preprocess_dataset(tables: dict[str, pd.DataFrame], config: PreprocessConfig | None = None) -> dict[str, pd.DataFrame]:
    """Preprocess every table of a split."""
    return {label: preprocess_table(df, config) for label, df in tables.items()}


def coverage_report(df: pd.DataFrame) -> dict[str, float]:
    """Fraction of non-empty values per normalized column (sanity diagnostic)."""
    report: dict[str, float] = {}
    for col in (COLS.name, COLS.name_core, COLS.address, COLS.address_core, COLS.country,
                COLS.pincode, COLS.house_number, COLS.city, COLS.state):
        if col in df.columns:
            series = df[col].fillna("").astype(str).str.strip()
            report[col] = round(float((series != "").mean()) if len(series) else 0.0, 4)
    return report
