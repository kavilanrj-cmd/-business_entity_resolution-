"""Schema validation for the input tables.

The goal is to fail early, loudly and with an actionable message rather than
producing a confusing error deep inside the pipeline.
"""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from ..config import ADDRESS_COLUMN, COUNTRY_COLUMN, ID_COLUMN, NAME_COLUMN

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (ID_COLUMN, NAME_COLUMN, ADDRESS_COLUMN, COUNTRY_COLUMN)
#: Columns that must be present; country/address may legitimately be absent in
#: some variants of the challenge data, so they are reported as warnings.
OPTIONAL_COLUMNS: tuple[str, ...] = (ADDRESS_COLUMN, COUNTRY_COLUMN)


class SchemaError(ValueError):
    """Raised when an input table cannot be used by the pipeline."""


def validate_table(df: pd.DataFrame, label: str, require_id: bool = True) -> list[str]:
    """Validate one record table; returns a list of human-readable warnings."""
    warnings: list[str] = []
    columns = {str(c).strip() for c in df.columns}
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    critical = [c for c in missing if c == ID_COLUMN or c == NAME_COLUMN] if require_id else []
    critical = [c for c in missing if c not in OPTIONAL_COLUMNS]
    if critical:
        raise SchemaError(
            f"{label}: missing required column(s) {critical}. Present columns: {sorted(columns)}"
        )
    for c in missing:
        warnings.append(f"{label}: optional column '{c}' is absent")

    if ID_COLUMN in columns and require_id:
        ids = df[ID_COLUMN]
        n_null = int(ids.isna().sum())
        if n_null:
            warnings.append(f"{label}: {n_null} rows have a missing {ID_COLUMN}")
        dup_ids = int(ids.dropna().duplicated().sum())
        if dup_ids:
            warnings.append(f"{label}: {dup_ids} duplicated {ID_COLUMN} values (keep first)")
    for col in (NAME_COLUMN, ADDRESS_COLUMN, COUNTRY_COLUMN):
        if col in columns:
            empty = int((df[col].fillna("").astype(str).str.strip() == "").sum())
            if empty:
                warnings.append(f"{label}: {empty} rows have an empty {col}")
    return warnings


def validate_dataset(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Validate every table of a split and check cross-split consistency."""
    warnings: list[str] = []
    for label, df in tables.items():
        warnings.extend(validate_table(df, label))
    s1_ids = set(tables["source1"][ID_COLUMN].dropna().astype(str))
    for source in ("source2", "source3"):
        other = set(tables[source][ID_COLUMN].dropna().astype(str))
        overlap = s1_ids & other
        if overlap:
            warnings.append(
                f"ID collision between source1 and {source}: {len(overlap)} shared ids (e.g. {sorted(overlap)[:3]})"
            )
    return warnings


def log_warnings(warnings: Iterable[str]) -> None:
    for w in warnings:
        LOGGER.warning(w)
