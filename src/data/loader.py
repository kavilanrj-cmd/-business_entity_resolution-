"""Loading the TSV challenge files.

Every file is read with ``pd.read_csv(path, sep="\\t")`` as required by the
challenge specification.  IDs are read as strings so that identifiers such as
``"007"`` survive round-tripping, and no assumption is made about the number of
rows in any file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import ADDRESS_COLUMN, COUNTRY_COLUMN, ID_COLUMN, NAME_COLUMN, SOURCE1, SOURCE2, SOURCE3

LOGGER = logging.getLogger(__name__)

#: Columns the pipeline operates on, in canonical order.
CANONICAL_COLUMNS: tuple[str, ...] = (ID_COLUMN, NAME_COLUMN, ADDRESS_COLUMN, COUNTRY_COLUMN)

#: Column aliases tolerated in the input files, mapped to the canonical name.
COLUMN_ALIASES: dict[str, str] = {
    "id": ID_COLUMN,
    "entityid": ID_COLUMN,
    "entity_id": ID_COLUMN,
    "businessid": ID_COLUMN,
    "business_id": ID_COLUMN,
    "name": NAME_COLUMN,
    "businessname": NAME_COLUMN,
    "business_name": NAME_COLUMN,
    "company_name": NAME_COLUMN,
    "companyname": NAME_COLUMN,
    "address": ADDRESS_COLUMN,
    "businessaddress": ADDRESS_COLUMN,
    "business_address": ADDRESS_COLUMN,
    "addr": ADDRESS_COLUMN,
    "country_code": COUNTRY_COLUMN,
    "countrycode": COUNTRY_COLUMN,
    "country": COUNTRY_COLUMN,
    "nation": COUNTRY_COLUMN,
}

GT_ID_ALIASES: dict[str, str] = {
    "source1_entity_id": "source1_id",
    "s1_id": "source1_id",
    "source_1_id": "source1_id",
    "entity_id": "source1_id",
    "source1": "source1_id",
    "matched_entity_ids": "matched_ids",
    "matched_ids": "matched_ids",
    "matched_entity_id": "matched_ids",
    "match_ids": "matched_ids",
    "matches": "matched_ids",
    "s2_s3_ids": "matched_ids",
    "s2_s3_id": "matched_ids",
}


@dataclass
class RawDataset:
    """The three record tables for a split, already string-typed."""

    source1: pd.DataFrame
    source2: pd.DataFrame
    source3: pd.DataFrame
    split: str

    def table(self, source: str) -> pd.DataFrame:
        return {"source1": self.source1, "source2": self.source2, "source3": self.source3}[source]

    def __iter__(self):
        for source in (SOURCE1, SOURCE2, SOURCE3):
            yield source, self.table(source)


def read_tsv(path: str | Path, name: str | None = None) -> pd.DataFrame:
    """Read a TSV file, coercing the entity-id column to ``str``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Expected layout:\n"
            "  dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv\n"
            "  dataset/test/{test_source1,test_source2,test_source3}.tsv"
        )
    label = name or path.name
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=["", "NA", "N/A", "null", "NULL", "nan", "NaN"])
    df = df.rename(columns=lambda c: COLUMN_ALIASES.get(str(c).strip().lower(), str(c).strip()))
    if ID_COLUMN in df.columns:
        df[ID_COLUMN] = df[ID_COLUMN].astype("string").str.strip()
    for col in (NAME_COLUMN, ADDRESS_COLUMN, COUNTRY_COLUMN):
        if col in df.columns:
            df[col] = df[col].astype("string")
    LOGGER.info("Loaded %-28s rows=%-8d cols=%s", label, len(df), list(df.columns))
    return df


def find_file(directory: str | Path, *patterns: str) -> Path:
    """Locate a file inside ``directory`` by trying a set of name patterns.

    Falls back to a fuzzy search (``*source1*.tsv``) so that the loader keeps
    working when the challenge files use slightly different prefixes.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    for pattern in patterns:
        candidate = directory / pattern
        if candidate.exists():
            return candidate
    for pattern in patterns:
        wildcard = "*" + pattern
        matches = sorted(directory.glob(wildcard))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"None of {list(patterns)} found in {directory}. Available: "
        f"{[p.name for p in sorted(directory.glob('*'))]}"
    )


def load_split(root: str | Path, split: str) -> RawDataset:
    """Load the three record tables for ``split`` (``"train"`` or ``"test"``)."""
    directory = Path(root)
    tables = {}
    for source in (SOURCE1, SOURCE2, SOURCE3):
        path = find_file(
            directory,
            f"{split}_{source}.tsv",
            f"{source}.tsv",
            f"{split}{source}.tsv",
        )
        tables[source] = read_tsv(path, name=f"{split}/{path.name}")
    return RawDataset(split=split, **tables)


def load_train(train_dir: str | Path) -> RawDataset:
    """Load the training split."""
    return load_split(train_dir, "train")


def load_test(test_dir: str | Path) -> RawDataset:
    """Load the test split."""
    return load_split(test_dir, "test")


def describe_dataframe(df: pd.DataFrame, label: str) -> dict[str, object]:
    """Collect shape / dtype / null / duplicate statistics for one table."""
    stats: dict[str, object] = {
        "label": label,
        "shape": tuple(int(x) for x in df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
        "null_fraction": {
            c: (float(df[c].isna().mean()) if len(df) else 0.0) for c in df.columns
        },
    }
    if ID_COLUMN in df.columns:
        ids = df[ID_COLUMN]
        stats["id_nulls"] = int(ids.isna().sum())
        stats["id_duplicates"] = int(ids.duplicated(keep=False).sum())
        stats["id_unique"] = int(ids.nunique(dropna=True))
    for col in (NAME_COLUMN, ADDRESS_COLUMN, COUNTRY_COLUMN):
        if col in df.columns:
            s = df[col].dropna().astype(str)
            stats[f"{col}_empty_string"] = int((s.str.strip() == "").sum())
            stats[f"{col}_unique"] = int(s.nunique())
            if len(s):
                stats[f"{col}_len_mean"] = round(float(s.str.len().mean()), 2)
                stats[f"{col}_len_median"] = float(s.str.len().median())
    if SOURCE1 not in label and ID_COLUMN in df.columns:
        # For noisy sources the same business legitimately appears twice.
        pass
    else:
        for col in (NAME_COLUMN, ADDRESS_COLUMN):
            if col in df.columns:
                stats[f"{col}_duplicate_rows"] = int(df.duplicated(subset=[ID_COLUMN, col], keep=False).sum())
    return stats
