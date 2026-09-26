"""Data loading, schema validation and ground-truth handling."""

from __future__ import annotations

from .ground_truth import GroundTruth, build_label_lookup, load_ground_truth, parse_ground_truth_frame
from .loader import CANONICAL_COLUMNS, RawDataset, describe_dataframe, load_split, load_test, load_train, read_tsv
from .validator import SchemaError, log_warnings, validate_dataset, validate_table

__all__ = [
    "CANONICAL_COLUMNS",
    "GroundTruth",
    "RawDataset",
    "SchemaError",
    "build_label_lookup",
    "describe_dataframe",
    "load_ground_truth",
    "load_split",
    "load_test",
    "load_train",
    "log_warnings",
    "parse_ground_truth_frame",
    "read_tsv",
    "validate_dataset",
    "validate_table",
]
