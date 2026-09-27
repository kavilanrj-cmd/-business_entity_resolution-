"""Data loading, schema validation and ground-truth handling."""

from __future__ import annotations

from .ground_truth import GroundTruth, build_label_lookup, load_ground_truth, parse_ground_truth_frame
from .loader import (
    CANONICAL_COLUMNS,
    DATASET_ROOTS,
    DEFAULT_CHUNK_ROWS,
    RawDataset,
    dataset_root,
    default_test_dir,
    default_train_dir,
    describe_dataframe,
    load_split,
    load_test,
    load_train,
    profile_tsv,
    read_tsv,
    read_tsv_chunks,
    resolve_dataset_root,
    source_path,
    stream_split,
)
from .validator import SchemaError, log_warnings, validate_dataset, validate_table

__all__ = [
    "CANONICAL_COLUMNS",
    "DATASET_ROOTS",
    "DEFAULT_CHUNK_ROWS",
    "GroundTruth",
    "RawDataset",
    "SchemaError",
    "build_label_lookup",
    "dataset_root",
    "default_test_dir",
    "default_train_dir",
    "describe_dataframe",
    "load_ground_truth",
    "load_split",
    "load_test",
    "load_train",
    "log_warnings",
    "parse_ground_truth_frame",
    "profile_tsv",
    "read_tsv",
    "read_tsv_chunks",
    "resolve_dataset_root",
    "source_path",
    "stream_split",
    "validate_dataset",
    "validate_table",
]
