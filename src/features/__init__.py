"""Pair-level feature engineering for entity resolution."""

from __future__ import annotations

from .address_features import ADDRESS_FEATURES, AddressFeatureBuilder
from .country_features import COUNTRY_FEATURES, country_block
from .feature_builder import (
    COMBINED_FEATURES,
    FeatureBuilder,
    FeatureMatrix,
    build_features,
    feature_names,
    label_from_ground_truth,
)
from .name_features import NAME_FEATURES, NameFeatureBuilder

__all__ = [
    "ADDRESS_FEATURES",
    "COUNTRY_FEATURES",
    "COMBINED_FEATURES",
    "NAME_FEATURES",
    "AddressFeatureBuilder",
    "FeatureBuilder",
    "FeatureMatrix",
    "NameFeatureBuilder",
    "build_features",
    "country_block",
    "feature_names",
    "label_from_ground_truth",
]
