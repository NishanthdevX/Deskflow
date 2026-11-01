"""Feature engineering, registry, target alignment, and dataset building."""

from deskflow.features.builder import (
    build_dataset,
    describe_dataset,
    merge_datasets,
    validate_dataset,
)
from deskflow.features.extract import extract_run_features
from deskflow.features.registry import (
    FEATURE_NAME_SET,
    FEATURE_NAMES,
    FEATURE_SET_VERSION,
    SLOW_THRESHOLD,
    FeatureSpec,
)
from deskflow.features.splits import assign_splits

__all__ = [
    "FEATURE_NAME_SET",
    "FEATURE_NAMES",
    "FEATURE_SET_VERSION",
    "SLOW_THRESHOLD",
    "FeatureSpec",
    "assign_splits",
    "build_dataset",
    "describe_dataset",
    "extract_run_features",
    "merge_datasets",
    "validate_dataset",
]
