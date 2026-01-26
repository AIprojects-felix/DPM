"""Tabular data preprocessing for clinical and laboratory features.

This module provides reference implementations for preprocessing clinical
tabular data used by the DPM model. These examples demonstrate common
preprocessing steps but should be adapted to your specific data format.

Example usage:
    >>> from dpm.preprocessing.tabular import preprocess_clinical_features
    >>> features = preprocess_clinical_features(age=65, psa=7.5, ...)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union
import numpy as np


# Default clinical feature specifications
# These are example features - adapt to your dataset
DEFAULT_CONTINUOUS_FEATURES = [
    "age",              # Patient age in years
    "psa",              # Prostate-specific antigen (ng/mL)
    "psa_density",      # PSA / prostate volume
    "prostate_volume",  # Volume in mL (from imaging)
    "free_psa_ratio",   # Free PSA / Total PSA
    "dre_score",        # Digital rectal exam score (if quantified)
]

DEFAULT_CATEGORICAL_FEATURES = [
    "family_history",   # Binary: prostate cancer in first-degree relatives
    "prior_biopsy",     # Binary: previous negative biopsy
    "race_ethnicity",   # Categorical: encoded as integers
    "smoking_status",   # Categorical: never/former/current
]

# Reference statistics for z-score normalization
# These should be computed from your training set
DEFAULT_FEATURE_STATS = {
    "age": {"mean": 65.0, "std": 10.0},
    "psa": {"mean": 8.0, "std": 15.0},
    "psa_density": {"mean": 0.15, "std": 0.1},
    "prostate_volume": {"mean": 50.0, "std": 30.0},
    "free_psa_ratio": {"mean": 0.15, "std": 0.08},
    "dre_score": {"mean": 1.0, "std": 0.5},
}


def preprocess_clinical_features(
    output_dim: int = 64,
    feature_stats: Optional[Dict] = None,
    **kwargs,
) -> np.ndarray:
    """Preprocess clinical features into a standardized vector.

    This function demonstrates how to transform raw clinical values into
    a fixed-size feature vector suitable for the DPM model.

    Args:
        output_dim: Target dimension for the output feature vector.
        feature_stats: Dict mapping feature names to {"mean", "std"} for
                       z-score normalization. Uses defaults if not provided.
        **kwargs: Clinical feature values as keyword arguments.

    Returns:
        numpy array of shape [output_dim] with standardized features.

    Example:
        >>> features = preprocess_clinical_features(
        ...     age=65,
        ...     psa=7.5,
        ...     prostate_volume=45.0,
        ...     family_history=True,
        ... )
        >>> features.shape
        (64,)

    Notes:
        - Missing values (None or np.nan) are imputed with 0 after normalization
        - Categorical features are one-hot encoded
        - The output is padded/truncated to match output_dim
    """
    stats = feature_stats or DEFAULT_FEATURE_STATS

    features = []

    # Process continuous features
    for feat_name in DEFAULT_CONTINUOUS_FEATURES:
        value = kwargs.get(feat_name, None)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            # Missing value: use 0 (mean after z-score)
            features.append(0.0)
        else:
            # Z-score normalization
            if feat_name in stats:
                mean = stats[feat_name]["mean"]
                std = stats[feat_name]["std"]
                normalized = (float(value) - mean) / (std + 1e-8)
            else:
                normalized = float(value)
            features.append(normalized)

    # Process categorical features (simple encoding)
    for feat_name in DEFAULT_CATEGORICAL_FEATURES:
        value = kwargs.get(feat_name, None)
        if value is None:
            features.append(0.0)
        elif isinstance(value, bool):
            features.append(1.0 if value else 0.0)
        else:
            features.append(float(value))

    # Convert to numpy array
    features = np.array(features, dtype=np.float32)

    # Pad or truncate to output_dim
    if len(features) < output_dim:
        features = np.pad(features, (0, output_dim - len(features)), mode="constant")
    elif len(features) > output_dim:
        features = features[:output_dim]

    return features


def standardize_labs(
    lab_values: Dict[str, float],
    reference_ranges: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """Standardize laboratory values using reference ranges.

    Converts raw lab values to standardized scores where 0 represents the
    middle of the normal range and values outside the range are flagged.

    Args:
        lab_values: Dict mapping lab names to their values.
        reference_ranges: Dict mapping lab names to {"low", "high"} normal ranges.

    Returns:
        Dict with standardized lab values.

    Example:
        >>> labs = standardize_labs({"psa": 4.5, "testosterone": 450})
        >>> labs["psa"]  # Standardized PSA value
    """
    # Default reference ranges (example values - use your institution's ranges)
    default_ranges = {
        "psa": {"low": 0.0, "high": 4.0},
        "testosterone": {"low": 300, "high": 1000},
        "hemoglobin": {"low": 13.5, "high": 17.5},
        "creatinine": {"low": 0.7, "high": 1.3},
    }
    ranges = reference_ranges or default_ranges

    standardized = {}
    for lab_name, value in lab_values.items():
        if lab_name in ranges:
            low = ranges[lab_name]["low"]
            high = ranges[lab_name]["high"]
            midpoint = (low + high) / 2
            half_range = (high - low) / 2
            standardized[lab_name] = (value - midpoint) / (half_range + 1e-8)
        else:
            standardized[lab_name] = value

    return standardized


def impute_missing_values(
    features: np.ndarray,
    method: str = "mean",
    fill_value: float = 0.0,
) -> np.ndarray:
    """Impute missing values (NaN) in feature array.

    Args:
        features: Input feature array with potential NaN values.
        method: Imputation method - "mean", "median", or "constant".
        fill_value: Value to use if method is "constant".

    Returns:
        Feature array with NaN values replaced.
    """
    mask = np.isnan(features)
    if not mask.any():
        return features

    result = features.copy()
    if method == "mean":
        fill = np.nanmean(features)
    elif method == "median":
        fill = np.nanmedian(features)
    else:
        fill = fill_value

    result[mask] = fill
    return result
