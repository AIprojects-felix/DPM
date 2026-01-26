"""MRI preprocessing utilities for prostate imaging.

This module provides reference implementations for preprocessing MRI volumes
used by the DPM model. These examples demonstrate common preprocessing steps
for multiparametric prostate MRI (mpMRI).

Required dependencies (install as needed):
    pip install nibabel SimpleITK

Example usage:
    >>> from dpm.preprocessing.mri import preprocess_mri_volume
    >>> volume = preprocess_mri_volume("/path/to/t2w.nii.gz")
"""
from __future__ import annotations

from typing import Optional, Tuple, Union
from pathlib import Path
import numpy as np


def preprocess_mri_volume(
    input_path: Union[str, Path],
    target_size: Tuple[int, int, int] = (64, 128, 128),
    normalize: bool = True,
    clip_percentile: Tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """Preprocess an MRI volume for the DPM model.

    This function loads a NIfTI file, resamples to the target size,
    and applies intensity normalization.

    Args:
        input_path: Path to the input NIfTI file (.nii or .nii.gz).
        target_size: Target volume dimensions (D, H, W).
        normalize: Whether to apply z-score normalization.
        clip_percentile: Percentile range for intensity clipping.

    Returns:
        numpy array of shape [1, D, H, W] (single channel).

    Example:
        >>> volume = preprocess_mri_volume(
        ...     "/path/to/t2w.nii.gz",
        ...     target_size=(64, 128, 128),
        ... )
        >>> volume.shape
        (1, 64, 128, 128)

    Notes:
        - Requires nibabel for NIfTI file reading
        - For production use, consider adding bias field correction
        - Registration to a template may improve consistency
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "nibabel is required for MRI preprocessing. "
            "Install with: pip install nibabel"
        )

    # Load NIfTI file
    img = nib.load(str(input_path))
    data = img.get_fdata().astype(np.float32)

    # Clip intensity outliers
    if clip_percentile is not None:
        low, high = np.percentile(data, clip_percentile)
        data = np.clip(data, low, high)

    # Resample to target size
    data = resample_volume(data, target_size)

    # Normalize intensity
    if normalize:
        data = normalize_intensity(data)

    # Add channel dimension [1, D, H, W]
    data = data[np.newaxis, ...]

    return data


def normalize_intensity(
    volume: np.ndarray,
    method: str = "zscore",
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normalize MRI intensity values.

    Args:
        volume: Input 3D volume.
        method: Normalization method - "zscore", "minmax", or "percentile".
        mask: Optional binary mask for computing statistics.

    Returns:
        Normalized volume.

    Methods:
        - zscore: (x - mean) / std, zero mean and unit variance
        - minmax: Scale to [0, 1] range
        - percentile: Scale using 1st and 99th percentiles
    """
    if mask is not None:
        values = volume[mask > 0]
    else:
        values = volume.flatten()

    if method == "zscore":
        mean = np.mean(values)
        std = np.std(values) + 1e-8
        normalized = (volume - mean) / std
    elif method == "minmax":
        vmin, vmax = np.min(values), np.max(values)
        normalized = (volume - vmin) / (vmax - vmin + 1e-8)
    elif method == "percentile":
        p1, p99 = np.percentile(values, [1, 99])
        normalized = (volume - p1) / (p99 - p1 + 1e-8)
        normalized = np.clip(normalized, 0, 1)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized.astype(np.float32)


def resample_volume(
    volume: np.ndarray,
    target_size: Tuple[int, int, int],
    order: int = 1,
) -> np.ndarray:
    """Resample a 3D volume to target size using interpolation.

    Args:
        volume: Input 3D volume.
        target_size: Target dimensions (D, H, W).
        order: Interpolation order (0=nearest, 1=linear, 3=cubic).

    Returns:
        Resampled volume.
    """
    try:
        from scipy.ndimage import zoom
    except ImportError:
        raise ImportError(
            "scipy is required for volume resampling. "
            "Install with: pip install scipy"
        )

    current_size = volume.shape
    zoom_factors = [t / c for t, c in zip(target_size, current_size)]
    resampled = zoom(volume, zoom_factors, order=order)

    return resampled.astype(np.float32)


def apply_bias_field_correction(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Apply N4 bias field correction to an MRI volume.

    This is optional but can improve consistency across different scanners.
    Requires SimpleITK.

    Args:
        input_path: Path to input NIfTI file.
        output_path: Optional path to save corrected volume.

    Returns:
        Bias-corrected volume as numpy array.
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError(
            "SimpleITK is required for bias field correction. "
            "Install with: pip install SimpleITK"
        )

    # Read image
    image = sitk.ReadImage(str(input_path), sitk.sitkFloat32)

    # Create a mask (simple thresholding)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)

    # N4 bias field correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(image, mask)

    # Save if output path provided
    if output_path is not None:
        sitk.WriteImage(corrected, str(output_path))

    # Convert to numpy
    return sitk.GetArrayFromImage(corrected)


def crop_to_prostate(
    volume: np.ndarray,
    mask: np.ndarray,
    margin: int = 10,
    target_size: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """Crop volume to prostate region using a segmentation mask.

    Args:
        volume: Input 3D volume.
        mask: Binary prostate segmentation mask.
        margin: Pixels to add around the bounding box.
        target_size: Optional target size for the cropped region.

    Returns:
        Cropped (and optionally resized) volume.
    """
    # Find bounding box from mask
    nonzero = np.argwhere(mask > 0)
    if len(nonzero) == 0:
        # No mask - return center crop
        return volume

    mins = nonzero.min(axis=0)
    maxs = nonzero.max(axis=0)

    # Add margin
    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, volume.shape)

    # Crop
    cropped = volume[
        mins[0]:maxs[0],
        mins[1]:maxs[1],
        mins[2]:maxs[2],
    ]

    # Resample to target size if specified
    if target_size is not None:
        cropped = resample_volume(cropped, target_size)

    return cropped
