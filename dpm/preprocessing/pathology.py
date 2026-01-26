"""Pathology image preprocessing for whole slide images (WSI).

This module provides reference implementations for preprocessing histopathology
images used by the DPM model. These examples demonstrate common preprocessing
steps for whole slide images.

Required dependencies (install as needed):
    pip install openslide-python pillow

System dependency:
    OpenSlide library must be installed on your system.
    Ubuntu: apt-get install openslide-tools
    macOS: brew install openslide

Example usage:
    >>> from dpm.preprocessing.pathology import preprocess_wsi
    >>> patches = preprocess_wsi("/path/to/slide.svs")
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union
from pathlib import Path
import numpy as np


def preprocess_wsi(
    wsi_path: Union[str, Path],
    patch_size: int = 224,
    target_magnification: float = 20.0,
    max_patches: int = 100,
    aggregate: str = "mean",
) -> np.ndarray:
    """Preprocess a whole slide image for the DPM model.

    This function extracts tissue patches from a WSI and aggregates them
    into a single representation suitable for the model.

    Args:
        wsi_path: Path to the WSI file (SVS, TIFF, etc.).
        patch_size: Size of extracted patches (square).
        target_magnification: Target magnification level (e.g., 20x).
        max_patches: Maximum number of patches to extract.
        aggregate: Aggregation method - "mean", "random", or "first".

    Returns:
        numpy array of shape [3, patch_size, patch_size].

    Example:
        >>> image = preprocess_wsi(
        ...     "/path/to/slide.svs",
        ...     patch_size=224,
        ...     target_magnification=20.0,
        ... )
        >>> image.shape
        (3, 224, 224)

    Notes:
        - For production, consider using attention-based aggregation
        - Stain normalization may improve cross-site consistency
    """
    # Extract patches
    patches = extract_tissue_patches(
        wsi_path,
        patch_size=patch_size,
        magnification=target_magnification,
        max_patches=max_patches,
    )

    if len(patches) == 0:
        # Return blank image if no patches found
        return np.zeros((3, patch_size, patch_size), dtype=np.float32)

    # Aggregate patches
    if aggregate == "mean":
        # Average all patches
        aggregated = np.mean(patches, axis=0)
    elif aggregate == "random":
        # Select a random patch
        idx = np.random.randint(len(patches))
        aggregated = patches[idx]
    elif aggregate == "first":
        aggregated = patches[0]
    else:
        raise ValueError(f"Unknown aggregation method: {aggregate}")

    return aggregated.astype(np.float32)


def extract_tissue_patches(
    wsi_path: Union[str, Path],
    patch_size: int = 224,
    magnification: float = 20.0,
    max_patches: int = 100,
    tissue_threshold: float = 0.5,
    min_tissue_fraction: float = 0.3,
) -> List[np.ndarray]:
    """Extract tissue patches from a whole slide image.

    Args:
        wsi_path: Path to the WSI file.
        patch_size: Size of extracted patches (square).
        magnification: Target magnification level.
        max_patches: Maximum number of patches to return.
        tissue_threshold: Threshold for tissue detection (0-1).
        min_tissue_fraction: Minimum fraction of tissue in a patch.

    Returns:
        List of numpy arrays, each of shape [3, patch_size, patch_size].

    Notes:
        - Patches are in RGB format, normalized to [0, 1]
        - Returns CHW format (channels first) for PyTorch compatibility
    """
    try:
        import openslide
    except ImportError:
        raise ImportError(
            "openslide-python is required for WSI processing. "
            "Install with: pip install openslide-python\n"
            "Also ensure OpenSlide is installed on your system."
        )

    # Open slide
    slide = openslide.OpenSlide(str(wsi_path))

    # Find the appropriate level for target magnification
    level, downsample = _find_magnification_level(slide, magnification)

    # Get dimensions at this level
    level_dims = slide.level_dimensions[level]

    # Compute patch size at this level
    patch_size_level = int(patch_size * downsample)

    # Get tissue mask at low resolution for efficiency
    tissue_mask = _get_tissue_mask(slide, downsample_factor=32)

    # Find candidate patch locations
    candidates = _find_patch_candidates(
        tissue_mask,
        patch_size=patch_size,
        slide_dims=slide.dimensions,
        downsample_factor=32,
        min_tissue_fraction=min_tissue_fraction,
    )

    # Extract patches
    patches = []
    for x, y in candidates[:max_patches]:
        # Read region at native resolution
        region = slide.read_region((x, y), level, (patch_size, patch_size))
        # Convert to RGB (remove alpha channel)
        region = region.convert("RGB")
        # Convert to numpy and normalize
        patch = np.array(region, dtype=np.float32) / 255.0
        # Convert to CHW format
        patch = patch.transpose(2, 0, 1)
        patches.append(patch)

    slide.close()
    return patches


def _find_magnification_level(
    slide,
    target_mag: float,
) -> Tuple[int, float]:
    """Find the best level for target magnification.

    Args:
        slide: OpenSlide object.
        target_mag: Target magnification (e.g., 20.0).

    Returns:
        Tuple of (level_index, downsample_factor).
    """
    # Try to get objective power from metadata
    try:
        objective_power = float(slide.properties.get(
            "openslide.objective-power", 40.0
        ))
    except (ValueError, TypeError):
        objective_power = 40.0  # Default assumption

    # Calculate target downsample
    target_downsample = objective_power / target_mag

    # Find closest level
    best_level = 0
    best_diff = float("inf")
    for i, downsample in enumerate(slide.level_downsamples):
        diff = abs(downsample - target_downsample)
        if diff < best_diff:
            best_diff = diff
            best_level = i

    return best_level, slide.level_downsamples[best_level]


def _get_tissue_mask(
    slide,
    downsample_factor: int = 32,
) -> np.ndarray:
    """Generate a binary tissue mask from the slide thumbnail.

    Args:
        slide: OpenSlide object.
        downsample_factor: Downsampling factor for thumbnail.

    Returns:
        Binary mask where 1 indicates tissue.
    """
    from PIL import Image

    # Get thumbnail
    thumb_size = (
        slide.dimensions[0] // downsample_factor,
        slide.dimensions[1] // downsample_factor,
    )
    thumbnail = slide.get_thumbnail(thumb_size)
    thumbnail = thumbnail.convert("RGB")
    thumb_array = np.array(thumbnail)

    # Simple tissue detection using saturation and value thresholds
    # Convert to HSV-like representation
    gray = np.mean(thumb_array, axis=2)
    saturation = np.std(thumb_array, axis=2)

    # Tissue is typically not too bright (background) and has some color
    tissue_mask = (gray < 220) & (saturation > 10)

    return tissue_mask.astype(np.uint8)


def _find_patch_candidates(
    tissue_mask: np.ndarray,
    patch_size: int,
    slide_dims: Tuple[int, int],
    downsample_factor: int,
    min_tissue_fraction: float,
) -> List[Tuple[int, int]]:
    """Find candidate patch locations based on tissue mask.

    Args:
        tissue_mask: Binary tissue mask.
        patch_size: Patch size in pixels at full resolution.
        slide_dims: Slide dimensions at full resolution.
        downsample_factor: Downsample factor used for tissue mask.
        min_tissue_fraction: Minimum tissue fraction in patch.

    Returns:
        List of (x, y) coordinates at full resolution.
    """
    candidates = []

    # Convert patch size to mask coordinates
    mask_patch_size = max(1, patch_size // downsample_factor)

    # Scan through mask
    h, w = tissue_mask.shape
    for my in range(0, h - mask_patch_size, mask_patch_size):
        for mx in range(0, w - mask_patch_size, mask_patch_size):
            # Check tissue fraction in this region
            region = tissue_mask[my:my+mask_patch_size, mx:mx+mask_patch_size]
            tissue_fraction = np.mean(region)

            if tissue_fraction >= min_tissue_fraction:
                # Convert back to full resolution coordinates
                x = mx * downsample_factor
                y = my * downsample_factor
                candidates.append((x, y))

    # Shuffle to randomize selection
    np.random.shuffle(candidates)

    return candidates


def apply_stain_normalization(
    image: np.ndarray,
    method: str = "macenko",
    target_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply stain normalization to a pathology image.

    Stain normalization reduces color variations across different
    scanners and staining protocols.

    Args:
        image: Input image of shape [3, H, W] in range [0, 1].
        method: Normalization method - "macenko" or "reinhard".
        target_image: Reference image for normalization (optional).

    Returns:
        Normalized image of shape [3, H, W].

    Notes:
        This is a simplified implementation. For production use,
        consider using dedicated libraries like StainTools or torchstain.
    """
    # This is a placeholder for stain normalization
    # In production, use proper implementations like:
    # - torchstain: pip install torchstain
    # - StainTools: pip install staintools

    # Simple contrast normalization as fallback
    image = image.copy()
    for c in range(3):
        channel = image[c]
        p1, p99 = np.percentile(channel, [1, 99])
        channel = np.clip((channel - p1) / (p99 - p1 + 1e-8), 0, 1)
        image[c] = channel

    return image
