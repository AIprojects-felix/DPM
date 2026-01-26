"""Data preprocessing utilities for the DPM model.

This module provides reference implementations for preprocessing different
data modalities used by the Digital Prostate Model (DPM). These are meant
to serve as examples and documentation rather than production-ready pipelines.

For actual deployment, preprocessing should be adapted to your specific
data formats, institutional standards, and computational infrastructure.

Modules:
    tabular: Clinical and laboratory feature preprocessing
    mri: MRI volume preprocessing and normalization
    pathology: Whole slide image (WSI) preprocessing
    omics: Genomic and proteomic data preprocessing

See the README.md in this directory for detailed documentation.
"""

from .tabular import preprocess_clinical_features, standardize_labs
from .mri import preprocess_mri_volume, normalize_intensity
from .pathology import preprocess_wsi, extract_tissue_patches
from .omics import preprocess_gene_expression, preprocess_protein_levels

__all__ = [
    # Tabular
    "preprocess_clinical_features",
    "standardize_labs",
    # MRI
    "preprocess_mri_volume",
    "normalize_intensity",
    # Pathology
    "preprocess_wsi",
    "extract_tissue_patches",
    # Omics
    "preprocess_gene_expression",
    "preprocess_protein_levels",
]
