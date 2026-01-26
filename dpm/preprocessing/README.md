# DPM Data Preprocessing Reference

This directory contains reference implementations for preprocessing the four data
modalities used by the Digital Prostate Model (DPM). These scripts are provided
as **documentation and examples**, not as production pipelines.

## Overview

The DPM model expects preprocessed data in specific formats. This module provides
example code showing how raw clinical data can be transformed into model-ready inputs.

**Important:** These are simplified examples. Real-world preprocessing should be
adapted to your specific data formats, quality control requirements, and
institutional standards.

## Modality-Specific Preprocessing

### 1. Clinical/Tabular Data (`tabular.py`)

Clinical features include demographics, lab values, and symptom scores.

**Input:** Raw clinical records (CSV, database export, etc.)
**Output:** Standardized feature vector of shape `[clinical_input_dim]`

Key steps:
- Handle missing values (imputation or flagging)
- Normalize continuous variables (z-score or min-max)
- One-hot encode categorical variables
- Apply feature selection based on clinical relevance

```python
from dpm.preprocessing import preprocess_clinical_features

# Example: Process a patient's clinical data
features = preprocess_clinical_features(
    age=65,
    psa=7.5,
    prostate_volume=45.0,
    # ... other features
)
# Result: numpy array of shape [64]
```

### 2. MRI Data (`mri.py`)

Multiparametric MRI (mpMRI) preprocessing for T2-weighted and DWI sequences.

**Input:** DICOM or NIfTI files
**Output:** Normalized 3D volume of shape `[C, D, H, W]` (default: `[1, 64, 128, 128]`)

Key steps:
- DICOM to NIfTI conversion (if needed)
- Resampling to isotropic resolution
- Intensity normalization (z-score per volume)
- Cropping/padding to target size
- Optional: bias field correction, registration

```python
from dpm.preprocessing import preprocess_mri_volume

# Example: Process an MRI volume
volume = preprocess_mri_volume(
    nifti_path="/path/to/t2w.nii.gz",
    target_size=(64, 128, 128),
    normalize=True,
)
# Result: numpy array of shape [1, 64, 128, 128]
```

### 3. Pathology Data (`pathology.py`)

Whole slide image (WSI) preprocessing for histopathology analysis.

**Input:** WSI files (SVS, TIFF, NDPI, etc.)
**Output:** Image patches or aggregated features of shape `[3, H, W]` (default: `[3, 224, 224]`)

Key steps:
- Tissue detection and segmentation
- Patch extraction at target magnification (e.g., 20x)
- Stain normalization (optional but recommended)
- Quality filtering (blur, artifacts, background)
- Patch selection strategy (random, grid, attention-based)

```python
from dpm.preprocessing import preprocess_wsi, extract_tissue_patches

# Example: Extract patches from a WSI
patches = extract_tissue_patches(
    wsi_path="/path/to/slide.svs",
    patch_size=224,
    magnification=20,
    max_patches=100,
)
# Result: list of numpy arrays, each shape [3, 224, 224]
```

### 4. Omics Data (`omics.py`)

Genomic (RNA-seq, mutations) and proteomic data preprocessing.

**Input:** Expression matrices, VCF files, or mass spectrometry data
**Output:** Feature sequence of shape `[seq_len, feature_dim]` (default: `[128, 128]`)

Key steps:
- Gene/protein ID mapping to standard nomenclature
- Log-transformation for expression data
- Batch effect correction (if combining datasets)
- Feature selection (e.g., top variable genes)
- Missing value imputation

```python
from dpm.preprocessing import preprocess_gene_expression

# Example: Process RNA-seq data
omics_features = preprocess_gene_expression(
    expression_file="/path/to/rnaseq.csv",
    gene_list=None,  # Use default gene panel
    seq_len=128,
    feature_dim=128,
)
# Result: numpy array of shape [128, 128]
```

## Expected Data Shapes

After preprocessing, data should match these shapes (configurable in `default_config.yaml`):

| Modality | Shape | Config Key |
|----------|-------|------------|
| Clinical | `[64]` | `model.clinical_input_dim` |
| Omics | `[128, 128]` | `data.omics_seq_len`, `model.omics_feature_dim` |
| MRI | `[1, 64, 128, 128]` | `model.mri_in_channels`, `data.mri_size` |
| Pathology | `[3, 224, 224]` | `data.image_size` |

## File Format for Training

The training script expects:
1. A manifest CSV listing all samples with paths and labels
2. Preprocessed data files (.npy recommended for speed)

Example manifest structure:
```csv
patient_id,visit_id,clinical_path,omics_path,mri_path,pathology_path,diag_label,...
P0001,V001,/data/clin/P0001_V001.npy,/data/omics/P0001_V001.npy,/data/mri/P0001_V001.npy,/data/path/P0001_V001.npy,2,...
```

## Dependencies

The preprocessing modules may require additional libraries not in the base requirements:
- MRI: `nibabel`, `SimpleITK`, `ANTsPy` (optional)
- Pathology: `openslide`, `pyvips`, `histolab` (optional)
- Omics: `scanpy`, `anndata` (optional)

Install as needed:
```bash
pip install nibabel SimpleITK  # MRI
pip install openslide-python   # Pathology (requires system OpenSlide)
pip install scanpy anndata     # Omics
```

## Notes

1. **These are reference implementations.** Adapt them to your specific needs.
2. **Quality control is essential.** Add appropriate QC steps for your data.
3. **Preprocessing is offline.** Run preprocessing once and save results.
4. **Missing data is handled by the model.** Set modality to absent if unavailable.
