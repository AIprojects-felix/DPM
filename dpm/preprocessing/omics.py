"""Omics data preprocessing for genomic and proteomic features.

This module provides reference implementations for preprocessing omics data
(RNA-seq, proteomics, etc.) used by the DPM model. These examples demonstrate
common preprocessing steps for high-dimensional biological data.

Required dependencies (install as needed):
    pip install scanpy anndata pandas

Example usage:
    >>> from dpm.preprocessing.omics import preprocess_gene_expression
    >>> features = preprocess_gene_expression("/path/to/rnaseq.csv")
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union
from pathlib import Path
import numpy as np


# Default gene panel for prostate cancer (example - adapt to your needs)
# These represent commonly studied genes in prostate cancer research
DEFAULT_GENE_PANEL = [
    # Androgen signaling
    "AR", "FOXA1", "TMPRSS2", "NKX3-1",
    # Tumor suppressors
    "TP53", "PTEN", "RB1", "BRCA1", "BRCA2",
    # Oncogenes
    "MYC", "ERG", "ETV1", "ETV4",
    # Cell cycle
    "CDK4", "CDK6", "CDKN1A", "CDKN2A",
    # DNA repair
    "ATM", "ATR", "CHEK1", "CHEK2",
    # PI3K pathway
    "PIK3CA", "AKT1", "MTOR",
    # Wnt pathway
    "APC", "CTNNB1",
    # Immune markers
    "CD274", "PDCD1", "CTLA4",
]


def preprocess_gene_expression(
    expression_file: Union[str, Path],
    gene_list: Optional[List[str]] = None,
    seq_len: int = 128,
    feature_dim: int = 128,
    log_transform: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Preprocess RNA-seq gene expression data.

    This function loads expression data, applies standard preprocessing,
    and formats it for the DPM model's OmicsEncoder.

    Args:
        expression_file: Path to expression file (CSV with genes as rows).
        gene_list: List of genes to include. Uses default panel if None.
        seq_len: Number of genes (sequence length) in output.
        feature_dim: Feature dimension for each gene.
        log_transform: Whether to apply log2(x+1) transformation.
        normalize: Whether to z-score normalize.

    Returns:
        numpy array of shape [seq_len, feature_dim].

    Example:
        >>> features = preprocess_gene_expression(
        ...     "/path/to/rnaseq.csv",
        ...     seq_len=128,
        ...     feature_dim=128,
        ... )
        >>> features.shape
        (128, 128)

    Notes:
        - Input format: CSV with gene names in first column, expression in second
        - Missing genes are filled with zeros
        - The output is padded/truncated to match seq_len
    """
    import pandas as pd

    # Load expression data
    df = pd.read_csv(expression_file, index_col=0)

    # Use default gene panel if not specified
    genes = gene_list or DEFAULT_GENE_PANEL

    # Extract expression values for target genes
    expression_values = []
    for gene in genes[:seq_len]:
        if gene in df.index:
            value = float(df.loc[gene].iloc[0])
        else:
            value = 0.0
        expression_values.append(value)

    # Convert to numpy
    expression = np.array(expression_values, dtype=np.float32)

    # Log transform
    if log_transform:
        expression = np.log2(expression + 1)

    # Z-score normalize
    if normalize:
        mean = np.mean(expression)
        std = np.std(expression) + 1e-8
        expression = (expression - mean) / std

    # Pad to seq_len
    if len(expression) < seq_len:
        expression = np.pad(
            expression,
            (0, seq_len - len(expression)),
            mode="constant",
        )

    # Create feature matrix [seq_len, feature_dim]
    # Each gene gets a feature vector with the expression value and positional encoding
    features = np.zeros((seq_len, feature_dim), dtype=np.float32)
    for i, value in enumerate(expression):
        # Simple feature: expression value repeated + position encoding
        features[i, 0] = value
        # Add sinusoidal position encoding
        for j in range(1, min(feature_dim, 64)):
            if j % 2 == 0:
                features[i, j] = np.sin(i / (10000 ** (j / 64)))
            else:
                features[i, j] = np.cos(i / (10000 ** ((j-1) / 64)))

    return features


def preprocess_protein_levels(
    protein_file: Union[str, Path],
    protein_list: Optional[List[str]] = None,
    seq_len: int = 128,
    feature_dim: int = 128,
    log_transform: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Preprocess proteomics data (mass spectrometry or immunoassay).

    Similar to gene expression preprocessing, but for protein measurements.

    Args:
        protein_file: Path to protein data file (CSV format).
        protein_list: List of proteins to include.
        seq_len: Number of proteins in output.
        feature_dim: Feature dimension for each protein.
        log_transform: Whether to apply log transformation.
        normalize: Whether to z-score normalize.

    Returns:
        numpy array of shape [seq_len, feature_dim].

    Example:
        >>> features = preprocess_protein_levels(
        ...     "/path/to/proteomics.csv",
        ...     seq_len=128,
        ... )
        >>> features.shape
        (128, 128)
    """
    # Same logic as gene expression
    return preprocess_gene_expression(
        protein_file,
        gene_list=protein_list,
        seq_len=seq_len,
        feature_dim=feature_dim,
        log_transform=log_transform,
        normalize=normalize,
    )


def load_anndata_expression(
    h5ad_path: Union[str, Path],
    gene_list: Optional[List[str]] = None,
    layer: Optional[str] = None,
) -> np.ndarray:
    """Load expression data from AnnData format (h5ad).

    AnnData is a common format for single-cell and bulk RNA-seq data.

    Args:
        h5ad_path: Path to .h5ad file.
        gene_list: List of genes to extract.
        layer: AnnData layer to use (None for .X).

    Returns:
        numpy array of expression values.

    Requires:
        pip install anndata
    """
    try:
        import anndata
    except ImportError:
        raise ImportError(
            "anndata is required for h5ad files. "
            "Install with: pip install anndata"
        )

    adata = anndata.read_h5ad(h5ad_path)

    # Get expression matrix
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    # Convert sparse to dense if needed
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Extract specific genes if requested
    if gene_list is not None:
        gene_indices = []
        for gene in gene_list:
            if gene in adata.var_names:
                gene_indices.append(adata.var_names.get_loc(gene))
            else:
                gene_indices.append(-1)  # Missing gene

        # Extract columns for each gene
        result = np.zeros((X.shape[0], len(gene_list)), dtype=np.float32)
        for i, idx in enumerate(gene_indices):
            if idx >= 0:
                result[:, i] = X[:, idx]

        return result

    return X.astype(np.float32)


def batch_correct(
    expression_matrices: List[np.ndarray],
    batch_labels: List[str],
    method: str = "combat",
) -> List[np.ndarray]:
    """Apply batch effect correction across multiple datasets.

    Batch effects are technical variations between samples processed
    at different times or locations.

    Args:
        expression_matrices: List of expression matrices to correct.
        batch_labels: List of batch identifiers for each matrix.
        method: Correction method - "combat" or "simple".

    Returns:
        List of batch-corrected matrices.

    Notes:
        For proper ComBat correction, use scanpy:
            pip install scanpy
    """
    if method == "simple":
        # Simple z-score per batch
        corrected = []
        for mat in expression_matrices:
            mean = np.mean(mat, axis=0, keepdims=True)
            std = np.std(mat, axis=0, keepdims=True) + 1e-8
            corrected.append((mat - mean) / std)
        return corrected

    elif method == "combat":
        # Use scanpy's ComBat implementation
        try:
            import scanpy as sc
            import anndata
        except ImportError:
            raise ImportError(
                "scanpy and anndata are required for ComBat correction. "
                "Install with: pip install scanpy anndata"
            )

        # Combine into single AnnData object
        combined = np.vstack(expression_matrices)
        batch_ids = []
        for i, (mat, label) in enumerate(zip(expression_matrices, batch_labels)):
            batch_ids.extend([label] * mat.shape[0])

        adata = anndata.AnnData(X=combined)
        adata.obs["batch"] = batch_ids

        # Run ComBat
        sc.pp.combat(adata, key="batch")

        # Split back into original matrices
        corrected = []
        idx = 0
        for mat in expression_matrices:
            n = mat.shape[0]
            corrected.append(adata.X[idx:idx+n])
            idx += n

        return corrected

    else:
        raise ValueError(f"Unknown correction method: {method}")


def select_variable_genes(
    expression: np.ndarray,
    n_top: int = 2000,
    method: str = "variance",
) -> np.ndarray:
    """Select top variable genes from expression matrix.

    Args:
        expression: Expression matrix of shape [samples, genes].
        n_top: Number of top genes to select.
        method: Selection method - "variance" or "dispersion".

    Returns:
        Indices of selected genes.
    """
    if method == "variance":
        # Simple variance-based selection
        variances = np.var(expression, axis=0)
        indices = np.argsort(variances)[::-1][:n_top]
    elif method == "dispersion":
        # Coefficient of variation
        means = np.mean(expression, axis=0) + 1e-8
        stds = np.std(expression, axis=0)
        cv = stds / means
        indices = np.argsort(cv)[::-1][:n_top]
    else:
        raise ValueError(f"Unknown selection method: {method}")

    return indices
