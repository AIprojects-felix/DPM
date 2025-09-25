"""Modality-specific encoders used by the DPM model.

Each encoder maps its modality input into a shared embedding space of size `embed_dim`.

Encoders:
- ClinicalEncoder: MLP over flattened tabular features.
- OmicsEncoder: TransformerEncoder over a sequence of feature tokens.
- MRIEncoder: Lightweight 3D CNN producing an embedding from a 3D volume.
- PathologyEncoder: Vision Transformer (via timm) producing an image embedding.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm  # type: ignore
except Exception:  # pragma: no cover - optional import handled by requirements
    timm = None


class ClinicalEncoder(nn.Module):
    """MLP encoder for tabular clinical or lab features.

    Args:
        input_dim: Number of input features.
        embed_dim: Output embedding dimension.
        hidden_dims: Hidden layer sizes for the MLP.
        dropout: Dropout probability applied after hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        dims = [input_dim] + hidden_dims
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(dims[-1], embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [B, input_dim].
        Returns:
            Tensor of shape [B, embed_dim].
        """
        return self.net(x)


class OmicsEncoder(nn.Module):
    """Transformer-based encoder for tokenized omics features.

    Input is a sequence of per-feature tokens; we use a small TransformerEncoder,
    then global average pool across the token dimension.

    Args:
        feature_dim: Dimension of each token's feature vector.
        embed_dim: Shared embedding dimension.
        n_layers: Number of TransformerEncoder layers.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(feature_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [B, T, feature_dim], where T is sequence length.
        Returns:
            Tensor of shape [B, embed_dim].
        """
        x = self.proj(x)  # [B, T, E]
        x = self.encoder(x)  # [B, T, E]
        x = x.mean(dim=1)  # global average pooling over tokens
        x = self.layer_norm(x)
        return x


class MRIEncoder(nn.Module):
    """Lightweight 3D CNN for MRI volumes.

    The network downsamples and aggregates a 3D input into a compact embedding.

    Args:
        in_channels: Number of input channels (e.g., 1).
        embed_dim: Shared embedding dimension.
        width: Base channel width for the CNN.
    """

    def __init__(self, in_channels: int, embed_dim: int, width: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(width, width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(width * 2, width * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),  # [B, C, 1, 1, 1]
        )
        self.proj = nn.Linear(width * 4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [B, C, D, H, W].
        Returns:
            Tensor of shape [B, embed_dim].
        """
        x = self.features(x)  # [B, C, 1, 1, 1]
        x = x.flatten(1)  # [B, C]
        x = self.proj(x)  # [B, E]
        return x

    def load_pretrained_mri_ptpca_weights(self, path: Optional[str] = None) -> None:
        """Placeholder for loading MRI-PTPCa pre-trained weights.

        Implement actual weight loading as needed. If `path` is None, you may
        integrate a registry or download routine in the future.
        """
        # Placeholder: no-op
        return


class PathologyEncoder(nn.Module):
    """Vision Transformer encoder for pathology images (WSI or tiles).

    Uses timm to instantiate a ViT backbone (no classification head) and projects
    its output to the shared embedding dimension if needed.

    Args:
        embed_dim: Shared embedding dimension.
        backbone: timm model name, e.g., "vit_base_patch16_224".
    """

    def __init__(self, embed_dim: int, backbone: str = "vit_base_patch16_224") -> None:
        super().__init__()
        if timm is None:
            raise ImportError(
                "timm is required for PathologyEncoder. Please install timm as specified in requirements.txt."
            )
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        if hasattr(self.backbone, "num_features"):
            in_dim = int(self.backbone.num_features)  # type: ignore[attr-defined]
        else:
            # Fallback: common ViT base dimension
            in_dim = 768
        self.proj = nn.Linear(in_dim, embed_dim) if in_dim != embed_dim else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [B, 3, H, W].
        Returns:
            Tensor of shape [B, embed_dim].
        """
        feats = self.backbone(x)  # [B, in_dim]
        out = self.proj(feats)
        out = self.norm(out)
        return out

    def load_pretrained_conch_weights(self, path: Optional[str] = None) -> None:
        """Placeholder for loading CONCH pre-trained weights.

        Implement actual weight loading as needed. If `path` is None, you may
        integrate a registry or download routine in the future.
        """
        # Placeholder: no-op
        return
