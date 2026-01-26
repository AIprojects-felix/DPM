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
                nn.LayerNorm(dims[i + 1]),  # LayerNorm works with batch size 1
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
        # Use GroupNorm instead of BatchNorm3d to handle batch size 1
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, width), width),  # GroupNorm works with batch size 1
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(width, width * 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(16, width * 2), width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(width * 2, width * 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, width * 4), width * 4),
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

    def load_pretrained_mri_ptpca_weights(self, path: Optional[str] = None) -> bool:
        """Load pre-trained MRI-PTPCa weights for the MRI encoder.

        MRI-PTPCa (Pre-Training for Prostate Cancer Analysis) is a foundation model
        pre-trained on large-scale prostate MRI datasets using self-supervised learning.
        Using these weights can significantly improve performance, especially when
        labeled training data is limited.

        **How to obtain weights:**
        The MRI-PTPCa weights are available from the original authors upon request
        for research purposes. Contact information and licensing details can be found
        in the associated publication.

        **Why pre-trained weights are optional:**
        - The model works with random initialization for testing/development
        - Pre-trained weights require separate download (~500MB)
        - Some institutions may have data use restrictions
        - Enables quick reproduction without external dependencies

        **Weight loading behavior:**
        - If path is None or file doesn't exist: returns False, uses random init
        - If path is valid: loads weights with strict=False to handle architecture
          differences, returns True

        Args:
            path: Path to the .pth file containing MRI-PTPCa weights.
                  If None, no weights are loaded (random initialization).

        Returns:
            True if weights were successfully loaded, False otherwise.

        Example:
            >>> encoder = MRIEncoder(in_channels=1, embed_dim=768)
            >>> loaded = encoder.load_pretrained_mri_ptpca_weights("path/to/mri_ptpca.pth")
            >>> if loaded:
            ...     print("MRI-PTPCa weights loaded successfully")
            ... else:
            ...     print("Using random initialization")
        """
        if path is None:
            return False

        import os
        if not os.path.exists(path):
            return False

        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            # Handle different checkpoint formats
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Load with strict=False to handle architecture differences
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if missing:
                import logging
                logging.info(f"MRI-PTPCa: Missing keys (will use random init): {missing}")
            return True
        except Exception as e:
            import logging
            logging.warning(f"Failed to load MRI-PTPCa weights from {path}: {e}")
            return False


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

    def load_pretrained_conch_weights(self, path: Optional[str] = None) -> bool:
        """Load pre-trained CONCH weights for the Pathology encoder.

        CONCH (Contrastive learning from Captions for Histopathology) is a vision-language
        foundation model pre-trained on over 1.17 million histopathology image-caption
        pairs. It provides strong representations for pathology image analysis tasks.

        **Reference:**
        Lu, M.Y., et al. "A visual-language foundation model for computational pathology."
        Nature Medicine (2024). https://doi.org/10.1038/s41591-024-02856-4

        **How to obtain weights:**
        CONCH weights are available through the Hugging Face model hub:
        https://huggingface.co/MahmoodLab/CONCH

        Access requires:
        1. Hugging Face account
        2. Agreement to the model's license terms
        3. Download via `huggingface_hub` or direct download

        **Why pre-trained weights are optional:**
        - The model works with random initialization for testing/development
        - CONCH requires agreeing to license terms before download
        - Pre-trained weights require significant storage (~1GB)
        - Enables quick code testing without external dependencies

        **Weight loading behavior:**
        - If path is None or file doesn't exist: returns False, uses random init
        - If path is valid: loads backbone weights with strict=False, returns True

        Args:
            path: Path to the .pth file containing CONCH weights.
                  If None, no weights are loaded (random initialization).

        Returns:
            True if weights were successfully loaded, False otherwise.

        Example:
            >>> encoder = PathologyEncoder(embed_dim=768)
            >>> loaded = encoder.load_pretrained_conch_weights("path/to/conch.pth")
            >>> if loaded:
            ...     print("CONCH weights loaded successfully")
            ... else:
            ...     print("Using random initialization")
        """
        if path is None:
            return False

        import os
        if not os.path.exists(path):
            return False

        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            # Handle different checkpoint formats (CONCH may use "model" or "visual" keys)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "visual" in state_dict:
                state_dict = state_dict["visual"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Filter to backbone-only weights
            backbone_state = {}
            for k, v in state_dict.items():
                # Remove common prefixes from foundation model checkpoints
                new_k = k.replace("visual.", "").replace("backbone.", "")
                backbone_state[new_k] = v

            # Load with strict=False to handle architecture differences
            missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
            if missing:
                import logging
                logging.info(f"CONCH: Missing keys (will use random init): {len(missing)} keys")
            return True
        except Exception as e:
            import logging
            logging.warning(f"Failed to load CONCH weights from {path}: {e}")
            return False
