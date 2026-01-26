"""Core DPM model with temporal modeling for multi-visit sequences."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .encoders import ClinicalEncoder, OmicsEncoder, MRIEncoder, PathologyEncoder
from .temporal import TimeGapEmbedding, TemporalTransformer


@dataclass
class DPMConfig:
    """Configuration for the DPM model.

    Default values match the paper description. Key hyperparameters:
    - embed_dim: 768 (shared embedding dimension across all modalities)
    - fusion_layers: 6 (Backbone Fusion Transformer depth, L=6 in paper)
    - temporal_layers: 4 (Temporal Transformer depth)
    - temporal_heads: 8 (attention heads for temporal modeling)
    """
    # Core embedding dimension (as described in paper)
    embed_dim: int = 768
    diag_num_classes: int = 8

    # Modality-specific encoder configs
    clinical_input_dim: int = 64
    omics_feature_dim: int = 128
    mri_in_channels: int = 1
    pathology_backbone: str = "vit_base_patch16_224"

    # Backbone Fusion Transformer (as described in paper: L=6)
    fusion_layers: int = 6
    fusion_heads: int = 8
    fusion_dropout: float = 0.1

    # Temporal Transformer for longitudinal modeling
    temporal_layers: int = 4
    temporal_heads: int = 8
    temporal_dropout: float = 0.1
    max_time_gap: float = 1825.0  # ~5 years in days

    # Staging tasks
    isup_num_classes: int = 5
    ipss_num_classes: int = 3
    nih_cpsi_num_classes: int = 3

    # Multi-label treatment
    num_treatments: int = 10


class DPM(nn.Module):
    """DPM: Multi-modal temporal fusion model."""

    def __init__(self, cfg: DPMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        E = cfg.embed_dim

        # Modality encoders
        self.clinical_encoder = ClinicalEncoder(cfg.clinical_input_dim, E)
        self.omics_encoder = OmicsEncoder(cfg.omics_feature_dim, E)
        self.mri_encoder = MRIEncoder(cfg.mri_in_channels, E)
        self.pathology_encoder = PathologyEncoder(E, backbone=cfg.pathology_backbone)

        # Fusion transformer (for single visit)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=E,
            nhead=cfg.fusion_heads,
            dim_feedforward=E * 4,
            dropout=cfg.fusion_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.backbone_fusion_transformer = nn.TransformerEncoder(enc_layer, cfg.fusion_layers)

        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, E) * 0.02)
        self.clinical_missing_token = nn.Parameter(torch.randn(1, 1, E) * 0.02)
        self.omics_missing_token = nn.Parameter(torch.randn(1, 1, E) * 0.02)
        self.mri_missing_token = nn.Parameter(torch.randn(1, 1, E) * 0.02)
        self.pathology_missing_token = nn.Parameter(torch.randn(1, 1, E) * 0.02)

        # Temporal modeling
        self.time_gap_embedding = TimeGapEmbedding(E, cfg.max_time_gap)
        self.temporal_transformer = TemporalTransformer(
            E, cfg.temporal_layers, cfg.temporal_heads, cfg.temporal_dropout
        )

        # Task heads
        self.diag_grading_head = nn.Linear(E, cfg.diag_num_classes)
        self.treatment_response_head = nn.Linear(E, cfg.num_treatments)  # Multi-label
        self.prognosis_outcome_head = nn.Linear(E, 1)

        # Staging heads
        self.isup_head = nn.Linear(E, cfg.isup_num_classes)
        self.ipss_head = nn.Linear(E, cfg.ipss_num_classes)
        self.nih_cpsi_head = nn.Linear(E, cfg.nih_cpsi_num_classes)

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _expand_token(token: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        return token.to(device).expand(batch_size, -1, -1)

    def _encode_modality(
        self,
        x: torch.Tensor,
        present: torch.Tensor,
        encoder: nn.Module,
        missing_token: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a modality, using missing token where not present.

        Args:
            x: [B, ...] input tensor
            present: [B] bool mask
            encoder: modality encoder
            missing_token: [1, 1, E] learned token

        Returns:
            [B, E] encoded features
        """
        B = x.shape[0]
        device = x.device
        dtype = x.dtype
        E = self.cfg.embed_dim
        out = torch.zeros(B, E, device=device, dtype=dtype)

        # Fill with missing token first
        out[:] = missing_token.squeeze(1).to(device=device, dtype=dtype)

        # Encode present samples
        idx = torch.nonzero(present, as_tuple=True)[0]
        if idx.numel() > 0:
            enc = encoder(x.index_select(0, idx))
            out.index_copy_(0, idx, enc.to(dtype=dtype))

        return out

    def _encode_single_visit(
        self,
        clinical: torch.Tensor,
        omics: torch.Tensor,
        mri: torch.Tensor,
        pathology: torch.Tensor,
        pm_clinical: torch.Tensor,
        pm_omics: torch.Tensor,
        pm_mri: torch.Tensor,
        pm_pathology: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a single visit and return visit-level embedding.

        Args:
            clinical: [B, clinical_dim]
            omics: [B, seq_len, feature_dim]
            mri: [B, C, D, H, W]
            pathology: [B, 3, H, W]
            pm_*: [B] presence masks

        Returns:
            [B, E] visit embedding
        """
        B = clinical.shape[0]
        device = clinical.device

        h_clin = self._encode_modality(clinical, pm_clinical, self.clinical_encoder, self.clinical_missing_token)
        h_omic = self._encode_modality(omics, pm_omics, self.omics_encoder, self.omics_missing_token)
        h_mri = self._encode_modality(mri, pm_mri, self.mri_encoder, self.mri_missing_token)
        h_pat = self._encode_modality(pathology, pm_pathology, self.pathology_encoder, self.pathology_missing_token)

        # Stack and add CLS token
        cls = self._expand_token(self.cls_token, B, device)
        tokens = torch.stack([h_clin, h_omic, h_mri, h_pat], dim=1)  # [B, 4, E]
        tokens = torch.cat([cls, tokens], dim=1)  # [B, 5, E]

        # Fusion
        fused = self.backbone_fusion_transformer(tokens)
        z_visit = fused[:, 0]  # CLS token output

        return z_visit

    def forward_with_logits(
        self,
        x: Dict[str, torch.Tensor],
        present_mask: Dict[str, torch.Tensor],
        time_gaps: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning raw logits.

        Args:
            x: Dict with keys 'clinical', 'omics', 'mri', 'pathology'
               Each tensor has shape [B, T, ...]
            present_mask: Dict with same keys, shape [B, T]
            time_gaps: [B, T] time gaps in days
            visit_mask: [B, T] bool, True=valid visit

        Returns:
            Dict with logits for each task
        """
        B, T = visit_mask.shape
        device = visit_mask.device

        # Encode each visit
        z_seq_list = []
        for t in range(T):
            # Extract time step t
            clinical_t = x["clinical"][:, t]  # [B, dim]
            omics_t = x["omics"][:, t]  # [B, seq, feat]
            mri_t = x["mri"][:, t]  # [B, C, D, H, W]
            path_t = x["pathology"][:, t]  # [B, 3, H, W]

            pm_clin_t = present_mask["clinical"][:, t]  # [B]
            pm_omic_t = present_mask["omics"][:, t]
            pm_mri_t = present_mask["mri"][:, t]
            pm_path_t = present_mask["pathology"][:, t]

            z_t = self._encode_single_visit(
                clinical_t, omics_t, mri_t, path_t,
                pm_clin_t, pm_omic_t, pm_mri_t, pm_path_t
            )
            z_seq_list.append(z_t)

        z_seq = torch.stack(z_seq_list, dim=1)  # [B, T, E]

        # Add time gap embeddings
        z_seq = z_seq + self.time_gap_embedding(time_gaps)

        # Temporal transformer
        h_global = self.temporal_transformer(z_seq, visit_mask)  # [B, E]

        # Task heads
        return {
            "diag": self.diag_grading_head(h_global),
            "treat": self.treatment_response_head(h_global),
            "risk": self.prognosis_outcome_head(h_global),
            "isup": self.isup_head(h_global),
            "ipss": self.ipss_head(h_global),
            "nih_cpsi": self.nih_cpsi_head(h_global),
        }

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        present_mask: Dict[str, torch.Tensor],
        time_gaps: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward returning probabilities."""
        outputs = self.forward_with_logits(x, present_mask, time_gaps, visit_mask)
        return {
            "diag": self.softmax(outputs["diag"]),
            "treat": self.sigmoid(outputs["treat"]),
            "risk": outputs["risk"],
            "isup": self.softmax(outputs["isup"]),
            "ipss": self.softmax(outputs["ipss"]),
            "nih_cpsi": self.softmax(outputs["nih_cpsi"]),
        }
