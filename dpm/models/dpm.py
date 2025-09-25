"""Core DPM model for multi-modal fusion and multi-task prediction.

This module defines the `DPM` class which integrates four modality encoders
(Clinical, Omics, MRI, Pathology), fuses their embeddings with a
TransformerEncoder, and outputs three heads: diagnosis grading, treatment
response, and prognosis risk score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .encoders import ClinicalEncoder, OmicsEncoder, MRIEncoder, PathologyEncoder


@dataclass
class DPMConfig:
    """Configuration for the DPM model."""

    embed_dim: int = 256
    diag_num_classes: int = 3

    clinical_input_dim: int = 64

    omics_feature_dim: int = 128

    mri_in_channels: int = 1

    pathology_backbone: str = "vit_base_patch16_224"

    fusion_layers: int = 2
    fusion_heads: int = 8
    fusion_dropout: float = 0.1


class DPM(nn.Module):
    """DPM: Multi-modal fusion model with learned missing-modality tokens."""

    def __init__(self, cfg: DPMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        E = cfg.embed_dim

        # Encoders
        self.clinical_encoder = ClinicalEncoder(cfg.clinical_input_dim, E)
        self.omics_encoder = OmicsEncoder(cfg.omics_feature_dim, E)
        self.mri_encoder = MRIEncoder(cfg.mri_in_channels, E)
        self.pathology_encoder = PathologyEncoder(E, backbone=cfg.pathology_backbone)

        # Fusion transformer
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

        # Heads
        self.diag_grading_head = nn.Linear(E, cfg.diag_num_classes)
        self.treatment_response_head = nn.Linear(E, 1)
        self.prognosis_outcome_head = nn.Linear(E, 1)

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _expand_token(token: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        return token.to(device).expand(batch_size, -1, -1)

    def _encode_or_missing(
        self,
        x: Optional[torch.Tensor],
        present: Optional[torch.Tensor],
        encoder: nn.Module,
        missing_token: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a modality or return a learned missing token per sample."""
        if x is None and present is None:
            B = 1
            device = next(self.parameters()).device
            return self._expand_token(missing_token, B, device).squeeze(1)

        assert x is not None, "Input tensor must be provided when present mask is given."
        B = x.shape[0]
        device = x.device
        E = self.cfg.embed_dim
        out = torch.empty(B, E, device=device)

        if present is None:
            return encoder(x)

        out[:] = self._expand_token(missing_token, B, device).squeeze(1)
        idx = torch.nonzero(present, as_tuple=True)[0]
        if idx.numel() > 0:
            enc = encoder(x.index_select(0, idx))
            out.index_copy_(0, idx, enc)
        return out

    def _fuse(self, tokens: torch.Tensor) -> torch.Tensor:
        fused = self.backbone_fusion_transformer(tokens)
        return fused[:, 0]

    def forward_with_logits(
        self,
        x: Dict[str, Optional[torch.Tensor]],
        present_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward returning raw logits: (diag, treat, risk)."""
        device = next(self.parameters()).device
        # Infer batch size
        B_opt = None
        for t in x.values():
            if t is not None:
                B_opt = int(t.shape[0])
                break
        if B_opt is None:
            raise ValueError("At least one modality must be provided.")
        B = B_opt

        pm = present_mask or {}
        p_clin = pm.get("clinical")
        p_omics = pm.get("omics")
        p_mri = pm.get("mri")
        p_path = pm.get("pathology")

        h_clin = self._encode_or_missing(x.get("clinical"), p_clin, self.clinical_encoder, self.clinical_missing_token)
        h_omic = self._encode_or_missing(x.get("omics"), p_omics, self.omics_encoder, self.omics_missing_token)
        h_mri = self._encode_or_missing(x.get("mri"), p_mri, self.mri_encoder, self.mri_missing_token)
        h_pat = self._encode_or_missing(x.get("pathology"), p_path, self.pathology_encoder, self.pathology_missing_token)

        cls = self._expand_token(self.cls_token, h_clin.shape[0], device)
        tokens = torch.stack([h_clin, h_omic, h_mri, h_pat], dim=1)
        tokens = torch.cat([cls, tokens], dim=1)

        h_global = self._fuse(tokens)

        diag_logits = self.diag_grading_head(h_global)
        treat_logits = self.treatment_response_head(h_global)
        risk_score = self.prognosis_outcome_head(h_global)
        return diag_logits, treat_logits, risk_score

    def forward(
        self,
        x: Dict[str, Optional[torch.Tensor]],
        present_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward returning probabilities for diag/treat and raw risk."""
        diag_logits, treat_logits, risk_score = self.forward_with_logits(x, present_mask)
        p_diag = self.softmax(diag_logits)
        p_treat = self.sigmoid(treat_logits)
        return p_diag, p_treat, risk_score
