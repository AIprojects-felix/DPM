"""Custom loss functions for multi-task learning in DPM."""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagnosisLoss(nn.Module):
    """Cross-entropy loss for multi-class diagnosis."""

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets)


class StagingLoss(nn.Module):
    """Cross-entropy loss for staging with ignore_index for N/A cases."""

    def __init__(self, ignore_index: int = -1) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Mask for valid (non-ignored) samples
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            # Return 0 if no valid samples to avoid NaN
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        loss = self.loss(logits, targets)
        # Average only over valid samples
        return loss[valid_mask].mean()


class TreatmentLoss(nn.Module):
    """Multi-label BCE loss for treatment prediction."""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, num_treatments]
            targets: [B, num_treatments] float
            mask: [B] optional mask
        """
        if mask is not None:
            logits = logits[mask]
            targets = targets[mask]
        if logits.numel() == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


class PrognosisLoss(nn.Module):
    """Cox Proportional Hazards negative log partial likelihood."""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        device = risk.device
        risk = risk.view(-1)
        time = time.view(-1)
        event = event.view(-1).to(dtype=torch.bool)

        # Sort by descending time
        order = torch.argsort(time, descending=True)
        risk = risk[order]
        event = event[order]

        # Cumulative log-sum-exp
        lse = torch.logcumsumexp(risk, dim=0)
        events_idx = torch.nonzero(event, as_tuple=True)[0]
        if events_idx.numel() == 0:
            return torch.zeros((), device=device)

        num = risk[events_idx]
        den = lse[events_idx]
        loss = -(num - den).mean()
        return loss

    def forward(self, risk_score: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        return self._cox_ph_loss(risk_score, time, event)
