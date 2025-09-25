"""Custom loss functions for multi-task learning in DPM.

- DiagnosisLoss: Cross-entropy for diagnosis grading.
- TreatmentLoss: BCE with logits for treatment response (supports masking).
- PrognosisLoss: Cox PH negative log partial likelihood for survival analysis.
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagnosisLoss(nn.Module):
    """Cross-entropy loss for multi-class diagnosis grading.

    Args:
        ignore_index: Label index to ignore (for missing labels).
    """

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets)


class TreatmentLoss(nn.Module):
    """BCE with logits for binary treatment response.

    If `mask` is provided (bool tensor of shape [B]), only masked samples
    contribute to the loss.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1)
        if mask is not None:
            mask = mask.view(-1)
            logits = logits[mask]
            targets = targets[mask]
        if logits.numel() == 0:
            return torch.zeros((), device=targets.device, dtype=targets.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


class PrognosisLoss(nn.Module):
    """Cox Proportional Hazards negative log partial likelihood.

    Implements Breslow's method for ties. Given risk scores r, event indicators e,
    and survival/censoring times t, the loss is:
        L = - sum_{i: e_i=1} ( r_i - log sum_{j: t_j >= t_i} exp(r_j) )
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        device = risk.device
        risk = risk.view(-1)
        time = time.view(-1)
        event = event.view(-1).to(dtype=torch.bool)

        # Sort by descending time so risk sets are cumulative
        order = torch.argsort(time, descending=True)
        risk = risk[order]
        event = event[order]

        # Compute cumulative log-sum-exp of risk (denominator terms)
        lse = torch.logcumsumexp(risk, dim=0)
        # For events, accumulate contribution
        events_idx = torch.nonzero(event, as_tuple=True)[0]
        if events_idx.numel() == 0:
            return torch.zeros((), device=device)
        # Numerator: risk for event cases
        num = risk[events_idx]
        # Denominator: log-sum-exp at corresponding indices
        den = lse[events_idx]
        loss = -(num - den).mean()
        return loss

    def forward(self, risk_score: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        return self._cox_ph_loss(risk_score, time, event)
