"""Temporal modeling components: TimeGapEmbedding + TemporalTransformer."""

import math
import torch
import torch.nn as nn


class TimeGapEmbedding(nn.Module):
    """Sinusoidal time gap encoding."""

    def __init__(self, embed_dim: int, max_time_gap: float = 1825.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_time_gap = max_time_gap
        self.time_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, time_gaps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_gaps: [B, T] time gaps in days
        Returns:
            [B, T, embed_dim]
        """
        B, T = time_gaps.shape
        device = time_gaps.device

        # Normalize time gaps
        t = time_gaps / self.max_time_gap
        dim_half = self.embed_dim // 2

        # Sinusoidal frequencies
        freqs = torch.exp(
            torch.arange(0, dim_half, device=device, dtype=torch.float32) *
            (-math.log(10000.0) / dim_half)
        )

        # [B, T, dim_half]
        angles = t.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0) * math.pi
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        return self.time_proj(pe)


class TemporalTransformer(nn.Module):
    """Transformer for visit sequences with causal masking."""

    def __init__(
        self,
        embed_dim: int,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, z_seq: torch.Tensor, visit_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_seq: [B, T, E] visit embeddings
            visit_mask: [B, T] bool, True=valid visit
        Returns:
            [B, E] global representation (last valid position)
        """
        B, T, E = z_seq.shape
        device = z_seq.device

        # Causal mask: upper triangular = True (blocked)
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

        # Padding mask: True = ignore
        padding_mask = ~visit_mask

        h_seq = self.encoder(z_seq, mask=causal_mask, src_key_padding_mask=padding_mask)
        h_seq = self.norm(h_seq)

        # Extract last valid position for each sample
        seq_lens = visit_mask.sum(dim=1)  # [B]
        last_indices = (seq_lens - 1).clamp(min=0)  # [B]
        h_global = h_seq[torch.arange(B, device=device), last_indices]  # [B, E]

        return h_global
