"""Hebbian FFN for AENCA V13 theta blocks — BDH-inspired cross-modal gating.

Same architecture as V7.1 HebbianFFN: takes pre-attention and post-attention
states, encodes each to concept space via ReLU, applies Hebbian product,
then decodes.

Each theta cycle has its own unique HebbianFFN instance (unique weights).

Architecture:
    x = ReLU(W_enc_x(RMSNorm_x(z_pre)))    — pre-attention concept encoding
    y = ReLU(W_enc_y(RMSNorm_y(z_post)))    — post-attention concept encoding
    xy = x ⊙ y                              — Hebbian product
    output = W_dec(xy)                       — decode (ZERO-INIT)

Initialization:
    W_enc_x: xavier_normal_(gain=√2)  — He init for ReLU
    W_enc_y: xavier_normal_(gain=√2)  — He init for ReLU
    W_dec:   zeros_()                 — zero-init cascade

Parameters: 3 × d_model × d_ff + 2 × d_model (norms) per theta block.

Reference: V7.1 hebbian.py — HebbianFFN (identical architecture).
Reference: Kosowski et al. (2025), "The Dragon Hatchling" — Hebbian gating.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HebbianFFN(nn.Module):
    """Hebbian feed-forward network — BDH-inspired cross-modal concept gating.

    Takes two inputs: pre-attention z_pre and post-attention z_post.
    Encodes each independently via ReLU to positive sparse concept vectors,
    multiplies (Hebbian product = "neurons that fire together wire together"),
    then decodes back to model dimension.

    The Hebbian product is more selective than SwiGLU: it requires concepts
    to be active in BOTH the input (pre-attn) AND the context (post-attn).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()

        # Pre-norms for each encoding path
        self.norm_x = nn.RMSNorm(d_model)
        self.norm_y = nn.RMSNorm(d_model)

        # Concept encoders: d_model → d_ff
        self.W_enc_x = nn.Linear(d_model, d_ff, bias=False)
        self.W_enc_y = nn.Linear(d_model, d_ff, bias=False)

        # Concept decoder: d_ff → d_model (ZERO-INIT)
        self.W_dec = nn.Linear(d_ff, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Encoders: He init (√2 gain for ReLU). Decoder: ZERO (cascade)."""
        nn.init.xavier_normal_(self.W_enc_x.weight, gain=math.sqrt(2.0))
        nn.init.xavier_normal_(self.W_enc_y.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.W_dec.weight)

    def forward(self, z_pre: torch.Tensor, z_post: torch.Tensor) -> torch.Tensor:
        """Compute Hebbian FFN output.

        Args:
            z_pre:  (B, L, d_model) pre-attention state (concepts in input)
            z_post: (B, L, d_model) post-attention state (concepts from context)

        Returns:
            output: (B, L, d_model) FFN output (to be added as residual)
        """
        x = F.relu(self.W_enc_x(self.norm_x(z_pre)))      # (B, L, d_ff)
        y = F.relu(self.W_enc_y(self.norm_y(z_post)))      # (B, L, d_ff)

        xy = x * y                                          # Hebbian product

        output = self.W_dec(self.dropout(xy))               # (B, L, d_model)

        return output
