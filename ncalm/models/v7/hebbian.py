"""Hebbian-gated modules for AENCA V7 — BDH-inspired concept encoding.

Implements two key ideas from Dragon Hatchling (BDH) architecture:

1. **ReLU-lowrank concept encoding**: Project activations to a concept space
   via learned linear projection + ReLU. This naturally produces positive,
   sparse activation vectors (~5% nonzero), where each dimension represents
   a "concept neuron." The sparsity provides inherent interpretability.

2. **Hebbian gating (x ⊙ y product)**: Gate signal flow using element-wise
   multiplication of two independently encoded signals. This implements
   a "neurons that fire together wire together" principle — only concepts
   that are active in BOTH signal paths contribute to the output.

Two modules:

- **HebbianKernel**: Replaces SlimKernel + CSS for gamma steps.
  Hebbian interaction between local perception and current state.
  "Only update concepts that are both locally perceived AND in the current state."

- **HebbianFFN**: Replaces SwiGLU FFN in theta blocks.
  Hebbian interaction between pre-attention and post-attention state.
  "Only update concepts that are both in the input AND supported by context."

Key property: Same parameter count and FLOPs as the SwiGLU modules they replace.
3 matrices each: (enc_x, enc_y, dec) ↔ (gate, up, down). The difference is
purely in the inductive bias:
  - SwiGLU: SiLU(W_gate(z)) ⊙ W_up(z) — both from SAME input
  - Hebbian: ReLU(W_x(signal_A)) ⊙ ReLU(W_y(signal_B)) — from DIFFERENT inputs,
    positive sparse, cross-modal gating

Reference: Kosowski et al. (2025), "The Dragon Hatchling: The Missing Link
between the Transformer and Models of the Brain." arXiv:2509.26507.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v7.config import V7Config


class HebbianKernel(nn.Module):
    """NCA kernel with Hebbian concept gating — BDH-inspired.

    Replaces SlimKernel + CSS. Two inputs (local perception P and current
    state z) are encoded to a shared concept space via ReLU-lowrank, then
    multiplied (Hebbian product), then decoded back to model dimension.

    Architecture:
        P = CausalConv1d(z)                    — local perception (computed outside)
        x = ReLU(W_enc_p(RMSNorm(P)))          — concept encoding of perception
        y = ReLU(W_enc_z(RMSNorm(z)))          — concept encoding of state
        xy = x ⊙ y                             — Hebbian product (positive sparse)
        delta = W_dec(Dropout(xy))             — decode to model dim (zero-init)
        z' = z + delta                         — residual update (applied in model)

    The Hebbian product acts as an implicit gate: only concepts that appear
    in BOTH the local perception AND the current state contribute to the
    update. This is more selective than SwiGLU, which gates based on a
    single input with learned gate/data factorization.

    Positive sparse activations (~5% nonzero) emerge naturally from ReLU
    applied to high-dimensional concept vectors, providing interpretability.

    Parameters: 3 × d_model × d_concept + norms ≈ 2304 × d_concept
    FLOPs/tok: 6 × d_model × d_concept (same ratio as SwiGLU of equivalent size)
    """

    def __init__(self, config: V7Config):
        super().__init__()
        d = config.d_model
        d_concept = config.d_concept

        # Pre-norms (separate for each encoding path)
        self.norm_p = nn.RMSNorm(d)
        self.norm_z = nn.RMSNorm(d)

        # Concept encoders: d_model → d_concept
        self.W_enc_p = nn.Linear(d, d_concept, bias=False)  # perception path
        self.W_enc_z = nn.Linear(d, d_concept, bias=False)  # state path

        # Concept decoder: d_concept → d_model (zero-init for cascade)
        self.W_dec = nn.Linear(d_concept, d, bias=False)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights — zero-init cascade on decoder.

        Encoders: Xavier normal (standard init for projection matrices).
        Decoder: ZERO → kernel starts producing zero delta (near-identity).
        The model begins as pure embedding pass-through and learns to update
        during training — consistent with NCA zero-init cascade principle.
        """
        nn.init.xavier_normal_(self.W_enc_p.weight, gain=1.0)
        nn.init.xavier_normal_(self.W_enc_z.weight, gain=1.0)
        nn.init.zeros_(self.W_dec.weight)  # zero-init cascade

    def forward(self, P: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute Hebbian-gated state update.

        Args:
            P: (B, L, d_model) local perception from CausalConv
            z: (B, L, d_model) current cell state

        Returns:
            delta: (B, L, d_model) state update proposal
        """
        # Encode to concept space (positive sparse via ReLU)
        x = F.relu(self.W_enc_p(self.norm_p(P)))   # (B, L, d_concept)
        y = F.relu(self.W_enc_z(self.norm_z(z)))    # (B, L, d_concept)

        # Hebbian product: only mutually active concepts pass through
        xy = x * y                                   # (B, L, d_concept)

        # Decode
        delta = self.W_dec(self.dropout(xy))          # (B, L, d_model)

        return delta


class HebbianFFN(nn.Module):
    """Hebbian feed-forward network for theta blocks — BDH-inspired.

    Replaces SwiGLU FFN. Takes TWO inputs: pre-attention and post-attention
    states. Encodes each to concept space via ReLU, applies Hebbian product,
    then decodes. This implements the key BDH insight:

      - x_sparse (from pre-attn): "which concepts are active in the input?"
      - y_sparse (from post-attn): "which concepts are supported by context?"
      - x ⊙ y: "only update concepts that are BOTH in input AND supported"

    Compare with SwiGLU: SiLU(W_gate(z)) ⊙ W_up(z) — both from SAME input.
    HebbianFFN uses DIFFERENT inputs for the two paths, providing strictly
    more informative cross-modal gating.

    Architecture:
        x = ReLU(W_enc_x(RMSNorm(z_pre)))    — pre-attention concept encoding
        y = ReLU(W_enc_y(RMSNorm(z_post)))    — post-attention concept encoding
        xy = x ⊙ y                            — Hebbian product
        output = W_dec(Dropout(xy))            — decode to model dim

    Parameters: 3 × d_model × d_ff_theta (identical to SwiGLU)
    FLOPs: 6 × d_model × d_ff_theta per token (identical to SwiGLU)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()

        # Pre-norms for each encoding path
        self.norm_x = nn.RMSNorm(d_model)
        self.norm_y = nn.RMSNorm(d_model)

        # Concept encoders: d_model → d_ff
        self.W_enc_x = nn.Linear(d_model, d_ff, bias=False)  # pre-attn path
        self.W_enc_y = nn.Linear(d_model, d_ff, bias=False)  # post-attn path

        # Concept decoder: d_ff → d_model
        self.W_dec = nn.Linear(d_ff, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights — zero-init cascade on decoder.

        Encoders: Xavier normal (good for ReLU — maintains variance).
        Decoder: ZERO → FFN starts as zero → model starts with only attention.
        """
        nn.init.xavier_normal_(self.W_enc_x.weight, gain=math.sqrt(2.0))  # He init for ReLU
        nn.init.xavier_normal_(self.W_enc_y.weight, gain=math.sqrt(2.0))  # He init for ReLU
        nn.init.zeros_(self.W_dec.weight)  # zero-init cascade

    def forward(self, z_pre: torch.Tensor, z_post: torch.Tensor) -> torch.Tensor:
        """Compute Hebbian FFN output.

        Args:
            z_pre:  (B, L, d_model) pre-attention state (concepts in input)
            z_post: (B, L, d_model) post-attention state (concepts from context)

        Returns:
            output: (B, L, d_model) FFN output (to be added as residual)
        """
        # Encode to concept space (positive sparse via ReLU)
        x = F.relu(self.W_enc_x(self.norm_x(z_pre)))    # (B, L, d_ff_theta)
        y = F.relu(self.W_enc_y(self.norm_y(z_post)))    # (B, L, d_ff_theta)

        # Hebbian product: only cross-confirmed concepts contribute
        xy = x * y                                         # (B, L, d_ff_theta)

        # Decode
        output = self.W_dec(self.dropout(xy))              # (B, L, d_model)

        return output
