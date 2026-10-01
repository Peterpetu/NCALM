"""Hebbian-gated modules for AENCA V8 — Growing Cellular Automata.

Extends V7.1 BDH-inspired Hebbian gating for the V8 Growing NCA paradigm.

Key V8 changes from V7:
  1. **HebbianKernel**: Extended perception input. V7 took (P, z) where P
     was just the CausalConv output. V8 takes a pre-concatenated perceived
     state ẑ ∈ R^{3(d+1)} = [z_conv ∥ z ∥ ∇z] (Spec Definition 7.1) and
     the cell state z ∈ R^{d+1} (includes alpha channel).

  2. **Dimension change**: All projections operate on d_full = d_model + 1
     to include the alpha channel in the growth dynamics. The alpha channel
     is updated jointly with hidden state but serves a purely gating role
     (Spec Section 5.3).

  3. **Perception RMSNorm**: A single RMSNorm(3(d+1)) on the concatenated
     perceived input replaces V7's separate norm_p and norm_z.

  4. **HebbianFFN**: Unchanged from V7. Operates on d_model only (within
     ThetaBlock, which slices off the alpha channel before attention).

Architecture (HebbianKernel, Spec Sections 7-8):
    Perceived state (computed in model.py):
        ẑ_i = RMSNorm([z_{conv,i} ∥ z_i ∥ ∇_i])       # (B, L, 3(d+1))

    Hebbian update (Spec Definition 8.1):
        x = ReLU(W_enc_p(ẑ_i))                         # (B, L, d_concept)
        y = ReLU(W_enc_z(RMSNorm(z_i)))                # (B, L, d_concept)
        Δz_i = W_dec(x ⊙ y)                            # (B, L, d+1)

    Properties (Spec Section 8.2-8.3):
        - Hebbian product: "neurons that fire together wire together"
        - ReLU sparsity: ~5% nonzero (positive sparse activations)
        - Zero-init decoder: model starts as identity (Class I still life)
        - Growth from simplicity: Class I → II → IV as W_dec grows

    Parameter count (Spec Section 8.4):
        |θ_Φ| = d_c × (3(d+1) + (d+1) + (d+1)) = 5 · d_c · (d+1)
        For d=768, d_c=2048: |θ_Φ| ≈ 7.88M

Reference: model_documents/AENCA_V8_GROWING_CELLULAR_AUTOMATA.md
  Definition 7.1 (Perception), Definition 8.1 (Hebbian Update Kernel),
  Proposition 8.1 (Hebbian Interpretation), Proposition 8.2 (Zero-Init),
  Proposition 8.3 (Growth from Simplicity), Section 8.4 (Parameter Count).

Reference: Kosowski et al. (2025), "The Dragon Hatchling: The Missing Link
between the Transformer and Models of the Brain." arXiv:2509.26507.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v8.config import V8Config


class HebbianKernel(nn.Module):
    """V8 NCA kernel with extended perception and Hebbian concept gating.

    V8 extension of V7's HebbianKernel. Two key changes:

    1. Perception input is now the full concatenated perceived state:
       ẑ = [z_conv ∥ z ∥ ∇z] ∈ R^{3(d+1)}  (Spec Def 7.1)
       Instead of V7's separate (P, z) inputs where P = CausalConv(z).

    2. All dimensions include the alpha channel (d+1 instead of d):
       W_enc_p: R^{3(d+1)} → R^{d_c}     (perception encoder)
       W_enc_z: R^{d+1}   → R^{d_c}      (state encoder)
       W_dec:   R^{d_c}   → R^{d+1}      (decoder, zero-init)

    Architecture (Spec Def 8.1):
        x = ReLU(W_enc_p(norm_perceived(ẑ)))   # concept encoding of perception
        y = ReLU(W_enc_z(norm_z(z)))            # concept encoding of state
        xy = x ⊙ y                              # Hebbian product (positive sparse)
        Δz = W_dec(Dropout(xy))                 # decode to full state dim

    The Hebbian product acts as an implicit gate: only concepts that appear
    in BOTH the perceived neighborhood AND the current state contribute.
    This is Hebb's rule in vector form (Spec Proposition 8.1).

    Zero-init decoder (Spec Proposition 8.2):
        At initialization, W_dec = 0 → Δz = 0 → all cells are still lifes.
        The model begins in Wolfram's Class I and learns to grow.

    Parameters: 5 · d_concept · (d_model + 1)
    For d=768, d_c=2048: ≈7.88M (shared across ALL positions and ALL steps)
    """

    def __init__(self, config: V8Config):
        super().__init__()
        df = config.d_full          # d + 1
        d_perceived = config.d_perceived  # 3(d + 1)
        d_concept = config.d_concept

        # Pre-norms
        # Spec Def 7.1: RMSNorm on concatenated [z_conv ∥ z ∥ ∇z]
        self.norm_perceived = nn.RMSNorm(d_perceived)
        # State norm (separate path)
        self.norm_z = nn.RMSNorm(df)

        # Concept encoders
        # W_p: R^{3(d+1)} → R^{d_c} (Xavier normal init)
        self.W_enc_p = nn.Linear(d_perceived, d_concept, bias=False)
        # W_z: R^{d+1} → R^{d_c} (Xavier normal init)
        self.W_enc_z = nn.Linear(df, d_concept, bias=False)

        # Concept decoder: R^{d_c} → R^{d+1} (ZERO-INIT for cascade)
        self.W_dec = nn.Linear(d_concept, df, bias=False)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights — zero-init cascade on decoder.

        Spec Proposition 8.2: W_dec = 0 at initialization.
        This guarantees that at init, Φ(ẑ, z) = 0 for all inputs.
        Every cell is a still life (Definition 1.5). The system begins
        in Wolfram's Class I.

        Spec Proposition 8.3: During training, as W_dec departs from zero,
        dynamics transition: Class I → Class II → Class IV → Class III.
        Edge-of-chaos regularization (Section 11) constrains to Class IV.

        Encoders: Xavier normal (standard for projection matrices).
        Decoder: ZERO → kernel starts producing zero delta (near-identity).
        """
        nn.init.xavier_normal_(self.W_enc_p.weight, gain=1.0)
        nn.init.xavier_normal_(self.W_enc_z.weight, gain=1.0)
        nn.init.zeros_(self.W_dec.weight)  # zero-init cascade

    def forward(
        self, perceived: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Compute Hebbian-gated state update (Spec Definition 8.1).

        Args:
            perceived: (B, L, 3(d+1)) concatenated perceived state
                       = [z_conv ∥ z ∥ ∇z] (Spec Def 7.1)
            z:         (B, L, d+1)    current cell state (hidden + alpha)

        Returns:
            delta: (B, L, d+1) state update proposal Δz_i
                   Includes both hidden and alpha channel updates.
        """
        # Encode to concept space (positive sparse via ReLU)
        # Spec Def 8.1: x = ReLU(W_p · ẑ), y = ReLU(W_z · z)
        x = F.relu(self.W_enc_p(self.norm_perceived(perceived)))  # (B, L, d_concept)
        y = F.relu(self.W_enc_z(self.norm_z(z)))                  # (B, L, d_concept)

        # Hebbian product: only mutually active concepts pass through
        # Spec Proposition 8.1: h_j = [ReLU(W_p·ẑ)]_j · [ReLU(W_z·z)]_j
        xy = x * y  # (B, L, d_concept)

        # Decode to full state dimension (d+1)
        delta = self.W_dec(self.dropout(xy))  # (B, L, d+1)

        return delta


class HebbianFFN(nn.Module):
    """Hebbian feed-forward network for theta blocks — BDH-inspired.

    UNCHANGED from V7. Operates on d_model only (alpha channel is sliced
    off before entering the ThetaBlock). Takes pre-attention and post-
    attention states as two independent inputs for cross-modal gating.

    Architecture (same as V7):
        x = ReLU(W_enc_x(RMSNorm(z_pre)))    — pre-attention concept encoding
        y = ReLU(W_enc_y(RMSNorm(z_post)))    — post-attention concept encoding
        xy = x ⊙ y                            — Hebbian product
        output = W_dec(Dropout(xy))            — decode to model dim

    Parameters: 3 × d_model × d_ff_theta (identical to SwiGLU)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()

        # Pre-norms for each encoding path
        self.norm_x = nn.RMSNorm(d_model)
        self.norm_y = nn.RMSNorm(d_model)

        # Concept encoders: d_model → d_ff
        self.W_enc_x = nn.Linear(d_model, d_ff, bias=False)
        self.W_enc_y = nn.Linear(d_model, d_ff, bias=False)

        # Concept decoder: d_ff → d_model
        self.W_dec = nn.Linear(d_ff, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights — zero-init cascade on decoder."""
        nn.init.xavier_normal_(self.W_enc_x.weight, gain=math.sqrt(2.0))
        nn.init.xavier_normal_(self.W_enc_y.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.W_dec.weight)

    def forward(self, z_pre: torch.Tensor, z_post: torch.Tensor) -> torch.Tensor:
        """Compute Hebbian FFN output.

        Args:
            z_pre:  (B, L, d_model) pre-attention state
            z_post: (B, L, d_model) post-attention state

        Returns:
            output: (B, L, d_model) FFN output (to be added as residual)
        """
        x = F.relu(self.W_enc_x(self.norm_x(z_pre)))
        y = F.relu(self.W_enc_y(self.norm_y(z_post)))
        xy = x * y
        return self.W_dec(self.dropout(xy))
