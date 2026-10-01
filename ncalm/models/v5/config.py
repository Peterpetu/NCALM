"""AENCA V5 configuration.

All hyperparameters for the Precision-Routed NCA Language Model.
Reference: model_documents/aenca_v5_architecture.md §14.3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class V5Config:
    """Complete configuration for AENCA V5.

    V5 uses unified d_model state, PRMP for global communication,
    phase-specialized kernel bank, and autoregressive training.
    All equation references are to the V5 architecture specification.
    """

    # ── Dimensions (§2.2, §3.1) ────────────────────────────────────────────
    d_model: int = 512          # Cell state dimension (= embedding dim)
    d_msg: int = 64             # PRMP message dimension (compressed)
    d_ff: int = 1024            # Kernel feed-forward dimension
    d_pe: int = 64              # Sinusoidal position encoding dimension

    # ── PRMP (§4.4) ────────────────────────────────────────────────────────
    n_heads: int = 4            # PRMP attention heads (H)

    # ── Kernel Bank (§5.2) ──────────────────────────────────────────────────
    n_trunk: int = 3            # SwiGLU residual blocks per kernel
    kernel_radius: int = 1      # Conv kernel radius (kernel_size = r+1 for causal)

    # ── NCA Steps Per Phase (§5.2) ──────────────────────────────────────────
    k_phase1: int = 4           # Propagation steps (K₁)
    k_phase2: int = 8           # Integration steps (K₂)
    k_phase3: int = 4           # Refinement steps (K₃)

    # ── Readout (§7) ────────────────────────────────────────────────────────
    mos_components: int = 1     # Mixture-of-Softmax components (1 = standard)
    tie_embeddings: bool = True # Tie readout to embedding matrix (§7.5)

    # ── Vocabulary and Sequence ─────────────────────────────────────────────
    vocab_size: int = 10_000    # |V| — BPE vocabulary size
    max_seq_len: int = 256      # L — max sequence length

    # ── Loss Weights (§9.1) ─────────────────────────────────────────────────
    lambda_reg: float = 0.01    # State regularization weight
    lambda_aux: float = 0.1     # Auxiliary early-exit loss weight

    # ── Gradient Stabilization (§8.3) ───────────────────────────────────────
    lambda_agc: float = 0.02    # Adaptive gradient clipping ratio

    # ── Gradient Checkpointing ──────────────────────────────────────────────
    checkpoint_interval: int = 4  # Checkpoint every C steps

    # ── Training (§8.3) ─────────────────────────────────────────────────────
    dropout: float = 0.0        # Dropout on PRMP output (§15.3)
    seq_len: int = 256          # Sequence length for training

    # ── Derived Properties ──────────────────────────────────────────────────

    @property
    def total_steps(self) -> int:
        """Total NCA steps K = K₁ + K₂ + K₃."""
        return self.k_phase1 + self.k_phase2 + self.k_phase3

    @property
    def K(self) -> int:
        """Alias for total_steps (backward compat with trainer logging)."""
        return self.total_steps

    @property
    def d_head(self) -> int:
        """Per-head dimension for PRMP (§4.4): d_head = d_msg / H."""
        return self.d_msg // self.n_heads

    @property
    def d_conv_in(self) -> int:
        """Conv input dimension: (r+1) * d_model for causal conv (§6.1)."""
        return (self.kernel_radius + 1) * self.d_model

    @property
    def d_kernel_in(self) -> int:
        """Kernel input dimension (§5.3): d_model + d_model + d_pe.

        Input = [P_i (conv); M_i (PRMP); PE(i)]
        """
        return self.d_model + self.d_model + self.d_pe

    @property
    def phase_steps(self) -> list[int]:
        """List of steps per phase: [K₁, K₂, K₃]."""
        return [self.k_phase1, self.k_phase2, self.k_phase3]

    @property
    def residual_scale(self) -> float:
        """Residual scaling α_res = 1/√N_trunk (§5.3, Eq 5.6)."""
        return 1.0 / math.sqrt(self.n_trunk) if self.n_trunk > 0 else 1.0

    def gate_bias_init(self, phase_steps: int) -> float:
        """Gate bias b_g = -ln(K_p - 1) for phase with K_p steps (§10.4, Eq 10.6).

        Gives σ(b_g) ≈ 1/K_p initial gate opening.
        """
        return -math.log(max(phase_steps - 1, 1))
