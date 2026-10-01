"""AENCA V6 configuration — Oscillator-Controlled Adaptive NCA.

V6 architecture: per-cell coupled Kuramoto oscillator network controlling
all temporal dynamics of a Neural Cellular Automaton language model.

Key design:
  - Single state vector z (no z_H/z_L split)
  - K=8 oscillator channels per cell, self-regulated via Kuramoto
    synchronization, learned relevance gates, and frequency diversity reg
  - Three coupling mechanisms: inter-channel (K×K Kuramoto), local spatial
    (depthwise causal Conv1d), global causal mean-field (cumsum)
  - Per-position readout accumulation (tokens halt independently)
  - T_max=256 maximum NCA steps with segment-based gradient checkpointing

Reference: Kuramoto (1984), "Chemical Oscillations, Waves, and Turbulence."
Reference: Banino et al. (2021), "PonderNet: Learning to Ponder."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class V6Config:
    """Complete configuration for AENCA V6: Oscillator-Controlled NCA.

    Architecture:
        embed → T_max NCA steps of:
            oscillator.step(z, φ) → temporal gates + slow context
            conv(z) → local perception P
            kernel([P; slow_ctx; pe]) → (Δz, gate)
            z ← z + pos_active · kernel_gate · osc_gate · Δz
            readout accumulation (per-position, learned when to contribute)
        → MoS Readout from accumulated z_out

    Oscillator dynamics (Kuramoto with three coupling terms):
        φ_k^(i)(t+1) = φ_k^(i)(t) + ω_k(z_i)
            + Σ_j C_kj sin(φ_j^(i) - φ_k^(i))       [inter-channel]
            + κ_local · Conv1d(φ_k)_i                  [local spatial]
            + κ_global · sin(mean_field_k^[0:i] - φ_k^(i))  [global]
    """

    # ── Dimensions ─────────────────────────────────────────────────────
    d_model: int = 512          # Cell state dimension (= embedding dim)
    d_msg: int = 64             # Slow updater attention message dim (dormant)
    d_ff: int = 1024            # Fast kernel feed-forward dimension
    d_pe: int = 64              # Sinusoidal position encoding dimension
    d_slow_ff: int = 1024       # Slow updater MLP dim (dormant)

    # ── Slow Updater (dormant — enabled via enable_attention) ──────────
    n_heads: int = 4            # Attention heads in slow updater

    # ── Fast Kernel ────────────────────────────────────────────────────
    n_trunk: int = 3            # SwiGLU residual blocks per fast kernel
    kernel_radius: int = 1      # Conv kernel radius (causal: size = r+1)

    # ── Oscillator Network ─────────────────────────────────────────────
    n_osc_channels: int = 8     # K: max oscillator channels per cell
    t_max: int = 256            # Maximum NCA steps
    t_min: int = 1              # Minimum steps before readout allowed
    readout_halt_eps: float = 0.01  # Per-position hard halt threshold

    # ── Oscillator Coupling ────────────────────────────────────────────
    osc_base_freq: float = 0.5      # Base natural frequency (radians/step)
    osc_freq_spread: float = 2.0    # Octave spread for frequency bias init
    osc_coupling_init: float = 0.1  # Inter-channel coupling magnitude init
    d_osc_hidden: int = 256             # Hidden dim for all oscillator MLPs
    enable_spatial_coupling: bool = True   # Local spatial phase coupling
    spatial_coupling_init: float = 0.05    # Local coupling strength init
    enable_mean_field: bool = True         # Global causal mean-field coupling

    # ── Oscillator Regularization ──────────────────────────────────────
    lambda_freq_reg: float = 0.001  # Frequency diversity reg weight

    # ── Attention Hook (dormant by default) ────────────────────────────
    enable_attention: bool = False  # Enable SlowUpdater (causal attention)

    # ── Readout ────────────────────────────────────────────────────────
    mos_components: int = 1     # Mixture-of-Softmax components (1 = standard)
    tie_embeddings: bool = True # Tie readout to embedding matrix

    # ── Vocabulary and Sequence ────────────────────────────────────────
    vocab_size: int = 10_000    # |V| — vocabulary size
    max_seq_len: int = 256      # L — max sequence length

    # ── Loss Weights ───────────────────────────────────────────────────
    lambda_reg: float = 0.01    # State regularization weight
    lambda_aux: float = 0.1     # Deep supervision weight

    # ── Gradient Stabilization ─────────────────────────────────────────
    lambda_agc: float = 0.02    # Adaptive gradient clipping ratio

    # ── Gradient Checkpointing ─────────────────────────────────────────
    checkpoint_segment_size: int = 8  # Steps per gradient checkpoint segment

    # ── Deep Supervision ───────────────────────────────────────────────
    deep_supervision_steps: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256)

    # ── Training ───────────────────────────────────────────────────────
    dropout: float = 0.0        # Dropout (reserved for future use)
    seq_len: int = 256          # Sequence length for training

    # ── Derived Properties ─────────────────────────────────────────────

    @property
    def total_steps(self) -> int:
        """Upper bound total NCA steps."""
        return self.t_max

    @property
    def K(self) -> int:
        """Alias for total_steps (backward compat with trainer logging)."""
        return self.t_max

    @property
    def d_head(self) -> int:
        """Per-head dimension for slow updater attention (dormant)."""
        return self.d_msg // self.n_heads

    @property
    def d_kernel_in(self) -> int:
        """Fast kernel input dimension: [P_i (conv); slow_ctx_i; pe_i].

        Second d_model component is oscillator-derived slow context
        (frequency-filtered view of z), replacing the old z_H.
        """
        return self.d_model + self.d_model + self.d_pe

    @property
    def residual_scale(self) -> float:
        """Residual scaling α_res = 1/√N_trunk."""
        return 1.0 / math.sqrt(self.n_trunk) if self.n_trunk > 0 else 1.0

    def gate_bias_init(self, num_steps: int) -> float:
        """Gate bias b_g = -ln(K-1) → σ(b_g) ≈ 1/K initial gate opening."""
        return -math.log(max(num_steps - 1, 1))
