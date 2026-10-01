"""AENCA V4/V4.1 configuration.

All hyperparameters for the Diffusion Neural Cellular Automata Language Model.
V4 reference: model_documents/aenca_v4_architecture.md §4.3, §4.6, §5.5, §6, §8.
V4.1 reference: model_documents/aenca_v4_1_architecture.md §1-§10.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class AENCAv4Config:
    """Complete configuration for AENCA V4.

    Small model (dev) targeting RTX 4080 Super (16 GB VRAM).
    All equation references are to the V4 architecture specification.
    """

    # ── Vocabulary & Embeddings ─────────────────────────────────────────────
    vocab_size: int = 10_000           # |V| — BPE vocabulary size
    d_emb: int = 384                   # Token embedding dimension

    # ── Latent Space (§4.3, Eq 4.4-4.5) ────────────────────────────────────
    d_z: int = 128                     # Latent channels (token representation)
    d_h: int = 256                     # Hidden channels (working memory)
    d_c: int = 32                      # Control channels (sensor state, gates)
    # D = d_z + d_h + d_c = 416       # Total cell state dimension

    # ── Convolution (§5.1-5.2) ──────────────────────────────────────────────
    kernel_radius: int = 1             # r; kernel_size = 2r+1 = 3
    # D_conv = D = 416                # Conv output dimension (same as state)

    # ── Kernel MLP (§5.5, Eq 5.17-5.23) ────────────────────────────────────
    d_hidden: int = 1024               # Kernel trunk hidden dimension
    n_trunk_layers: int = 3            # Number of trunk MLP layers
    d_ff: int = 512                    # SwiGLU intermediate dim (V4.1 Eq 4.11)
    alpha_res: float = 0.5            # Residual scaling 1/sqrt(N_trunk) (V4.1 Eq 4.14)

    # ── Positional & Timestep Encoding (§5.4, §4.5) ────────────────────────
    d_pos: int = 32                    # Position encoding dimension
    d_t: int = 32                      # Timestep encoding dimension
    L_max: int = 262_144               # Max position for encoding (2^18 = 250K+)

    # ── Active Sensing (§6) ─────────────────────────────────────────────────
    n_sensor_heads: int = 2            # h — sensor heads per cell
    n_actions: int = 9                 # |A| = 3×3 (direction × scale change)
    L_scales: int = 18                 # Scale levels (covers 250K context)

    # ── NCA Dynamics (§5.3, §5.2) ───────────────────────────────────────────
    K: int = 24                        # NCA steps (smoke=24, overnight=32)
    M: int = 8                         # Dilation cycle length ⌈log₂(L)⌉
    C_clamp: float = 10.0              # State clamping bound (Eq 5.15)

    # ── Gradient Checkpointing (§8.4) ───────────────────────────────────────
    checkpoint_interval: int = 4       # C — save checkpoint every C steps

    # ── Diffusion (§3) ──────────────────────────────────────────────────────
    masking_eps: float = 1e-4          # ε for endpoint clamping (Eq 3.12)
    p_random: float = 0.5             # Random masking probability (Def 3.5)
    p_causal: float = 0.4             # Causal masking probability
    p_block: float = 0.1              # Block masking probability

    # ── EFE & Sensing (§6.6-6.8) ────────────────────────────────────────────
    tau_pi: float = 1.0                # Policy temperature (Eq 6.16)
    tau_G: float = 1.0                 # EFE temperature (Eq 6.19)
    beta_efe: float = 0.5             # Epistemic (info gain) weight (Eq 6.21)
    kappa_cost: float = 0.1           # Movement cost weight (Eq 6.27)
    p_efe: float = 0.05               # Fraction of masked positions for Tier-2 EFE
    d_g_hidden: int = 256              # Observation model hidden dim (§6.5)

    # ── Anytime Supervision (§8.2b) ─────────────────────────────────────────
    # Set in __post_init__: defaults to (K//4, K//2, 3*K//4)
    anytime_steps: Optional[Tuple[int, ...]] = None

    # ── Loss Weights (§8.2, Eq 8.1; V4.1 §9.4) ──────────────────────────────
    lambda_any: float = 0.5           # Anytime supervision weight
    lambda_obs: float = 0.1           # Observation model weight
    lambda_pg: float = 0.01           # Policy gradient weight
    lambda_h: float = 0.001           # Hidden state penalty (Eq 7.17b)
    lambda_c: float = 0.0001          # Consensus penalty (Eq 7.17c)
    lambda_pred: float = 0.1          # Prediction error loss (V4.1 Eq 9.1b)
    lambda_div: float = 0.01          # Sensor diversity loss (V4.1 Eq 7.5)
    lambda_prec: float = 0.001        # Precision prior loss (V4.1 Eq 9.1g)
    tau_div: float = 0.3              # Diversity cosine threshold (V4.1 Eq 7.5)

    # ── REINFORCE Baseline (§6.8) ────────────────────────────────────────────
    pg_baseline_ema: float = 0.99     # EMA decay for running baseline (Eq 6.31)

    # ── Training (§8.5) ──────────────────────────────────────────────────────
    seq_len: int = 256                 # L — sequence length

    # ── Warmup Schedules ─────────────────────────────────────────────────────
    obs_warmup_steps: int = 100        # Ramp λ_obs from 0 → λ_obs
    efe_warmup_steps: int = 500        # Ramp τ_π from 2.0 → 1.0

    # ── V4.1 Stochastic Dynamics (§8, Eq 8.2) ─────────────────────────────
    sigma_noise_init: float = 0.01    # Initial noise scale (learned)

    # ── V4.1 Gradient Clipping (§4.1, Eq 4.2-4.3) ────────────────────────
    lambda_agc: float = 0.02          # Adaptive gradient clipping ratio

    # ── V4.1 Curriculum Masking (§6.3, Eq 6.11-6.13) ─────────────────────
    curriculum_easy_steps: int = 5000  # Phase 1 boundary (Beta(2,5))
    curriculum_mid_steps: int = 20000  # Phase 2 boundary (uniform)

    def __post_init__(self) -> None:
        if self.anytime_steps is None:
            # Default: readout at K/4, K/2, 3K/4 (Eq 8.3)
            self.anytime_steps = (self.K // 4, self.K // 2, 3 * self.K // 4)
        # Validate masking probabilities sum to 1
        total_p = self.p_random + self.p_causal + self.p_block
        assert abs(total_p - 1.0) < 1e-6, (
            f"Masking probs must sum to 1, got {total_p}"
        )

    # ── Derived Properties ───────────────────────────────────────────────────

    @property
    def D(self) -> int:
        """Total cell state dimension (Eq 4.5): D = d_z + d_h + d_c."""
        return self.d_z + self.d_h + self.d_c

    @property
    def D_conv(self) -> int:
        """Conv output dimension = D (§5.6)."""
        return self.D

    @property
    def kernel_size(self) -> int:
        """Convolution kernel size = 2r + 1 (Eq 5.3)."""
        return 2 * self.kernel_radius + 1

    @property
    def d_in(self) -> int:
        """Kernel input dimension (Eq 5.17):
        d_in = D_conv + h·d_emb + d_pos + d_t
        """
        return (
            self.D_conv
            + self.n_sensor_heads * self.d_emb
            + self.d_pos
            + self.d_t
        )

    @property
    def mask_token_id(self) -> int:
        """[MASK] token ID = vocab_size (appended after vocabulary)."""
        return self.vocab_size

    @property
    def gate_bias_init(self) -> float:
        """Gate bias b_g = -ln(K - 1) so sigma(b_g) ≈ 1/K (V4.1 Eq 4.27).

        V4 used -ln(√(K·L) - 1) which gave σ ≈ 0.013 — too conservative.
        V4.1 corrects to -ln(K - 1) giving σ ≈ 0.042 for K=24.
        """
        return -math.log(self.K - 1)

    @property
    def delta_s_scale(self) -> float:
        """W_Δs scaling factor = 1/√(2K) (Eq 5.30)."""
        return 1.0 / math.sqrt(2 * self.K)

    @property
    def conv_xavier_gain(self) -> float:
        """Conv Xavier gain = 1/√(2r+1) (Eq 5.32)."""
        return 1.0 / math.sqrt(self.kernel_size)

    @property
    def n_checkpoints(self) -> int:
        """Number of gradient checkpoint segments."""
        return math.ceil(self.K / self.checkpoint_interval)
