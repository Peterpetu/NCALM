"""AENCA V8 configuration — Growing Cellular Automata.

V8 architecture: extends V7.1 (BDH-inspired Hebbian NCA) with Growing NCA
mechanisms derived from three theoretical pillars:
  - Wolfram: complexity from simple rules, Class IV edge-of-chaos
  - Levin: structure IS function, bioelectric morphogenesis, regeneration
  - Zenil: algorithmic complexity, BDM/CTM, DSG monitoring

Key V8 extensions over V7:
  - Alpha channel: state extended from d to d+1 (lifecycle control)
  - Alive masking: cells alive only if neighbor alpha > tau_alpha
  - Three-state cell lifecycle: DEAD → ACTIVE → SETTLED
  - Settlement: EMA of update magnitudes → soft sigmoid score
  - Awakening: settled cells reactivate when gradient shifts > tau_w
  - Stochastic masking: Bernoulli(0.5) per cell per gamma step
  - Growth-gated update: z += alive · (1 - settled) · mask · Δz
  - Perception kernel: [CausalConv ∥ state ∥ left-gradient] → RMSNorm
  - Edge-of-chaos regularization: Lyapunov exponent band [-0.05, +0.05]
  - Spectral entropy complexity proxy: SVD-based Class IV band
  - Permanence loss: encourage settlement
  - Regeneration loss: corruption → recovery contraction training
  - Deep supervision preserved from V7

Reference: model_documents/AENCA_V8_GROWING_CELLULAR_AUTOMATA.md
  Sections 5-6 (state space, alive masking),
  Sections 7-9 (perception, update kernel, stochastic masking),
  Section 10 (permanence/settlement/awakening),
  Section 11 (edge-of-chaos, spectral entropy),
  Section 16 (complete loss function with 6 terms),
  Section 21 (parameter budget: ZERO additional learned parameters).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class V8Config:
    """Complete configuration for AENCA V8: Growing Cellular Automata.

    Architecture (per forward pass):
        z⁰ = Seed(x): Embed + PE → (B, L, d), α=1.0 → (B, L, d+1)
        For θ in 1..n_theta:                            [THETA CYCLE]
            For γ in 1..t_gamma:                        [GAMMA STEPS — shared weights]
                alive = 1[max_neighbor α > τ_α]         [alive masking]
                awaken settled cells if ‖∇−∇_settle‖>τ_w [awakening]
                ẑ = RMSNorm([Conv(z) ∥ z ∥ ∇z])        [perception: 3(d+1) dims]
                Δz = W_dec(ReLU(W_p·ẑ) ⊙ ReLU(W_z·z)) [Hebbian kernel]
                m ~ Bernoulli(p_update)                  [stochastic mask]
                z += alive·(1−settled)·m·Δz              [growth-gated update]
                update settlement EMA, settled score      [convergence tracking]
            z = ThetaBlock_θ(z, settled)                 [θ step, settled=read-only]
            halt_prob = HaltHead(z_hidden)               [readout accumulation]
            z_out += min(halt_prob, remaining) · z_hidden
        logits = Readout(z_out)

    Loss: L_CE + λ_halt·L_halt + λ_perm·L_perm + λ_regen·L_regen
          + λ_edge·L_edge + λ_aux·L_aux

    Parameter overhead over V7: ZERO new learned parameters.
    All V8 mechanisms (alive masking, settlement, stochastic masking,
    awakening, edge-of-chaos) are computed from existing state variables.
    Only dimension change: d → d+1 in conv and HebbianKernel (alpha channel).
    """

    # ── Dimensions ─────────────────────────────────────────────────────
    d_model: int = 768          # Hidden cell state dimension (embedding dim)
    d_pe: int = 64              # Sinusoidal position encoding dimension (legacy)

    # ── Hebbian Kernel (BDH-inspired, extended for V8 perception) ─────
    d_concept: int = 2048       # Concept space dim for Hebbian gamma kernel
    kernel_radius: int = 2      # Causal conv radius (kernel_size = r + 1)

    # ── Hierarchical Timing ────────────────────────────────────────────
    n_theta: int = 4            # Theta cycles (each has unique attn+FFN)
    t_gamma: int = 2            # Gamma steps per theta cycle (shared kernel)

    # ── Theta Attention ────────────────────────────────────────────────
    n_heads: int = 12           # Attention heads per theta block
    d_head: int = 64            # Per-head dimension
    d_ff_theta: int = 0         # Theta FFN width (0 = auto-size to fill param budget)

    # ── Adaptive Halting (PonderNet, preserved from V7) ────────────────
    enable_halting: bool = True     # Per-position readout accumulation
    halt_eps: float = 0.01          # Hard halt threshold (remaining mass)
    lambda_halt: float = 0.01       # PonderNet KL regularization weight

    # ── Readout (preserved from V7) ────────────────────────────────────
    mos_components: int = 1     # MoS components (1 = standard softmax)
    tie_embeddings: bool = True # Tie readout to embedding matrix

    # ── Vocabulary and Sequence ────────────────────────────────────────
    vocab_size: int = 10_000    # |V| — vocabulary size
    max_seq_len: int = 256      # L — max sequence length

    # ══════════════════════════════════════════════════════════════════
    # V8-SPECIFIC: Growing Cellular Automata Mechanisms
    # (Spec Sections 5-12, 16-19)
    # ══════════════════════════════════════════════════════════════════

    # ── Alpha Channel & Alive Masking (Spec Sec 5-6) ──────────────────
    # Def 6.1: alive_i = 1[max_{j∈N(i)} z_j^α > τ_α]
    tau_alpha: float = 0.1      # Alive threshold (Def 6.1, default from spec)

    # ── Settlement (Spec Sec 6.3, 10.2, 10.3) ─────────────────────────
    # Def 10.2: ā_i = β·ā_i + (1−β)·‖Δz_i‖₂  (EMA of update magnitudes)
    # Def 10.3: settled_i = σ((τ_s − ā_i) / η)  (soft settlement score)
    settlement_beta: float = 0.9        # β: EMA decay rate (Def 10.2)
    tau_s: float = 0.05                 # τ_s: settlement threshold (Def 10.3)
    settlement_eta: float = 0.1         # η: temperature for sigmoid sharpness (Def 10.3)
    tau_c: float = 0.5                  # τ_c: computation cutoff threshold (Def 6.2)

    # ── Awakening (Spec Sec 10.3) ──────────────────────────────────────
    # Def 10.4: awaken_i = 1[‖∇_i − ∇_i^{settle}‖₂ > τ_w]
    tau_w: float = 0.5          # τ_w: awakening threshold (Def 10.4)

    # ── Stochastic Masking (Spec Sec 9) ────────────────────────────────
    # Def 9.1: m_i ~ Bernoulli(p_update) per cell per step
    p_update: float = 0.5      # Stochastic update probability (default from spec)

    # ── Edge-of-Chaos Regulation (Spec Sec 11) ────────────────────────
    # Def 11.2: L_edge = α_edge·[ReLU(λ̂_max − λ₊) + ReLU(λ₋ − λ̂_max)]
    enable_edge_chaos: bool = True      # Enable Lyapunov regularization
    edge_lambda_plus: float = 0.05      # λ₊: upper chaos threshold (Def 11.2)
    edge_lambda_minus: float = -0.05    # λ₋: lower order threshold (Def 11.2)
    lyapunov_eps: float = 1e-4          # σ_ε: perturbation noise scale (Def 11.1)
    edge_batch_fraction: float = 0.25   # Fraction of batch for Lyapunov computation

    # ── Spectral Entropy (Spec Sec 11.3, 19) ──────────────────────────
    # Def 11.3: H_sv = −Σ σ̂_j log₂ σ̂_j (normalized SVD singular values)
    # Def 19.1: target band [η_min, η_max] = [0.3, 0.7] × log₂(min(L,d))
    enable_spectral_entropy: bool = True
    spectral_eta_min_frac: float = 0.3  # Lower bound fraction of max entropy
    spectral_eta_max_frac: float = 0.7  # Upper bound fraction of max entropy
    alpha_sv: float = 0.01              # α_sv: spectral entropy loss weight

    # ── Loss Weights (Spec Sec 16.8) ──────────────────────────────────
    lambda_perm: float = 0.1    # Permanence loss weight (Sec 16.4)
    lambda_regen: float = 0.01  # Regeneration loss weight (Sec 16.5)
    lambda_edge: float = 0.1    # Edge-of-chaos loss weight (Sec 16.6)
    lambda_aux: float = 0.5     # Deep supervision weight (Sec 16.7)
    lambda_reg: float = 0.0     # State regularization (zeroed; replaced by L_perm)

    # ── Regeneration Training (Spec Sec 16.5, 18) ─────────────────────
    # Protocol A: Gaussian noise corruption
    enable_regen: bool = True           # Enable regeneration training
    p_regen: float = 0.1               # Probability of corruption per batch
    sigma_corrupt: float = 0.1         # σ_corrupt: Gaussian noise scale (Sec 16.5)
    T_r: int = 2                        # Recovery steps (default: t_gamma)

    # ── Gradient Stabilization (preserved from V7) ─────────────────────
    lambda_agc: float = 0.02    # Adaptive gradient clipping ratio

    # ── Pool-Based NCA Training (Spec Sec 17) ─────────────────────────
    # State pool: persistent buffer of partially grown states
    enable_pool: bool = False           # Enable pool-based training
    pool_size: int = 256                # P: number of states in pool
    pool_rejuv_schedule: tuple[float, ...] = (1.0, 0.5, 0.2, 0.1)
    # p_rejuv at warmup/early/mid/late phases (Def 17.2)
    pool_phase_boundaries: tuple[float, ...] = (0.05, 0.20, 0.60)
    # Fraction-of-training thresholds for phase transitions
    pool_growth_multipliers: tuple[int, ...] = (1, 1, 2, 3)
    # K_step multiplier at each phase (× n_theta·t_gamma)

    # ── Training ───────────────────────────────────────────────────────
    dropout: float = 0.0        # Dropout rate
    seq_len: int = 256          # Sequence length for training
    checkpoint_interval: int = 4  # For trainer compat

    # ── CSS legacy fields (unused in V8, kept for config compat) ───────
    n_streams: int = 8
    d_state: int = 32
    d_css: int = 256
    d_ff: int = 768

    # ══════════════════════════════════════════════════════════════════
    # Derived Properties
    # ══════════════════════════════════════════════════════════════════

    @property
    def d_full(self) -> int:
        """Full cell state dimension including alpha channel.

        Spec Def 5.1: z_i = [z_i^hidden; z_i^α] ∈ R^{d+1}
        """
        return self.d_model + 1

    @property
    def d_perceived(self) -> int:
        """Perceived state dimension after concatenation.

        Spec Def 7.1: ẑ_i = RMSNorm([z_conv,i ∥ z_i ∥ ∇_i]) ∈ R^{3(d+1)}
        """
        return 3 * self.d_full

    @property
    def a_bar_init(self) -> float:
        """Initial value for settlement EMA (ā).

        Spec Def 10.2: ā^(0) = ‖Δz^(0)‖ (first update magnitude).
        Spec Sec 15.1 Step 1: ā^(0) = ∞ (no settlement history).

        We resolve this by initializing to τ_s + 3η, which gives:
          settled_init = σ((τ_s − a_bar_init) / η) = σ(−3) ≈ 0.047
        Cells start clearly unsettled but can settle within available
        gamma steps as dynamics converge (after ~20 steps with zero Δz).
        """
        return self.tau_s + 3.0 * self.settlement_eta

    @property
    def total_gamma_steps(self) -> int:
        """Total gamma steps across all theta cycles."""
        return self.n_theta * self.t_gamma

    @property
    def total_steps(self) -> int:
        """Total NCA-equivalent steps (gamma + theta)."""
        return self.n_theta * (self.t_gamma + 1)

    @property
    def K(self) -> int:
        """Backward compat alias for total_steps."""
        return self.total_steps

    @property
    def residual_scale(self) -> float:
        """1/√(2·n_theta) scaling for theta block output projections."""
        return 1.0 / math.sqrt(2.0 * self.n_theta) if self.n_theta > 0 else 1.0

    @property
    def css_state_dim(self) -> int:
        """Legacy compat: total CSS state dimension."""
        return self.n_streams * self.d_state

    def gate_bias_init(self, num_steps: int) -> float:
        """Gate bias b_g = −ln(K−1) → σ(b_g) ≈ 1/K initial opening."""
        return -math.log(max(num_steps - 1, 1))

    def compute_d_ff_theta(self, target_params: int = 162_000_000) -> int:
        """Auto-compute d_ff_theta to reach target total parameter count.

        V8 parameter budget (Spec Sec 21.1):
            embed = vocab × d_model                        (tied → shared with readout)
            hebbian_kernel = d_concept × (3·(d+1) + (d+1) + (d+1)) + norms
                           = d_concept × 5·(d+1)           (Spec Sec 8.4)
            conv = (d+1) × (d+1) × (kernel_radius + 1)     (operates on d+1 channels)
            per theta block:
                attn_qkv = 3 × d × d                       (operates on d_model only)
                attn_out = d × d
                hebbian_ffn = 3 × d × d_ff_theta            (enc_x + enc_y + dec)
                norms ≈ 3 × d
            halt = d + 1 + d                                (W + bias + norm)
            norms ≈ small

        V8 overhead: ZERO additional learned parameters.
        Dim changes: conv and HebbianKernel use d+1 instead of d.

        Returns:
            d_ff_theta: auto-computed FFN width for theta blocks
        """
        d = self.d_model
        df = self.d_full  # d + 1
        V = self.vocab_size
        n_th = self.n_theta

        # Embedding (tied → appears once, shared with readout)
        embed_params = V * d
        if not self.tie_embeddings:
            embed_params += V * d

        # Readout norm
        readout_norm = d

        # V8 HebbianKernel (Spec Sec 8.4):
        # W_enc_p: 3(d+1) × d_c, W_enc_z: (d+1) × d_c, W_dec: d_c × (d+1)
        # = d_c × (3(d+1) + (d+1) + (d+1)) = d_c × 5(d+1)
        # Plus norms: RMSNorm(3(d+1)) + RMSNorm(d+1)
        dc = self.d_concept
        kernel_params = (
            dc * 3 * df           # W_enc_p
            + dc * df              # W_enc_z
            + dc * df              # W_dec
            + 3 * df               # norm_perceived (3(d+1))
            + df                   # norm_z (d+1)
        )

        # V8 CausalConv1d: operates on d+1 channels (alpha participates)
        conv_params = df * df * (self.kernel_radius + 1) + df  # conv weight + norm

        # Per theta block (attention + HebbianFFN), excluding d_ff_theta:
        attn_params_per_block = (
            3 * d * d              # Q, K, V projections (on d_model)
            + d * d                # output projection
            + 3 * d                # 3 RMSNorms (pre-attn + 2 Hebbian norms)
        )

        # Halt head
        halt_params = (d + 1 + d) if self.enable_halting else 0

        # Fixed params (everything except theta FFN)
        fixed = (
            embed_params
            + readout_norm
            + kernel_params
            + conv_params
            + n_th * attn_params_per_block
            + halt_params
        )

        # Remaining budget: n_theta × 3 × d × d_ff_theta
        remaining = target_params - fixed
        ffn_per_unit = 3 * d
        d_ff_theta = max(remaining // (n_th * ffn_per_unit), 256)

        return int(d_ff_theta)
