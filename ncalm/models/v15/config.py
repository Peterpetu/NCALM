"""AENCA V15 configuration — 2D Latent Reaction-Diffusion Morphogenetic Network.

V15 replaces V14's 1D single-species NCA with a 2D dual-species
reaction-diffusion NCA on a folded latent grid. Turing instability
from differential diffusion (activator 3×3, inhibitor 5×5) guarantees
position differentiation — the morphogen MUST form spatial patterns.

Key design:
  - 1D→2D folding: N tokens → H×W grid (row-major)
  - Dual species: activator a (d_a=128, k=3) + inhibitor b (d_b=64, k=5)
  - Causal 2D masked depthwise neighborhood (no dilation, no 1D conv)
  - K=8 parallel rules with state+step routing
  - Content conditioning c at every NCA step (c-NCA persistent signal)
  - 2D sinusoidal position encoding (row + column)
  - Content-conditioned bilinear gate per species
  - Stochastic async updates (p=0.5 training, 1.0 inference)
  - Morphogen-informed PonderNet halting
  - Zero-init cascade: fresh model = bag-of-embeddings

Dimensions (Phase 1, L=1):
  d_p = d_a + d_b + d_a + d_b + d_c + d_pos = 544
  d_rule = 384 → ρ = d_rule/d_s = 2.0 (edge of chaos)
  d_ff ≈ 13,024 (auto-computed from target_params)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class V15Config:
    """Complete configuration for AENCA V15: 2D RD-NCA.

    Architecture (per forward pass):
        embed + SinPE₁D → z₀ (B, N, d_model)
        a = tanh(W_seed_a(z₀)), b = tanh(W_seed_b(z₀))
        a, b = Fold(a), Fold(b)                    → (B, H, W, d_a/d_b)
        For θ in 0..n_theta-1:
            c = W_cond(Norm(z)), c = Fold(c)        [CONTENT CONDITIONING]
            For m in 0..M-1:                         [RD-NCA EVOLUTION]
                D_a = P_3×3(a), D_b = P_5×5(b)     [CAUSAL 2D PERCEPTION]
                p = [a, b, D_a, D_b, c, π]         [PERCEPTION VECTOR]
                δ = Σ_k w_k · rule_k(Norm(p))       [K PARALLEL RULES]
                g_a, g_b = BilinearGate(a, b, c)    [CONTENT-COND GATE]
                a += mask · g_a ⊙ δ_a               [STOCHASTIC UPDATE]
                b += mask · g_b ⊙ δ_b
            z += W_read(Norm([a;b]))                 [READOUT → CONTENT]
            z += SwiGLU_FFN(z)                       [INTEGRATION FFN]
            a += Fold(tanh(W_fb_a(Norm(z))))         [CONTENT → MORPHOGEN]
            b += Fold(tanh(W_fb_b(Norm(z))))
            PonderNet halting                        [ADAPTIVE HALT]
        logits = Readout(z_out)
    """

    # ── Core Dimensions ────────────────────────────────────────────────
    d_model: int = 768
    d_a: int = 128              # activator channels (narrow diffusion)
    d_b: int = 64               # inhibitor channels (wide diffusion)
    d_c: int = 128              # content conditioning dimension
    d_pos: int = 32             # 2D position encoding dim (16 row + 16 col)
    d_rule: int = 384           # rule MLP hidden dim; ρ = d_rule/d_s = 2.0
    n_rules: int = 8            # K parallel NCA rules
    n_morph_steps: int = 8      # M NCA evolution steps per theta
    n_theta: int = 4            # theta cycles
    d_ff: int = 0               # 0 = auto-compute from target_params
    step_embed_dim: int = 16    # step embedding for router

    # ── Kernel Sizes ───────────────────────────────────────────────────
    k_a: int = 3                # activator kernel (3×3, 4 causal + self)
    k_b: int = 5                # inhibitor kernel (5×5, 12 causal + self)

    # ── Stochastic Update ──────────────────────────────────────────────
    p_async: float = 0.5        # update probability (training only)

    # ── Vocabulary / Sequence ──────────────────────────────────────────
    vocab_size: int = 50304
    max_seq_len: int = 1024

    # ── Halting ────────────────────────────────────────────────────────
    enable_halting: bool = True
    halt_eps: float = 0.01
    lambda_halt: float = 0.01

    # ── Readout ────────────────────────────────────────────────────────
    tie_embeddings: bool = False
    mos_components: int = 1

    # ── Loss Weights ───────────────────────────────────────────────────
    lambda_reg: float = 0.005       # drift regularization
    lambda_aux: float = 0.05        # deep supervision
    lambda_div: float = 0.1         # routing entropy
    lambda_bal: float = 0.1         # Switch Transformer load-balance
    lambda_dyn: float = 0.01        # dynamics magnitude
    lambda_spatial: float = 0.1     # spatial diversity (NEW)
    lambda_temporal: float = 0.05   # temporal diversity (NEW)

    # ── Training ───────────────────────────────────────────────────────
    dropout: float = 0.0
    checkpoint_interval: int = 4
    target_params: int = 162_000_000

    # ── Hierarchy (Phase 2 — Phase 1 uses L=1) ────────────────────────
    n_levels: int = 1           # L=1 for Phase 1
    d_cross: int = 32           # cross-scale communication dim (Phase 2)
    r_pool: int = 2             # pooling factor between levels

    # ── Derived Properties ─────────────────────────────────────────────

    @property
    def d_s(self) -> int:
        """Total morphogen state dimension: d_a + d_b."""
        return self.d_a + self.d_b

    @property
    def grid_width(self) -> int:
        """W = ceil(sqrt(max_seq_len))."""
        return math.ceil(math.sqrt(self.max_seq_len))

    @property
    def grid_height(self) -> int:
        """H = ceil(max_seq_len / W)."""
        return math.ceil(self.max_seq_len / self.grid_width)

    @property
    def grid_size(self) -> int:
        """H * W (may be >= max_seq_len due to padding)."""
        return self.grid_height * self.grid_width

    @property
    def d_p(self) -> int:
        """Perception vector dimension (Phase 1: no cross-scale).

        p = [a, b, D_a, D_b, c, π]
        d_p = d_a + d_b + d_a + d_b + d_c + d_pos
            = 128 + 64 + 128 + 64 + 128 + 32 = 544  (Phase 1)
        """
        base = 2 * self.d_a + 2 * self.d_b + self.d_c + self.d_pos
        if self.n_levels > 1:
            base += 2 * self.d_cross  # parent + child signals
        return base

    @property
    def total_steps(self) -> int:
        """Total effective steps per forward pass."""
        return self.n_theta * (self.n_morph_steps + 1)

    @property
    def K(self) -> int:
        """Backward compat alias for total_steps (trainer logging)."""
        return self.total_steps

    def validate(self) -> None:
        """Validate config consistency."""
        assert self.d_a >= 16, f"d_a must be >= 16, got {self.d_a}"
        assert self.d_b >= 8, f"d_b must be >= 8, got {self.d_b}"
        assert self.k_a < self.k_b, (
            f"Turing condition requires k_a < k_b, got k_a={self.k_a}, k_b={self.k_b}"
        )
        assert self.n_rules >= 1, f"n_rules must be >= 1, got {self.n_rules}"
        assert self.n_morph_steps >= 1, f"n_morph_steps >= 1, got {self.n_morph_steps}"
        assert self.n_theta >= 1, f"n_theta >= 1, got {self.n_theta}"
        assert self.d_rule > self.d_s, (
            f"Edge of chaos requires d_rule > d_s, got {self.d_rule} vs {self.d_s}"
        )
        assert 0.0 < self.p_async <= 1.0, f"p_async must be in (0, 1], got {self.p_async}"
        assert self.k_a % 2 == 1, f"k_a must be odd, got {self.k_a}"
        assert self.k_b % 2 == 1, f"k_b must be odd, got {self.k_b}"

    def compute_d_ff(self) -> int:
        """Auto-compute d_ff for IntegrationFFN to reach target_params.

        IntegrationFFN per theta block: SwiGLU
            W_gate(d_model → d_ff) + W_up(d_model → d_ff) + W_down(d_ff → d_model)
            = 3 * d_model * d_ff per block
        Total FFN params = n_theta * 3 * d_model * d_ff

        Solves: target_params = fixed_params + n_theta * 3 * d_model * d_ff
        """
        d = self.d_model
        da = self.d_a
        db = self.d_b
        ds = self.d_s
        dc = self.d_c
        dp = self.d_p
        dr = self.d_rule
        V = self.vocab_size
        K = self.n_rules
        M = self.n_morph_steps
        n_th = self.n_theta

        # Token embedding: V * d
        embed_params = V * d

        # Readout: if tied, just norm; else projection + norm
        if self.tie_embeddings:
            readout_params = d  # just norm
        else:
            readout_params = V * d + d

        # ── Seeding ──
        # W_seed_a(d → d_a, no bias) + W_seed_b(d → d_b, no bias)
        seed_params = d * da + d * db

        # ── Content Conditioning ──
        # W_cond(d → d_c, no bias) + RMSNorm(d)
        cond_params = d * dc + d

        # ── Perception (Activator 3×3) ──
        # Masked depthwise Conv2d: d_a channels, kernel k_a×k_a
        # Only causal entries have gradients, but full kernel stored
        perc_a_params = da * self.k_a * self.k_a

        # ── Perception (Inhibitor 5×5) ──
        perc_b_params = db * self.k_b * self.k_b

        # ── 2D Position Encoding ──
        # Non-learnable (sinusoidal) → 0 params

        # ── Rules (×K, shared across theta) ──
        # RMSNorm(d_p)
        rule_norm_params = dp
        # K × W_r1(d_p → d_rule, with bias)
        rule_w1_params = K * (dp * dr + dr)
        # K × W_r2(d_rule → d_s, no bias) [ZERO INIT]
        rule_w2_params = K * (dr * ds)

        # ── Routing ──
        # W_route(d_s → K, no bias) [ZERO INIT]
        route_params = ds * K
        # StepEmbed(M, step_embed_dim)
        step_embed_params = M * self.step_embed_dim
        # W_step_route(step_embed_dim → K, no bias)
        step_route_params = self.step_embed_dim * K

        # ── Gates ──
        # Activator: W_ga1(d_a+d_c → d_a) + W_ga2(d_a+d_c → d_a) + bias_ga(d_a)
        gate_a_params = 2 * (da + dc) * da + da
        # Inhibitor: W_gb1(d_b+d_c → d_b) + W_gb2(d_b+d_c → d_b) + bias_gb(d_b)
        gate_b_params = 2 * (db + dc) * db + db

        # ── Feedback ──
        # W_fb_a(d → d_a) [ZERO] + W_fb_b(d → d_b) [ZERO] + RMSNorm(d)
        feedback_params = d * da + d * db + d

        # ── Shared NCA subtotal ──
        shared_nca = (
            seed_params
            + cond_params
            + perc_a_params
            + perc_b_params
            + rule_norm_params
            + rule_w1_params
            + rule_w2_params
            + route_params
            + step_embed_params
            + step_route_params
            + gate_a_params
            + gate_b_params
            + feedback_params
        )

        # ── Per-θ Readouts (×n_theta) ──
        # W_read(d_s → d, no bias) [ZERO] + RMSNorm(d_s) + λ scale logits (1)
        readout_per_theta = ds * d + ds + 1
        total_readouts = n_th * readout_per_theta

        # ── Per-θ FFN norms ──
        # RMSNorm(d) per theta (inside IntegrationFFN)
        ffn_norms = n_th * d

        # ── Halting ──
        # W_halt(d + d_a → 1, with bias) + RMSNorm(d)
        halt_params = (d + da) + 1 + d if self.enable_halting else 0

        # Sum all non-FFN params
        fixed = (
            embed_params
            + readout_params
            + shared_nca
            + total_readouts
            + ffn_norms
            + halt_params
        )

        # Solve: target = fixed + n_theta * 3 * d * d_ff
        remaining = self.target_params - fixed
        d_ff = remaining // (n_th * 3 * d)

        return max(d_ff, 64)  # minimum d_ff

    def __post_init__(self) -> None:
        """Auto-compute d_ff if set to 0, then validate."""
        if self.d_ff == 0:
            self.d_ff = self.compute_d_ff()
        self.validate()
