"""AENCA V14 configuration — Emergent Dynamics (NCA Communication).

V14 replaces V13's explicit multi-head attention with Content-Enriched
NCA Communication: information routing emerges from morphogen NCA dynamics
rather than explicit Q·K·V dot-product attention.

Key design:
  - Morphogen field h seeded from content z, wider than V13 (d_h=256)
  - Content perception: z projected into morphogen space at every NCA step
  - K=8 parallel NCA rules with softmax routing, shared across theta cycles
  - Exponential dilation schedule covers full sequence in O(log N) steps
  - Content readout: h → z update via zero-init projection
  - Integration FFN (SwiGLU-like) replaces HebbianFFN
  - NO explicit Q/K/V projections, NO dot-product attention, NO RoPE
  - PonderNet adaptive halting at theta boundaries (unchanged from V13)
  - Zero-init cascade: fresh V14 produces near-zero residuals

Inspired by:
  - ANCA paper (Kvalsund et al., 2026): emergent attention from NCA movement
  - NCA dynamics paper (Xu et al., 2024): channel/neuron ratio → dynamic behavior
  - V13: proven theta-loop + morphogen-field architecture
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _default_dilations() -> tuple[int, ...]:
    """Exponential dilation schedule: covers 2048+ positions in 12 steps."""
    return (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


@dataclass
class V14Config:
    """Complete configuration for AENCA V14: Emergent Dynamics.

    Architecture (per forward pass):
        embed + SinPE → z₀ (B, L, d_model)
        h = tanh(W_h_init(z₀))  — morphogen seed (B, L, d_h)
        For θ in 0..n_theta-1:
            For m in 0..n_morph_steps-1:              [NCA COMMUNICATION]
                c_m = content_conv(W_c(norm(z)), d_m)  [CONTENT PERCEPTION]
                h = h + gate * Σ_k w_k · rule_k([h, h_nbr, c_m])
                h_multi += α_m · h                    [MULTI-SCALE AGGREGATE]
            z = z + W_readout(norm(h_multi))           [CONTENT READOUT]
            z = z + IntegrationFFN(z)                 [POST-NCA PROCESSING]
            h = h + feedback(z)                       [CONTENT → MORPHOGEN]
            halt_prob, z_out accumulation              [PONDERNET]
        logits = Readout(z_out)
    """

    # ── Core Dimensions ────────────────────────────────────────────────
    d_model: int = 768
    d_h: int = 256              # wider morphogen (was 128 in V13)
    n_rules: int = 8            # more rules for diverse dynamics (was 4)
    morph_rule_hidden: int = 0  # 0 = same as d_h; >0 for dynamic behavior ratio
    n_morph_steps: int = 12     # enough for ~2048-pos coverage (was 4)
    n_theta: int = 4
    d_ff: int = 0               # 0 = auto-compute from target_params

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

    # ── Morphogen / NCA Communication ──────────────────────────────────
    morph_dilations: tuple[int, ...] = field(default_factory=_default_dilations)
    morph_gate_bias: float = 0.0   # sigmoid(0)=0.5; gate must learn to open
    morph_conv_kernel: int = 3
    step_embed_dim: int = 16       # step embedding injected into router

    # ── Loss Weights ───────────────────────────────────────────────────
    lambda_reg: float = 0.005
    lambda_aux: float = 0.05
    lambda_div: float = 0.1         # routing entropy (stronger to prevent collapse)
    lambda_bal: float = 0.1         # Switch Transformer load-balance loss
    lambda_dyn: float = 0.01        # dynamics magnitude (10x stronger)
    lambda_step_div: float = 0.01   # step cosine-diversity (penalize repetition)

    # ── Training ───────────────────────────────────────────────────────
    dropout: float = 0.0
    checkpoint_interval: int = 4
    target_params: int = 162_000_000

    # ── Derived Properties ─────────────────────────────────────────────

    @property
    def effective_rule_hidden(self) -> int:
        """Hidden dim for NCA rules (default = d_h)."""
        return self.morph_rule_hidden if self.morph_rule_hidden > 0 else self.d_h

    @property
    def total_steps(self) -> int:
        """Total effective steps per forward pass (for trainer logging)."""
        return self.n_theta * (self.n_morph_steps + 1)

    @property
    def K(self) -> int:
        """Backward compat alias for total_steps (trainer logging)."""
        return self.total_steps

    def validate(self) -> None:
        """Validate config consistency."""
        assert self.n_rules >= 1, f"n_rules must be >= 1, got {self.n_rules}"
        assert self.n_morph_steps >= 1, f"n_morph_steps >= 1, got {self.n_morph_steps}"
        assert self.n_theta >= 1, f"n_theta >= 1, got {self.n_theta}"
        assert len(self.morph_dilations) >= 1, "morph_dilations must not be empty"
        assert self.d_h >= 32, f"d_h must be >= 32, got {self.d_h}"

    def compute_d_ff(self) -> int:
        """Auto-compute d_ff for IntegrationFFN to reach target_params.

        IntegrationFFN per theta block: SwiGLU-like
            W_gate(d_model → d_ff) + W_up(d_model → d_ff) + W_down(d_ff → d_model)
            = 3 * d_model * d_ff per block
        Total FFN params = n_theta * 3 * d_model * d_ff

        Solves: target_params = fixed_params + n_theta * 3 * d_model * d_ff
        """
        d = self.d_model
        dh = self.d_h
        V = self.vocab_size
        K = self.n_rules
        n_th = self.n_theta
        rh = self.effective_rule_hidden

        # Token embedding: V * d
        embed_params = V * d

        # Readout: projection + norm
        if self.tie_embeddings:
            readout_params = d
        else:
            readout_params = V * d + d

        # ── Morphogen (NCA Communication) — shared across theta ──

        # Seed: W_h_init(d → d_h, no bias)
        seed_params = d * dh

        # Morphogen conv: depthwise(d_h, 1, kernel_size)
        morph_conv_params = dh * self.morph_conv_kernel

        # Content perception:
        #   norm_z_proj: d_model
        #   W_content_proj(d → d_h, no bias): d * d_h
        #   content_conv: depthwise(d_h, 1, kernel_size): d_h * kernel_size
        content_perc_params = d + d * dh + dh * self.morph_conv_kernel

        # Rules (shared):
        #   norm(3*d_h): 3*d_h
        #   K x W_h1(3*d_h → rule_hidden, with bias): K * (3*d_h*rh + rh)
        #   K x W_h2(rule_hidden → d_h, no bias): K * rh*d_h
        #   W_router(d_h → K, no bias): d_h*K
        morph_norm_params = 3 * dh
        rule_params = K * (3 * dh * rh + rh) + K * (rh * dh) + dh * K

        # Bilinear gate: W_gate_a(d_h→d_h) + W_gate_b(d_h→d_h) + bias(d_h)
        gate_params = dh * dh + dh * dh + dh

        # Step embedding for router: embed(n_morph_steps, step_embed_dim)
        #   + W_step_route(step_embed_dim → K, no bias)
        step_embed_params = self.n_morph_steps * self.step_embed_dim + self.step_embed_dim * K

        # Content readout per theta (UNIQUE per theta):
        #   norm_readout(d_h): d_h
        #   W_readout(d_h → d, no bias): d_h * d
        #   scale_logits(n_morph_steps): n_morph_steps
        readout_per_theta = dh + dh * d + self.n_morph_steps

        # Content→morph feedback (SHARED):
        #   norm_fb(d): d
        #   W_h_feedback(d → d_h, no bias): d * d_h
        feedback_params = d + d * dh

        # Per theta IntegrationFFN norms: norm(d)
        ffn_norms_per_theta = d

        # Halt head: W_halt(d→1, with bias) + norm(d)
        halt_params = d + 1 + d if self.enable_halting else 0

        # Sum all fixed params
        fixed = (
            embed_params
            + readout_params
            + seed_params
            + morph_conv_params
            + content_perc_params
            + morph_norm_params
            + rule_params
            + gate_params
            + step_embed_params
            + feedback_params
            + n_th * (readout_per_theta + ffn_norms_per_theta)
            + halt_params
        )

        # Solve: target = fixed + n_theta * 3 * d_model * d_ff
        d_ff = (self.target_params - fixed) // (n_th * 3 * d)

        return max(d_ff, 64)  # safety floor

    def __post_init__(self) -> None:
        """Auto-compute d_ff if set to 0, then validate."""
        if isinstance(self.morph_dilations, list):
            self.morph_dilations = tuple(self.morph_dilations)
        if self.d_ff == 0:
            self.d_ff = self.compute_d_ff()
        self.validate()
