"""AENCA V13 configuration — Morphogenetic Attention.

V13 adds a lightweight morphogen field (h ∈ R^{L×d_h}) that evolves via
local NCA dynamics and gates Q/K projections of V7.1's attention. At
initialization, V13 behaves identically to V7.1 (zero-init cascade).

Key design:
  - Morphogen field h seeded from content z, evolved via K parallel local rules
  - Q gate = 1 + tanh(W_Q_h(norm(h))), K gate same — range [0, 2], starts at 1.0
  - Attention + HebbianFFN unique per theta cycle (V7.1 pattern)
  - Morphogen rules + gates shared across theta cycles
  - PonderNet adaptive halting at theta boundaries
  - No HebbianKernel / CausalConv from V7.1 (subsumed by morphogen)
  - Freed ~6.5M params → wider HebbianFFN (d_ff: 7464→~8100)

Reference: ANCA paper (Kvalsund et al., 2026) — emergent attention from NCA.
Reference: V7.1 — proven theta-block structure (96% of V7.1's 162M params).
Reference: V11 — morphogen field, BilinearGate, DepthwiseCausalConv.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class V13Config:
    """Complete configuration for AENCA V13: Morphogenetic Attention.

    Architecture (per forward pass):
        embed + SinPE → z₀ (B, L, d_model)
        h = tanh(W_h_init(z₀))  — morphogen seed (B, L, d_h)
        For θ in 0..n_theta-1:
            For m in 0..n_morph_steps-1:             [MORPHOGEN EVOLUTION]
                h = h + gate * Σ_k w_k · rule_k(h)
            q_gate, k_gate = generate_gates(h)       [MORPHOGEN → ATTENTION]
            z = ThetaAttention_θ(z, q_gate, k_gate)  [GATED ATTENTION]
            z = z + HebbianFFN_θ(z_pre, z)            [CROSS-MODAL FFN]
            h = h + feedback(z)                       [CONTENT → MORPHOGEN]
            halt_prob, z_out accumulation              [PONDERNET]
        logits = Readout(z_out)
    """

    # ── Core Dimensions ────────────────────────────────────────────────
    d_model: int = 768
    d_h: int = 128
    n_rules: int = 4
    n_morph_steps: int = 4
    n_theta: int = 4
    n_heads: int = 12
    d_head: int = 64
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

    # ── Morphogen ──────────────────────────────────────────────────────
    morph_dilations: tuple[int, ...] = (1, 2, 4, 8)
    morph_gate_bias: float = 3.0
    morph_conv_kernel: int = 3

    # ── Loss Weights ───────────────────────────────────────────────────
    lambda_reg: float = 0.005
    lambda_aux: float = 0.05
    lambda_div: float = 0.01

    # ── Training ───────────────────────────────────────────────────────
    dropout: float = 0.0
    checkpoint_interval: int = 4
    target_params: int = 162_000_000

    # ── Derived Properties ─────────────────────────────────────────────

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
        assert self.d_model == self.n_heads * self.d_head, (
            f"d_model ({self.d_model}) != n_heads ({self.n_heads}) * d_head ({self.d_head})"
        )
        assert self.n_rules >= 1, f"n_rules must be >= 1, got {self.n_rules}"
        assert self.n_morph_steps >= 1, f"n_morph_steps must be >= 1, got {self.n_morph_steps}"
        assert self.n_theta >= 1, f"n_theta must be >= 1, got {self.n_theta}"
        assert len(self.morph_dilations) >= 1, "morph_dilations must not be empty"

    def compute_d_ff(self) -> int:
        """Auto-compute d_ff to reach target_params.

        Solves: target_params = fixed_params + n_theta * 3 * d_model * d_ff
        for d_ff, where fixed_params accounts for everything except HebbianFFN.

        Returns:
            d_ff: auto-computed FFN width for theta blocks
        """
        d = self.d_model
        dh = self.d_h
        V = self.vocab_size
        K = self.n_rules
        n_th = self.n_theta

        # Token embedding
        embed_params = V * d

        # Readout: projection + norm
        if self.tie_embeddings:
            readout_params = d  # just norm
        else:
            readout_params = V * d + d  # separate projection + norm

        # Morphogen seed: W_h_init(d_model → d_h, no bias)
        seed_params = d * dh

        # Morphogen conv: depthwise (d_h, 1, kernel_size)
        conv_params = dh * self.morph_conv_kernel

        # Morphogen rules:
        #   shared RMSNorm(2*d_h): 2*d_h
        #   K x W_h1(2*d_h → d_h, with bias): K * (2*d_h*d_h + d_h)
        #   K x W_h2(d_h → d_h, no bias): K * d_h*d_h
        #   W_router(d_h → K, no bias): d_h*K
        morph_norm_params = 2 * dh
        rule_params = K * (2 * dh * dh + dh) + K * (dh * dh) + dh * K

        # Bilinear gate: W_gate_a(d_h→d_h) + W_gate_b(d_h→d_h) + bias(d_h)
        gate_params = dh * dh + dh * dh + dh

        # Q/K morph gates:
        #   2 x RMSNorm(d_h): 2*d_h
        #   2 x W_Q/K_h(d_h → d_model, no bias): 2*d_h*d_model
        qk_gate_params = 2 * dh + 2 * dh * d

        # Content→morph feedback:
        #   RMSNorm(d_model): d
        #   W_h_feedback(d_model → d_h, no bias): d*d_h
        feedback_params = d + d * dh

        # Per theta block attention:
        #   norm(d): d
        #   Q/K/V projections: 3 * d * d
        #   out_proj: d * d
        #   Total: 4*d*d + d (RMSNorm has d params)
        attn_per_theta = 4 * d * d + d

        # Per theta block HebbianFFN norms (norm_x + norm_y): 2*d
        # The W_enc_x, W_enc_y, W_dec are the unknowns we're solving for
        ffn_norms_per_theta = 2 * d

        # Halt head: W_halt(d→1, with bias) + norm(d)
        halt_params = d + 1 + d if self.enable_halting else 0

        # Sum all fixed params
        fixed = (
            embed_params
            + readout_params
            + seed_params
            + conv_params
            + morph_norm_params
            + rule_params
            + gate_params
            + qk_gate_params
            + feedback_params
            + n_th * (attn_per_theta + ffn_norms_per_theta)
            + halt_params
        )

        # Solve: target = fixed + n_theta * 3 * d_model * d_ff
        d_ff = (self.target_params - fixed) // (n_th * 3 * d)

        return max(d_ff, 64)  # safety floor

    def __post_init__(self) -> None:
        """Auto-compute d_ff if set to 0, then validate."""
        # Convert morph_dilations from list to tuple if needed
        if isinstance(self.morph_dilations, list):
            self.morph_dilations = tuple(self.morph_dilations)
        if self.d_ff == 0:
            self.d_ff = self.compute_d_ff()
        self.validate()
