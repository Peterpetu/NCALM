"""AENCA V7 configuration — Stream-Hierarchical NCA.

V7 architecture: two-timescale hierarchical NCA with Causal State Streams
for O(L) global communication and sparse full attention at theta boundaries.

Key design:
  - Two-timescale hierarchy: n_theta theta cycles × t_gamma gamma steps
  - Gamma steps: shared-weight NCA kernel (local conv + CSS + SlimKernel)
  - Theta steps: unique-weight attention + ultra-wide SwiGLU FFN
  - Causal State Stream (CSS): novel O(L) global communication with
    state-dependent decay, multi-stream architecture, zero-init
  - Per-position adaptive halting at theta boundaries (PonderNet-style)
  - Slim 1-SwiGLU kernel: "depth from iteration" — t_gamma iterations
    of 1 block = effective depth t_gamma per theta cycle

Architectural thesis:
  The NCA kernel + CSS iterate within each theta cycle (gamma loop),
  refining local + streamed-global representations. At theta boundaries,
  full causal attention + wide FFN provides high-bandwidth global mixing.
  Halting at theta boundaries allows easy tokens to skip later cycles.

  Total effective depth = n_theta × (t_gamma + 1 attention + 1 FFN)
  With n_theta=4, t_gamma=3: 4 × (3 kernel + 1 attn + 1 FFN) = 20 ops
  Compare GPT-2: 12 × (1 attn + 1 FFN) = 24 ops — similar compute.

Reference: "Depth from Iteration" — NCA kernel is shallow; loop provides depth.
Reference: DeepSeek — ultra-wide FFN outperforms deep narrow at equal FLOPs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class V7Config:
    """Complete configuration for AENCA V7: Stream-Hierarchical NCA.

    Architecture (per forward pass):
        embed + SinPE → z₀ (B, L, d_model)
        For θ in 1..n_theta:
            For γ in 1..t_gamma:                       [GAMMA: shared weights]
                P = CausalConv1d(RMSNorm(z))           [local perception]
                css_ctx = CSS(z)                       [O(L) global context]
                Δz, gate = SlimKernel([P; css_ctx; pe]) [1 SwiGLU update]
                z = z + gate · Δz                       [NCA gated residual]
            z = ThetaBlock_θ(z)                        [THETA: unique attention+FFN]
            halt_prob = σ(W_halt(z))                   [readout accumulation]
            z_out += min(halt_prob, remaining) · z
        logits = Readout(z_out)

    Causal State Stream (CSS) — novel global communication:
        N parallel streams, each with d_state dimensions.
        Per position: write gated content, apply state-dependent decay,
        read compressed summary. O(L) per step, applied at every gamma step.
        Key novelty: decay is state-dependent (learned from z), not fixed.
    """

    # ── Dimensions ─────────────────────────────────────────────────────
    d_model: int = 768          # Cell state dimension (= embedding dim)
    d_ff: int = 768             # SlimKernel feed-forward dimension
    d_pe: int = 64              # Sinusoidal position encoding dimension

    # ── Causal State Stream (CSS) ──────────────────────────────────────
    n_streams: int = 8          # Number of parallel state streams
    d_state: int = 32           # Per-stream hidden state dimension
    d_css: int = 256            # CSS output projection dimension

    # ── Hierarchical Timing ────────────────────────────────────────────
    n_theta: int = 4            # Theta cycles (each has unique attn+FFN)
    t_gamma: int = 3            # Gamma steps per theta cycle (shared kernel)

    # ── Theta Attention ────────────────────────────────────────────────
    n_heads: int = 12           # Attention heads per theta block
    d_head: int = 64            # Per-head dimension
    d_ff_theta: int = 0         # Theta FFN width (0 = auto-size to fill param budget)

    # ── Hebbian Kernel (BDH-inspired, replaces SlimKernel + CSS) ──────
    d_concept: int = 2048       # Concept space dim for Hebbian gamma kernel
    kernel_radius: int = 2      # Causal conv radius (kernel_size = r + 1)

    # ── Adaptive Halting ───────────────────────────────────────────────
    enable_halting: bool = True     # Per-position readout accumulation
    halt_eps: float = 0.01          # Hard halt threshold (remaining mass)
    lambda_halt: float = 0.01       # PonderNet KL regularization weight

    # ── Readout ────────────────────────────────────────────────────────
    mos_components: int = 1     # MoS components (1 = standard softmax)
    tie_embeddings: bool = True # Tie readout to embedding matrix

    # ── Vocabulary and Sequence ────────────────────────────────────────
    vocab_size: int = 10_000    # |V| — vocabulary size
    max_seq_len: int = 256      # L — max sequence length

    # ── Loss Weights ───────────────────────────────────────────────────
    lambda_reg: float = 0.01    # State regularization: ‖z_out - z₀‖²/d
    lambda_aux: float = 0.05    # Deep supervision at theta boundaries

    # ── Gradient Stabilization ─────────────────────────────────────────
    lambda_agc: float = 0.02    # Adaptive gradient clipping ratio

    # ── Training ───────────────────────────────────────────────────────
    dropout: float = 0.0        # Dropout (reserved for future use)
    seq_len: int = 256          # Sequence length for training (profile override)
    checkpoint_interval: int = 4  # For GPT2 compat in trainer

    # ── Derived Properties ─────────────────────────────────────────────

    @property
    def total_gamma_steps(self) -> int:
        """Total gamma (shared kernel) steps across all theta cycles."""
        return self.n_theta * self.t_gamma

    @property
    def total_steps(self) -> int:
        """Total NCA-equivalent steps (gamma steps + theta attention steps).

        Used by trainer for logging compatibility with V5/V6.
        """
        return self.n_theta * (self.t_gamma + 1)  # +1 for theta attention per cycle

    @property
    def K(self) -> int:
        """Backward compat alias for total_steps (trainer logging)."""
        return self.total_steps

    @property
    def d_kernel_in(self) -> int:
        """HebbianKernel input dimension (legacy compat).

        V7.1: HebbianKernel takes P (d_model) and z (d_model) separately.
        This property is kept for backward compatibility.
        """
        return self.d_model + self.d_css + self.d_pe

    @property
    def residual_scale(self) -> float:
        """Residual scaling for theta block output projections.

        1/√(2 × n_theta): scaling for attention + FFN output projections,
        analogous to GPT-2's 1/√(2 × n_layer).
        """
        return 1.0 / math.sqrt(2.0 * self.n_theta) if self.n_theta > 0 else 1.0

    @property
    def css_state_dim(self) -> int:
        """Total CSS state dimension: n_streams × d_state."""
        return self.n_streams * self.d_state

    def gate_bias_init(self, num_steps: int) -> float:
        """Gate bias b_g = -ln(K-1) → σ(b_g) ≈ 1/K initial gate opening.

        For SlimKernel: num_steps = total_gamma_steps.
        For halt head: num_steps = n_theta.
        """
        return -math.log(max(num_steps - 1, 1))

    def compute_d_ff_theta(self, target_params: int = 162_000_000) -> int:
        """Auto-compute d_ff_theta to reach target total parameter count.

        Given all other dimensions, solves for d_ff_theta such that
        total model params ≈ target_params.

        V7.1 model parameter budget:
            embed = vocab × d_model                        (tied → no separate readout)
            hebbian_kernel = 3 × d_model × d_concept + norms  (replaces SlimKernel+CSS)
            conv = d_model × d_model × (kernel_radius + 1) + norms
            per theta block:
                attn_qkv = 3 × d_model × d_model
                attn_out = d_model × d_model
                hebbian_ffn = 3 × d_model × d_ff_theta  (enc_x + enc_y + dec)
                norms ≈ 3 × d_model  (pre-attn norm + 2 Hebbian norms)
            halt = d_model + 1 (if halting)
            norms & biases ≈ small

        Returns:
            d_ff_theta: auto-computed FFN width for theta blocks
        """
        d = self.d_model
        V = self.vocab_size
        n_th = self.n_theta

        # Embedding (tied → appears once, shared with readout)
        embed_params = V * d

        # If not tied, readout has separate weight + norm
        if not self.tie_embeddings:
            embed_params += V * d  # separate readout projection

        # Readout norm
        readout_norm = d

        # HebbianKernel: W_enc_p + W_enc_z + W_dec + norms
        dc = self.d_concept
        kernel_params = (
            d * dc                    # W_enc_p
            + d * dc                   # W_enc_z
            + dc * d                   # W_dec
            + 2 * d                    # norm_p + norm_z (RMSNorm)
        )

        # CausalConv1d + norm
        conv_params = d * d * (self.kernel_radius + 1) + d

        # Per theta block (attention + HebbianFFN), excluding d_ff_theta part:
        attn_params_per_block = (
            3 * d * d                  # Q, K, V projections
            + d * d                    # output projection
            + 3 * d                    # 3 RMSNorms (pre-attn + 2 Hebbian norms)
        )

        # Halt head (if enabled)
        halt_params = (d + 1 + d) if self.enable_halting else 0  # W + bias + norm

        # Fixed params (everything except theta FFN)
        fixed = (
            embed_params
            + readout_norm
            + kernel_params
            + conv_params
            + n_th * attn_params_per_block
            + halt_params
        )

        # Remaining budget for theta HebbianFFN: n_theta × 3 × d × d_ff_theta
        # (HebbianFFN: W_enc_x + W_enc_y + W_dec = 3 × d × d_ff_theta)
        remaining = target_params - fixed
        ffn_per_unit = 3 * d  # params per unit of d_ff_theta per theta block
        d_ff_theta = max(remaining // (n_th * ffn_per_unit), 256)

        return int(d_ff_theta)
