"""AENCA V9 configuration — Wolfram Cellular Automata Language Model.

V9 architecture: radical simplification based on three first principles:

  - Wolfram: Complex behavior from a SINGLE simple local rule applied uniformly.
    One shared HebbianKernel (the "genome") is applied 3 times with increasing
    dilation — always the same rule, at different scales.

  - Levin: Structure IS function. Wide cells (d=2048) carry rich representations.
    The grown cell state z IS the language model representation — no separate
    attention/FFN hierarchy needed.

  - Zenil: The simplest program that generates observed complexity is most probable.
    Only 9 GEMMs total (3 steps × 3 matmuls). No attention, no halting,
    no alpha channel, no settlement gating.

Design principles:
  - d_model=2048: 2.7× GPT-2's 768. Wide cells = rich substrate for complexity.
  - ONE shared kernel: ~59M params shared across all 3 steps (the genome).
  - Dilated causal conv: [1, r, r²] geometric dilations for full receptive field.
  - Zero-init W_dec: starts as Class I still life, learns to grow.
  - Tied embeddings: 103M params (63%) — the vocabulary IS the readout.
  - 9 GEMMs total: matches GPT-2 throughput (~45K tok/s on RTX 4080).

Forward pass (3 shared CA steps):
    z = Embed(x) + PE                              # Seed
    For step i with dilation d_i:
        perceived = DepthwiseCausalConv(z, d=d_i)   # Local neighborhood
        p = ReLU(W_enc_p · RMSNorm(perceived))      # Perception encoding
        s = ReLU(W_enc_z · RMSNorm(z))              # Self encoding
        z = z + W_dec · (p ⊙ s)                     # Hebbian update
    logits = RMSNorm(z) · Embed^T                   # Tied readout
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class V9Config:
    """Complete configuration for AENCA V9: Wolfram Cellular Automata.

    Parameter budget (auto-sized):
        embed (tied)   = vocab_size × d_model            (~103M at defaults)
        pos_embed      = max_seq_len × d_model           (~2M at defaults)
        kernel         = 3 × d_model × d_concept         (~59M auto-sized)
        conv (shared)  = d_model × kernel_size            (~6K)
        norms          = 3 × d_model                      (~6K)
        ─────────────────────────────────────────────────────
        Total          ≈ target_params                    (~162M at defaults)

    d_concept is auto-sized to fill the remaining budget after embedding.
    """

    # ── Core Dimensions ───────────────────────────────────────────────
    d_model: int = 2048         # Cell state width (2.7× GPT-2)
    d_concept: int = 0          # Concept space dim (0 = auto-size from target_params)
    n_steps: int = 3            # CA iterations (each with different dilation)

    # ── Causal Convolution ────────────────────────────────────────────
    conv_kernel_size: int = 3   # Depthwise conv kernel size (shared across steps)

    # ── Vocabulary and Sequence ───────────────────────────────────────
    vocab_size: int = 50304     # |V| — vocabulary size
    max_seq_len: int = 1024     # L — max sequence length

    # ── Embeddings ────────────────────────────────────────────────────
    tie_embeddings: bool = True # Tie readout to embedding (saves 103M params)

    # ── Regularization ────────────────────────────────────────────────
    dropout: float = 0.0        # Dropout rate in Hebbian product
    lambda_perm: float = 0.01   # Permanence loss weight (encourages convergence)
    lambda_spectral: float = 0.001  # Spectral norm loss weight (bounds dynamics)

    # ── Parameter Budget ──────────────────────────────────────────────
    target_params: int = 0      # 0 = don't auto-size d_concept

    # ── Training ──────────────────────────────────────────────────────
    checkpoint_interval: int = 4    # Gradient checkpointing interval

    # ══════════════════════════════════════════════════════════════════
    # Derived Properties
    # ══════════════════════════════════════════════════════════════════

    @property
    def dilations(self) -> list[int]:
        """Auto-compute geometric dilations for full receptive field coverage.

        Uses geometric progression [1, r, r², ...] where r is chosen so that
        the composed receptive field covers the full sequence length.

        Receptive field = 1 + (k-1) × Σ d_i  for composed causal convolutions.
        """
        k = self.conv_kernel_size
        n = self.n_steps
        L = self.max_seq_len

        if n <= 1:
            return [1]

        # Need: 1 + (k-1) * sum(dilations) >= L
        # With geometric [1, r, r^2, ...], sum = (r^n - 1) / (r - 1)
        # Approximate: r^(n-1) ≈ (L - 1) / (k - 1)
        target_sum = (L - 1) / (k - 1)
        r = max(2, int(math.ceil(target_sum ** (1.0 / (n - 1)))))
        return [r ** i for i in range(n)]

    @property
    def receptive_field(self) -> int:
        """Total receptive field of the composed CA steps."""
        return 1 + (self.conv_kernel_size - 1) * sum(self.dilations)

    def compute_d_concept(self, target_params: int = 0) -> int:
        """Auto-compute d_concept to reach target total parameter count.

        Budget breakdown:
            embed (tied)   = vocab_size × d_model
            pos_embed      = max_seq_len × d_model
            kernel W_enc_p = d_model × d_concept
            kernel W_enc_z = d_model × d_concept
            kernel W_dec   = d_concept × d_model
            conv weight    = d_model × kernel_size  (depthwise, shared)
            norms          = 3 × d_model  (norm_perceived, norm_z, final_norm)

        Returns:
            d_concept: Auto-computed concept space dimension.
        """
        target = target_params or self.target_params
        if target <= 0:
            return self.d_concept

        d = self.d_model
        V = self.vocab_size

        # Fixed params (everything except kernel linear layers)
        embed_params = V * d                          # tied embedding
        if not self.tie_embeddings:
            embed_params += V * d                     # separate readout head
        pe_params = self.max_seq_len * d              # positional embedding
        conv_params = d * self.conv_kernel_size       # depthwise (groups=d)
        norm_params = 3 * d                           # 2 kernel norms + final norm
        fixed = embed_params + pe_params + conv_params + norm_params

        # Kernel linear layers: 3 × d × d_concept
        remaining = target - fixed
        d_concept = remaining // (3 * d)

        return max(d_concept, 64)  # Floor at 64 to avoid degenerate models
