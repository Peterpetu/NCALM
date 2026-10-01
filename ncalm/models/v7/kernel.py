"""Slim kernel for AENCA V7 NCA gamma steps.

The SlimKernel is a deliberately shallow network (1 SwiGLU residual block)
that is applied repeatedly at each gamma step. The design follows the NCA
principle of "depth from iteration": instead of stacking 3 SwiGLU blocks
like V6 (which gives 3 × 16 = 48 effective depth — massive overkill), V7
uses 1 block × t_gamma × n_theta iterations = effective depth proportional
to the number of NCA steps.

Architecture:
    Input:  [P; css_ctx; pe]  → (B, L, d_kernel_in)
    W_proj: d_kernel_in → d_ff        (input compression)
    1× SwiGLU residual block at d_ff  (the atomic update function)
    W_delta: d_ff → d_model           (state update proposal, zero-init)
    W_gate:  d_ff → d_model → σ(·)    (per-dimension gate, bias ≈ 1/K)

Gated update (applied in model.py):
    z^(t+1) = z^(t) + gate · Δz

The kernel receives:
    P:       Local perception from causal conv (d_model)
    css_ctx: Global context from Causal State Streams (d_css)
    pe:      Sinusoidal position encoding (d_pe)

Zero-init cascade:
    - SwiGLU W_down: zeros → block initially outputs zeros
    - W_delta: zeros → Δz starts at zero
    - W_gate bias: -ln(total_gamma_steps - 1) → σ ≈ 1/total_gamma_steps
    Model begins as near-identity and learns to update during training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v7.config import V7Config


class SwiGLUBlock(nn.Module):
    """Single SwiGLU residual block.

    h_out = h_in + W_down · (SiLU(W_gate · RMSNorm(h_in)) ⊙ W_up · RMSNorm(h_in))

    No residual scaling (α_res=1.0) since V7 uses only one block.
    """

    def __init__(self, d_ff: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_ff)
        self.W_gate = nn.Linear(d_ff, d_ff, bias=False)
        self.W_up = nn.Linear(d_ff, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_ff, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_norm = self.norm(h)
        return h + self.W_down(F.silu(self.W_gate(h_norm)) * self.W_up(h_norm))


class SlimKernel(nn.Module):
    """NCA kernel with single SwiGLU block — depth from iteration.

    Unlike V6's FastKernel (3 SwiGLU blocks), the SlimKernel uses exactly
    one SwiGLU block. The effective depth comes from iterating the kernel
    across t_gamma gamma steps per theta cycle.

    Input composition:
        [P_i; css_ctx_i; pe_i]
        where P_i = CausalConv1d(z_i) — local spatial perception
              css_ctx_i = CSS(z_i)     — O(L) streamed global context
              pe_i = SinPE(i)          — position encoding

    Output:
        delta: (B, L, d_model) — state update proposal
        gate:  (B, L, d_model) — per-dimension sigmoid gate ∈ (0, 1)

    Parameter count (d_ff=768, d_model=768, d_kernel_in=1088):
        W_proj:  1088 × 768 = 835K
        SwiGLU:  3 × 768²   = 1.77M
        W_delta: 768 × 768   = 590K
        W_gate:  768 × 768   = 590K + 768 bias
        Norms:   ~2 × 768    = 1.5K
        Total:   ~3.8M (vs V6's 11.6M — 3× cheaper)
    """

    def __init__(self, config: V7Config):
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.d_kernel_in = config.d_kernel_in

        # Normalize CSS context before concatenation
        self.ctx_norm = nn.RMSNorm(config.d_css)

        # Input projection: concatenated inputs → d_ff
        self.W_proj = nn.Linear(config.d_kernel_in, config.d_ff, bias=False)

        # Single SwiGLU residual block
        self.swiglu = SwiGLUBlock(config.d_ff)

        # Output heads
        self.W_delta = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.W_gate = nn.Linear(config.d_ff, config.d_model, bias=True)

        # Initialize
        self._init_weights(config)

    def _init_weights(self, config: V7Config) -> None:
        """Initialize kernel weights — zero-init cascade.

        Follows NCA principle: model starts as near-identity.
        All output paths initialized to produce ~zero initially.
        """
        # W_proj: Xavier normal, attenuated gain for stability
        nn.init.xavier_normal_(
            self.W_proj.weight,
            gain=1.0 / math.sqrt(config.d_kernel_in),
        )

        # SwiGLU internals
        nn.init.xavier_normal_(self.swiglu.W_gate.weight)
        nn.init.xavier_normal_(self.swiglu.W_up.weight)
        # W_down: ZERO → SwiGLU block initially outputs zero
        nn.init.zeros_(self.swiglu.W_down.weight)

        # W_delta: ZERO → Δz starts at zero (zero-init cascade)
        nn.init.zeros_(self.W_delta.weight)

        # W_gate: small Xavier + bias for initial gate opening
        # σ(-ln(total_gamma_steps - 1)) ≈ 1/total_gamma_steps
        nn.init.xavier_normal_(self.W_gate.weight, gain=0.1)
        gate_bias = config.gate_bias_init(config.total_gamma_steps)
        nn.init.constant_(self.W_gate.bias, gate_bias)

    def forward(
        self,
        P: torch.Tensor,
        css_ctx: torch.Tensor,
        pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute state update for one NCA gamma step.

        Args:
            P:       (B, L, d_model)  local perception (causal conv output)
            css_ctx: (B, L, d_css)    CSS global context
            pe:      (B, L, d_pe)     position encoding

        Returns:
            delta: (B, L, d_model) state update proposal
            gate:  (B, L, d_model) per-dimension gate ∈ (0, 1)
        """
        # Normalize CSS context for stable concatenation
        ctx = self.ctx_norm(css_ctx)

        # Concatenate inputs: [P; css_ctx_norm; pe]
        x_in = torch.cat([P, ctx, pe], dim=-1)    # (B, L, d_kernel_in)

        # Input projection
        h = self.W_proj(x_in)                      # (B, L, d_ff)

        # Single SwiGLU block
        h = self.swiglu(h)

        # Output heads
        delta = self.W_delta(h)                     # (B, L, d_model)
        gate = torch.sigmoid(self.W_gate(h))        # (B, L, d_model) ∈ (0, 1)

        return delta, gate
