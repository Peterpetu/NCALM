"""Fast kernel for AENCA V6 NCA step.

The fast kernel operates at every NCA step. It receives local perception
(conv output), oscillator-derived slow context, and position encoding as
inputs, and produces a gated state update for the cell state z.

V6 uses ONE shared kernel — temporal dynamics are controlled externally
by the coupled oscillator network, not by internal phase structure.

Architecture:
    Input: [P_i; slow_ctx_i; PE(i)] → W_proj → N_trunk SwiGLU blocks → [Δz, g]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v6.config import V6Config


class SwiGLUBlock(nn.Module):
    """Single SwiGLU residual block (same structure as V5).

    h_ℓ = h_{ℓ-1} + α_res · W_down · (SiLU(W_gate · RMSNorm(h)) ⊙ W_up · RMSNorm(h))
    """

    def __init__(self, d_ff: int, alpha_res: float):
        super().__init__()
        self.norm = nn.RMSNorm(d_ff)
        self.W_gate = nn.Linear(d_ff, d_ff, bias=False)
        self.W_up = nn.Linear(d_ff, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_ff, bias=False)
        self.alpha_res = alpha_res

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        h_norm = self.norm(h)
        gate = F.silu(self.W_gate(h_norm))
        up = self.W_up(h_norm)
        return h + self.alpha_res * self.W_down(gate * up)


class FastKernel(nn.Module):
    """NCA kernel with SwiGLU trunk.

    Single shared kernel for all NCA steps. Receives oscillator-derived
    slow context as conditioning instead of a separate z_H state vector.
    The d_kernel_in dimension is d_model + d_model + d_pe.

    Architecture:
        Input: [P_i; slow_ctx_i; PE(i)] → W_proj → N_trunk SwiGLU → [Δz, gate]

    The kernel produces:
        - Δz ∈ ℝ^d_model: state update proposal
        - gate ∈ (0,1)^d_model: per-dimension gate

    The gated update (before oscillator gating):
        z^(t+1) = z^(t) + pos_active · osc_gate · kernel_gate · Δz
    """

    def __init__(self, config: V6Config):
        """Initialize the fast kernel.

        Args:
            config: V6 configuration
        """
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.d_kernel_in = config.d_kernel_in

        # Normalize slow context before concatenation
        self.ctx_norm = nn.RMSNorm(config.d_model)

        # Input projection
        self.W_proj = nn.Linear(config.d_kernel_in, config.d_ff, bias=False)

        # SwiGLU trunk
        alpha_res = config.residual_scale
        self.trunk = nn.ModuleList([
            SwiGLUBlock(config.d_ff, alpha_res)
            for _ in range(config.n_trunk)
        ])

        # Output heads
        self.W_delta = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.W_gate = nn.Linear(config.d_ff, config.d_model, bias=True)

        # Initialize
        self._init_weights(config)

    def _init_weights(self, config: V6Config) -> None:
        """Initialize kernel weights — zero-init cascade."""
        # W_proj: Xavier normal, gain = 1/√d_in
        nn.init.xavier_normal_(
            self.W_proj.weight,
            gain=1.0 / math.sqrt(config.d_kernel_in)
        )

        # SwiGLU blocks
        for block in self.trunk:
            nn.init.xavier_normal_(block.W_gate.weight)
            nn.init.xavier_normal_(block.W_up.weight)
            # W_down: zero-init → kernel output starts near zero
            nn.init.zeros_(block.W_down.weight)

        # W_delta: zero-init (kernel starts producing Δz ≈ 0)
        nn.init.zeros_(self.W_delta.weight)

        # W_gate + bias: gate bias for per-step opening
        # σ(-ln(T_max - 1)) ≈ 1/T_max initial gate opening.
        nn.init.xavier_normal_(self.W_gate.weight, gain=0.1)
        gate_bias = config.gate_bias_init(config.t_max)
        nn.init.constant_(self.W_gate.bias, gate_bias)

    def forward(
        self, P: torch.Tensor, slow_ctx: torch.Tensor, pe: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute state update for one NCA step.

        Args:
            P:        (B, L, d_model) local perception (causal conv output)
            slow_ctx: (B, L, d_model) oscillator-derived slow context
            pe:       (B, L, d_pe)    position encoding

        Returns:
            delta: (B, L, d_model) state update proposal
            gate:  (B, L, d_model) per-dimension gate ∈ (0, 1)
        """
        # Normalize slow context for stable kernel input
        ctx = self.ctx_norm(slow_ctx)

        # Concatenate inputs: [P; slow_ctx_norm; pe]
        x_in = torch.cat([P, ctx, pe], dim=-1)  # (B, L, d_kernel_in)

        # Input projection
        h = self.W_proj(x_in)  # (B, L, d_ff)

        # SwiGLU trunk
        for block in self.trunk:
            h = block(h)

        # Output heads
        delta = self.W_delta(h)                  # (B, L, d_model)
        gate = torch.sigmoid(self.W_gate(h))     # (B, L, d_model) ∈ (0, 1)

        return delta, gate
