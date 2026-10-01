"""Phase-Specialized Kernel Bank for AENCA V5.

Implements the SwiGLU trunk with per-dimension gating (§5).
Each phase has an independent kernel with the same architecture but
separate parameters. The kernel takes concatenated local perception,
PRMP messages, and position encoding as input.

Reference: model_documents/aenca_v5_architecture.md §5.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v5.config import V5Config


class SwiGLUBlock(nn.Module):
    """Single SwiGLU residual block (§5.3, Eq 5.5-5.6).

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
        """Forward pass with residual connection.

        Args:
            h: (*, d_ff) input hidden state

        Returns:
            h_out: (*, d_ff) output with residual
        """
        h_norm = self.norm(h)
        gate = F.silu(self.W_gate(h_norm))
        up = self.W_up(h_norm)
        return h + self.alpha_res * self.W_down(gate * up)


class PhaseKernel(nn.Module):
    """Phase-specialized NCA kernel with SwiGLU trunk (§5.3).

    Architecture:
        Input: [P_i; M_i; PE(i)] → W_proj → N_trunk SwiGLU blocks → [Δs, g]

    Each kernel produces:
        - Δs_i ∈ ℝ^{d_model}: state update proposal
        - g_i ∈ (0,1)^{d_model}: per-dimension gate

    The gated update is: s^(k+1) = s^(k) + g_i ⊙ Δs_i  (Eq 6.5)
    """

    def __init__(self, config: V5Config, phase_steps: int):
        """Initialize a single phase kernel.

        Args:
            config: V5 configuration
            phase_steps: number of NCA steps in this phase (for gate bias init)
        """
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.d_kernel_in = config.d_kernel_in

        # Input projection (§5.3, Eq 5.4)
        self.W_proj = nn.Linear(config.d_kernel_in, config.d_ff, bias=False)

        # SwiGLU trunk (§5.3, Eq 5.5-5.6)
        alpha_res = config.residual_scale
        self.trunk = nn.ModuleList([
            SwiGLUBlock(config.d_ff, alpha_res)
            for _ in range(config.n_trunk)
        ])

        # Output heads (§5.3, Eq 5.7-5.8)
        self.W_delta_s = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.W_gate = nn.Linear(config.d_ff, config.d_model, bias=True)

        # Initialize (§10.4)
        self._init_weights(config, phase_steps)

    def _init_weights(self, config: V5Config, phase_steps: int) -> None:
        """Initialize kernel weights per §10.4."""
        # W_proj: Xavier normal, gain=1/√d_in (Eq 10.5)
        nn.init.xavier_normal_(
            self.W_proj.weight,
            gain=1.0 / math.sqrt(config.d_kernel_in)
        )

        # SwiGLU blocks: Xavier normal for W_gate, W_up (Eq 10.5)
        for block in self.trunk:
            nn.init.xavier_normal_(block.W_gate.weight)
            nn.init.xavier_normal_(block.W_up.weight)
            # W_down: Zero initialization (Eq 10.5) — kernel output starts near zero
            nn.init.zeros_(block.W_down.weight)

        # W_Δs: Zero initialization (kernel starts with Δs ≈ 0)
        nn.init.zeros_(self.W_delta_s.weight)

        # W_gate + bias: gate bias = -ln(K_p - 1) (Eq 10.6)
        nn.init.xavier_normal_(self.W_gate.weight, gain=0.1)
        gate_bias = config.gate_bias_init(phase_steps)
        nn.init.constant_(self.W_gate.bias, gate_bias)

    def forward(
        self, P: torch.Tensor, M: torch.Tensor, pe: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute state update and gate for one NCA step.

        Args:
            P:  (B, L, d_model) local perception (causal conv output)
            M:  (B, L, d_model) PRMP message aggregate
            pe: (B, L, d_pe)    position encoding

        Returns:
            delta_s: (B, L, d_model) state update proposal
            gate:    (B, L, d_model) per-dimension gate ∈ (0, 1)
        """
        # Concatenate inputs (§5.3, Eq 5.3)
        x_in = torch.cat([P, M, pe], dim=-1)  # (B, L, d_kernel_in)

        # Input projection (Eq 5.4)
        h = self.W_proj(x_in)  # (B, L, d_ff)

        # SwiGLU trunk (Eq 5.5-5.6)
        for block in self.trunk:
            h = block(h)

        # Output heads (Eq 5.7-5.8)
        delta_s = self.W_delta_s(h)             # (B, L, d_model)
        gate = torch.sigmoid(self.W_gate(h))    # (B, L, d_model) ∈ (0, 1)

        return delta_s, gate
