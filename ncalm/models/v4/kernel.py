"""Shared NCA kernel f_theta for AENCA V4.1.

The kernel is a SwiGLU-based MLP applied identically at every position and every
NCA step — it IS the uniform local rule that makes this a true NCA.

Architecture (V4.1 Eq 4.8-4.17):
    Input:  x_in = [P_i, o_i, psi_i]  in R^{d_in=2016}
    Projection: Linear(d_in, d_hidden=768)
    Trunk:  4x SwiGLUBlock with RMSNorm pre-norm + residual (alpha_res=0.5)
    Output: (Delta_s, l_g, l_a) — state update, gate logits, action logits

SwiGLU block (Eq 4.11-4.14):
    Pre-norm: RMSNorm(d_hidden)
    Gate:     SiLU(W_gate @ h)   W_gate: (d_ff, d_hidden), no bias
    Up:       W_up @ h           W_up:   (d_ff, d_hidden), no bias
    Down:     W_down @ (gate * up)  W_down: (d_hidden, d_ff), no bias
    Residual: x + alpha_res * down_output

Initialization (V4.1 §4.5, Eq 4.27, 5.30):
    - Input projection: Xavier uniform, gain=0.5
    - SwiGLU weights: Xavier uniform, gain=0.5
    - RMSNorm: default (ones)
    - W_Delta_s: Xavier then scale by 1/sqrt(2K)
    - Gate bias: b_g = -ln(K - 1) -> sigma ~ 0.042 for K=24
    - Gate weight: zero-initialized
    - Action head: default Xavier

Reference: model_documents/aenca_v4_1_architecture.md §4.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ncalm.models.v4.config import AENCAv4Config


class SwiGLUBlock(nn.Module):
    """SwiGLU feed-forward block with RMSNorm pre-norm and residual (V4.1 Eq 4.11-4.14).

    Pre-norm: RMSNorm(d_hidden)
    SwiGLU:   down(SiLU(gate(x)) * up(x))
    Residual: x + alpha_res * output
    """

    def __init__(self, d_hidden: int, d_ff: int, alpha_res: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_hidden)
        self.w_gate = nn.Linear(d_hidden, d_ff, bias=False)
        self.w_up = nn.Linear(d_hidden, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_hidden, bias=False)
        self.alpha_res = alpha_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU block with pre-norm and residual.

        Args:
            x: (B, L, d_hidden)

        Returns:
            x_out: (B, L, d_hidden)
        """
        h = self.norm(x)
        gate = torch.nn.functional.silu(self.w_gate(h))
        up = self.w_up(h)
        out = self.w_down(gate * up)
        return x + self.alpha_res * out


class NCAKernel(nn.Module):
    """Shared NCA kernel f_theta (V4.1 Definition 5.6).

    This single module is called at every position and every NCA step.
    It has NO step index or position index as input — locality and uniformity
    are architectural guarantees (NCA Axioms 1, 3).

    Output heads:
        Delta_s:  R^D        — proposed state update (Eq 5.21)
        l_g:      R^D        — pre-sigmoid gate logits (Eq 5.22)
        l_a:      R^{h*|A|}  — action logits for all sensor heads (Eq 5.23)
    """

    def __init__(self, config: AENCAv4Config) -> None:
        super().__init__()
        self.config = config

        # ── Input projection: d_in → d_hidden (Eq 4.8) ────────────────
        self.input_proj = nn.Linear(config.d_in, config.d_hidden)

        # ── Trunk: N SwiGLU blocks with pre-norm residual (Eq 4.10) ───
        self.trunk = nn.ModuleList([
            SwiGLUBlock(config.d_hidden, config.d_ff, config.alpha_res)
            for _ in range(config.n_trunk_layers)
        ])

        # ── Output heads (Eq 5.21-5.23) ───────────────────────────────
        self.head_delta_s = nn.Linear(config.d_hidden, config.D)
        self.head_gate = nn.Linear(config.d_hidden, config.D)
        self.head_action = nn.Linear(
            config.d_hidden, config.n_sensor_heads * config.n_actions
        )

        self._init_weights(config)

    def _init_weights(self, config: AENCAv4Config) -> None:
        """Apply V4.1 initialization scheme (§4.5, Eq 4.27, 5.30)."""

        # Input projection: Xavier uniform, gain=0.5
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.5)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

        # SwiGLU weights: Xavier uniform, gain=0.5
        for block in self.trunk:
            for linear in [block.w_gate, block.w_up, block.w_down]:
                nn.init.xavier_uniform_(linear.weight, gain=0.5)
            # RMSNorm: default init (ones) — no action needed

        # W_Delta_s: Xavier init then scale by 1/sqrt(2K) (Eq 5.30)
        nn.init.xavier_uniform_(self.head_delta_s.weight)
        self.head_delta_s.weight.data.mul_(config.delta_s_scale)
        if self.head_delta_s.bias is not None:
            nn.init.zeros_(self.head_delta_s.bias)

        # Gate: weight = 0, bias = -ln(K-1) (V4.1 Eq 4.27)
        nn.init.zeros_(self.head_gate.weight)
        nn.init.constant_(self.head_gate.bias, config.gate_bias_init)

        # Action head: Xavier uniform
        nn.init.xavier_uniform_(self.head_action.weight)
        if self.head_action.bias is not None:
            nn.init.zeros_(self.head_action.bias)

    def forward(
        self, x_in: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply kernel to concatenated perception/observation/encoding input.

        Args:
            x_in: Shape (B, L, d_in) — concatenation of [P_i, o_i, psi_i].

        Returns:
            delta_s:       Shape (B, L, D) — proposed state update.
            gate_logits:   Shape (B, L, D) — pre-sigmoid gate logits.
            action_logits: Shape (B, L, h*|A|) — action logits for sensors.
        """
        h = self.input_proj(x_in)  # (B, L, d_hidden)
        for block in self.trunk:
            h = block(h)  # residual SwiGLU inside
        delta_s = self.head_delta_s(h)  # (B, L, D)
        gate_logits = self.head_gate(h)  # (B, L, D)
        action_logits = self.head_action(h)  # (B, L, h*|A|)
        return delta_s, gate_logits, action_logits
