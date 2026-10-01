"""SwiGLU-like Integration FFN for AENCA V14 — Emergent Dynamics.

Replaces V13's HebbianFFN. Post-NCA-communication integration:
the NCA morphogen evolution already performs cross-modal mixing (content
perception + morphogen dynamics), so the FFN only needs to integrate
the readout. SwiGLU structure: W_down(SiLU(W_gate(z)) * W_up(z)).

Architecture:
    z_norm = RMSNorm(z)
    gate   = SiLU(W_gate(z_norm))       — gating path
    up     = W_up(z_norm)               — value path
    output = W_down(gate * up)          — decode (ZERO-INIT)

Unlike V13 HebbianFFN which takes two inputs (pre/post attention),
IntegrationFFN takes only the current content state since the NCA
communication has already mixed cross-position information.

Reference: V13 HebbianFFN — 3 × d_model × d_ff parameter pattern
Reference: PaLM / LLaMA — SwiGLU gating pattern
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntegrationFFN(nn.Module):
    """SwiGLU-like post-NCA integration FFN.

    Takes a single input z (content after NCA readout has been applied)
    and produces a residual update.

    Parameters: 3 × d_model × d_ff + d_model (norm) per theta block.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()

        self.norm = nn.RMSNorm(d_model)

        # Gating and value paths
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Gate/Up: xavier normal. Down: ZERO (cascade)."""
        nn.init.xavier_normal_(self.W_gate.weight, gain=1.0)
        nn.init.xavier_normal_(self.W_up.weight, gain=1.0)
        nn.init.zeros_(self.W_down.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute SwiGLU FFN output.

        Args:
            z: (B, L, d_model) content state after NCA readout

        Returns:
            output: (B, L, d_model) FFN output (to be added as residual)
        """
        z_norm = self.norm(z)
        gate = F.silu(self.W_gate(z_norm))          # (B, L, d_ff)
        up = self.W_up(z_norm)                       # (B, L, d_ff)
        hidden = self.dropout(gate * up)             # (B, L, d_ff)
        return self.W_down(hidden)                   # (B, L, d_model)
