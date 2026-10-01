"""SwiGLU-like Integration FFN for AENCA V15.

Identical to V14's IntegrationFFN. Post-NCA integration:
the RD-NCA evolution already performs cross-position communication,
so the FFN only needs local per-position computation.

Architecture:
    z_norm = RMSNorm(z)
    gate   = SiLU(W_gate(z_norm))
    up     = W_up(z_norm)
    output = W_down(gate * up)          — ZERO-INIT

Reference: V14 IntegrationFFN (identical structure).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntegrationFFN(nn.Module):
    """SwiGLU-like post-NCA integration FFN.

    Parameters: 3 × d_model × d_ff + d_model (norm) per theta block.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()

        self.norm = nn.RMSNorm(d_model)

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
        gate = F.silu(self.W_gate(z_norm))
        up = self.W_up(z_norm)
        hidden = self.dropout(gate * up)
        return self.W_down(hidden)
