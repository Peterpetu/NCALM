"""Simple Hebbian rule and causal convolution for AENCA V10.

The rule is the "genome" — ONE shared set of weights applied identically
at every position, at every step, at every scale. Only the dilation changes.

V10's kernel is intentionally SMALL (d_rule=800 vs V9's d_concept=9255).
Wolfram: Rule 110 is 8 bits. Complexity comes from iteration, not rule size.
The kernel is 4.9M params — 17× smaller than V9's 59M.

Architecture (per CA step):
    perceived = DepthwiseCausalConv(z, dilation)         # neighborhood
    p = ReLU(W_p · RMSNorm(perceived))                   # perception encoding
    s = ReLU(W_s · RMSNorm(z))                           # self encoding
    delta = W_d · (p ⊙ s)                                # Hebbian update

Zero-init on W_d guarantees Class I still life at initialization.
All structure must be GROWN through training.

Parameters (~4.9M at d=2048, d_rule=800):
    W_p:   d_model × d_rule     (1.6M)
    W_s:   d_model × d_rule     (1.6M)
    W_d:   d_rule × d_model     (1.6M)  ZERO-INIT
    conv:  d_model × kernel_size (6K)
    norms: 2 × d_model          (4K)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v10.config import V10Config


class DepthwiseCausalConv(nn.Module):
    """Shared depthwise causal convolution with runtime dilation.

    Identical to V9's implementation — same local rule, different scales.
    Each channel has its own independent kernel. Causal left-padding
    ensures no future information leak.

    Parameters: d_model × kernel_size (one 1D kernel per channel).
    """

    def __init__(self, d_model: int, kernel_size: int = 3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_model, 1, kernel_size))
        self.kernel_size = kernel_size
        self.d_model = d_model
        nn.init.kaiming_uniform_(self.weight, a=0)

    def forward(self, x: torch.Tensor, dilation: int = 1) -> torch.Tensor:
        """Apply causal depthwise conv with given dilation.

        Args:
            x: (B, L, d_model) cell states
            dilation: Dilation factor for this CA step

        Returns:
            perceived: (B, L, d_model) neighborhood-aggregated states
        """
        h = x.transpose(1, 2)  # (B, d_model, L)
        pad = (self.kernel_size - 1) * dilation
        h = F.pad(h, (pad, 0))
        h = F.conv1d(h, self.weight, groups=self.d_model, dilation=dilation)
        return h.transpose(1, 2)  # (B, L, d_model)


class SimpleHebbianRule(nn.Module):
    """The genome: ONE shared tiny Hebbian update rule for all 64 CA steps.

    This is V10's soul. A deliberately small rule (~5M params) that defines
    the local update, applied 64 times with cyclic dilations. Complexity
    from iteration, not from rule size.

    Same Hebbian product as V9 (p ⊙ s binding), but with d_rule=800
    instead of d_concept=9255. The rule is 17× simpler.

    Zero-init cascade (W_d = 0):
        At init, delta = 0 everywhere → all cells are still lifes (Class I).
        Growth must be LEARNED through training: I → II → IV.
    """

    def __init__(self, config: V10Config):
        super().__init__()
        d = config.d_model
        dr = config.d_rule

        # Perception: depthwise causal conv (shared across all dilations)
        self.conv = DepthwiseCausalConv(d, config.conv_kernel_size)

        # Pre-norms
        self.norm_perceived = nn.RMSNorm(d)
        self.norm_z = nn.RMSNorm(d)

        # Encoders: d_model → d_rule (small!)
        self.W_p = nn.Linear(d, dr, bias=False)
        self.W_s = nn.Linear(d, dr, bias=False)

        # Decoder: d_rule → d_model (ZERO-INIT)
        self.W_d = nn.Linear(dr, d, bias=False)

        # Dropout on Hebbian product
        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Encoders: Xavier normal. Decoder: ZERO (Class I still life start)."""
        nn.init.xavier_normal_(self.W_p.weight, gain=1.0)
        nn.init.xavier_normal_(self.W_s.weight, gain=1.0)
        nn.init.zeros_(self.W_d.weight)  # THE zero-init cascade

    def forward(self, z: torch.Tensor, dilation: int = 1) -> torch.Tensor:
        """Apply one CA step: perceive, encode, bind, decode.

        Args:
            z: (B, L, d_model) current cell states
            dilation: Conv dilation for this step's receptive field scale

        Returns:
            delta: (B, L, d_model) state update proposal Δz
        """
        perceived = self.conv(z, dilation=dilation)

        p = F.relu(self.W_p(self.norm_perceived(perceived)))  # (B, L, d_rule)
        s = F.relu(self.W_s(self.norm_z(z)))                  # (B, L, d_rule)

        # Hebbian binding: only mutually active concepts fire
        delta = self.W_d(self.dropout(p * s))  # (B, L, d_model)

        return delta
