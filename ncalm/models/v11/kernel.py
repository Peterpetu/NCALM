"""Hebbian rule, morphogen rule, and causal convolution for AENCA V11.

V11 has TWO coupled local rules:

  1. Content rule (SimpleHebbianRule): The genome for content z.
     Same Hebbian product as V10, but now the perceived neighborhood is
     GATED by the morphogen field, enabling emergent selective routing.

  2. Morphogen rule (MorphogenRule): A tiny MLP that evolves morphogen h.
     h modulates HOW cells perceive each other's content. The morphogen
     creates a substrate for emergent information routing without prescribing
     what computational structures form.

Both rules share the same DepthwiseCausalConv architecture (one instance per
field, with field-appropriate channel count). Both are local. Both are shared
across all 64 time steps. All complexity emerges from coupled iteration.

Content rule parameters (~4.9M at d=2048, d_rule=800):
    W_p:   d_model × d_rule     (1.6M)
    W_s:   d_model × d_rule     (1.6M)
    W_d:   d_rule × d_model     (1.6M)  ZERO-INIT
    conv:  d_model × kernel_size (6K)
    norms: 2 × d_model          (4K)

Morphogen rule parameters (~49K at d_h=128):
    W_h1:  2·d_h × d_h + d_h    (33K)   Xavier init
    W_h2:  d_h × d_h            (16K)   ZERO-INIT
    conv:  d_h × kernel_size    (384)
    norm:  2·d_h                (256)

Zero-init cascade:
    W_d = 0 → content delta = 0 at init (all cells are Class I still lifes)
    W_h2 = 0 → morphogen delta = 0 at init (morphogen starts frozen too)
    Both fields must be GROWN through training. I → II → IV.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v11.config import V11Config


class DepthwiseCausalConv(nn.Module):
    """Shared depthwise causal convolution with runtime dilation.

    Each channel has its own independent kernel. Causal left-padding
    ensures no future information leak.

    Used for BOTH content perception (d_model channels) and morphogen
    perception (d_h channels). One instance per field.

    Parameters: channels × kernel_size.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))
        self.kernel_size = kernel_size
        self.channels = channels
        nn.init.kaiming_uniform_(self.weight, a=0)

    def forward(self, x: torch.Tensor, dilation: int = 1) -> torch.Tensor:
        """Apply causal depthwise conv with given dilation.

        Args:
            x: (B, L, channels) cell states
            dilation: Dilation factor for this CA step

        Returns:
            perceived: (B, L, channels) neighborhood-aggregated states
        """
        h = x.transpose(1, 2)  # (B, channels, L)
        pad = (self.kernel_size - 1) * dilation
        h = F.pad(h, (pad, 0))
        h = F.conv1d(h, self.weight, groups=self.channels, dilation=dilation)
        return h.transpose(1, 2)  # (B, L, channels)


class SimpleHebbianRule(nn.Module):
    """The content genome: ONE shared Hebbian update rule for all 64 CA steps.

    V11 extension: the perceived neighborhood can be GATED by an external
    signal (the morphogen gate). When gate=None, behavior is identical to V10.

    Architecture (per step):
        perceived = DepthwiseCausalConv(z, dilation)          # neighborhood
        perceived = perceived * gate   (if gate provided)     # MORPHOGEN GATING
        p = ReLU(W_p · RMSNorm(perceived))                   # perception encoding
        s = ReLU(W_s · RMSNorm(z))                           # self encoding
        delta = W_d · (p ⊙ s)                                # Hebbian update

    The gate is the KEY mechanism: morphogen controls WHAT each cell perceives
    from its neighbors. Different morphogen states → different perception →
    different information flow. Emergent routing substrate.

    Zero-init cascade (W_d = 0):
        At init, delta = 0 everywhere → all cells are still lifes (Class I).
        Growth must be LEARNED through training.
    """

    def __init__(self, config: V11Config):
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

    def forward(
        self,
        z: torch.Tensor,
        dilation: int = 1,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply one CA step: perceive (optionally gated), encode, bind, decode.

        Args:
            z: (B, L, d_model) current cell states
            dilation: Conv dilation for this step's receptive field scale
            gate: (B, L, d_model) optional morphogen perception gate ∈ [0, 1].
                  When provided, perceived = perceived * gate, enabling the
                  morphogen field to control what each cell sees from neighbors.
                  When None, behavior is identical to V10.

        Returns:
            delta: (B, L, d_model) state update proposal Δz
        """
        perceived = self.conv(z, dilation=dilation)

        # ── Morphogen gating: selective perception ────────────────────
        if gate is not None:
            perceived = perceived * gate

        p = F.relu(self.W_p(self.norm_perceived(perceived)))  # (B, L, d_rule)
        s = F.relu(self.W_s(self.norm_z(z)))                  # (B, L, d_rule)

        # Hebbian binding: only mutually active concepts fire
        delta = self.W_d(self.dropout(p * s))  # (B, L, d_model)

        return delta


class MorphogenRule(nn.Module):
    """The morphogen genome: one shared tiny MLP evolving the morphogen field.

    The morphogen field h ∈ R^{d_h} is the second field in V11's coupled CA.
    It evolves via its own local rule, perceiving its own neighborhood through
    a separate DepthwiseCausalConv, and produces updates via a tiny MLP.

    The morphogen enables the CA to develop internal signaling pathways:
    reaction-diffusion dynamics, Turing patterns, gradient-based differentiation.
    Its state gates content perception (in SimpleHebbianRule), creating the
    substrate for emergent information routing.

    Architecture (per step):
        h_nbr = DepthwiseCausalConv_h(h, dilation)          # morphogen neighborhood
        combined = [h; h_nbr]                                # concatenate
        hidden = tanh(W_h1 · RMSNorm(combined))              # hidden layer (bounded)
        delta_h = W_h2 · hidden                              # output (ZERO-INIT)

    tanh activation:
        Bounded [-1, 1]. Enables both positive and negative morphogen dynamics
        (activation + inhibition). Prevents morphogen explosion. Enables
        reaction-diffusion-like patterns.

    Zero-init cascade (W_h2 = 0):
        At init, delta_h = 0 → morphogen starts frozen. The morphogen field
        must be GROWN through training, just like content. Both fields start
        as Class I still lifes and evolve together. I → II → IV.

    Parameters (~49K at d_h=128):
        W_h1:  256 × 128 + 128 (bias)  = 33K
        W_h2:  128 × 128               = 16K   ZERO-INIT
        conv:  128 × 3                  = 384
        norm:  256                      = 256
    """

    def __init__(self, config: V11Config):
        super().__init__()
        dh = config.d_h

        # Morphogen perception: own causal conv (d_h channels)
        self.conv = DepthwiseCausalConv(dh, config.conv_kernel_size)

        # Pre-norm on concatenated [h, h_nbr]
        self.norm = nn.RMSNorm(2 * dh)

        # Tiny MLP: [h; h_nbr] → delta_h
        self.W_h1 = nn.Linear(2 * dh, dh, bias=True)   # Hidden layer with bias
        self.W_h2 = nn.Linear(dh, dh, bias=False)       # Output: ZERO-INIT

        self._init_weights()

    def _init_weights(self) -> None:
        """Hidden: Xavier normal. Output: ZERO (morphogen starts frozen)."""
        nn.init.xavier_normal_(self.W_h1.weight, gain=1.0)
        nn.init.zeros_(self.W_h1.bias)
        nn.init.zeros_(self.W_h2.weight)  # THE zero-init cascade for morphogen

    def forward(self, h: torch.Tensor, dilation: int = 1) -> torch.Tensor:
        """Evolve morphogen field: perceive, encode, update.

        Args:
            h: (B, L, d_h) current morphogen state
            dilation: Conv dilation for this step (same schedule as content)

        Returns:
            delta_h: (B, L, d_h) morphogen update proposal Δh
        """
        h_nbr = self.conv(h, dilation=dilation)  # (B, L, d_h)

        # Concatenate self and neighborhood
        combined = torch.cat([h, h_nbr], dim=-1)  # (B, L, 2·d_h)

        # Tiny MLP with tanh (bounded, enables inhibition + activation)
        hidden = torch.tanh(self.W_h1(self.norm(combined)))  # (B, L, d_h)
        delta_h = self.W_h2(hidden)  # (B, L, d_h)

        return delta_h
