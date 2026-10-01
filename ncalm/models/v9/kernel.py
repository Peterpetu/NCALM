"""Hebbian kernel and causal convolution for AENCA V9 — Wolfram CA.

The kernel is the "genome" of the cellular automaton: ONE shared set of weights
applied identically at every position, at every step, at every scale. Only the
dilation of the causal convolution changes between steps, giving the kernel
progressively wider receptive fields.

Architecture (per CA step):
    perceived = DepthwiseCausalConv(z, dilation=d_i)     # neighborhood
    p = ReLU(W_enc_p · RMSNorm(perceived))               # perception encoding
    s = ReLU(W_enc_z · RMSNorm(z))                       # self encoding
    delta = W_dec · (p ⊙ s)                              # Hebbian update

The Hebbian product (p ⊙ s) implements Hebb's rule in vector form:
"Concepts that appear in BOTH the neighborhood AND the self fire together."
ReLU ensures positive sparse activations (~5% nonzero).

Zero-init on W_dec guarantees:
  - At init: delta = 0 for all inputs → every cell is a still life (Class I)
  - During training: W_dec departs from zero → dynamics emerge (Class I → IV)
  - This IS Wolfram's growth from simplicity

Parameters:
    W_enc_p:  d_model × d_concept     (perception encoder)
    W_enc_z:  d_model × d_concept     (self encoder)
    W_dec:    d_concept × d_model     (decoder, ZERO-INIT)
    conv:     d_model × kernel_size   (depthwise causal, shared across dilations)
    norms:    2 × d_model             (RMSNorm for perceived and self)

Total: 3 × d_model × d_concept + d_model × kernel_size + 2 × d_model
For d=2048, d_concept=9256: ≈56.9M (the genome)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v9.config import V9Config


class DepthwiseCausalConv(nn.Module):
    """Shared depthwise causal convolution with runtime dilation.

    Each channel has its own independent kernel of size k. The same weights
    are reused at each CA step with a different dilation — same rule,
    different scale. This is pure Wolfram: identical local rule everywhere.

    Causal: left-padding of (k-1)*dilation ensures no future information leak.
    Depthwise: groups=d_model, each channel convolves independently.

    Parameters: d_model × kernel_size (one 1D kernel per channel).
    For d=2048, k=3: 6,144 params.
    """

    def __init__(self, d_model: int, kernel_size: int = 3):
        super().__init__()
        # Weight: (d_model, 1, kernel_size) for depthwise conv
        self.weight = nn.Parameter(torch.empty(d_model, 1, kernel_size))
        self.kernel_size = kernel_size
        self.d_model = d_model

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize conv weights with small values for stable start.

        Kaiming uniform is the default for Conv1d. We use it here for
        consistency. The conv is just a local mixing operation.
        """
        nn.init.kaiming_uniform_(self.weight, a=0)

    def forward(self, x: torch.Tensor, dilation: int = 1) -> torch.Tensor:
        """Apply causal depthwise conv with given dilation.

        Args:
            x: (B, L, d_model) cell states
            dilation: Dilation factor for this CA step

        Returns:
            perceived: (B, L, d_model) neighborhood-aggregated states
        """
        # Conv1d expects (B, C, L) format
        h = x.transpose(1, 2)  # (B, d_model, L)

        # Causal left-padding: ensures output[t] depends only on input[≤t]
        pad = (self.kernel_size - 1) * dilation
        h = F.pad(h, (pad, 0))

        # Depthwise conv with shared weights, variable dilation
        h = F.conv1d(h, self.weight, groups=self.d_model, dilation=dilation)

        return h.transpose(1, 2)  # (B, L, d_model)


class HebbianKernel(nn.Module):
    """The genome: ONE shared Hebbian update kernel for all CA steps.

    This is the soul of V9. A single set of weights that defines the local
    update rule, applied identically at every position and every step.
    Only the conv dilation changes between steps, giving the kernel
    progressively wider neighborhoods to perceive.

    Architecture:
        perceived = DepthwiseCausalConv(z, dilation)      # perceive neighbors
        p = ReLU(W_enc_p · RMSNorm(perceived))             # encode perception
        s = ReLU(W_enc_z · RMSNorm(z))                     # encode self
        delta = W_dec · Dropout(p ⊙ s)                     # decode Hebbian product

    The Hebbian product (p ⊙ s) is the binding operation:
        - p encodes what the NEIGHBORHOOD looks like (conv output)
        - s encodes what THIS CELL currently looks like (cell state)
        - Their element-wise product fires only where BOTH are active
        - ReLU ensures positive sparse activations (~5% nonzero)
        - This sparsity gives the model implicit selectivity

    Zero-init cascade (W_dec = 0):
        At initialization, delta = 0 everywhere → all cells are still lifes.
        The model begins in Wolfram's Class I. As training progresses and
        W_dec departs from zero, dynamics emerge: I → II → IV.

    Parameters: ~59M for d=2048, d_concept≈9256 (36% of 162M budget)
    """

    def __init__(self, config: V9Config):
        super().__init__()
        d = config.d_model
        dc = config.d_concept

        # ── Perception: depthwise causal conv (shared across dilations) ──
        self.conv = DepthwiseCausalConv(d, config.conv_kernel_size)

        # ── Pre-norms ────────────────────────────────────────────────────
        self.norm_perceived = nn.RMSNorm(d)   # normalize conv output
        self.norm_z = nn.RMSNorm(d)           # normalize cell state

        # ── Concept encoders ─────────────────────────────────────────────
        # W_enc_p: d_model → d_concept (perception → concept space)
        self.W_enc_p = nn.Linear(d, dc, bias=False)
        # W_enc_z: d_model → d_concept (self-state → concept space)
        self.W_enc_z = nn.Linear(d, dc, bias=False)

        # ── Concept decoder ──────────────────────────────────────────────
        # W_dec: d_concept → d_model (ZERO-INIT for Class I start)
        self.W_dec = nn.Linear(dc, d, bias=False)

        # ── Dropout (on Hebbian product) ─────────────────────────────────
        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights — zero-init cascade on decoder.

        Encoders: Xavier normal (standard for projection matrices).
        Decoder: ZERO → kernel starts producing zero delta (still life).

        This initialization guarantees:
          - At step 0: Φ(z, d) = 0 for all z, all dilations
          - Every cell is a Class I still life (Wolfram)
          - Growth must be LEARNED, not built in
        """
        nn.init.xavier_normal_(self.W_enc_p.weight, gain=1.0)
        nn.init.xavier_normal_(self.W_enc_z.weight, gain=1.0)
        nn.init.zeros_(self.W_dec.weight)  # THE zero-init cascade

    def forward(
        self, z: torch.Tensor, dilation: int = 1
    ) -> torch.Tensor:
        """Apply one CA step: perceive, encode, bind, decode.

        Args:
            z: (B, L, d_model) current cell states
            dilation: Conv dilation for this step's receptive field scale

        Returns:
            delta: (B, L, d_model) state update proposal Δz
        """
        # Perceive: local neighborhood via dilated causal conv
        perceived = self.conv(z, dilation=dilation)     # (B, L, d)

        # Encode to concept space (positive sparse via ReLU)
        p = F.relu(self.W_enc_p(self.norm_perceived(perceived)))  # (B, L, d_concept)
        s = F.relu(self.W_enc_z(self.norm_z(z)))                  # (B, L, d_concept)

        # Hebbian product: only mutually active concepts contribute
        # This IS Hebb's rule: "neurons that fire together wire together"
        ps = p * s  # (B, L, d_concept) — sparse binding

        # Decode to cell update (zero-init → starts as zero → still life)
        delta = self.W_dec(self.dropout(ps))  # (B, L, d_model)

        return delta
