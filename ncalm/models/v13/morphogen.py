"""Morphogen field module for AENCA V13 — Morphogenetic Attention.

The morphogen field h ∈ R^{L×d_h} is a lightweight second field that evolves
via local NCA dynamics and steers attention Q/K projections.

Components:
  - DepthwiseCausalConv: neighborhood perception with runtime dilation
  - MorphogenRules: K parallel tiny MLPs with softmax routing
  - BilinearGate: controls morphogen update magnitude
  - Q/K gate generation: 1 + tanh(W_h(norm(h))) → [0, 2] range
  - Content→morphogen feedback: tanh(W_feedback(norm(z)))

Key properties:
  - All morphogen components are SHARED across theta cycles
  - Zero-init cascade: W_h2=0, W_router=0, W_Q_h=0, W_K_h=0, W_feedback=0
  - At init: gates = 1.0, morphogen stays at seed, V13 = V7.1

Reference: V11 kernel.py — DepthwiseCausalConv, MorphogenRule, BilinearGate
Reference: V12 kernel.py — K parallel rules with routing
Reference: ANCA paper — belief-state dynamics driving attention
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v13.config import V13Config


class DepthwiseCausalConv(nn.Module):
    """Depthwise causal convolution with runtime dilation.

    Each channel has its own independent 1D kernel. Causal left-padding
    ensures no future information leak. Dilation is a forward() argument,
    NOT fixed at construction — the same weight tensor is used for all
    dilation values.

    Architecture:
        h → transpose → left-pad → conv1d(groups=channels, dilation=d) → transpose

    Parameters: channels × kernel_size (no bias).
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
            x: (B, L, channels) input
            dilation: dilation factor (scales receptive field)

        Returns:
            out: (B, L, channels) neighborhood-aggregated output
        """
        h = x.transpose(1, 2)                              # (B, C, L)
        pad = (self.kernel_size - 1) * dilation
        h = F.pad(h, (pad, 0))                              # left-pad = causal
        h = F.conv1d(h, self.weight, groups=self.channels, dilation=dilation)
        return h.transpose(1, 2)                             # (B, L, C)


class MorphogenField(nn.Module):
    """Complete morphogen field module for V13.

    Manages:
      1. Seeding h from content z
      2. Evolving h via K parallel rules with routing + bilinear gate
      3. Generating Q/K attention gates from h
      4. Receiving content→morphogen feedback

    All weights are SHARED across theta cycles — the morphogen state h
    varies per theta because it evolves through the loop, while the
    rules themselves are reused (standard NCA principle).
    """

    def __init__(self, config: V13Config):
        super().__init__()
        d = config.d_model
        dh = config.d_h
        K = config.n_rules

        # ── Morphogen Seed ───────────────────────────────────────────
        self.W_h_init = nn.Linear(d, dh, bias=False)

        # ── Depthwise Causal Conv (shared across all morph steps) ────
        self.morph_conv = DepthwiseCausalConv(dh, config.morph_conv_kernel)

        # ── K Parallel Morphogen Rules ───────────────────────────────
        self.morph_norm = nn.RMSNorm(2 * dh)

        # Per-rule W_h1 (with bias) and W_h2 (zero-init, no bias)
        self.W_h1 = nn.ModuleList([
            nn.Linear(2 * dh, dh, bias=True) for _ in range(K)
        ])
        self.W_h2 = nn.ModuleList([
            nn.Linear(dh, dh, bias=False) for _ in range(K)
        ])

        # Router: h → K logits (zero-init → uniform routing at start)
        self.W_router = nn.Linear(dh, K, bias=False)

        # ── Bilinear Gate ────────────────────────────────────────────
        self.W_gate_a = nn.Linear(dh, dh, bias=False)
        self.W_gate_b = nn.Linear(dh, dh, bias=False)
        self.gate_bias = nn.Parameter(torch.full((dh,), config.morph_gate_bias))

        # ── Q/K Morph Gates ──────────────────────────────────────────
        self.norm_qh = nn.RMSNorm(dh)
        self.norm_kh = nn.RMSNorm(dh)
        self.W_Q_h = nn.Linear(dh, d, bias=False)  # d_h → d_model
        self.W_K_h = nn.Linear(dh, d, bias=False)

        # ── Content → Morphogen Feedback ─────────────────────────────
        self.norm_fb = nn.RMSNorm(d)
        self.W_h_feedback = nn.Linear(d, dh, bias=False)

        # Store config
        self.n_rules = K
        self.morph_dilations = config.morph_dilations

        self._init_weights(config)

    def _init_weights(self, config: V13Config) -> None:
        """Initialize all morphogen weights per the zero-init cascade spec.

        Zero-init cascade ensures V13 = V7.1 at initialization:
          - W_h2 = 0 → delta_h = 0, morphogen stays at seed
          - W_router = 0 → uniform routing (1/K)
          - W_Q_h = 0 → q_gate = 1 + tanh(0) = 1.0
          - W_K_h = 0 → k_gate = 1.0
          - W_h_feedback = 0 → no content→morph interaction at init
        """
        # Seed projection
        nn.init.xavier_normal_(self.W_h_init.weight, gain=1.0)

        # Per-rule weights
        for k in range(self.n_rules):
            nn.init.xavier_normal_(self.W_h1[k].weight, gain=1.0)
            nn.init.zeros_(self.W_h1[k].bias)
            nn.init.zeros_(self.W_h2[k].weight)  # ZERO-INIT cascade

        # Router: zero-init → uniform softmax
        nn.init.zeros_(self.W_router.weight)

        # Bilinear gate branches
        nn.init.xavier_normal_(self.W_gate_a.weight, gain=1.0)
        nn.init.xavier_normal_(self.W_gate_b.weight, gain=1.0)
        # gate_bias already initialized to +3.0 in constructor

        # Q/K morph gates: ZERO-INIT → gates = 1.0 at start
        nn.init.zeros_(self.W_Q_h.weight)
        nn.init.zeros_(self.W_K_h.weight)

        # Content→morph feedback: ZERO-INIT
        nn.init.zeros_(self.W_h_feedback.weight)

    def seed(self, z: torch.Tensor) -> torch.Tensor:
        """Create initial morphogen field from content embedding.

        Args:
            z: (B, L, d_model) initial content state

        Returns:
            h: (B, L, d_h) initial morphogen state
        """
        return torch.tanh(self.W_h_init(z))

    def evolve(
        self,
        h: torch.Tensor,
        n_steps: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Evolve morphogen field for n_steps via local NCA dynamics.

        Each step:
          1. Depthwise causal conv → neighborhood h_nbr
          2. Concatenate [h, h_nbr] → normalize
          3. K parallel rules: tanh(W_h1(h_cat)) → W_h2(·)
          4. Softmax routing over K rules
          5. Bilinear gate controls update magnitude
          6. Residual: h = h + gate * delta_h

        Args:
            h: (B, L, d_h) current morphogen state
            n_steps: number of evolution steps (M)

        Returns:
            h: (B, L, d_h) evolved morphogen state
            routing_weights: list of (B, L, K) routing weight tensors per step
        """
        routing_weights = []

        for m in range(n_steps):
            dilation = self.morph_dilations[m % len(self.morph_dilations)]

            # Neighborhood perception
            h_nbr = self.morph_conv(h, dilation=dilation)     # (B, L, d_h)

            # Concatenate and normalize
            h_cat = self.morph_norm(
                torch.cat([h, h_nbr], dim=-1)
            )                                                  # (B, L, 2*d_h)

            # K parallel rules
            deltas = []
            for k in range(self.n_rules):
                hidden = torch.tanh(self.W_h1[k](h_cat))      # (B, L, d_h)
                delta_k = self.W_h2[k](hidden)                 # (B, L, d_h)
                deltas.append(delta_k)
            delta_stack = torch.stack(deltas, dim=2)           # (B, L, K, d_h)

            # Softmax routing
            route_logits = self.W_router(h)                    # (B, L, K)
            weights = F.softmax(route_logits, dim=-1)          # (B, L, K)
            routing_weights.append(weights)

            # Weighted sum of rule outputs
            delta_h = (
                weights.unsqueeze(-1) * delta_stack
            ).sum(dim=2)                                       # (B, L, d_h)

            # Bilinear gate
            gate = torch.sigmoid(
                F.silu(self.W_gate_a(h)) * self.W_gate_b(h) + self.gate_bias
            )                                                  # (B, L, d_h)

            # Residual update
            h = h + gate * delta_h

        return h, routing_weights

    def generate_gates(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate Q/K attention gates from morphogen state.

        Gates are in [0, 2] range, starting at 1.0 (due to W_Q_h, W_K_h
        being zero-initialized → tanh(0) = 0 → 1 + 0 = 1).

        Applied in d_model space BEFORE head reshape.

        Args:
            h: (B, L, d_h) current morphogen state

        Returns:
            q_gate: (B, L, d_model) query gate values
            k_gate: (B, L, d_model) key gate values
        """
        q_gate = 1.0 + torch.tanh(self.W_Q_h(self.norm_qh(h)))  # (B, L, d_model)
        k_gate = 1.0 + torch.tanh(self.W_K_h(self.norm_kh(h)))  # (B, L, d_model)
        return q_gate, k_gate

    def update_from_content(
        self, h: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Update morphogen from content state (content→morph feedback).

        Zero-init on W_h_feedback means this is a no-op at initialization.

        Args:
            h: (B, L, d_h) current morphogen state
            z: (B, L, d_model) current content state

        Returns:
            h: (B, L, d_h) updated morphogen state
        """
        return h + torch.tanh(self.W_h_feedback(self.norm_fb(z)))
