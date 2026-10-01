"""Content-Enriched Morphogen Field for AENCA V14 — Emergent Dynamics.

The morphogen field h ∈ R^{L×d_h} evolves via local NCA dynamics enriched
with content perception at every step. Unlike V13's morphogen which only
generates Q/K gates, V14's morphogen IS the information routing mechanism.

Architecture per evolution step m:
    1. h_nbr = morph_conv(h, dilation=d_m)          — morphogen neighborhood
    2. c_m   = content_conv(W_c(norm(z)), dilation=d_m) — content perception
    3. h_cat = norm([h, h_nbr, c_m])                 — 3-channel input
    4. K parallel rules: delta_k = W_h2(tanh(W_h1(h_cat)))
    5. Softmax routing over K rules
    6. Bilinear gate controls update magnitude
    7. Residual: h = h + gate * Σ_k w_k · delta_k

Key properties:
  - All morphogen components SHARED across theta cycles (NCA principle)
  - Content perception injects z at each step → content-dependent routing
  - Exponential dilation schedule: effective receptive field grows 2^M
  - Zero-init cascade: W_h2=0, W_router=0 → no evolution at init
  - Content readout: separate per-theta projections (unique)

Reference: ANCA paper — sensor movement = content perception at each step
Reference: V13 morphogen.py — DepthwiseCausalConv, bilinear gate pattern
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v14.config import V14Config


class DepthwiseCausalConv(nn.Module):
    """Depthwise causal convolution with runtime dilation.

    Identical to V13: each channel has its own 1D kernel, causal left-padding,
    dilation is a forward() argument. Same weight for all dilation values.
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


class ContentEnrichedMorphogenField(nn.Module):
    """Complete morphogen field module for V14 with content perception.

    Manages:
      1. Seeding h from content z
      2. Evolving h via K parallel rules with content perception at each step
      3. Content readout: h → z update (via per-theta readout projections)
      4. Content → morphogen feedback

    Morphogen evolution rules are SHARED across theta cycles.
    Content readout projections are UNIQUE per theta (created externally).
    """

    def __init__(self, config: V14Config):
        super().__init__()
        d = config.d_model
        dh = config.d_h
        K = config.n_rules
        rh = config.effective_rule_hidden

        # ── Morphogen Seed ───────────────────────────────────────────
        self.W_h_init = nn.Linear(d, dh, bias=False)

        # ── Morphogen Neighborhood Conv (shared across steps) ────────
        self.morph_conv = DepthwiseCausalConv(dh, config.morph_conv_kernel)

        # ── Content Perception ───────────────────────────────────────
        # Project z into morphogen space, then perceive via dilated conv
        self.norm_z_proj = nn.RMSNorm(d)
        self.W_content_proj = nn.Linear(d, dh, bias=False)
        self.content_conv = DepthwiseCausalConv(dh, config.morph_conv_kernel)

        # ── K Parallel NCA Rules ─────────────────────────────────────
        # Input: [h, h_nbr, c_m] → 3*d_h dimensions
        self.rule_norm = nn.RMSNorm(3 * dh)

        self.W_h1 = nn.ModuleList([
            nn.Linear(3 * dh, rh, bias=True) for _ in range(K)
        ])
        self.W_h2 = nn.ModuleList([
            nn.Linear(rh, dh, bias=False) for _ in range(K)
        ])

        # Router: h → K logits (zero-init → uniform routing at start)
        self.W_router = nn.Linear(dh, K, bias=False)

        # Step embedding: allows different routing per NCA step
        # Without this, the router sees nearly-identical h at each step
        # and produces identical routing → repetitive dynamics
        self.step_embed = nn.Embedding(config.n_morph_steps, config.step_embed_dim)
        self.W_step_route = nn.Linear(config.step_embed_dim, K, bias=False)

        # ── Bilinear Gate ────────────────────────────────────────────
        self.W_gate_a = nn.Linear(dh, dh, bias=False)
        self.W_gate_b = nn.Linear(dh, dh, bias=False)
        self.gate_bias = nn.Parameter(torch.full((dh,), config.morph_gate_bias))

        # ── Content → Morphogen Feedback ─────────────────────────────
        self.norm_fb = nn.RMSNorm(d)
        self.W_h_feedback = nn.Linear(d, dh, bias=False)

        # Store config
        self.n_rules = K
        self.morph_dilations = config.morph_dilations
        self.d_h = dh

        self._init_weights(config)

    def _init_weights(self, config: V14Config) -> None:
        """Initialize all morphogen weights per the zero-init cascade spec.

        Zero-init cascade ensures V14 produces near-zero residuals at init:
          - W_h2 = 0 → delta_h = 0, morphogen stays at seed
          - W_router = 0 → uniform routing (1/K)
          - W_h_feedback = 0 → no content→morph interaction at init
          - (External readout W_readout = 0 → no morph→content at init)
        """
        # Seed projection
        nn.init.xavier_normal_(self.W_h_init.weight, gain=1.0)

        # Content perception
        nn.init.xavier_normal_(self.W_content_proj.weight, gain=1.0)

        # Per-rule weights
        for k in range(self.n_rules):
            nn.init.xavier_normal_(self.W_h1[k].weight, gain=1.0)
            nn.init.zeros_(self.W_h1[k].bias)
            nn.init.zeros_(self.W_h2[k].weight)  # ZERO-INIT cascade

        # Router: zero-init → uniform softmax
        nn.init.zeros_(self.W_router.weight)

        # Step embedding: small init so step bias starts mild
        nn.init.normal_(self.step_embed.weight, std=0.02)
        nn.init.zeros_(self.W_step_route.weight)

        # Bilinear gate branches
        nn.init.xavier_normal_(self.W_gate_a.weight, gain=1.0)
        nn.init.xavier_normal_(self.W_gate_b.weight, gain=1.0)
        # gate_bias already initialized to morph_gate_bias in constructor

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
        z: torch.Tensor,
        n_steps: int,
        scale_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Evolve morphogen field for n_steps with content perception.

        Each step:
          1. Depthwise causal conv → morphogen neighborhood h_nbr
          2. Content projection + depthwise causal conv → content perception c_m
          3. Concatenate [h, h_nbr, c_m] → normalize
          4. K parallel rules → route → gate → residual update

        Multi-scale readout (Phase C): if scale_weights is provided,
        accumulates a weighted sum of h states across all steps:
            h_multi = Σ_m scale_weights[m] · h^(m)
        This captures information at different dilation scales without
        storing all intermediate states (memory-efficient running sum).

        The exponential dilation schedule means step m perceives content at
        distance ~2^m, giving O(log N) full-sequence coverage.

        Args:
            h:             (B, L, d_h) current morphogen state
            z:             (B, L, d_model) current content state
            n_steps:       number of evolution steps (M)
            scale_weights: (M,) softmax weights for multi-scale aggregation,
                           or None to use final h only

        Returns:
            h:                (B, L, d_h) final evolved morphogen state
            h_multi:          (B, L, d_h) multi-scale aggregated state
                              (equals h if scale_weights is None)
            routing_weights:  list of (B, L, K) routing per step
            step_deltas:      list of (B, L) delta norms per step (for dynamics loss)
        """
        routing_weights = []
        step_deltas = []

        # Multi-scale accumulator (memory-efficient running sum)
        if scale_weights is not None:
            h_multi = torch.zeros_like(h)
        else:
            h_multi = None

        # Project content once (reused across steps with different dilations)
        z_proj = self.W_content_proj(self.norm_z_proj(z))  # (B, L, d_h)

        for m in range(n_steps):
            dilation = self.morph_dilations[m % len(self.morph_dilations)]
            # Clamp dilation so conv doesn't reach entirely into padding
            # (for short sequences, high dilations see only zeros)
            eff_dilation = min(dilation, max(1, z.shape[1] // 2))

            # 1. Morphogen neighborhood perception
            h_nbr = self.morph_conv(h, dilation=eff_dilation)       # (B, L, d_h)

            # 2. Content perception at same dilation
            c_m = self.content_conv(z_proj, dilation=eff_dilation)  # (B, L, d_h)

            # 3. Concatenate and normalize
            h_cat = self.rule_norm(
                torch.cat([h, h_nbr, c_m], dim=-1)
            )                                                    # (B, L, 3*d_h)

            # 4. K parallel rules
            deltas = []
            for k in range(self.n_rules):
                hidden = torch.tanh(self.W_h1[k](h_cat))        # (B, L, rh)
                delta_k = self.W_h2[k](hidden)                  # (B, L, d_h)
                deltas.append(delta_k)
            delta_stack = torch.stack(deltas, dim=2)             # (B, L, K, d_h)

            # 5. Softmax routing with step-dependent bias
            route_logits = self.W_router(h)                      # (B, L, K)
            # Add step-specific bias so different steps can prefer different rules
            step_idx = torch.tensor(m, device=h.device)
            step_bias = self.W_step_route(self.step_embed(step_idx))  # (K,)
            route_logits = route_logits + step_bias
            weights = F.softmax(route_logits, dim=-1)            # (B, L, K)
            routing_weights.append(weights)

            # 6. Weighted sum of rule outputs
            delta_h = (
                weights.unsqueeze(-1) * delta_stack
            ).sum(dim=2)                                         # (B, L, d_h)

            # 7. Bilinear gate
            gate = torch.sigmoid(
                F.silu(self.W_gate_a(h)) * self.W_gate_b(h) + self.gate_bias
            )                                                    # (B, L, d_h)

            # 8. Residual update
            update = gate * delta_h
            h = h + update

            # 9. Multi-scale accumulation (running weighted sum)
            if h_multi is not None:
                h_multi = h_multi + scale_weights[m] * h

            # Track dynamics strength for diversity loss
            # Use smooth norm (sqrt(sum_sq + eps)) to avoid NaN grad at zero
            # Do NOT detach — gradient must flow for dyn_loss to train morphogen
            smooth_norm = (update.pow(2).sum(dim=-1) + 1e-8).sqrt()  # (B, L)
            step_deltas.append(smooth_norm)

        # If no multi-scale, readout from final state
        if h_multi is None:
            h_multi = h

        return h, h_multi, routing_weights, step_deltas

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


class ContentReadout(nn.Module):
    """Project evolved morphogen state back into content space.

    One instance per theta cycle (unique weights) — each theta can extract
    different information from the morphogen field.

    Multi-scale readout (Phase C): learned scale_logits select which
    morphogen evolution steps to read from. softmax(scale_logits) gives
    per-step weights α_m. These are passed to evolve() which accumulates
    h_multi = Σ_m α_m · h^(m) as a memory-efficient running sum.

    At initialization:
      - W_readout = 0 (zero-init cascade → no morph→content)
      - scale_logits = 0 (uniform softmax → equal weight per step)
    """

    def __init__(self, d_h: int, d_model: int, n_morph_steps: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_h)
        self.W_readout = nn.Linear(d_h, d_model, bias=False)
        nn.init.zeros_(self.W_readout.weight)  # ZERO-INIT cascade

        # Multi-scale readout: learned per-step aggregation weights
        # Zeros → uniform softmax(0,...,0) = (1/M, ..., 1/M)
        self.scale_logits = nn.Parameter(torch.zeros(n_morph_steps))

    def get_scale_weights(self) -> torch.Tensor:
        """Return softmax-normalized scale weights for evolve().

        Returns:
            scale_weights: (n_morph_steps,) weights summing to 1
        """
        return F.softmax(self.scale_logits, dim=0)

    def forward(self, h_multi: torch.Tensor) -> torch.Tensor:
        """Read content update from multi-scale aggregated morphogen state.

        Args:
            h_multi: (B, L, d_h) multi-scale aggregated morphogen state

        Returns:
            z_update: (B, L, d_model) content update (to be added as residual)
        """
        return self.W_readout(self.norm(h_multi))
