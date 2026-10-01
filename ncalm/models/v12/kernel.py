"""Kernels for AENCA V12: MixtureHebbianRules, MorphogenRule, DepthwiseCausalConv.

V12 replaces V11's single SimpleHebbianRule with MixtureHebbianRules:
K=8 parallel Hebbian rules, probabilistically routed per-cell-per-step by
the morphogen field h. This is the direct implementation of MNCA §3.1 within
the two-field NCA paradigm.

MNCA paper mapping (arxiv 2506.20486):
  §3.1 Eq.4a:   z ~ Cat(π(s_i, η))
                → weights = Gumbel_softmax(W_router · h / τ(t))
  §3.1 Eq.4b:   s_i^{t+1} = s_i^t + Σ_k φ_k^{z_k}
                → Δz = Σ_k w_k · δ_k  (soft, differentiable mixture)
  §3.2 Eq.5b:   x_k ~ N(0,1) injected into each rule
                → noise_sigma · randn_like(bound) before W_d_k projection
  Appendix F:   effective Lipschitz L_F ≤ Σ_k π_k·L_k ≤ max_k L_k
                → mixture smooths and stabilises the Hebbian update dynamics

Architecture per step:
  1. router_logits  = W_router · h                          (B, L, K)
  2. weights        = Gumbel_softmax(logits / τ(t))         (B, L, K)  [train]
                    OR  softmax(logits / τ_min)                         [eval]
  3. perceived      = DepthwiseCausalConv(z, dilation) ⊙ gate
  4. p              = relu(W_p · RMSNorm(perceived))         (B, L, d_rule)  SHARED
  5. s_all          = relu(einsum('bld,krd→blkr', z_norm, W_s))  (B,L,K,d_rule)
  6. bound          = p.expand_k ⊙ s_all + noise            (B, L, K, d_rule)
  7. delta_all      = einsum('blkr,kdr→blkd', bound, W_d)   (B, L, K, d_model)
  8. delta_z        = Σ_k w_k · delta_all_k                  (B, L, d_model)

Zero-init cascade:
  W_d  = zeros → delta_all = 0 at init → V12 starts as a K-way averaged zero
  W_router = zeros → logits = 0 → weights = 1/K (uniform prior)
  → V12 starts identically to a zero-delta V11 (safe, stable initialisation)

Weight sharing insight (vs MNCA paper's fully-independent rules):
  Shared W_p (perception) — all rules see the SAME filtered neighbourhood.
  The morphogen gate already controls WHAT is perceived. Rule identity controls
  HOW the perceived content is BOUND to self-state (W_s_k, W_d_k).
  This shares 1/3 of mature rule cost while preserving full specialisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v12.config import V12Config


class DepthwiseCausalConv(nn.Module):
    """Shared depthwise causal convolution with runtime dilation.

    Each channel has its own independent kernel. Left-padding ensures
    strict causality (no future information). Identical to V11.

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
            x:        (B, L, channels) cell states
            dilation: Dilation factor for this CA step

        Returns:
            perceived: (B, L, channels) neighbourhood-aggregated states
        """
        h = x.transpose(1, 2)              # (B, channels, L)
        pad = (self.kernel_size - 1) * dilation
        h = F.pad(h, (pad, 0))
        h = F.conv1d(h, self.weight, groups=self.channels, dilation=dilation)
        return h.transpose(1, 2)           # (B, L, channels)


class MixtureHebbianRules(nn.Module):
    """K parallel Hebbian rules with morphogen-driven probabilistic routing.

    Implements MNCA §3.1 mixture model within the two-field NCA paradigm.
    K rules φ_k each have a specialised binding pair (W_s_k, W_d_k), while
    sharing a common perception encoder W_p. The morphogen h routes each cell
    to a soft mixture of rules at each CA step via W_router.

    "Attention Growth" concept:
        At step t=0, h is freshly seeded from content → W_router(h) ≈ 0
        → weights ≈ 1/K (all rules equal). As h evolves through 64 coupled
        CA steps, routing sharpens. By step 63, cells have differentiated into
        linguistic roles (lexical, syntactic, semantic, boundary, etc.).
        This is attention that GROWS through dynamics, not prescribed from day 1.

    Parameter layout:
        W_p:      nn.Linear(d_model, d_rule, bias=False) — SHARED, Kaiming init
        W_s:      nn.Parameter(K, d_rule, d_model)       — per-rule, Xavier per slice
        W_d:      nn.Parameter(K, d_model, d_rule)       — per-rule, ZERO init (cascade)
        W_router: nn.Linear(d_h, K, bias=False)          — ZERO init (uniform prior)

    Einsum dimensions throughout:
        b = batch, l = seq_len, k = rule index, r = d_rule, d = d_model

    Batched matmuls (GPU-parallel across K rules — NOT K sequential ops):
        s_all  = relu(einsum('bld,krd→blkr', z_norm, W_s))
        δ_all  = einsum('blkr,kdr→blkd', bound, W_d)
    """

    def __init__(self, config: V12Config):
        super().__init__()
        d = config.d_model
        dr = config.d_rule
        dh = config.d_h
        K = config.n_rules

        self.n_rules = K
        self.noise_sigma = config.noise_sigma
        self.route_temp_min = config.route_temp_min
        self.route_temp_max = config.route_temp_max

        # ── Shared content perception ──────────────────────────────────
        self.conv = DepthwiseCausalConv(d, config.conv_kernel_size)
        self.norm_perceived = nn.RMSNorm(d)
        self.norm_z = nn.RMSNorm(d)
        self.W_p = nn.Linear(d, dr, bias=False)      # Kaiming init (default)

        # ── Per-rule binding weights (stacked for batched einsum) ──────
        # W_s[k]: (d_rule, d_model) — like Linear(d_model, d_rule, bias=False).weight
        # W_d[k]: (d_model, d_rule) — like Linear(d_rule, d_model, bias=False).weight
        # W_d starts at zero → delta_all = 0 at init (Class I still-life)
        self.W_s = nn.Parameter(torch.empty(K, dr, d))
        self.W_d = nn.Parameter(torch.zeros(K, d, dr))   # ZERO cascade

        # ── Morphogen → rule router ────────────────────────────────────
        # ZERO init → logits = 0 → softmax = 1/K → uniform prior at start
        # Rules diverge only as routing signal grows through gradient updates.
        self.W_router = nn.Linear(dh, K, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise W_s with Xavier normal per rule slice. W_d and W_router stay zero."""
        for k in range(self.n_rules):
            nn.init.xavier_normal_(self.W_s[k], gain=1.0)
        # W_d already zeros; W_router already zeros from nn.Linear default=Kaiming
        # but we explicitly zero W_router for the uniform-prior guarantee:
        nn.init.zeros_(self.W_router.weight)

    def forward(
        self,
        z: torch.Tensor,              # (B, L, d_model)
        h: torch.Tensor,              # (B, L, d_h)
        dilation: int,
        gate: torch.Tensor | None,    # (B, L, d_model) morphogen gate ∈ [0,1]
        training: bool,
        step_frac: float,             # t / (n_steps-1) ∈ [0,1] — controls τ(t)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run K-rule mixture Hebbian update.

        Args:
            z:          (B, L, d_model) current content state
            h:          (B, L, d_h) current morphogen state
            dilation:   convolution dilation for this CA step
            gate:       (B, L, d_model) morphogen gate, or None
            training:   True → Gumbel-softmax + noise; False → softmax, no noise
            step_frac:  position within CA steps for temperature annealing

        Returns:
            delta_z: (B, L, d_model) — mixture content update proposal
            weights: (B, L, K) — routing weights (in graph for diversity loss)
        """
        # ── 1. Morphogen-driven routing ───────────────────────────────
        router_logits = self.W_router(h)      # (B, L, K)

        # Temperature anneal: starts high (uniform), sharpens as morphogen matures
        tau = self.route_temp_max - (self.route_temp_max - self.route_temp_min) * step_frac
        tau = max(tau, 1e-3)

        if training:
            # Gumbel-softmax: differentiable stochastic rule assignment (MNCA §3.1)
            weights = F.gumbel_softmax(router_logits, tau=tau, hard=False, dim=-1)
        else:
            weights = F.softmax(router_logits / tau, dim=-1)   # (B, L, K)

        # ── 2. Shared gated perception ────────────────────────────────
        perceived = self.conv(z, dilation=dilation)      # (B, L, d_model)
        if gate is not None:
            perceived = perceived * gate                  # morphogen gates what is seen

        # Shared perception encoding (same for all K rules)
        p = F.relu(self.W_p(self.norm_perceived(perceived)))   # (B, L, d_rule)

        # ── 3. Per-rule self-encoding (batched) ───────────────────────
        z_norm = self.norm_z(z)                          # (B, L, d_model)
        # W_s: (K, d_rule, d_model). Einsum: contract d_model → (B, L, K, d_rule)
        s_all = F.relu(torch.einsum('bld,krd->blkr', z_norm, self.W_s))

        # ── 4. Hebbian binding: p ⊙ s_k per rule ─────────────────────
        p_exp = p.unsqueeze(2).expand(-1, -1, self.n_rules, -1)  # (B, L, K, d_rule)
        bound = self.dropout(p_exp * s_all)                       # (B, L, K, d_rule)

        # Intrinsic noise per rule during training (MNCA §3.2 — x_k ~ N(0,1))
        # Applied additively to the Hebbian product before output projection.
        if training and self.noise_sigma > 0.0:
            bound = bound + self.noise_sigma * torch.randn_like(bound)

        # ── 5. Per-rule output projection (batched) ───────────────────
        # W_d: (K, d_model, d_rule). Einsum: contract d_rule → (B, L, K, d_model)
        delta_all = torch.einsum('blkr,kdr->blkd', bound, self.W_d)

        # ── 6. Soft weighted mixture ──────────────────────────────────
        # weights: (B, L, K) → (B, L, K, 1) for broadcast across d_model
        delta_z = (weights.unsqueeze(-1) * delta_all).sum(dim=2)  # (B, L, d_model)

        return delta_z, weights


class MorphogenRule(nn.Module):
    """Morphogen evolution rule — same structure as V11, expanded to d_h=512.

    The morphogen field h ∈ R^{d_h} is the second field in V12's coupled CA.
    It evolves through its own local rule (perceive neighbourhood, tiny MLP).
    At d_h=512, the morphogen becomes the PRIMARY routing substrate, with
    bilinear gate capacity d_h(d_h+1)/2 ≈ 131K independent patterns.

    Architecture (per step):
        h_nbr    = DepthwiseCausalConv_h(h, dilation)    # morphogen neighbourhood
        combined = [h; h_nbr]                             ∈ R^{1024}
        hidden   = tanh(W_h1 · RMSNorm(combined))         (bounded: ±1)
        delta_h  = W_h2 · hidden                          ZERO-INIT

    tanh activation: enables inhibitory dynamics (negative delta_h) and
    prevents morphogen explosion. Identical rationale to V11.

    Zero-init cascade (W_h2 = 0):
        Morphogen starts frozen. Both fields grow from Class I still-lifes together.

    Parameters (~0.79M at d_h=512):
        W_h1: 1024×512+512 = 0.52M   Xavier init
        W_h2:  512×512     = 0.26M   ZERO init
        conv:  512×3        = 1.5K
        norm:  1024         = 1K
    """

    def __init__(self, config: V12Config):
        super().__init__()
        dh = config.d_h

        self.conv = DepthwiseCausalConv(dh, config.conv_kernel_size)
        self.norm = nn.RMSNorm(2 * dh)
        self.W_h1 = nn.Linear(2 * dh, dh, bias=True)    # Xavier init
        self.W_h2 = nn.Linear(dh, dh, bias=False)        # ZERO init

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.W_h1.weight, gain=1.0)
        nn.init.zeros_(self.W_h1.bias)
        nn.init.zeros_(self.W_h2.weight)

    def forward(self, h: torch.Tensor, dilation: int = 1) -> torch.Tensor:
        """Evolve morphogen: perceive neighbourhood, encode, update.

        Args:
            h:        (B, L, d_h) current morphogen state
            dilation: Conv dilation (same schedule as content)

        Returns:
            delta_h: (B, L, d_h) morphogen update proposal
        """
        h_nbr = self.conv(h, dilation=dilation)             # (B, L, d_h)
        combined = torch.cat([h, h_nbr], dim=-1)            # (B, L, 2·d_h)
        hidden = torch.tanh(self.W_h1(self.norm(combined))) # (B, L, d_h)
        return self.W_h2(hidden)                            # (B, L, d_h)
