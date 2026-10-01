"""AENCA V8 main model: Growing Cellular Automata Language Model.

Architecture (Spec Section 15 — Complete Forward Pass):

  Step 1: Seed Initialization (Spec Def 5.2)
      z_hidden = W_embed · x + PE(i)    ∀i          # (B, L, d)
      z_alpha  = 1.0                      ∀i          # all cells alive
      z        = [z_hidden; z_alpha]                   # (B, L, d+1)
      ā_i      = ∞  (no settlement history)
      settled  = 0   (no cells settled)

  Step 2: Hierarchical Growth Loop (Spec Def 13.1-13.3)
      For θ ∈ 1..n_theta:                              [THETA CYCLE]

        Step 2a: Gamma steps (repeat t_gamma times):    [GAMMA STEPS]
          For each cell i:
            (i)    alive = 1[max_{j∈N(i)} z_j^α > τ_α]  (Def 6.1)
            (ii)   Dead cells → zero out
            (iii)  Awakening check: settled & gradient shift > τ_w → reset
            (iv)   Settled cells → skip
            (v)    Perceive: ẑ = RMSNorm([conv(z) ∥ z ∥ ∇z])  (Def 7.1)
            (vi)   Update: Δz = Φ(ẑ, z)                        (Def 8.1)
            (vii)  Stochastic: m ~ Bernoulli(0.5)               (Def 9.1)
            (viii) Apply: z += (1−settled)·m·Δz                  (Sec 9.1)
            (ix)   Settlement: update EMA, settled score         (Def 10.2-10.3)
            (x)    Record ∇ for newly settled cells              (Def 10.4)

        Step 2b: Theta step (Spec Def 13.2)
          ThetaBlock(z, settled)  — settled cells attend but not modified

        Step 2c: Halting (Spec Sec 16.3)
          halt_prob = σ(W_h · z_hidden + b_h)
          Accumulate weighted output

  Step 3: Readout (Spec Sec 15.1)
      ŷ = MoS-Readout(Σ w^(n) · z_hidden^(n))

  Loss (Spec Section 16 — Complete Loss Function):
      L_V8 = L_CE + λ_halt·L_halt + λ_perm·L_perm
             + λ_regen·L_regen + λ_edge·L_edge + λ_aux·L_aux

  Six terms enforce the Seven Axioms:
      L_CE    → Axiom 6 (Structure=Function)
      L_halt  → Axiom 1 (Growth), Axiom 5 (Universality)
      L_perm  → Axiom 2 (Permanence), Axiom 3 (Sparsity)
      L_regen → Axiom 2 (Permanence), Axiom 6 (Structure=Function)
      L_edge  → Axiom 5 (Universality), Axiom 4 (Simplicity)
      L_aux   → Axiom 1 (Growth), Axiom 7 (Self-Organization)

  V8 overhead: ZERO additional learned parameters.
  All V8 mechanisms (alive masking, settlement, stochastic masking,
  awakening, edge-of-chaos) are computed from existing state variables.

Reference: model_documents/AENCA_V8_GROWING_CELLULAR_AUTOMATA.md
  Full specification: Sections 5-24.

V7.1 mechanisms preserved:
  - CausalConv1D local perception (extended to d+1)
  - HebbianKernel Hebbian product gating (extended perception input)
  - ThetaBlock attention + HebbianFFN (settlement-aware)
  - AdaptiveHalting PonderNet readout
  - MixtureOfSoftmaxReadout
  - SinusoidalPositionEncoding
  - Gradient checkpointing at theta boundaries
  - Deep supervision auxiliary losses
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ncalm.model import NCALMModel, register_model
from ncalm.models.v8.config import V8Config
from ncalm.models.v8.hebbian import HebbianKernel
from ncalm.models.v8.attention import ThetaBlock
from ncalm.models.v8.halting import (
    AdaptiveHalting,
    EdgeOfChaosRegularizer,
    SpectralEntropyRegularizer,
)

# Reuse readout from V5 — operates on d_model hidden dims
from ncalm.models.v5.readout import MixtureOfSoftmaxReadout


# ═══════════════════════════════════════════════════════════════════════
# Utility Modules
# ═══════════════════════════════════════════════════════════════════════

class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal position encoding (non-parametric).

    Produces d_model-dimensional sinusoidal features. Applied to the
    hidden state only (not the alpha channel). Preserved from V7.

    Spec Section 5.2: PE(i) ∈ R^d (sinusoidal, added to embedding).
    """

    def __init__(self, d: int, max_seq_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_seq_len, d)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d, 2).float() * -(math.log(10000.0) / d)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:seq_len]


class CausalConv1D(nn.Module):
    """Causal 1D convolution for local spatial perception.

    V8 extension: operates on d_full = d_model + 1 channels, so the
    alpha channel participates in the convolution (Spec Def 7.2).
    This means the convolution output includes alpha-channel
    information from the local neighborhood, which feeds into the
    perception kernel.

    Spec Definition 7.2:
        z_{conv,i} = Σ_{j=0}^{w-1} W_conv^{(j)} · z_{i-j} + b_conv
        where W_conv^{(j)} ∈ R^{(d+1)×(d+1)}, w = kernel_radius + 1

    Left-padded convolution: position i sees only positions i-r, ..., i.
    Shared across all gamma steps (NCA principle).
    """

    def __init__(self, d_full: int, kernel_radius: int):
        super().__init__()
        kernel_size = kernel_radius + 1
        self.padding = kernel_radius
        self.conv = nn.Conv1d(
            in_channels=d_full,
            out_channels=d_full,
            kernel_size=kernel_size,
            padding=0,
            bias=False,
        )
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply causal conv (Spec Def 7.2).

        Args:
            z: (B, L, d+1) full cell states (hidden + alpha)

        Returns:
            z_conv: (B, L, d+1) local convolution output
        """
        # No pre-norm here — RMSNorm is applied after concatenation
        # in HebbianKernel.norm_perceived (Spec Def 7.1)
        x = z.transpose(1, 2)                   # (B, d+1, L)
        x = F.pad(x, (self.padding, 0))         # left-pad for causal
        z_conv = self.conv(x)                    # (B, d+1, L)
        return z_conv.transpose(1, 2)            # (B, L, d+1)


# ═══════════════════════════════════════════════════════════════════════
# Growth State Container
# ═══════════════════════════════════════════════════════════════════════

class StatePool:
    """Persistent buffer of partially grown states for NCA training.

    Spec Section 17 (Pool-Based NCA Training):
        P = {(Z_j, k_j, x_j)}_{j=1}^{P}

    The pool stores detached GrowthState snapshots along with the associated
    input tokens and growth-step count. At each training iteration the model
    either (a) continues growing a pool sample or (b) starts from a fresh
    seed, depending on the rejuvenation probability for the current phase.

    The pool is stored on CPU to avoid occupying GPU memory.
    """

    def __init__(self, pool_size: int) -> None:
        self.pool_size = pool_size
        self.entries: list[dict[str, Any]] = []  # each: {z, a_bar, settled, settle_grads, has_settled, tokens, k}

    def __len__(self) -> int:
        return len(self.entries)

    def _detach_cpu(self, t: torch.Tensor) -> torch.Tensor:
        return t.detach().cpu()

    def store(self, gs: "GrowthState", tokens: torch.Tensor, k: int) -> None:
        """Store a grown state in the pool. Evicts oldest if full."""
        entry = {
            "z": self._detach_cpu(gs.z),
            "a_bar": self._detach_cpu(gs.a_bar),
            "settled": self._detach_cpu(gs.settled),
            "settle_grads": self._detach_cpu(gs.settle_grads),
            "has_settled": self._detach_cpu(gs.has_settled),
            "tokens": self._detach_cpu(tokens),
            "k": k,
        }
        if len(self.entries) < self.pool_size:
            self.entries.append(entry)
        else:
            idx = torch.randint(len(self.entries), (1,)).item()
            self.entries[idx] = entry

    def sample(
        self, device: torch.device, tau_c: float = 0.5, a_bar_init: float = 0.35
    ) -> tuple["GrowthState", torch.Tensor, int]:
        """Sample a random entry, move to device, and rejuvenate frontier.

        Pool states may contain dead cells (alpha=0) that were not settled.
        On resume, these non-settled dead cells need rejuvenation:
          - Alpha reset to 1.0 (so they participate in the growth frontier)
          - a_bar reset to a_bar_init (so they don't immediately re-settle)
        Settled cells (settled >= tau_c) are left untouched (Permanence Axiom).
        """
        idx = torch.randint(len(self.entries), (1,)).item()
        e = self.entries[idx]
        gs = GrowthState(
            z=e["z"].to(device),
            a_bar=e["a_bar"].to(device),
            settled=e["settled"].to(device),
            settle_grads=e["settle_grads"].to(device),
            has_settled=e["has_settled"].to(device),
        )

        # Rejuvenate non-settled cells: restore alpha and reset settlement EMA
        not_settled = gs.settled < tau_c  # (B, L)
        z = gs.z.clone()
        z[..., -1] = torch.where(not_settled, torch.ones_like(z[..., -1]), z[..., -1])
        gs.z = z
        gs.a_bar = torch.where(
            not_settled, torch.full_like(gs.a_bar, a_bar_init), gs.a_bar
        )

        return gs, e["tokens"].to(device), e["k"]


class GrowthState:
    """Container for V8 growth dynamics state variables.

    Tracks the cell lifecycle state across gamma steps:
    - z:              (B, L, d+1) full cell state (hidden + alpha)
    - a_bar:          (B, L) EMA of update magnitudes (Spec Def 10.2)
    - settled:        (B, L) soft settlement scores ∈ [0,1] (Spec Def 10.3)
    - settle_grads:   (B, L, d+1) gradient at time of settlement (Spec Def 10.4)
    - has_settled:    (B, L) bool mask of cells that have ever settled

    These are NOT parameters — they are computed from the state at each step.
    """

    __slots__ = ("z", "a_bar", "settled", "settle_grads", "has_settled")

    def __init__(
        self,
        z: torch.Tensor,
        a_bar: torch.Tensor,
        settled: torch.Tensor,
        settle_grads: torch.Tensor,
        has_settled: torch.Tensor,
    ):
        self.z = z
        self.a_bar = a_bar
        self.settled = settled
        self.settle_grads = settle_grads
        self.has_settled = has_settled


# ═══════════════════════════════════════════════════════════════════════
# Main Model
# ═══════════════════════════════════════════════════════════════════════

@register_model("v8")
class AENCAv8Model(NCALMModel):
    """AENCA V8: Growing Cellular Automata Language Model.

    Extends V7.1's BDH-inspired Hebbian NCA with Growing NCA mechanisms:

    Growing NCA mechanisms (Spec Sections 5-12):
        ✓ Alpha channel: z_i = [z_hidden; z_alpha] ∈ R^{d+1}
        ✓ Alive masking: alive_i = 1[max_neighbor α > τ_α]
        ✓ Three-state lifecycle: DEAD → ACTIVE → SETTLED
        ✓ Settlement: EMA of update magnitudes → soft sigmoid score
        ✓ Awakening: reactivate if gradient shifts > τ_w
        ✓ Stochastic masking: Bernoulli(0.5) per cell per step
        ✓ Growth-gated update: z += alive·(1−settled)·m·Δz
        ✓ Perception kernel: [Conv ∥ State ∥ ∇] → RMSNorm
        ✓ Edge-of-chaos regulation (Lyapunov band loss)
        ✓ Spectral entropy complexity proxy
        ✓ Permanence loss, regeneration loss

    NCA Purity (maintained in gamma loop):
        ✓ Shared kernel weights applied to all cells
        ✓ Local computation (causal conv, radius r)
        ✓ Iterative refinement (t_gamma steps per theta cycle)
        ✓ Residual update with growth gating
        ✓ Per-position output with independent halting
        ✓ Zero-init cascade (model starts as Class I still life)
        ✓ Positive sparse activations (BDH-inspired ReLU)

    Hybrid extension (theta level):
        ✓ Unique attention + HebbianFFN per theta cycle
        ✓ Settlement-aware: settled cells attend but not modified
        ✓ Cross-modal concept gating (pre-attn × post-attn)

    Seven Axioms enforced:
        1. GROWTH:        alpha channel, alive masking, seed init
        2. PERMANENCE:    settlement EMA, permanence loss
        3. SPARSITY:      frontier-only gating, settlement skipping
        4. SIMPLICITY:    shared Hebbian kernel (7.9M for all)
        5. UNIVERSALITY:  edge-of-chaos + spectral entropy regulation
        6. STRUCTURE=FUNC: readout on grown state directly
        7. SELF-ORG:      6-term loss shapes growth dynamics
    """

    def __init__(self, config_dict: dict) -> None:
        super().__init__()

        # Parse config
        v8_keys = {f.name for f in V8Config.__dataclass_fields__.values()}
        v8_params = {k: v for k, v in config_dict.items() if k in v8_keys}
        self.config = V8Config(**v8_params)
        cfg = self.config

        # Auto-size theta FFN if not specified
        if cfg.d_ff_theta == 0:
            target = config_dict.get("target_params", 162_000_000)
            cfg.d_ff_theta = cfg.compute_d_ff_theta(target)

        # ── Token Embedding (Spec Def 5.2) ───────────────────────────
        # W_embed ∈ R^{|V| × d_model}
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.token_embedding.weight, std=1.0 / math.sqrt(cfg.d_model))

        # ── Position Encoding (Spec Def 5.2) ─────────────────────────
        # PE(i) ∈ R^{d_model} — sinusoidal, added to hidden state only
        self.state_pe = SinusoidalPositionEncoding(cfg.d_model, cfg.max_seq_len * 2)

        # ── Shared NCA Components (gamma level) ─────────────────────
        # CausalConv1D: operates on d_full = d+1 channels (Spec Def 7.2)
        self.conv = CausalConv1D(cfg.d_full, cfg.kernel_radius)

        # HebbianKernel: V8 extended perception (Spec Def 8.1)
        self.hebbian_kernel = HebbianKernel(cfg)

        # ── Unique Theta Blocks (one per theta cycle) ────────────────
        self.theta_blocks = nn.ModuleList([
            ThetaBlock(cfg, block_idx=i) for i in range(cfg.n_theta)
        ])

        # ── Adaptive Halting (Spec Sec 16.3) ─────────────────────────
        if cfg.enable_halting:
            self.halting = AdaptiveHalting(cfg)

        # ── Readout (V5, operates on d_model) ────────────────────────
        emb_weight = self.token_embedding.weight if cfg.tie_embeddings else None
        self.readout = MixtureOfSoftmaxReadout(cfg, emb_weight)

        # ── V8 Regularizers ──────────────────────────────────────────
        if cfg.enable_edge_chaos:
            self.edge_regularizer = EdgeOfChaosRegularizer(cfg)

        if cfg.enable_spectral_entropy:
            self.spectral_regularizer = SpectralEntropyRegularizer(cfg)

        # ── Pool-Based NCA Training (Spec Sec 17) ────────────────────
        self._pool: StatePool | None = None
        self._pool_phase: int = 0  # 0=warmup, 1=early, 2=mid, 3=late
        self._pool_p_rejuv: float = 1.0
        self._pool_growth_mult: int = 1
        if cfg.enable_pool:
            self._pool = StatePool(cfg.pool_size)

    # ══════════════════════════════════════════════════════════════════
    # V8 Core: Seed, Perception, Alive Mask, Settlement
    # ══════════════════════════════════════════════════════════════════

    def _seed_state(self, batch: torch.Tensor) -> GrowthState:
        """Create initial seed state from input tokens.

        Spec Definition 5.2:
            z_i^{(0),hidden} = W_embed · x_i + PE(i)    ∀i
            z_i^{(0),alpha}  = 1.0                       ∀i
            ā_i^{(0)}        = ∞  (no settlement history yet)

        All cells begin alive (α = 1.0). Settlement begins with
        no history (ā = large value → settled ≈ 0 initially).

        Args:
            batch: (B, L) token IDs

        Returns:
            GrowthState with initialized z, a_bar, settled, etc.
        """
        B, L = batch.shape
        device = batch.device
        cfg = self.config

        # Hidden state: embedding + position encoding
        z_hidden = self.token_embedding(batch)               # (B, L, d)
        z_hidden = z_hidden + self.state_pe(L).unsqueeze(0)  # (B, L, d)

        # Alpha channel: all alive (Spec Def 5.2: z_i^α = 1.0 ∀i)
        z_alpha = torch.ones(B, L, 1, device=device, dtype=z_hidden.dtype)

        # Full state: [hidden; alpha]
        z = torch.cat([z_hidden, z_alpha], dim=-1)  # (B, L, d+1)

        # Settlement EMA: neutral starting point for realistic settlement dynamics
        # Spec Def 10.2: ā^(0) = ‖Δz^(0)‖; Spec Step 1: ā^(0) = ∞
        # Initialized to τ_s + 3η so cells start clearly unsettled (settled ≈ 0.047)
        # but can settle within available gamma steps as dynamics converge.
        a_bar = torch.full((B, L), cfg.a_bar_init, device=device, dtype=z_hidden.dtype)

        # Settlement scores: all zero (no cells settled)
        settled = torch.zeros(B, L, device=device, dtype=z_hidden.dtype)

        # Gradient at settlement time: undefined initially
        settle_grads = torch.zeros(B, L, cfg.d_full, device=device, dtype=z_hidden.dtype)

        # Has-settled mask: no cells have ever settled
        has_settled = torch.zeros(B, L, device=device, dtype=torch.bool)

        return GrowthState(z, a_bar, settled, settle_grads, has_settled)

    def _compute_alive_mask(self, z: torch.Tensor) -> torch.Tensor:
        """Compute alive mask from alpha channel.

        Spec Definition 6.1:
            alive_i = 1[max_{j ∈ N(i)} z_j^α > τ_α]
        where N(i) = {i-1, i} for causal (autoregressive) models.

        Implementation: use 1D max-pool on the alpha channel with
        causal (left) padding and kernel_size = kernel_radius + 1.

        Spec Proposition 6.1: Dead cells are zeroed out. They contribute
        nothing to computation and consume no FLOPs.

        Spec Remark (Sec 6.1): In V8 language modeling, all positions start
        alive (α=1.0), so dead cells arise only through alpha decay.

        Args:
            z: (B, L, d+1) full cell states

        Returns:
            alive: (B, L) binary alive mask (float for multiplication)
        """
        alpha = z[..., -1]  # (B, L)
        kr = self.config.kernel_radius

        # Causal max-pool: left-pad, then 1D max-pool
        alpha_padded = F.pad(alpha.unsqueeze(1), (kr, 0), value=0.0)  # (B, 1, L+kr)
        max_alpha = F.max_pool1d(
            alpha_padded, kernel_size=kr + 1, stride=1
        ).squeeze(1)  # (B, L)

        alive = (max_alpha > self.config.tau_alpha).float()
        return alive

    def _compute_left_gradient(self, z: torch.Tensor) -> torch.Tensor:
        """Compute left-difference gradient for perception kernel.

        Spec Definition 7.1:
            ∇_i = z_i − z_{i-1}    (discrete spatial derivative)

        This gradient serves as (Spec Proposition 7.1):
        1. Boundary detection: large ‖∇_i‖ = representational boundary
        2. Growth direction: sign/magnitude indicate flow direction
        3. Gap junction analogy: analogous to bioelectric gap junction current

        For position i=0: ∇_0 = z_0 (no left neighbor → gradient is state itself,
        equivalent to padding with zeros).

        Args:
            z: (B, L, d+1) full cell states

        Returns:
            grad: (B, L, d+1) left-difference gradient
        """
        # z_shifted[i] = z[i-1], with z[-1] = 0 (zero-padding at left boundary)
        z_shifted = F.pad(z[:, :-1, :], (0, 0, 1, 0), value=0.0)  # (B, L, d+1)
        return z - z_shifted

    def _perceive(self, z: torch.Tensor) -> torch.Tensor:
        """Compute perceived state by concatenating three local signals.

        Spec Definition 7.1:
            ẑ_i = RMSNorm([z_{conv,i} ∥ z_i ∥ ∇_i])

        Three components (each ∈ R^{d+1}):
        1. z_conv: CausalConv1D output (V7 local perception, extended to d+1)
        2. z:      cell's own state (identity connection)
        3. ∇z:     left-difference gradient (boundary/flow information)

        The RMSNorm is applied inside HebbianKernel.

        Args:
            z: (B, L, d+1) full cell states

        Returns:
            perceived: (B, L, 3(d+1)) concatenated perceived state
        """
        z_conv = self.conv(z)                      # (B, L, d+1)
        z_grad = self._compute_left_gradient(z)    # (B, L, d+1)
        perceived = torch.cat([z_conv, z, z_grad], dim=-1)  # (B, L, 3(d+1))
        return perceived

    def _update_settlement(
        self,
        gs: GrowthState,
        delta_norm: torch.Tensor,
        current_grad: torch.Tensor,
        alive: torch.Tensor,
    ) -> None:
        """Update settlement EMA, scores, and awakening state.

        Three operations:

        1. EMA update (Spec Def 10.2):
           ā_i = β · ā_i + (1−β) · ‖Δz_i‖₂

        2. Settlement score (Spec Def 10.3):
           settled_i = σ((τ_s − ā_i) / η)
           Smooth, differentiable transition ∈ [0, 1].

        3. Record settlement gradient (Spec Def 10.4):
           When settled_i crosses τ_c for the first time, store ∇_i
           for future awakening comparisons.

        Args:
            gs:           GrowthState (mutated in-place)
            delta_norm:   (B, L) ‖Δz_i‖₂ update magnitudes
            current_grad: (B, L, d+1) current left-difference gradient
            alive:        (B, L) alive mask
        """
        cfg = self.config

        # 1. EMA update (Spec Def 10.2)
        # Only update for alive cells (dead cells keep their a_bar)
        new_a_bar = cfg.settlement_beta * gs.a_bar + (1.0 - cfg.settlement_beta) * delta_norm
        gs.a_bar = torch.where(alive.bool(), new_a_bar, gs.a_bar)

        # 2. Settlement score (Spec Def 10.3)
        # settled = σ((τ_s − ā) / η)
        # When ā >> τ_s → settled ≈ 0 (actively changing)
        # When ā << τ_s → settled ≈ 1 (converged)
        gs.settled = torch.sigmoid((cfg.tau_s - gs.a_bar) / cfg.settlement_eta)

        # 3. Record settlement gradient for newly settled cells
        # A cell is "newly settled" when settled >= τ_c AND it hasn't
        # previously recorded a settlement gradient.
        # NOTE: No .any() guard — torch.where is a no-op when mask is all-False,
        # and avoiding CPU-GPU sync is critical for throughput (48 gamma steps/fwd).
        has_settled_bool = gs.has_settled.bool() if gs.has_settled.dtype != torch.bool else gs.has_settled
        newly_settled = (gs.settled >= cfg.tau_c) & (~has_settled_bool) & alive.bool()
        gs.settle_grads = torch.where(
            newly_settled.unsqueeze(-1),
            current_grad,
            gs.settle_grads,
        )
        gs.has_settled = (has_settled_bool | newly_settled)

    def _check_awakening(
        self, gs: GrowthState, current_grad: torch.Tensor
    ) -> None:
        """Check and process awakening of settled cells.

        Spec Definition 10.4:
            awaken_i = 1[‖∇_i − ∇_i^{settle}‖₂ > τ_w]

        A settled cell reawakens when its local context has changed
        significantly since it settled. Upon awakening:
            settled_i → 0
            ā_i → reset (will be updated on next gamma step)

        Spec Proposition 10.2: Awakening occurs when the "landscape"
        around a settled cell changes — the cell's relationship to its
        neighbors has shifted from what it was at settlement time.
        Corresponds to Levin's regeneration.

        Args:
            gs:           GrowthState (mutated in-place)
            current_grad: (B, L, d+1) current left-difference gradient
        """
        cfg = self.config

        # Ensure has_settled is boolean (may be float from checkpoint)
        has_settled_bool = gs.has_settled.bool() if gs.has_settled.dtype != torch.bool else gs.has_settled

        # Only check cells that are currently settled (above computation cutoff)
        is_settled = has_settled_bool & (gs.settled >= cfg.tau_c)

        # Compute gradient shift: ‖∇_current − ∇_settle‖₂
        # NOTE: No .any() guards — torch.where handles empty masks without
        # CPU-GPU sync, critical for throughput in the 48-step gamma loop.
        grad_diff = (current_grad - gs.settle_grads).norm(dim=-1)  # (B, L)

        # Awakening: gradient shift exceeds threshold
        should_awaken = is_settled & (grad_diff > cfg.tau_w)

        # Reset settlement for awakened cells (no-op if mask is all-False)
        gs.settled = torch.where(
            should_awaken, torch.zeros_like(gs.settled), gs.settled
        )
        gs.a_bar = torch.where(
            should_awaken,
            torch.full_like(gs.a_bar, self.config.a_bar_init),
            gs.a_bar,
        )
        gs.has_settled = has_settled_bool & (~should_awaken)

    # ══════════════════════════════════════════════════════════════════
    # V8 Core: Gamma Step
    # ══════════════════════════════════════════════════════════════════

    def _gamma_step(
        self,
        gs: GrowthState,
        is_training: bool,
    ) -> torch.Tensor:
        """Single V8 NCA gamma step with growth dynamics.

        Implements the complete per-step algorithm from Spec Section 15.1
        (Step 2a), integrating all V8 mechanisms:

        (i)    Alive mask from alpha channel (Def 6.1)
        (ii)   Dead cell enforcement (Prop 6.1)
        (iii)  Awakening check for settled cells (Def 10.4)
        (iv)   Skip settled cells (settlement gating)
        (v)    Perception: [Conv ∥ State ∥ ∇] (Def 7.1)
        (vi)   Hebbian update: Δz = Φ(ẑ, z) (Def 8.1)
        (vii)  Stochastic mask: m ~ Bernoulli(0.5) (Def 9.1)
        (viii) Growth-gated apply: z += alive·(1−settled)·m·Δz
        (ix)   Update settlement EMA and scores (Def 10.2-10.3)
        (x)    Record settlement gradient (Def 10.4)

        Args:
            gs:          GrowthState (mutated in-place)
            is_training: whether in training mode (affects stochastic mask)

        Returns:
            delta_z: (B, L, d+1) the raw update before gating (for loss computation)
        """
        cfg = self.config
        z = gs.z

        # NOTE: Removed early-exit .all() check here — it triggered a
        # CPU-GPU sync every gamma step (~1500×/optimizer step), killing
        # throughput. Settlement gating in step (viii) already makes the
        # update near-zero for settled cells. The FLOP saving from early
        # exit is negligible vs the pipeline stall cost.

        # (i) Compute alive mask (Spec Def 6.1)
        alive = self._compute_alive_mask(z)  # (B, L)

        # (ii) Dead cell enforcement (Spec Prop 6.1): zero out dead cells
        z = z * alive.unsqueeze(-1)
        gs.z = z

        # (iii) Compute current gradient for awakening check
        current_grad = self._compute_left_gradient(z)  # (B, L, d+1)

        # Check awakening: settled cells whose context changed (Spec Def 10.4)
        self._check_awakening(gs, current_grad)

        # (v) Perception: [Conv ∥ State ∥ ∇] → (B, L, 3(d+1))
        perceived = self._perceive(z)

        # (vi) Hebbian update: Δz = Φ(ẑ, z) (Spec Def 8.1)
        delta_z = self.hebbian_kernel(perceived, z)  # (B, L, d+1)

        # (vii) Stochastic mask (Spec Def 9.1)
        if is_training:
            # m_i ~ Bernoulli(p_update) independently per cell
            stoch_mask = torch.bernoulli(
                torch.full_like(alive, cfg.p_update)
            )  # (B, L)
        else:
            # Inference: deterministic mode (Spec Sec 9.3, option 1)
            # All cells update with magnitude scaled by p_update
            stoch_mask = torch.full_like(alive, cfg.p_update)

        # (viii) Growth-gated update (Spec Sec 9.1):
        #   z_i += alive_i · (1 − settled_i) · m_i · Δz_i
        gate = alive * (1.0 - gs.settled) * stoch_mask  # (B, L)
        gs.z = z + gate.unsqueeze(-1) * delta_z

        # (ix) Update settlement (Spec Def 10.2-10.3)
        # Compute ‖Δz_i‖₂ for EMA: raw kernel output magnitude (Spec Def 10.2)
        # Uses raw Δz (before gating) to track the kernel's proposed update,
        # not the gated update. This prevents positive feedback where
        # (1-settled) ≈ 0 → gated_delta ≈ 0 → EMA drops → settled rises.
        # Normalized by √d_full so tau_s is scale-independent across d_model.
        delta_norm = delta_z.norm(dim=-1) / math.sqrt(cfg.d_full)  # (B, L)
        self._update_settlement(gs, delta_norm, current_grad, alive)

        return delta_z

    # ══════════════════════════════════════════════════════════════════
    # V8 Core: Theta Cycle and Hierarchical Loop
    # ══════════════════════════════════════════════════════════════════

    def _run_theta_cycle(
        self,
        gs: GrowthState,
        theta_block: ThetaBlock,
    ) -> tuple[GrowthState, torch.Tensor]:
        """Run one theta cycle: t_gamma gamma steps + theta attention.

        Spec Definition 13.2:
            Z_post_gamma = Gamma^{t_gamma}(Z)
            Z_post_theta = ThetaBlock(Z_post_gamma, settled)

        Also accumulates per-gamma permanence loss (Spec Sec 16.4):
            perm_k = alive_i · (1 − settled_i)  for each gamma step k

        Args:
            gs:          GrowthState (mutated across gamma steps)
            theta_block: the unique ThetaBlock for this cycle

        Returns:
            gs:         GrowthState after the full theta cycle
            perm_accum: scalar sum of permanence contributions from
                        this cycle's t_gamma gamma steps
        """
        device = gs.z.device
        perm_accum = torch.tensor(0.0, device=device, dtype=gs.z.dtype)

        # Gamma steps (shared kernel, iterative refinement)
        for _ in range(self.config.t_gamma):
            self._gamma_step(gs, is_training=self.training)
            # Per-gamma permanence (Spec Sec 16.4: sum over ALL K steps)
            alive = self._compute_alive_mask(gs.z)
            perm_accum = perm_accum + (alive * (1.0 - gs.settled)).mean()

        # Theta step (unique attention + HebbianFFN)
        # Settlement-aware: settled cells attend but not modified
        gs.z = theta_block(gs.z, gs.settled)

        return gs, perm_accum

    def _run_theta_cycle_for_checkpoint(
        self,
        z: torch.Tensor,
        a_bar: torch.Tensor,
        settled: torch.Tensor,
        settle_grads: torch.Tensor,
        has_settled: torch.Tensor,
        theta_block: ThetaBlock,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Checkpoint-compatible wrapper for _run_theta_cycle.

        torch.utils.checkpoint requires plain tensor arguments/returns
        (no custom objects). This wrapper unpacks/repacks the GrowthState.

        Args:
            z, a_bar, settled, settle_grads, has_settled: GrowthState fields
            theta_block: ThetaBlock for this cycle

        Returns:
            Tuple of updated GrowthState fields + permanence accumulator
        """
        gs = GrowthState(z, a_bar, settled, settle_grads, has_settled)
        gs, perm_accum = self._run_theta_cycle(gs, theta_block)
        return gs.z, gs.a_bar, gs.settled, gs.settle_grads, gs.has_settled, perm_accum

    def _run_hierarchical_loop(
        self,
        gs: GrowthState,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run the full two-timescale hierarchical loop with V8 growth.

        Spec Section 15.1 (Step 2):
            For n = 1..n_theta:
                Z = ThetaCycle_n(Z)     [gamma steps + theta attention]
                h^(n) = Halt(Z)         [PonderNet readout accumulation]

        With gradient checkpointing at theta boundaries (Spec Prop 20.3).

        Args:
            gs:    GrowthState with seeded initial state
            batch: (B, L) token IDs for aux loss (None during inference)

        Returns:
            z_out:     (B, L, d_model) halting-weighted accumulated hidden state
            loop_info: diagnostics dict with loss components
        """
        cfg = self.config
        B, L, _ = gs.z.shape
        d = cfg.d_model
        device = gs.z.device
        n_theta = cfg.n_theta
        use_halting = cfg.enable_halting

        # ── Readout accumulators (on d_model, not d_full) ────────────
        z_out = torch.zeros(B, L, d, device=device, dtype=gs.z.dtype)
        remaining = torch.ones(B, L, device=device, dtype=gs.z.dtype)
        halt_probs: list[torch.Tensor] = []

        # ── Deep supervision substrates ──────────────────────────────
        aux_substrates: list[torch.Tensor] = []

        # ── Settlement monitoring ────────────────────────────────────
        frontier_sizes: list = []
        settled_fracs: list = []

        # ── Intermediate state snapshots (for regeneration, Spec Sec 16.5)
        z_snapshots: list[torch.Tensor] = []

        # ── Permanence loss accumulator (Spec Sec 16.4) ──────────────
        # L_perm = (1/(L·K)) Σ_k Σ_i alive_i · (1 − settled_i)
        perm_loss_accum = torch.tensor(0.0, device=device, dtype=gs.z.dtype)
        perm_loss_count = 0

        # ── Main theta loop ──────────────────────────────────────────
        n_active_cycles = 0

        for theta_idx in range(n_theta):
            theta_block = self.theta_blocks[theta_idx]
            is_last = (theta_idx == n_theta - 1)

            # Early exit if all positions halted
            # NOTE: .all() sync is acceptable here (once per theta cycle,
            # not per gamma step — 12× vs 48×16 = 768× per forward pass)
            if use_halting and (remaining <= cfg.halt_eps).all():
                break

            # --- Pre-cycle monitoring + snapshot ---
            # Defer .item() calls to avoid per-theta CPU-GPU syncs.
            # Store tensors and extract values after the loop.
            with torch.no_grad():
                alive_now = self._compute_alive_mask(gs.z)
                frontier_sizes.append(
                    (alive_now * (gs.settled < cfg.tau_c).float()).sum() / max(L * B, 1)
                )
                settled_fracs.append(
                    (gs.settled >= cfg.tau_c).float().mean()
                )

            # Save z snapshot for mid-trajectory regeneration (Spec Sec 16.5)
            z_snapshots.append(gs.z.detach().clone())

            # Run theta cycle (gamma steps + theta attention + perm accumulation)
            if self.training:
                z_new, a_bar_new, settled_new, sg_new, hs_new, perm_cycle = checkpoint(
                    self._run_theta_cycle_for_checkpoint,
                    gs.z, gs.a_bar, gs.settled, gs.settle_grads,
                    gs.has_settled.float(),  # checkpoint needs float tensors
                    theta_block,
                    use_reentrant=False,
                )
                gs.z = z_new
                gs.a_bar = a_bar_new
                gs.settled = settled_new
                gs.settle_grads = sg_new
                gs.has_settled = hs_new > 0.5  # back to bool
            else:
                gs, perm_cycle = self._run_theta_cycle(gs, theta_block)

            # Accumulate per-gamma permanence (Spec Sec 16.4)
            perm_loss_accum = perm_loss_accum + perm_cycle
            perm_loss_count += cfg.t_gamma

            n_active_cycles += 1

            # ── Halting / readout accumulation ───────────────────────
            z_hidden = gs.z[..., :d]  # slice off alpha for readout

            if use_halting:
                z_out, remaining, hp = self.halting.accumulate(
                    z_hidden, z_out, remaining, is_last,
                )
                halt_probs.append(hp)
            else:
                if is_last:
                    z_out = z_out + remaining.unsqueeze(-1) * z_hidden
                    remaining = torch.zeros_like(remaining)
                else:
                    weight = 1.0 / n_theta
                    z_out = z_out + weight * z_hidden
                    remaining = remaining - weight

            # Deep supervision snapshot: raw per-cycle state (Spec Sec 16.7)
            # Spec: L_aux = (1/n_θ) Σ_{n=1}^{n_θ} L_CE^(n) — ALL cycles, uniform weight
            if batch is not None and cfg.lambda_aux > 0:
                aux_substrates.append(gs.z[..., :d].clone())

        # ── Absorb any remaining mass (safety) ───────────────────────
        if remaining.sum() > 0.001:
            z_out = z_out + remaining.unsqueeze(-1) * gs.z[..., :d]

        # ── Permanence loss (Spec Sec 16.4) ──────────────────────────
        perm_loss = perm_loss_accum / max(perm_loss_count, 1)

        # ── Deep supervision auxiliary losses (Spec Sec 16.7) ────────
        # Spec: L_aux = (1/n_θ) Σ_{n=1}^{n_θ} L_CE^(n)  — uniform weight, all cycles
        # Apply readout to raw per-cycle grown states, not halting-accumulated z_out
        aux_loss: torch.Tensor | None = None
        if batch is not None and cfg.lambda_aux > 0 and aux_substrates:
            targets_flat = batch[:, 1:].reshape(-1)
            aux_total = torch.tensor(0.0, device=device)

            for z_snap in aux_substrates:
                aux_logits = self.readout(z_snap)
                aux_flat = aux_logits[:, :-1].reshape(-1, cfg.vocab_size)

                if self.readout.returns_log_probs:
                    aux_ce = F.nll_loss(aux_flat, targets_flat, reduction="mean")
                else:
                    aux_ce = F.cross_entropy(aux_flat, targets_flat, reduction="mean")

                aux_total = aux_total + aux_ce

            aux_loss = aux_total / len(aux_substrates)

        # ── Halting KL ───────────────────────────────────────────────
        halt_kl: torch.Tensor | None = None
        if use_halting and halt_probs and cfg.lambda_halt > 0:
            halt_kl = self.halting.compute_kl_loss(halt_probs)

        # ── Monitoring metrics ───────────────────────────────────────
        with torch.no_grad():
            halted_mask = (remaining <= cfg.halt_eps)
            pct_halted = halted_mask.float().mean().item()
            expected_steps = n_active_cycles * (cfg.t_gamma + 1)
            final_settled = (gs.settled >= cfg.tau_c).float().mean().item()
            # Convert deferred tensor monitoring to Python floats (single sync)
            avg_frontier = float(sum(frontier_sizes) / max(len(frontier_sizes), 1))
            settled_fracs = [float(x) for x in settled_fracs]

        loop_info: dict[str, Any] = {
            "aux_loss": aux_loss,
            "halt_kl": halt_kl,
            "perm_loss": perm_loss,
            "expected_steps": expected_steps,
            "pct_halted_early": pct_halted,
            "n_active_cycles": n_active_cycles,
            "pct_settled": final_settled,
            "avg_frontier_frac": avg_frontier,
            "settled_trajectory": settled_fracs,
            "growth_state": gs,  # needed for edge-of-chaos
            "z_snapshots": z_snapshots,  # for mid-trajectory regeneration (Spec Sec 16.5)
        }

        return z_out, loop_info

    # ══════════════════════════════════════════════════════════════════
    # V8 Loss: Edge-of-Chaos (Spec Sec 16.6)
    # ══════════════════════════════════════════════════════════════════

    def _compute_edge_of_chaos_loss(
        self,
        batch: torch.Tensor,
        gs_clean: GrowthState,
    ) -> torch.Tensor:
        """Compute edge-of-chaos Lyapunov band loss.

        Spec Section 16.6: Requires a second forward pass with perturbed
        initial state. Computed on a subset of the batch for efficiency.

        Procedure:
            1. Take a subset of the batch (edge_batch_fraction)
            2. Create perturbation ε ~ N(0, σ_ε² I)
            3. Run perturbed state through one full theta cycle
            4. Estimate λ̂_max from divergence
            5. Compute band loss

        Args:
            batch:    (B, L) token IDs
            gs_clean: GrowthState at the end of clean forward pass

        Returns:
            edge_loss: scalar edge-of-chaos loss
        """
        cfg = self.config
        B = batch.shape[0]
        n_sub = max(int(B * cfg.edge_batch_fraction), 1)

        # Take subset of clean state
        z_sub = gs_clean.z[:n_sub].detach()

        # Create perturbation (Spec Def 11.1: ε ~ N(0, σ_ε² I))
        epsilon = torch.randn_like(z_sub) * cfg.lyapunov_eps
        epsilon_norm = epsilon.flatten(1).norm(dim=1)  # (n_sub,)

        # Run perturbed state through one theta cycle of gamma steps
        # (We use the first theta block for the Lyapunov estimation)
        z_perturbed = z_sub + epsilon

        # Run t_gamma gamma steps on both clean and perturbed
        # (minimal version: just the gamma dynamics, no theta attention)
        gs_test_clean = GrowthState(
            z=z_sub.clone(),
            a_bar=torch.full((n_sub, z_sub.shape[1]), cfg.a_bar_init, device=z_sub.device, dtype=z_sub.dtype),
            settled=torch.zeros(n_sub, z_sub.shape[1], device=z_sub.device, dtype=z_sub.dtype),
            settle_grads=torch.zeros_like(z_sub),
            has_settled=torch.zeros(n_sub, z_sub.shape[1], device=z_sub.device, dtype=torch.bool),
        )
        gs_test_perturbed = GrowthState(
            z=z_perturbed,
            a_bar=torch.full((n_sub, z_sub.shape[1]), cfg.a_bar_init, device=z_sub.device, dtype=z_sub.dtype),
            settled=torch.zeros(n_sub, z_sub.shape[1], device=z_sub.device, dtype=z_sub.dtype),
            settle_grads=torch.zeros_like(z_perturbed),
            has_settled=torch.zeros(n_sub, z_sub.shape[1], device=z_sub.device, dtype=torch.bool),
        )

        n_steps = cfg.t_gamma
        for _ in range(n_steps):
            self._gamma_step(gs_test_clean, is_training=False)
            self._gamma_step(gs_test_perturbed, is_training=False)

        # Estimate Lyapunov exponent
        lambda_max = self.edge_regularizer.estimate_lyapunov(
            gs_test_clean.z, gs_test_perturbed.z, epsilon_norm, n_steps
        )

        return self.edge_regularizer.compute_loss(lambda_max)

    # ══════════════════════════════════════════════════════════════════
    # V8 Loss: Spectral Entropy (Spec Sec 19)
    # ══════════════════════════════════════════════════════════════════

    def _compute_spectral_entropy_loss(
        self, z_grown: torch.Tensor
    ) -> torch.Tensor:
        """Compute spectral entropy complexity band loss.

        Spec Section 19.3:
            L_complexity = ReLU(η_min − H_sv) + ReLU(H_sv − η_max)

        Applied to Z^(K) — the final grown state (Spec Sec 19.3),
        NOT the halting-weighted readout mixture.

        Args:
            z_grown: (B, L, d_model) final grown hidden state

        Returns:
            complexity_loss: scalar spectral entropy band loss
        """
        cfg = self.config
        B, L, d = z_grown.shape

        H_sv = self.spectral_regularizer.compute_spectral_entropy(z_grown)
        return self.spectral_regularizer.compute_loss(H_sv, L, d)

    # ══════════════════════════════════════════════════════════════════
    # V8 Loss: Regeneration (Spec Sec 16.5, 18)
    # ══════════════════════════════════════════════════════════════════

    def _compute_regeneration_loss(
        self,
        z_snapshots: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute regeneration loss via mid-trajectory corruption→recovery.

        Spec Section 16.5 + Section 18 (Corruption Protocols A/B/C):
            1. Pick a random intermediate state from z_snapshots
            2. Apply one of three corruption protocols (chosen randomly):
               A: Gaussian noise — z_corrupt = z + δ, δ ~ N(0, σ²I)
               B: Zeroing — randomly zero 50% of positions
               C: Local shuffle — permute cells within ±2 positions
            3. Run T_r gamma steps on both corrupted and clean copies
            4. MSE between recovered and clean-advanced states

        Spec Proposition 16.1: Minimizing L_regen is equivalent to training
        contractive dynamics near the natural trajectory.

        This creates Levin's morphogenetic attractor dynamics: the system
        has a basin of attraction around each natural state, and
        perturbations are exponentially damped.

        Args:
            z_snapshots: list of detached z tensors from theta cycle boundaries

        Returns:
            regen_loss: scalar regeneration loss
        """
        cfg = self.config
        device = z_snapshots[0].device

        # Pick random mid-trajectory snapshot (Spec Sec 16.5)
        snap_idx = torch.randint(len(z_snapshots), (1,)).item()
        z_clean = z_snapshots[snap_idx]  # already detached
        B, L, df = z_clean.shape

        # Pick corruption protocol (Spec Sec 18: A, B, or C)
        protocol = torch.randint(3, (1,)).item()
        if protocol == 0:
            # Protocol A: Gaussian noise (Spec Sec 18.1)
            z_corrupted = z_clean + torch.randn_like(z_clean) * cfg.sigma_corrupt
        elif protocol == 1:
            # Protocol B: Zero random positions (Spec Sec 18.2)
            mask = (torch.rand(B, L, 1, device=device) > 0.5).float()
            z_corrupted = z_clean * mask
        else:
            # Protocol C: Local shuffle within ±2 positions (Spec Sec 18.3)
            shift = torch.randint(-2, 3, (B, L), device=device)
            src_idx = (torch.arange(L, device=device).unsqueeze(0) + shift).clamp(0, L - 1)
            z_corrupted = z_clean.gather(1, src_idx.unsqueeze(-1).expand_as(z_clean))

        # Create recovery and clean-continuation GrowthStates
        a_bar_val = cfg.a_bar_init
        gs_recover = GrowthState(
            z=z_corrupted.clone(),
            a_bar=torch.full((B, L), a_bar_val, device=device, dtype=z_clean.dtype),
            settled=torch.zeros(B, L, device=device, dtype=z_clean.dtype),
            settle_grads=torch.zeros_like(z_corrupted),
            has_settled=torch.zeros(B, L, device=device, dtype=torch.bool),
        )
        gs_clean_cont = GrowthState(
            z=z_clean.clone(),
            a_bar=torch.full((B, L), a_bar_val, device=device, dtype=z_clean.dtype),
            settled=torch.zeros(B, L, device=device, dtype=z_clean.dtype),
            settle_grads=torch.zeros_like(z_clean),
            has_settled=torch.zeros(B, L, device=device, dtype=torch.bool),
        )

        # Run T_r recovery steps (Spec Sec 16.5)
        # Both trajectories advance T_r steps; contraction means they converge
        for _ in range(cfg.T_r):
            self._gamma_step(gs_recover, is_training=self.training)
            self._gamma_step(gs_clean_cont, is_training=False)  # deterministic clean

        # Recovery loss: MSE between recovered and clean-advanced
        # Spec: L_regen = (1/|C|) Σ ‖z_recovered(k+T_r) − z_clean(k+T_r)‖²
        regen_loss = (gs_recover.z - gs_clean_cont.z).pow(2).mean()

        return regen_loss

    # ══════════════════════════════════════════════════════════════════
    # Training Forward Pass
    # ══════════════════════════════════════════════════════════════════

    def forward_train(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Complete V8 training step with 6-term loss function.

        Spec Section 16.1:
            L_V8 = L_CE + λ_halt·L_halt + λ_perm·L_perm
                   + λ_regen·L_regen + λ_edge·L_edge + λ_aux·L_aux

        Each term enforces one or more of the Seven Axioms (Spec Table 16.1).

        Args:
            batch: (B, L) token IDs, dtype long.

        Returns:
            loss:  Scalar total training loss.
            info:  Dict with loss components and monitoring metrics.
        """
        B, L = batch.shape
        cfg = self.config

        # ── Pool-Based Training (Spec Sec 17) ────────────────────────
        # Either continue growing a pool sample or start from fresh seed
        use_pool_sample = (
            cfg.enable_pool
            and self._pool is not None
            and len(self._pool) > 0
            and torch.rand(1).item() >= self._pool_p_rejuv
        )

        if use_pool_sample:
            gs, pool_tokens, pool_k = self._pool.sample(
                batch.device, tau_c=cfg.tau_c, a_bar_init=cfg.a_bar_init
            )
            # Use the pool's associated tokens for loss computation
            batch = pool_tokens
            B, L = batch.shape
        else:
            gs = self._seed_state(batch)

        # ── Step 2: Hierarchical Growth Loop (Spec Sec 15.1) ─────────
        z_out, loop_info = self._run_hierarchical_loop(gs, batch)

        # ── Pool update: store grown state back for future continuation
        if cfg.enable_pool and self._pool is not None:
            k_new = (pool_k if use_pool_sample else 0) + cfg.total_gamma_steps
            gs_final_pool: GrowthState = loop_info["growth_state"]
            self._pool.store(gs_final_pool, batch, k_new)

        # ── Step 3: Readout (Spec Sec 15.1 Step 3) ──────────────────
        logits = self.readout(z_out)

        # ══════════════════════════════════════════════════════════════
        # Step 4: Compute 6-term loss (Spec Section 16)
        # ══════════════════════════════════════════════════════════════

        logits_flat = logits[:, :-1].reshape(-1, cfg.vocab_size)
        targets_flat = batch[:, 1:].reshape(-1)

        # ── L_CE: Cross-Entropy (Spec Sec 16.2) ─────────────────────
        # Axiom 6 (Structure=Function): next-token prediction from grown state
        if self.readout.returns_log_probs:
            ce_loss = F.nll_loss(logits_flat, targets_flat, reduction="mean")
        else:
            ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")

        total_loss = ce_loss
        info: dict[str, Any] = {"ce": ce_loss.detach()}

        # ── L_perm: Permanence Loss (Spec Sec 16.4) ─────────────────
        # Axiom 2 (Permanence), Axiom 3 (Sparsity)
        # Fraction of alive cells still in active frontier
        if cfg.lambda_perm > 0:
            perm_loss = loop_info["perm_loss"]
            total_loss = total_loss + cfg.lambda_perm * perm_loss
            info["perm"] = perm_loss.detach()

        # ── L_aux: Deep Supervision (Spec Sec 16.7) ─────────────────
        # Axiom 1 (Growth), Axiom 7 (Self-Organization)
        if cfg.lambda_aux > 0 and loop_info["aux_loss"] is not None:
            total_loss = total_loss + cfg.lambda_aux * loop_info["aux_loss"]
            info["aux"] = loop_info["aux_loss"].detach()

        # ── L_halt: PonderNet KL (Spec Sec 16.3) ────────────────────
        # Axiom 1 (Growth), Axiom 5 (Universality)
        if cfg.enable_halting and cfg.lambda_halt > 0 and loop_info["halt_kl"] is not None:
            total_loss = total_loss + cfg.lambda_halt * loop_info["halt_kl"]
            info["halt_kl"] = loop_info["halt_kl"].detach()

        # ── L_edge: Edge-of-Chaos (Spec Sec 16.6) ───────────────────
        # Axiom 5 (Universality), Axiom 4 (Simplicity)
        # Computed on subset of batch for efficiency
        gs_final: GrowthState = loop_info["growth_state"]
        if cfg.enable_edge_chaos and cfg.lambda_edge > 0:
            edge_loss = self._compute_edge_of_chaos_loss(batch, gs_final)
            total_loss = total_loss + cfg.lambda_edge * edge_loss
            info["edge"] = edge_loss.detach()

            # Optional: spectral entropy complexity band
            # Spec Sec 19.3: H_sv applied to Z^(K) — the final grown state,
            # not the halting-weighted mixture z_out
            if cfg.enable_spectral_entropy and cfg.alpha_sv > 0:
                z_grown = gs_final.z[..., :cfg.d_model]  # (B, L, d)
                sv_loss = self._compute_spectral_entropy_loss(z_grown)
                total_loss = total_loss + cfg.alpha_sv * sv_loss
                info["spectral_entropy"] = sv_loss.detach()

        # ── L_regen: Regeneration (Spec Sec 16.5, 18) ──────────────
        # Axiom 2 (Permanence), Axiom 6 (Structure=Function)
        # Applied stochastically: with probability p_regen per batch
        # Uses mid-trajectory snapshots with corruption protocols A/B/C
        if cfg.enable_regen and cfg.lambda_regen > 0:
            z_snapshots = loop_info["z_snapshots"]
            if torch.rand(1).item() < cfg.p_regen and z_snapshots:
                regen_loss = self._compute_regeneration_loss(z_snapshots)
                total_loss = total_loss + cfg.lambda_regen * regen_loss
                info["regen"] = regen_loss.detach()

        # ── Monitoring metrics ───────────────────────────────────────
        info["expected_steps"] = loop_info["expected_steps"]
        info["pct_halted_early"] = loop_info["pct_halted_early"]
        info["pct_settled"] = loop_info["pct_settled"]
        info["avg_frontier_frac"] = loop_info["avg_frontier_frac"]

        return total_loss, info

    # ══════════════════════════════════════════════════════════════════
    # Inference
    # ══════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Autoregressive inference with V8 growth dynamics.

        Spec Section 9.3 (Inference):
            Option 1 (deterministic): m_i = p_update for all cells.
            Update magnitude is scaled but all cells update every step.

        Settled cells are read-only in attention, providing cached context.
        Spec Section 24.1-24.2 (Permanence-Accelerated Inference).

        Args:
            input_ids: (B, L) token IDs.

        Returns:
            logits: (B, L, vocab_size) predictions at all positions.
        """
        cfg = self.config

        gs = self._seed_state(input_ids)
        z_out, _ = self._run_hierarchical_loop(gs, batch=None)

        return self.readout(z_out)

    # ══════════════════════════════════════════════════════════════════
    # Parameter Groups
    # ══════════════════════════════════════════════════════════════════

    def get_param_groups(self) -> list[dict]:
        """Parameter groups for optimizer.

        decay: 2D+ weight matrices (kernels, attention, FFN)
        no_decay: biases, norms, embeddings
        """
        decay = []
        no_decay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2 or "norm" in name or "token_embedding" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {"params": decay, "label": "decay"},
            {"params": no_decay, "label": "no_decay", "weight_decay": 0.0},
        ]

    # ══════════════════════════════════════════════════════════════════
    # Pool-Based Training Hooks (Spec Sec 17)
    # ══════════════════════════════════════════════════════════════════

    def on_step_start(self, step: int, max_steps: int) -> None:
        """Update pool training phase based on growth curriculum (Spec Def 17.2).

        Phase boundaries (fraction of training):
            Warmup (0–5%):  p_rejuv=1.0,  growth_mult=1× (always fresh)
            Early  (5–20%): p_rejuv=0.5,  growth_mult=1× (mix fresh/continued)
            Mid   (20–60%): p_rejuv=0.2,  growth_mult=2× (longer growth)
            Late  (60–100%): p_rejuv=0.1, growth_mult=3× (extended growth)
        """
        cfg = self.config
        if not cfg.enable_pool:
            return

        frac = step / max(max_steps, 1)
        boundaries = cfg.pool_phase_boundaries  # (0.05, 0.20, 0.60)

        if frac < boundaries[0]:
            phase = 0
        elif frac < boundaries[1]:
            phase = 1
        elif frac < boundaries[2]:
            phase = 2
        else:
            phase = 3

        self._pool_phase = phase
        self._pool_p_rejuv = cfg.pool_rejuv_schedule[phase]
        self._pool_growth_mult = cfg.pool_growth_multipliers[phase]

    # ══════════════════════════════════════════════════════════════════
    # Description
    # ══════════════════════════════════════════════════════════════════

    def describe(self) -> dict[str, Any]:
        """V8 architecture description for analysis.json and logging."""
        cfg = self.config
        total = sum(p.numel() for p in self.parameters())

        kernel_params = sum(p.numel() for p in self.hebbian_kernel.parameters())
        conv_params = sum(p.numel() for p in self.conv.parameters())
        embed_params = sum(p.numel() for p in self.token_embedding.parameters())
        readout_params = sum(p.numel() for p in self.readout.parameters())
        theta_params = sum(p.numel() for p in self.theta_blocks.parameters())
        halt_params = (
            sum(p.numel() for p in self.halting.parameters())
            if cfg.enable_halting else 0
        )

        return {
            "architecture": "AENCA V8.0 (Growing Cellular Automata)",
            "total_params": total,
            "hebbian_kernel_params": kernel_params,
            "conv_params": conv_params,
            "embed_params": embed_params,
            "readout_params": readout_params,
            "theta_block_params": theta_params,
            "halt_params": halt_params,
            "d_model": cfg.d_model,
            "d_full": cfg.d_full,
            "d_concept": cfg.d_concept,
            "d_ff_theta": cfg.d_ff_theta,
            "n_theta": cfg.n_theta,
            "t_gamma": cfg.t_gamma,
            "n_heads": cfg.n_heads,
            "d_head": cfg.d_head,
            "vocab_size": cfg.vocab_size,
            "max_seq_len": cfg.max_seq_len,
            "v8_mechanisms": {
                "alpha_channel": True,
                "alive_masking": True,
                "settlement": True,
                "awakening": True,
                "stochastic_masking": True,
                "edge_of_chaos": cfg.enable_edge_chaos,
                "spectral_entropy": cfg.enable_spectral_entropy,
                "regeneration": cfg.enable_regen,
                "permanence_loss": cfg.lambda_perm > 0,
            },
            "v8_hyperparameters": {
                "tau_alpha": cfg.tau_alpha,
                "tau_s": cfg.tau_s,
                "tau_c": cfg.tau_c,
                "tau_w": cfg.tau_w,
                "settlement_beta": cfg.settlement_beta,
                "settlement_eta": cfg.settlement_eta,
                "p_update": cfg.p_update,
                "lambda_perm": cfg.lambda_perm,
                "lambda_regen": cfg.lambda_regen,
                "lambda_edge": cfg.lambda_edge,
                "lambda_halt": cfg.lambda_halt,
                "lambda_aux": cfg.lambda_aux,
            },
        }
