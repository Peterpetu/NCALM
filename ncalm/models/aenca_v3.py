"""AENCA v3: Active Embedding Neural Cellular Automata v3.

A pure Neural Cellular Automaton (NCA) language model based on the
Active Inference / Free Energy Principle framework.

Architecture summary:
  - World: token tape of length L (external, read-only via active sensors)
  - Substrate: 2D grid of Nx×Ny cells, each with state s_u ∈ R^D
  - Kernel f_θ: single shared MLP applied identically to every cell (NCA rule)
  - Sensors: each cell has a discrete read pointer p_u ∈ {0,...,pos} (causal)
  - Multi-scale movement: step sizes 2^ℓ for ℓ ∈ {0,...,L_scales-1}
  - Action selection: Expected Free Energy minimisation (active inference)
  - Readout: distributed mean vote W_out over belief channels

Fixes v3.1 (2026-02-21):
  Fix A: Variance-preserving gate bias init (-3.9) + layer scaling (1/√6)
  Fix B: TBPTT K=32 — prevents gradient explosion through 2550 recurrent steps
  Fix C: L_scales=18 (250K context coverage) + logarithmic movement cost
  Fix D: RelativePositionEncoder replaces absolute PointerPositionEncoder
         — zero learned parameters, provable zero-shot length generalization

References:
  - model_documents/aenca_v3_report.md  (concept + active inference maths)
  - model_documents/aeca_v3_design_decisions.md  (design rationale)
"""

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.checkpoint import checkpoint

from ncalm.model import NCALMModel, register_model


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AENCAv3Config:
    """All hyperparameters for the AENCA v3 model.

    Defaults give ~14.78M parameters on a 10K-vocab BPE tokenizer.
    All fields can be overridden via --model-config JSON in train.py.
    """
    # Vocabulary
    vocab_size: int = 10_000

    # Token embedding
    d_emb: int = 384

    # Substrate topology (2D Moore neighbourhood)
    Nx: int = 8          # grid rows
    Ny: int = 8          # grid columns  (m = Nx*Ny = 64 cells total)

    # State channel decomposition (D = d_h + d_b + d_c)
    d_h: int = 200       # hidden / working memory channels
    d_b: int = 80        # belief channels (readable by readout head)
    d_c: int = 40        # control channels (scale, gates, auxiliary)

    # Position encoding output dimension
    d_pos: int = 32      # kernel receives d_pos features per cell (per encoder use)

    # Local kernel (f_θ) hidden width
    d_kernel_h: int = 1280

    # Multi-scale sensor movement — Fix C: 18 scales covers 250K in T=8 steps
    # Step sizes: 2^0=1, 2^1=2, ..., 2^17=131072
    # Max reach: T × 2^(L_scales-1) = 8 × 131072 = 1,048,576 >> 250K
    L_scales: int = 18

    # Sensing / relaxation schedule
    T_steps: int = 8     # sensing steps per outer position
    R_steps: int = 2     # relaxation steps after each sensing step (training)
    R_max_inference: int = 20  # max relaxation steps at inference (adaptive)

    # Adaptive stopping thresholds (inference only)
    tau_H_stop: float = 0.5   # entropy threshold (nats) for early stop
    eps_E_stop: float = 1e-4  # energy-delta threshold for early stop

    # EFE hyperparameters
    tau_efe: float = 1.0      # EFE temperature (softmax denominator)
    beta_efe: float = 0.5     # epistemic value weight
    kappa_cost: float = 0.1   # movement cost weight

    # Loss weights
    lambda_obs: float = 0.1   # observation model (g_θ) loss weight
    lambda_pg: float = 0.01   # REINFORCE policy gradient loss weight
    lambda_h: float = 1e-4    # hidden state regulariser in local energy
    lambda_c: float = 1e-4    # consensus term in local energy

    # Anytime supervision (1-indexed sensing step numbers, weights sum to 1)
    anytime_steps: tuple = (2, 4, 6, 8)
    anytime_weights: tuple = (0.05, 0.10, 0.25, 0.60)

    # EMA decay for REINFORCE baseline
    pg_baseline_ema: float = 0.99

    # Fix B: TBPTT chunk size — breaks gradient graph every K positions
    # BPTT depth = K × (T + R) = 32 × 10 = 320 recurrent steps (vs 2550 full)
    tbptt_chunk_size: int = 32

    # Observation model warmup (ramps lambda_obs 0 → lambda_obs over N steps)
    obs_warmup_steps: int = 100

    @property
    def D(self) -> int:
        """Total state dimensionality per cell."""
        return self.d_h + self.d_b + self.d_c

    @property
    def m(self) -> int:
        """Total number of cells in the substrate."""
        return self.Nx * self.Ny


# ─────────────────────────────────────────────────────────────────────────────
# Utility: gather Moore neighbourhood
# ─────────────────────────────────────────────────────────────────────────────

def gather_neighborhood(substrate: torch.Tensor) -> torch.Tensor:
    """Gather the Moore neighbourhood (8 neighbours + self) for every cell.

    Document-faithful implementation: the perception operator is
        P_u(x^(t)) = {x_v^(t) : v ∈ Nbr(u)}
    represented as a concatenated vector in fixed canonical order:
        [NW, N, NE, W, Self, E, SW, S, SE]

    Boundary cells are zero-padded (standard NCA boundary condition).
    This is NOT a Sobel convolution; it is a direct neighbour concatenation.

    Args:
        substrate: (B, Nx, Ny, D) — current cell states.

    Returns:
        P: (B, Nx, Ny, 9*D) — concatenated 9-cell neighbourhood per cell.
    """
    B, Nx, Ny, D = substrate.shape

    # Move channel dim to position 1 for F.pad convenience: (B, D, Nx, Ny)
    x = substrate.permute(0, 3, 1, 2).contiguous()

    # Zero-pad by 1 on every spatial side → (B, D, Nx+2, Ny+2)
    x_pad = F.pad(x, (1, 1, 1, 1), mode='constant', value=0.0)

    # Collect the 9 neighbour slices in clockwise order from NW:
    #   (row_offset, col_offset) in the padded tensor:
    #   (0,0)=NW  (0,1)=N  (0,2)=NE
    #   (1,0)=W   (1,1)=self (1,2)=E
    #   (2,0)=SW  (2,1)=S  (2,2)=SE
    parts = []
    for dr in range(3):
        for dc in range(3):
            nb = x_pad[:, :, dr:dr + Nx, dc:dc + Ny]   # (B, D, Nx, Ny)
            parts.append(nb.permute(0, 2, 3, 1))         # (B, Nx, Ny, D)

    return torch.cat(parts, dim=-1)   # (B, Nx, Ny, 9*D)


# ─────────────────────────────────────────────────────────────────────────────
# Module: Relative position encoder (Fix D — replaces PointerPositionEncoder)
# ─────────────────────────────────────────────────────────────────────────────

class RelativePositionEncoder(nn.Module):
    """Relative logarithmic position encoding.

    Fix D: Replaces absolute PointerPositionEncoder to achieve zero-shot
    length generalization and NCA translation invariance.

    Encodes relative distance Δp = pos − pointer_pos using:
    - 1 normalized log feature: d_log = ln(Δp+1) / ln(L_max+1) ∈ [0,1]
    - n_sin = d_pos // 2 = 16 sin features (RoPE-style frequencies)
    - n_cos = d_pos - 1 - n_sin = 15 cos features

    Total output: 1 + 16 + 15 = 32 = d_pos features.

    Zero learned parameters — all sinusoidal frequencies are fixed buffers.
    Dual use: pass pointer relative distances OR scale step sizes (2^ℓ).

    Length generalization proof:
        π_rel(Δp) depends only on Δp, NOT on total sequence length n.
        Sensor 10 tokens back: same encoding for n=256 and n=250,000. ✓
        Sensor at Δp=100,000 (250K only): continuous sinusoidal extrapolation. ✓

    Weber-Fechner dual use for scales:
        step_size = 2^ℓ → d_log(2^ℓ) ≈ ℓ/L_scales (normalized scale level)
        Naturally encodes logarithmic scale structure.
    """

    def __init__(self, d_pos: int, L_max: int = 262144):
        super().__init__()
        self.d_pos = d_pos
        self._log_L_max_p1 = math.log(L_max + 1)

        # n_sin = d_pos // 2 = 16, n_cos = d_pos - 1 - n_sin = 15 → total 32
        n_sin = d_pos // 2
        n_cos = d_pos - 1 - n_sin
        # Frequencies: 10000^{2k/d_pos} for k=0..n_sin-1 and k=0..n_cos-1
        sin_freqs = 10000.0 ** (torch.arange(n_sin).float() * 2.0 / d_pos)
        cos_freqs = 10000.0 ** (torch.arange(n_cos).float() * 2.0 / d_pos)
        self.register_buffer('sin_freqs', sin_freqs)   # (n_sin,) = (16,)
        self.register_buffer('cos_freqs', cos_freqs)   # (n_cos,) = (15,)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """Encode relative distance.

        Args:
            delta: (...) long or int tensor — relative distance ≥ 0.
                   For pointer position: Δp = pos − pointer_pos
                   For scale level:      step_size = 2^{scale_level}

        Returns:
            (..., d_pos) float — [d_log, sin_0, cos_0, ..., sin_15, cos_14]
        """
        d = delta.float().clamp(min=0)                          # (...,)
        d_log = torch.log1p(d) / self._log_L_max_p1            # (...,) ∈ [0,1]
        sins = torch.sin(d.unsqueeze(-1) / self.sin_freqs)      # (..., n_sin)
        coss = torch.cos(d.unsqueeze(-1) / self.cos_freqs)      # (..., n_cos)
        return torch.cat([d_log.unsqueeze(-1), sins, coss], dim=-1)  # (..., d_pos)


# ─────────────────────────────────────────────────────────────────────────────
# Module: Substrate initialiser
# ─────────────────────────────────────────────────────────────────────────────

class SubstrateInitializer(nn.Module):
    """Generates the initial substrate state from learnable per-cell embeddings.

    Each cell u ∈ {0,...,m-1} has a unique learnable coordinate embedding
    coord_u ∈ R^{d_coord}.  A shared MLP maps coord_u → s_u^{(0)} ∈ R^D.

    This breaks the cell symmetry (each cell has a unique initial state)
    while preserving NCA parameter sharing (f_θ itself is still shared).
    """

    D_COORD = 32  # coordinate embedding dimension (fixed)

    def __init__(self, cfg: AENCAv3Config):
        super().__init__()
        self.Nx = cfg.Nx
        self.Ny = cfg.Ny
        m = cfg.m
        D = cfg.D

        # Learnable coordinate embeddings — one per cell
        self.coord_embed = nn.Embedding(m, self.D_COORD)

        # MLP: coord → initial cell state
        self.init_mlp = nn.Sequential(
            nn.Linear(self.D_COORD, 128),
            nn.GELU(),
            nn.Linear(128, D),
        )

    def forward(self, B: int, device: torch.device) -> torch.Tensor:
        """Return initial substrate of shape (B, Nx, Ny, D)."""
        m = self.Nx * self.Ny
        ids = torch.arange(m, device=device)     # (m,)
        coords = self.coord_embed(ids)            # (m, D_COORD)
        states = self.init_mlp(coords)            # (m, D)
        states = states.view(self.Nx, self.Ny, -1)  # (Nx, Ny, D)
        # Expand to batch without copying: (1, Nx, Ny, D) → (B, Nx, Ny, D)
        return states.unsqueeze(0).expand(B, -1, -1, -1).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Module: Local kernel f_θ
# ─────────────────────────────────────────────────────────────────────────────

class LocalKernel(nn.Module):
    """Shared local update kernel f_θ (the NCA rule).

    Single MLP applied identically to every cell at every sensing step.
    All parameter sharing is enforced by having exactly one set of weights.

    Input (per cell u) with Fix D (scale_enc added):
        concat(P_u ∈ R^{9D}, obs_u ∈ R^{d_emb}, pos_enc_u ∈ R^{d_pos}, scale_enc_u ∈ R^{d_pos})
        = 9*320 + 384 + 32 + 32 = 3328 floats   (with defaults)

    Architecture: 3 hidden layers with LayerNorm + GELU, then 6 output heads.

    Output heads (all computed from the shared 1280-dim representation):
        Δh         ∈ R^{d_h}       hidden state increment
        Δb         ∈ R^{d_b}       belief increment
        Δc         ∈ R^{d_c}       control increment
        gate       ∈ R^{D}         sigmoid gate (FM1 stabiliser)
        α_logits   ∈ R^3           direction action logits {-1, 0, +1}
        β_logits   ∈ R^3           scale-change action logits {-1, 0, +1}

    State update (gated residual — design_decisions.md Section 5, FM1):
        Δs = concat([Δh, Δb, Δc])         ∈ R^D
        g  = sigmoid(gate)                 ∈ (0,1)^D
        s_u^{t+1} = s_u^t + g ⊙ Δs       (bounded residual update)
    """

    def __init__(self, cfg: AENCAv3Config):
        super().__init__()
        D = cfg.D
        # Fix D: kernel now receives pos_enc AND scale_enc (2 × d_pos)
        d_in = 9 * D + cfg.d_emb + 2 * cfg.d_pos   # 3328 with defaults
        h = cfg.d_kernel_h

        # 3-layer shared trunk
        self.layers = nn.Sequential(
            nn.Linear(d_in, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
        )

        # Output heads (separate linear projections from shared representation)
        self.head_delta_h = nn.Linear(h, cfg.d_h)
        self.head_delta_b = nn.Linear(h, cfg.d_b)
        self.head_delta_c = nn.Linear(h, cfg.d_c)
        self.head_gate    = nn.Linear(h, D)
        self.head_alpha   = nn.Linear(h, 3)   # direction: {-1, 0, +1}
        self.head_beta    = nn.Linear(h, 3)   # scale change: {-1, 0, +1}

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (..., d_in) — batched kernel input.

        Returns:
            delta_s:      (..., D)  — raw state increment (before gating)
            gate_logits:  (..., D)  — pre-sigmoid gate values
            alpha_logits: (..., 3)  — direction action logits
            beta_logits:  (..., 3)  — scale-change action logits
        """
        h = self.layers(x)
        delta_h = self.head_delta_h(h)
        delta_b = self.head_delta_b(h)
        delta_c = self.head_delta_c(h)
        gate    = self.head_gate(h)
        alpha   = self.head_alpha(h)
        beta    = self.head_beta(h)
        delta_s = torch.cat([delta_h, delta_b, delta_c], dim=-1)
        return delta_s, gate, alpha, beta


# ─────────────────────────────────────────────────────────────────────────────
# Module: Observation model g_θ
# ─────────────────────────────────────────────────────────────────────────────

class ObservationModel(nn.Module):
    """Observation model g_θ used for EFE computation.

    Predicts the next observed token embedding given:
      - Current neighbourhood P_u ∈ R^{9D} (same format as f_θ perception)
      - Position encoding of the candidate next pointer: pos_enc ∈ R^{d_pos}
      - Scale encoding of the candidate next scale: scale_enc ∈ R^{d_pos}

    Fix D: Now takes both pos_enc and scale_enc (dual relative encoding).
    Input: concat(P_u, pos_enc, scale_enc) ∈ R^{9D + 2*d_pos} = 2944 floats (defaults)
    Output: ê_u(a) ∈ R^{d_emb} = 384 floats

    Note: g_θ is a SEPARATE network from f_θ (different weights, different
    purpose: world prediction vs state update).  Both share the same
    neighbourhood input format, enabling reuse of the computed P_u.

    Training signal (report Section 5.3):
        L_obs = ||g_θ(P_u^t, π_rel(Δp'), scale_enc') - e_{p_u^{t+1}}||_2^2
    where e_{p_u^{t+1}} is the actual tape embedding at the cell's new pointer.

    DIRAC DELTA APPROXIMATION: The full EFE requires E_{q(o'|a)}[...], but we
    approximate q(o'|a) ≈ δ(o' - ê_u(a)) (Dirac delta at g_θ's point estimate).
    This trades epistemic uncertainty for computational tractability and is a
    standard mean-field approximation in active inference implementations.
    """

    def __init__(self, cfg: AENCAv3Config):
        super().__init__()
        D = cfg.D
        # Fix D: obs model receives pos_enc AND scale_enc (2 × d_pos)
        d_in = 9 * D + 2 * cfg.d_pos   # 2944 with defaults

        self.net = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.GELU(),
            nn.Linear(512, cfg.d_emb),
        )

    def forward(self, P: torch.Tensor, pos_enc: torch.Tensor, scale_enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            P:          (..., 9*D)   — neighbourhood states
            pos_enc:    (..., d_pos) — relative position encoding of candidate next pointer
            scale_enc:  (..., d_pos) — scale encoding of candidate next scale (2^ℓ)

        Returns:
            emb_pred: (..., d_emb) — predicted token embedding
        """
        x = torch.cat([P, pos_enc, scale_enc], dim=-1)
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Module: Readout head
# ─────────────────────────────────────────────────────────────────────────────

class ReadoutHead(nn.Module):
    """Distributed readout head W_out.

    Implements the distributed vote from report Section 3.8:
        z_u^t = W_out b_u^t          (per-cell logits from belief channels)
        z^t   = (1/m) Σ_u z_u^t     (mean vote across all cells)
        q^t(y) = softmax(z^t)        (next-token distribution)

    No attention, no pooling, no cross-cell interaction — pure mean vote.
    Note: belief channels are normalized by AENCAv3Model.belief_norm BEFORE
    being passed to W_out (enhancing stability over long training runs).
    """

    def __init__(self, cfg: AENCAv3Config):
        super().__init__()
        self.d_h = cfg.d_h
        self.d_b = cfg.d_b
        # W_out ∈ R^{vocab_size × d_b}
        self.W_out = nn.Linear(cfg.d_b, cfg.vocab_size)

    def forward(self, substrate: torch.Tensor) -> torch.Tensor:
        """Compute mean-vote logits over all cells (without belief_norm).

        Note: Callers should apply belief_norm to substrate before calling
        this, or use W_out directly with normed beliefs.

        Args:
            substrate: (B, Nx, Ny, D) — current substrate state.

        Returns:
            logits: (B, vocab_size) — ensemble logits (mean over cells).
        """
        # Slice belief channels: s[d_h : d_h + d_b]
        belief = substrate[..., self.d_h : self.d_h + self.d_b]  # (B, Nx, Ny, d_b)
        # Per-cell logits
        cell_logits = self.W_out(belief)                           # (B, Nx, Ny, vocab_size)
        # Distributed mean vote (report Section 3.8 — exact mean, not attention)
        return cell_logits.mean(dim=(1, 2))                        # (B, vocab_size)

    def cell_logits(self, substrate: torch.Tensor) -> torch.Tensor:
        """Return per-cell logits (B, Nx, Ny, vocab_size) for EFE computation."""
        belief = substrate[..., self.d_h : self.d_h + self.d_b]
        return self.W_out(belief)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

@register_model("aenca_v3")
class AENCAv3Model(NCALMModel):
    """AENCA v3 — Active Embedding Neural Cellular Automata v3.

    Key design decisions:
      - 2D 8×8 substrate: O(√m) communication diameter vs O(m) for 1D
      - Raw neighbour concatenation (NOT Sobel): document-faithful perception
      - Full one-step EFE with dual training/inference modes:
          Training: risk = -log q̃(y*)    (CE with known target)
          Inference: risk = KL(q_curr || q̃) (no y* required)
      - Action masking: EFE=+∞ for out-of-bounds actions (REINFORCE variance)
      - Adaptive relaxation stopping at inference (entropy / energy-delta)
      - TBPTT K=32 + gradient checkpointing at position boundaries

    v3.1 fixes:
      - Fix A: Gate bias=-3.9 (variance-preserving, σ≈0.02) + layer scaling 1/√6
      - Fix B: TBPTT K=32 — gradient depth 2550 → 320 recurrent steps
      - Fix C: L_scales=18 (reaches 250K) + logarithmic movement cost
      - Fix D: RelativePositionEncoder (zero-shot length generalization)
      - Enhancements: belief_norm, substrate clamping, obs warmup, EFE τ annealing
    """

    def __init__(self, config: dict):
        """
        Args:
            config: dict from training profile + optional --model-config.
                    AENCA-specific keys (d_emb, Nx, Ny, etc.) override defaults.
        """
        super().__init__()

        # Build config from dict, falling back to dataclass defaults
        cfg = AENCAv3Config()
        for key in vars(cfg):
            if key in config:
                setattr(cfg, key, config[key])
        self.cfg = cfg

        # Derived constants (frequently used)
        self._D = cfg.D          # 320
        self._m = cfg.m          # 64

        # ── Components ───────────────────────────────────────────────────────
        self.token_embed    = nn.Embedding(cfg.vocab_size, cfg.d_emb)
        # Fix D: RelativePositionEncoder (L_max = 2^L_scales)
        self.pos_encoder    = RelativePositionEncoder(cfg.d_pos, L_max=2 ** cfg.L_scales)
        self.substrate_init = SubstrateInitializer(cfg)
        self.kernel         = LocalKernel(cfg)
        self.obs_model      = ObservationModel(cfg)
        self.readout        = ReadoutHead(cfg)
        # Stability enhancement: belief channel LayerNorm before readout
        self.belief_norm    = nn.LayerNorm(cfg.d_b)

        # ── Joint action table ────────────────────────────────────────────────
        # action_idx = alpha_idx * 3 + beta_idx,  alpha/beta_idx ∈ {0,1,2} → value ∈ {-1,0,+1}
        # Layout (9 actions, row = alpha ∈ {-1,0,+1}, col = beta ∈ {-1,0,+1}):
        #   idx: 0  1  2  3  4  5  6  7  8
        #   α:  -1 -1 -1  0  0  0 +1 +1 +1
        #   β:  -1  0 +1 -1  0 +1 -1  0 +1
        alphas = torch.tensor([-1, -1, -1,  0,  0,  0,  1,  1,  1], dtype=torch.long)
        betas  = torch.tensor([-1,  0,  1, -1,  0,  1, -1,  0,  1], dtype=torch.long)
        self.register_buffer('action_alphas', alphas)
        self.register_buffer('action_betas',  betas)

        # ── REINFORCE baseline (EMA of per-position final energy) ─────────────
        self.register_buffer('pg_baseline', torch.zeros(1))

        # ── Anytime step index set (0-indexed sensing step t) ─────────────────
        # anytime_steps=(2,4,6,8) means log energy after sensing step t=1,3,5,7
        self._anytime_t_set = frozenset(s - 1 for s in cfg.anytime_steps)
        self._n_anytime = len(cfg.anytime_steps)

        # ── Scheduled hyperparameters (updated via on_step_start) ─────────────
        self._current_step = 0
        self._lambda_obs_eff = 0.0          # obs warmup: starts at 0, ramps to lambda_obs
        self._tau_efe_eff = 2.0 * cfg.tau_efe  # EFE temp annealing: starts 2× target
        self._training_warmup_steps = config.get('warmup_steps', 500)

        # ── Weight initialisation ─────────────────────────────────────────────
        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        """Xavier uniform for linear layers; variance-preserving gate init (Fix A)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Fix A: Variance-preserving gate bias initialization.
        # With N = (L-1) × (T + R) = 255 × 10 = 2550 sequential kernel steps,
        # σ(b_gate) = 1/√N = 1/√2550 ≈ 0.0198 gives bounded state variance.
        # b_gate = -3.9 achieves σ(-3.9) ≈ 0.0198. Gate weight stays at 0.
        nn.init.zeros_(self.kernel.head_gate.weight)
        nn.init.constant_(self.kernel.head_gate.bias, -3.9)

        # Fix A complement: Layer scaling 1/√(2×3) ≈ 0.408 on output heads
        # (GPT-2 technique — reduces initial Δs magnitude to complement gate init)
        _layer_scale = 1.0 / math.sqrt(2 * 3)
        for head in [self.kernel.head_delta_h, self.kernel.head_delta_b, self.kernel.head_delta_c]:
            head.weight.data.mul_(_layer_scale)

    # ── Scheduled hyperparameter updates ─────────────────────────────────────

    def on_step_start(self, step: int, max_steps: int) -> None:
        """Update scheduled hyperparameters before each training step.

        Called by Trainer.train_step() before the forward pass.
        - Observation model warmup: λ_obs_eff ramps 0 → lambda_obs over obs_warmup_steps
        - EFE temperature annealing: τ_eff ramps 2×tau_efe → tau_efe over warmup_steps
        """
        self._current_step = step

        # Observation model warmup (Fix §7.1)
        obs_ramp = min(1.0, step / max(1, self.cfg.obs_warmup_steps))
        self._lambda_obs_eff = obs_ramp * self.cfg.lambda_obs

        # EFE temperature annealing: explore early, exploit later (Fix §7.2)
        warmup = self._training_warmup_steps
        t_ann = min(1.0, step / max(1, warmup))
        self._tau_efe_eff = (
            2.0 * self.cfg.tau_efe * (1.0 - t_ann)
            + self.cfg.tau_efe * t_ann
        )

    # ── Utility: tape embedding gather ────────────────────────────────────────

    def _gather_tape(
        self,
        emb: torch.Tensor,      # (B, seq_len, d_emb)
        pointers: torch.Tensor, # (B, Nx, Ny) long
        max_pos: int,           # upper bound for safety clamp
    ) -> torch.Tensor:
        """Gather token embeddings at each cell's pointer position.

        Returns: (B, Nx, Ny, d_emb)
        """
        B, Nx, Ny = pointers.shape
        d_emb = emb.shape[-1]

        ptrs = pointers.clamp(0, max_pos).reshape(B, Nx * Ny)  # (B, m)
        # Gather along seq_len dimension: emb[b, ptrs[b, i], :]
        idx = ptrs.unsqueeze(-1).expand(-1, -1, d_emb)          # (B, m, d_emb)
        gathered = emb.gather(1, idx)                            # (B, m, d_emb)
        return gathered.reshape(B, Nx, Ny, d_emb)

    # ── Utility: pointer initialisation ───────────────────────────────────────

    def _init_pointers(self, B: int, device: torch.device) -> torch.Tensor:
        """Initialise all sensor pointers to position 0.

        At the first outer step (pos=0), only token 0 is visible, so all
        sensors start at position 0.  Thereafter, sensors persist and move
        via their action policy across the outer loop.

        Returns: (B, Nx, Ny) long — all zeros.
        """
        return torch.zeros(B, self.cfg.Nx, self.cfg.Ny, device=device, dtype=torch.long)

    # ── Local energy ──────────────────────────────────────────────────────────

    def _local_energy(
        self,
        substrate: torch.Tensor,          # (B, Nx, Ny, D)
        targets: Optional[torch.Tensor],   # (B,) long — None at inference
    ) -> tuple:
        """Compute local energy E_t and pure CE for metrics.

        Energy formula (report Section 6.3):
            E_t = (1/m) Σ_u CE_u
                + λ_h × (1/m) Σ_u ||h_u||_2^2
                + λ_c × (1/m) Σ_u Σ_{v∈N(u)} ||h_u - h_v||_2^2

        Returns:
            (energy, ce_mean)  — both scalars.
            ce_mean is the pure ensemble CE (from mean logits) for PPL tracking.
        """
        cfg = self.cfg
        B, Nx, Ny, D = substrate.shape

        h = substrate[..., :cfg.d_h]                           # (B, Nx, Ny, d_h)
        b = substrate[..., cfg.d_h : cfg.d_h + cfg.d_b]       # (B, Nx, Ny, d_b)

        # Stability enhancement: apply belief_norm before readout
        b_normed = self.belief_norm(b)                         # (B, Nx, Ny, d_b)

        # ── CE term ───────────────────────────────────────────────────────────
        if targets is not None:
            # Per-cell CE: all cells predict the same next token y*
            logits_per_cell = self.readout.W_out(b_normed)     # (B, Nx, Ny, V)
            tgt_exp = targets.view(B, 1, 1).expand(-1, Nx, Ny)
            ce_per_cell = F.cross_entropy(
                logits_per_cell.reshape(B * Nx * Ny, cfg.vocab_size),
                tgt_exp.reshape(B * Nx * Ny),
                reduction='mean',
            )   # scalar (0-dim)
            # Ensemble CE from mean logits (for PPL tracking)
            ensemble_logits = logits_per_cell.mean(dim=(1, 2))   # (B, V)
            ce_ensemble = F.cross_entropy(ensemble_logits, targets, reduction='mean')
        else:
            # At inference, no target — energy is regularisation only
            zero = substrate.new_zeros([])      # 0-dim scalar tensor
            ce_per_cell = zero
            ce_ensemble = zero

        # ── Hidden state regulariser: (1/m) Σ_u ||h_u||^2 ────────────────────
        h_reg = (h * h).mean()   # 0-dim scalar: mean over B, Nx, Ny, d_h

        # ── Consensus term: (1/m) Σ_u Σ_{v∈N(u)} ||h_u - h_v||^2 ─────────────
        # Use zero-padded Moore neighbourhood for consistency with gather_neighborhood
        h_ch = h.permute(0, 3, 1, 2)                 # (B, d_h, Nx, Ny)
        h_pad = F.pad(h_ch, (1, 1, 1, 1))             # (B, d_h, Nx+2, Ny+2)
        consensus_parts: list[torch.Tensor] = []
        for dr in range(3):
            for dc in range(3):
                if dr == 1 and dc == 1:
                    continue  # skip self
                h_nb = h_pad[:, :, dr:dr + Nx, dc:dc + Ny].permute(0, 2, 3, 1)
                diff = h - h_nb                        # (B, Nx, Ny, d_h)
                consensus_parts.append((diff * diff).mean())   # 0-dim scalar
        consensus = torch.stack(consensus_parts).sum()   # 0-dim scalar

        energy = (
            ce_per_cell
            + cfg.lambda_h * h_reg
            + cfg.lambda_c * consensus
        )   # 0-dim scalar
        return energy, ce_ensemble

    # ── Expected Free Energy ───────────────────────────────────────────────────

    def _compute_efe(
        self,
        P: torch.Tensor,                          # (B, Nx, Ny, 9*D)
        substrate: torch.Tensor,                  # (B, Nx, Ny, D)
        pointers: torch.Tensor,                   # (B, Nx, Ny) long
        scale_lvl: torch.Tensor,                  # (B, Nx, Ny) long
        pos: int,                                 # causal upper bound (0-indexed)
        targets: Optional[torch.Tensor],          # (B,) long — None at inference
        training_mode: bool,
    ) -> tuple:
        """Compute one-step EFE for all 9 joint actions (batched).

        EFE formula (report Section 7.3):
            G_u(a) = R_u(a) - β_efe × I_u(a) + κ_cost × c(a)

        Where:
            R_u(a) [TRAINING]:  -log q̃_u(a)(y*)          (CE risk w/ known y*)
            R_u(a) [INFERENCE]: KL(q_u^t || q̃_u(a))       (KL risk, no y* needed)
            I_u(a):  H(q_u^t) - H(q̃_u(a))                (information gain)
            c(a):    |α| × ℓ_u / (L_scales − 1)           (logarithmic cost, Fix C)

        Fix C — Weber-Fechner logarithmic cost:
            OLD (broken): c = |α| × 2^ℓ / (pos+1) → cost ≈ 512 at ℓ=17 → never explored
            NEW (fixed):  c = |α| × ℓ / (L_scales-1) ∈ [0, 1] → uniformly explorable

        Fix D — Relative position encoding:
            Candidate pointer encoded as Δp = pos − cand_ptr (relative distance)
            Candidate scale encoded as 2^{cand_scale} (step size)
            Both use the same RelativePositionEncoder (dual-use, zero learned params)

        ACTION MASKING: EFE = +∞ for out-of-bounds actions (α≠0 but clip hits bound).
        This prevents REINFORCE from assigning probability to geometrically
        null actions (FM3 mitigation, design_decisions.md Section 8).

        Args:
            P:            (B, Nx, Ny, 9D) — current neighbourhood (before update)
            substrate:    (B, Nx, Ny, D)
            pointers:     (B, Nx, Ny) long
            scale_lvl:    (B, Nx, Ny) long
            pos:          int — current sequence position (causal bound)
            targets:      (B,) long or None
            training_mode: True → use CE risk; False → use KL risk

        Returns:
            efe:         (B, Nx, Ny, 9) — EFE values (+inf for masked actions)
            next_ptrs:   (B, Nx, Ny, 9) long — clipped candidate next pointers
            next_scales: (B, Nx, Ny, 9) long — clipped candidate next scales
        """
        cfg = self.cfg
        B, Nx, Ny, D = substrate.shape
        n_actions = 9

        alphas = self.action_alphas   # (9,) long, values ∈ {-1, 0, +1}
        betas  = self.action_betas    # (9,) long

        # ── Candidate next scales and pointers (all 9 actions) ────────────────
        sl = scale_lvl.unsqueeze(-1)              # (B, Nx, Ny, 1)
        pt = pointers.unsqueeze(-1)               # (B, Nx, Ny, 1)
        step_sizes = (2 ** sl).long()             # (B, Nx, Ny, 1) step sizes

        # Candidate next scales: clip to [0, L_scales-1]
        next_scales = (sl + betas).clamp(0, cfg.L_scales - 1)  # (B, Nx, Ny, 9)

        # Candidate next pointers: clip to [0, pos]
        next_ptrs_raw = pt + alphas * step_sizes               # (B, Nx, Ny, 9) unclipped
        next_ptrs     = next_ptrs_raw.clamp(0, pos)            # (B, Nx, Ny, 9) clipped

        # ── Action mask: out-of-bounds moves with α ≠ 0 ─────────────────────
        # These actions "do nothing" due to clipping but have different log-probs
        # than the "stay" action (α=0) — injecting spurious REINFORCE variance.
        alpha_nonzero = (alphas != 0)                                 # (9,) bool
        out_of_bounds = (next_ptrs_raw < 0) | (next_ptrs_raw > pos)  # (B, Nx, Ny, 9)
        action_mask   = out_of_bounds & alpha_nonzero                  # (B, Nx, Ny, 9)

        # ── Current belief distribution q_{u,t}(y) ────────────────────────────
        b_curr = substrate[..., cfg.d_h : cfg.d_h + cfg.d_b]   # (B, Nx, Ny, d_b)
        b_curr_normed  = self.belief_norm(b_curr)               # (B, Nx, Ny, d_b)
        curr_logits    = self.readout.W_out(b_curr_normed)      # (B, Nx, Ny, V)
        curr_log_probs = F.log_softmax(curr_logits, dim=-1)      # (B, Nx, Ny, V)
        curr_probs     = curr_log_probs.exp()                    # (B, Nx, Ny, V)
        # Per-cell entropy H(q_u^t) ∈ [0, log V]
        curr_entropy = -(curr_probs * curr_log_probs).sum(-1)    # (B, Nx, Ny)

        # ── Batch over 9 actions: B_eff = B × Nx × Ny × 9 ───────────────────
        B_eff = B * Nx * Ny * n_actions

        # Fix D: Relative position encoding for candidate pointers
        # Δp = pos − cand_ptr (how far back from current outer position)
        next_ptrs_flat  = next_ptrs.reshape(B_eff)                   # (B_eff,) long
        next_delta_flat = pos - next_ptrs_flat                        # (B_eff,) long ≥ 0
        next_pos_enc_flat = self.pos_encoder(next_delta_flat)         # (B_eff, d_pos)

        # Fix D: Scale encoding for candidate scales
        # step_size = 2^{cand_scale} (encoded like a "pseudo-Δp" distance)
        next_scales_flat    = next_scales.reshape(B_eff)               # (B_eff,) long
        next_step_sizes_flat = (1 << next_scales_flat.clamp(0, cfg.L_scales - 1))  # (B_eff,) long
        next_scale_enc_flat = self.pos_encoder(next_step_sizes_flat)   # (B_eff, d_pos)

        # Neighbourhood: same P for all 9 actions (expand then flatten)
        P_exp  = P.unsqueeze(-2).expand(-1, -1, -1, n_actions, -1)  # (B,Nx,Ny,9,9D)
        P_flat = P_exp.reshape(B_eff, 9 * D)                        # (B_eff, 9D)

        # ── Step 1: predict next observation via g_θ ─────────────────────────
        # g_θ input = concat(P_u, π_rel(Δp'(a)), scale_enc(a))
        pred_emb_flat = self.obs_model(P_flat, next_pos_enc_flat, next_scale_enc_flat)  # (B_eff, d_emb)

        # ── Step 2: simulate next belief state via f_θ ────────────────────────
        # f_θ input = concat(P_u, ê_u(a), π_rel(Δp'(a)), scale_enc(a))
        kernel_in_sim = torch.cat(
            [P_flat, pred_emb_flat, next_pos_enc_flat, next_scale_enc_flat], dim=-1
        )
        delta_s_sim, gate_sim, _, _ = self.kernel(kernel_in_sim)     # (B_eff, D) each

        # Current substrate expanded to B_eff
        sub_flat     = substrate.reshape(B * Nx * Ny, D)             # (B*m, D)
        sub_flat_exp = sub_flat.unsqueeze(1).expand(-1, n_actions, -1).reshape(B_eff, D)

        # Simulated next state (gated residual)
        sim_state = sub_flat_exp + torch.sigmoid(gate_sim) * delta_s_sim  # (B_eff, D)

        # Simulated belief distribution q̃_u(a)(y) — with belief_norm
        sim_b     = sim_state[:, cfg.d_h : cfg.d_h + cfg.d_b]          # (B_eff, d_b)
        sim_b_normed  = self.belief_norm(sim_b)                          # (B_eff, d_b)
        sim_logits    = self.readout.W_out(sim_b_normed)                 # (B_eff, V)
        sim_log_probs = F.log_softmax(sim_logits, dim=-1)               # (B_eff, V)
        sim_probs     = sim_log_probs.exp()                              # (B_eff, V)

        V = cfg.vocab_size
        sim_log_probs_4d = sim_log_probs.reshape(B, Nx, Ny, n_actions, V)
        sim_probs_4d     = sim_probs.reshape(B, Nx, Ny, n_actions, V)

        # ── Risk ─────────────────────────────────────────────────────────────
        if training_mode and targets is not None:
            # TRAINING MODE: R_u(a) = -log q̃_u(a)(y*)
            # (CE under simulated next state w.r.t. known target)
            tgt_exp = targets.view(B, 1, 1, 1, 1).expand(-1, Nx, Ny, n_actions, 1)
            risk = -sim_log_probs_4d.gather(-1, tgt_exp).squeeze(-1)  # (B, Nx, Ny, 9)
        else:
            # INFERENCE MODE: R_u(a) = KL(q_curr || q̃_u(a))
            # Uses current belief as preference — no y* required.
            curr_log_probs_exp = curr_log_probs.unsqueeze(-2)  # (B, Nx, Ny, 1, V)
            curr_probs_exp     = curr_probs.unsqueeze(-2)       # (B, Nx, Ny, 1, V)
            kl = (curr_probs_exp * (curr_log_probs_exp - sim_log_probs_4d)).sum(-1)
            risk = kl.clamp(min=0.0)  # KL is always ≥ 0; numerical guard

        # ── Epistemic value: I_u(a) = H(q_curr) - H(q̃_u(a)) ─────────────────
        sim_entropy = -(sim_probs_4d * sim_log_probs_4d).sum(-1)      # (B, Nx, Ny, 9)
        info_gain   = curr_entropy.unsqueeze(-1) - sim_entropy         # (B, Nx, Ny, 9)

        # ── Fix C: Logarithmic movement cost (Weber-Fechner Law) ─────────────
        # OLD (broken): c = |α| × 2^ℓ / (pos+1) → explodes for large ℓ
        # NEW (fixed):  c = |α| × ℓ / (L_scales-1) ∈ [0, 1]
        # Current scale level ℓ_u encodes the step size we operate at.
        sl_float = sl.float()    # (B, Nx, Ny, 1) — current scale level
        cost = (alphas.float().abs() * sl_float) / max(1, cfg.L_scales - 1)  # (B, Nx, Ny, 9)

        # ── Full EFE ─────────────────────────────────────────────────────────
        efe = risk - cfg.beta_efe * info_gain + cfg.kappa_cost * cost  # (B, Nx, Ny, 9)

        # ── Apply action mask (+∞ for out-of-bounds actions) ──────────────────
        efe = efe.masked_fill(action_mask, float('inf'))

        return efe, next_ptrs, next_scales

    # ── Inner sensing loop ────────────────────────────────────────────────────

    def _process_position(
        self,
        substrate:  torch.Tensor,   # (B, Nx, Ny, D)
        pointers:   torch.Tensor,   # (B, Nx, Ny) long
        scale_lvl:  torch.Tensor,   # (B, Nx, Ny) long
        emb:        torch.Tensor,   # (B, seq_len, d_emb)
        pos_tensor: torch.Tensor,   # scalar long — current sequence position
        targets:    torch.Tensor,   # (B,) long — next token to predict
    ) -> tuple:
        """T sensing steps + R relaxation steps for one outer-loop position.

        NOTE: pos_tensor is passed as a tensor (not int) so that
        torch.utils.checkpoint can capture it as a tensor argument.

        Fix D: Uses relative position encoding Δp = pos − pointer_pos.
        Fix C: Uses scale encoding via pos_encoder(2^ℓ).
        Stability: substrate clamped to [-50, 50] after each update.

        Returns a fixed-size flat tuple of tensors:
            (new_substrate,          # (B, Nx, Ny, D)
             new_pointers,           # (B, Nx, Ny) long
             new_scale_lvl,          # (B, Nx, Ny) long
             *energies,              # len(anytime_steps) scalars
             *lp_means,              # T_steps scalars (mean log-prob over B,Nx,Ny)
             obs_loss_mean)          # scalar
        """
        cfg = self.cfg
        pos = int(pos_tensor.item())
        B, Nx, Ny, D = substrate.shape

        # Enforce causal constraint: no sensor can see beyond current position
        pointers = pointers.clamp(0, pos)

        energies  = []   # len = n_anytime (one per supervised sensing step)
        lp_means  = []   # len = T_steps (mean action log-prob per sensing step)
        obs_losses = []  # len = T_steps

        training_mode = self.training

        for t in range(cfg.T_steps):

            # ── 1. Perception: gather Moore neighbourhood ──────────────────
            P = gather_neighborhood(substrate)   # (B, Nx, Ny, 9*D)

            # ── 2. Sense: read tape embedding at current pointer ───────────
            obs = self._gather_tape(emb, pointers, pos)  # (B, Nx, Ny, d_emb)

            # ── 3. Fix D: Relative position encoding ──────────────────────
            # pos_enc encodes HOW FAR BACK we're looking (relative distance)
            delta_pos = pos - pointers                    # (B, Nx, Ny) long ≥ 0
            pos_enc   = self.pos_encoder(delta_pos)       # (B, Nx, Ny, d_pos)
            # scale_enc encodes the current step size (2^ℓ)
            step_sizes_curr = (1 << scale_lvl.clamp(0, cfg.L_scales - 1))  # (B, Nx, Ny)
            scale_enc = self.pos_encoder(step_sizes_curr) # (B, Nx, Ny, d_pos)

            # ── 4. Compute EFE for all 9 joint actions (batched) ──────────
            efe, next_ptrs, next_scales = self._compute_efe(
                P=P,
                substrate=substrate,
                pointers=pointers,
                scale_lvl=scale_lvl,
                pos=pos,
                targets=targets,
                training_mode=training_mode,
            )
            # efe: (B, Nx, Ny, 9),  next_ptrs/next_scales: (B, Nx, Ny, 9) long

            # ── 5. Policy: q_u(a) ∝ exp(-G_u(a) / τ_eff) ────────────────
            # Use _tau_efe_eff for annealed temperature (exploration → exploitation)
            # Masked actions have efe=+∞ → logit=-∞ → zero probability
            tau = self._tau_efe_eff if training_mode else self.cfg.tau_efe
            policy_logits = (-efe / tau).masked_fill(
                efe == float('inf'), float('-inf')
            )   # (B, Nx, Ny, 9)
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)

            # Sample action: training = stochastic; eval = greedy (deterministic)
            if training_mode:
                flat_actions = Categorical(
                    logits=policy_logits.reshape(-1, 9)
                ).sample()                                    # (B*Nx*Ny,) long
                actions = flat_actions.reshape(B, Nx, Ny)    # (B, Nx, Ny)
            else:
                actions = policy_logits.argmax(dim=-1)        # (B, Nx, Ny) greedy

            # Sampled action log-probabilities for REINFORCE
            sampled_lp = policy_log_probs.gather(
                -1, actions.unsqueeze(-1)
            ).squeeze(-1)  # (B, Nx, Ny)
            lp_means.append(sampled_lp.mean())   # scalar

            # ── 6. Apply kernel with ACTUAL current observation ────────────
            # Fix D: kernel receives pos_enc AND scale_enc
            kernel_in = torch.cat([P, obs, pos_enc, scale_enc], dim=-1)
            delta_s, gate, _, _ = self.kernel(
                kernel_in.reshape(B * Nx * Ny, -1)
            )
            delta_s = delta_s.reshape(B, Nx, Ny, D)
            gate    = gate.reshape(B, Nx, Ny, D)
            substrate = substrate + torch.sigmoid(gate) * delta_s
            # Stability enhancement: clamp substrate to prevent catastrophic divergence
            substrate = substrate.clamp(-50.0, 50.0)

            # ── 7. Observation model loss (teach g_θ to predict next obs) ─
            # g_θ uses P from BEFORE the kernel update (current state).
            # Target: actual tape embedding at the chosen next pointer.
            chosen_next_ptrs   = next_ptrs.gather(
                -1, actions.unsqueeze(-1)
            ).squeeze(-1)                          # (B, Nx, Ny) long
            chosen_next_scales = next_scales.gather(
                -1, actions.unsqueeze(-1)
            ).squeeze(-1)                          # (B, Nx, Ny) long

            # Fix D: relative position and scale encoding for obs model
            next_delta_pos = pos - chosen_next_ptrs    # (B, Nx, Ny) long ≥ 0
            next_pos_enc   = self.pos_encoder(next_delta_pos)    # (B, Nx, Ny, d_pos)
            next_step_sizes = (1 << chosen_next_scales.clamp(0, cfg.L_scales - 1))
            next_scale_enc  = self.pos_encoder(next_step_sizes)  # (B, Nx, Ny, d_pos)

            pred_emb = self.obs_model(P, next_pos_enc, next_scale_enc)  # (B, Nx, Ny, d_emb)
            true_emb = self._gather_tape(emb, chosen_next_ptrs, pos)
            obs_loss_t = F.mse_loss(pred_emb, true_emb.detach())
            obs_losses.append(obs_loss_t)

            # ── 8. Move sensors ───────────────────────────────────────────
            pointers  = chosen_next_ptrs
            scale_lvl = chosen_next_scales

            # ── 9. Log energy at anytime supervision steps ────────────────
            # Anytime steps are 1-indexed (e.g., 2,4,6,8 means after step 2,4,6,8)
            # In 0-indexed loop: log after t=1,3,5,7 for anytime_steps=(2,4,6,8)
            if t in self._anytime_t_set:
                E_t, _ = self._local_energy(substrate, targets)
                energies.append(E_t)

            # ── 10. Relaxation steps (FIXED R during training) ────────────
            # The kernel runs with FIXED observations from this sensing step
            # (no pointer movement; promotes convergence toward local energy min)
            for _ in range(cfg.R_steps):
                P_r = gather_neighborhood(substrate)
                # Fix D: recompute relative encodings for current pointers/scales
                delta_pos_r   = pos - pointers                              # (B, Nx, Ny)
                pos_enc_r     = self.pos_encoder(delta_pos_r)               # (B, Nx, Ny, d_pos)
                step_sizes_r  = (1 << scale_lvl.clamp(0, cfg.L_scales - 1))
                scale_enc_r   = self.pos_encoder(step_sizes_r)              # (B, Nx, Ny, d_pos)
                kin_r = torch.cat(
                    [P_r, obs.detach(), pos_enc_r.detach(), scale_enc_r.detach()], dim=-1
                )
                ds_r, g_r, _, _ = self.kernel(kin_r.reshape(B * Nx * Ny, -1))
                ds_r = ds_r.reshape(B, Nx, Ny, D)
                g_r  = g_r.reshape(B, Nx, Ny, D)
                substrate = substrate + torch.sigmoid(g_r) * ds_r
                # Stability: clamp after relaxation steps too
                substrate = substrate.clamp(-50.0, 50.0)

        # ── Pack return values into flat tuple ────────────────────────────────
        obs_loss_mean = torch.stack(obs_losses).mean() if obs_losses else \
            torch.zeros(1, device=substrate.device).squeeze()

        return (
            substrate,
            pointers,
            scale_lvl,
            *energies,      # exactly self._n_anytime scalars
            *lp_means,      # exactly T_steps scalars
            obs_loss_mean,  # 1 scalar
        )

    # ── Training forward pass (outer loop) with TBPTT K=32 ───────────────────

    def forward_train(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """TBPTT outer loop over all sequence positions.

        Fix B: Truncated Backpropagation Through Time (K=32 chunk size).
        Substrate is detached at chunk boundaries, breaking gradient flow
        between chunks. Sensor state (pointers, scale_lvl) is integer — no
        gradient — and flows continuously across ALL chunks, enabling
        long-range retrieval via the active sensing policy.

        Why TBPTT is valid with Active Inference:
          - Substrate s is the Markov Blanket: local temporal consistency
            (K positions) is sufficient for training transition dynamics.
          - Long-range recall is delegated to the EFE policy (active perception).
            If token 0 is needed at position 200, the EFE policy directs a sensor
            back and the REINFORCE gradient flows locally without BPTT depth.
          - Total BPTT depth: K × (T+R) = 32 × 10 = 320 steps (vs 2550 full).

        Args:
            batch: (B, L) long — token IDs (B × L prediction targets per step)

        Returns:
            loss: scalar — total training loss (task + obs + policy gradient)
            info: dict with at minimum 'ce' (float, for PPL tracking)
        """
        B, L = batch.shape
        if L < 2:
            raise ValueError(f"Sequence length must be ≥ 2, got {L}")

        device = batch.device
        cfg = self.cfg

        # Precompute full tape embeddings (reused across all outer positions)
        emb = self.token_embed(batch)   # (B, L, d_emb)

        # Initialise substrate and sensor state
        substrate = self.substrate_init(B, device)                          # (B, Nx, Ny, D)
        pointers  = self._init_pointers(B, device)                          # (B, Nx, Ny) long
        scale_lvl = torch.zeros(B, cfg.Nx, cfg.Ny, device=device, dtype=torch.long)

        n_pos     = L - 1   # number of prediction positions
        n_anytime = self._n_anytime
        T         = cfg.T_steps
        K         = cfg.tbptt_chunk_size   # 32

        all_chunk_losses = []   # one loss tensor per TBPTT chunk
        all_ce_ens       = []   # float values for PPL tracking (all positions)
        last_info        = {}

        for chunk_start in range(0, n_pos, K):
            chunk_end = min(chunk_start + K, n_pos)

            # ── TBPTT boundary: detach substrate ──────────────────────────
            # Breaks gradient graph between chunks. Sensors (pointers, scale_lvl)
            # are integer tensors — carry state continuously, no gradient.
            substrate = substrate.detach()

            chunk_energies   = []
            chunk_lp_means   = []
            chunk_obs_losses = []
            chunk_ce_ens     = []

            for pos in range(chunk_start, chunk_end):
                targets    = batch[:, pos + 1]                   # (B,) next token
                pos_tensor = torch.tensor(pos, device=device)

                # Gradient-checkpointed inner loop within each TBPTT chunk
                result = checkpoint(
                    self._process_position,
                    substrate, pointers, scale_lvl, emb, pos_tensor, targets,
                    use_reentrant=False,
                )

                # Unpack fixed-structure return tuple
                substrate  = result[0]
                pointers   = result[1]
                scale_lvl  = result[2]
                energies   = result[3 : 3 + n_anytime]
                lp_means   = result[3 + n_anytime : 3 + n_anytime + T]
                obs_loss   = result[3 + n_anytime + T]

                chunk_energies.extend(energies)
                chunk_lp_means.extend(lp_means)
                chunk_obs_losses.append(obs_loss)

                # Ensemble CE for PPL tracking (no gradient needed)
                with torch.no_grad():
                    _, ce_ens = self._local_energy(substrate, targets)
                    chunk_ce_ens.append(ce_ens.item())

            # ── Compute chunk loss (independent autograd graph per chunk) ─
            chunk_n_pos = chunk_end - chunk_start
            chunk_loss, chunk_info = self._compute_total_loss(
                chunk_energies, chunk_lp_means, chunk_obs_losses,
                chunk_ce_ens, chunk_n_pos
            )
            all_chunk_losses.append(chunk_loss)
            all_ce_ens.extend(chunk_ce_ens)
            last_info = chunk_info

        # Sum independent chunk losses.
        # backward() traverses all K independent computation graphs simultaneously,
        # accumulating gradients additively.
        total_loss = torch.stack(all_chunk_losses).mean()

        info = {
            "ce":        sum(all_ce_ens) / max(1, len(all_ce_ens)),
            "task_loss": last_info.get("task_loss", 0.0),
            "obs_loss":  last_info.get("obs_loss", 0.0),
            "pg_loss":   last_info.get("pg_loss", 0.0),
        }
        return total_loss, info

    def _compute_total_loss(
        self,
        all_energies:   list,   # n_pos × n_anytime scalars
        all_lp_means:   list,   # n_pos × T scalars
        all_obs_losses: list,   # n_pos scalars
        all_ce_ens:     list,   # n_pos floats (plain Python, for info dict)
        n_pos:          int,
    ) -> tuple:
        """Compute unified training objective (report Section 8.1):

            J = E[  Σ_{n,t} w_t E_t^{(n)}      (anytime task loss, weighted)
                  + λ_obs_eff × L_obs           (observation model loss, warmed up)
                  + λ_pg  × L_pg                (REINFORCE policy gradient)
                ]

        Returns: (total_loss, info_dict)
        """
        cfg = self.cfg
        T = cfg.T_steps
        n_anytime = self._n_anytime
        weights = torch.tensor(
            list(cfg.anytime_weights),
            device=all_energies[0].device,
            dtype=torch.float,
        )

        # ── Anytime task loss ─────────────────────────────────────────────────
        # Reshape energies to (n_pos, n_anytime) then weight by anytime_weights
        energy_stack = torch.stack(all_energies)           # (n_pos × n_anytime,)
        energy_2d    = energy_stack.reshape(n_pos, n_anytime)   # (n_pos, n_anytime)
        weighted_pos = (energy_2d * weights.unsqueeze(0)).sum(-1)  # (n_pos,)
        task_loss    = weighted_pos.mean()

        # ── Observation model loss ────────────────────────────────────────────
        obs_loss = torch.stack(all_obs_losses).mean()

        # ── REINFORCE policy gradient loss ────────────────────────────────────
        # Reward signal per position: E_final = energy at last anytime step
        # (detached — it is the reward, not part of the differentiable path here)
        E_finals = energy_2d[:, -1].detach()   # (n_pos,)

        # Update EMA baseline (online, during training only)
        if self.training:
            batch_mean = E_finals.mean()
            self.pg_baseline.copy_(
                cfg.pg_baseline_ema * self.pg_baseline
                + (1 - cfg.pg_baseline_ema) * batch_mean
            )

        # Advantages: E_final - baseline (scalar per position)
        advantages = E_finals - self.pg_baseline.item()   # (n_pos,)

        # L_pg = -Σ_{pos,t} advantage_{pos} × mean_{B,u}[log π_θ(a_{B,u}^{pos,t})]
        # all_lp_means is laid out as [pos0_t0, pos0_t1, ..., pos0_tT, pos1_t0, ...]
        lp_stack = torch.stack(all_lp_means).reshape(n_pos, T)  # (n_pos, T)
        pg_terms = advantages.unsqueeze(-1) * lp_stack           # (n_pos, T)
        pg_loss  = -pg_terms.mean()    # normalised over positions and sensing steps

        # ── Total loss ────────────────────────────────────────────────────────
        # Use warmed-up lambda_obs (§7.1 obs_warmup)
        total_loss = (
            task_loss
            + self._lambda_obs_eff * obs_loss
            + cfg.lambda_pg * pg_loss
        )

        # ── Info dict ─────────────────────────────────────────────────────────
        # 'ce' = mean ensemble CE across all positions (for PPL tracking)
        mean_ce_ens = sum(all_ce_ens) / max(1, len(all_ce_ens))

        info = {
            "ce":        mean_ce_ens,      # float — used by Trainer for PPL
            "task_loss": task_loss.item(),
            "obs_loss":  obs_loss.item(),
            "pg_loss":   pg_loss.item(),
            "e_final":   E_finals.mean().item(),
            "pg_base":   self.pg_baseline.item(),
        }
        return total_loss, info

    # ── Inference forward pass ─────────────────────────────────────────────────

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Inference-mode forward pass.

        Uses inference EFE (KL divergence risk — no y* needed).
        Action selection is greedy (argmin EFE) for determinism.
        Relaxation uses adaptive stopping (entropy / energy-delta criteria).

        Fix D: Uses relative position encoding throughout.
        Belief_norm applied before readout for consistency with training.

        Args:
            input_ids: (B, L) long — token IDs (can be a prefix for generation)

        Returns:
            logits: (B, L, vocab_size) — next-token logits for all positions
        """
        B, L = input_ids.shape
        device = input_ids.device
        cfg = self.cfg

        emb = self.token_embed(input_ids)   # (B, L, d_emb)

        substrate = self.substrate_init(B, device)
        pointers  = self._init_pointers(B, device)
        scale_lvl = torch.zeros(B, cfg.Nx, cfg.Ny, device=device, dtype=torch.long)

        all_logits = []

        for pos in range(L):
            pointers = pointers.clamp(0, pos)

            # T sensing steps (inference mode: greedy + inference EFE)
            for t in range(cfg.T_steps):
                P       = gather_neighborhood(substrate)
                obs     = self._gather_tape(emb, pointers, pos)
                # Fix D: relative encoding
                delta_pos  = pos - pointers                              # (B, Nx, Ny) ≥ 0
                pos_enc    = self.pos_encoder(delta_pos)                 # (B, Nx, Ny, d_pos)
                step_sizes = (1 << scale_lvl.clamp(0, cfg.L_scales - 1))
                scale_enc  = self.pos_encoder(step_sizes)                # (B, Nx, Ny, d_pos)

                # Inference-mode EFE (no targets)
                efe, next_ptrs, next_scales = self._compute_efe(
                    P=P, substrate=substrate,
                    pointers=pointers, scale_lvl=scale_lvl,
                    pos=pos, targets=None, training_mode=False,
                )

                # Greedy action selection
                policy_logits = (-efe / cfg.tau_efe).masked_fill(
                    efe == float('inf'), float('-inf')
                )
                actions = policy_logits.argmax(dim=-1)   # (B, Nx, Ny)

                # Apply kernel with pos_enc + scale_enc
                kernel_in = torch.cat([P, obs, pos_enc, scale_enc], dim=-1)
                delta_s, gate, _, _ = self.kernel(
                    kernel_in.reshape(B * cfg.Nx * cfg.Ny, -1)
                )
                delta_s = delta_s.reshape(B, cfg.Nx, cfg.Ny, self._D)
                gate    = gate.reshape(B, cfg.Nx, cfg.Ny, self._D)
                substrate = substrate + torch.sigmoid(gate) * delta_s
                substrate = substrate.clamp(-50.0, 50.0)

                # Move sensors
                pointers  = next_ptrs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
                scale_lvl = next_scales.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

            # Adaptive relaxation (inference only — design_decisions.md Section 9)
            prev_energy_proxy = None
            for r in range(cfg.R_max_inference):
                # Fix D: relative encoding for relaxation
                delta_pos_r   = pos - pointers
                pos_enc_r     = self.pos_encoder(delta_pos_r)
                step_sizes_r  = (1 << scale_lvl.clamp(0, cfg.L_scales - 1))
                scale_enc_r   = self.pos_encoder(step_sizes_r)
                obs_fixed     = self._gather_tape(emb, pointers, pos)

                P_r   = gather_neighborhood(substrate)
                kin_r = torch.cat([P_r, obs_fixed, pos_enc_r, scale_enc_r], dim=-1)
                ds_r, g_r, _, _ = self.kernel(
                    kin_r.reshape(B * cfg.Nx * cfg.Ny, -1)
                )
                ds_r = ds_r.reshape(B, cfg.Nx, cfg.Ny, self._D)
                g_r  = g_r.reshape(B, cfg.Nx, cfg.Ny, self._D)
                substrate = substrate + torch.sigmoid(g_r) * ds_r
                substrate = substrate.clamp(-50.0, 50.0)

                # Adaptive stopping: entropy threshold OR energy-delta threshold
                b_curr    = substrate[..., cfg.d_h : cfg.d_h + cfg.d_b]
                b_normed  = self.belief_norm(b_curr)
                agg_logits = self.readout.W_out(b_normed).mean(dim=(1, 2))  # (B, V)
                q_agg      = F.softmax(agg_logits, dim=-1)
                entropy    = -(q_agg * (q_agg + 1e-8).log()).sum(-1).mean()

                # Energy proxy at inference: substrate state Frobenius norm change
                energy_proxy = (substrate * substrate).mean().item()
                delta_E = abs(energy_proxy - prev_energy_proxy) \
                    if prev_energy_proxy is not None else float('inf')
                prev_energy_proxy = energy_proxy

                if entropy.item() < cfg.tau_H_stop or delta_E < cfg.eps_E_stop:
                    break

            # Compute ensemble logits at this position — with belief_norm
            b_curr   = substrate[..., cfg.d_h : cfg.d_h + cfg.d_b]  # (B, Nx, Ny, d_b)
            b_normed = self.belief_norm(b_curr)
            logits   = self.readout.W_out(b_normed).mean(dim=(1, 2))  # (B, V)
            all_logits.append(logits)

        return torch.stack(all_logits, dim=1)   # (B, L, vocab_size)

    # ── NCALMModel ABC methods ─────────────────────────────────────────────────

    def get_param_groups(self) -> list[dict]:
        """Single parameter group (standard Adam setup).

        For future ablations, the kernel, obs_model, and readout could be
        given different learning rates or weight decay.
        """
        return [{"params": list(self.parameters()), "label": "all"}]

    def describe(self) -> dict[str, Any]:
        """Return architecture description for analysis.json and logging."""
        cfg = self.cfg
        total_params = sum(p.numel() for p in self.parameters())

        # Count kernel params (verifies NCA parameter sharing)
        kernel_params = sum(p.numel() for p in self.kernel.parameters())
        obs_params    = sum(p.numel() for p in self.obs_model.parameters())
        readout_params = sum(p.numel() for p in self.readout.parameters())
        embed_params  = sum(p.numel() for p in self.token_embed.parameters())

        return {
            "architecture":    "AENCA v3.1",
            "total_params":    total_params,
            "substrate":       f"{cfg.Nx}×{cfg.Ny}={cfg.m} cells",
            "state_dim":       f"D={cfg.D} (d_h={cfg.d_h}, d_b={cfg.d_b}, d_c={cfg.d_c})",
            "kernel_params":   kernel_params,
            "obs_model_params": obs_params,
            "readout_params":  readout_params,
            "embed_params":    embed_params,
            "T_steps":         cfg.T_steps,
            "R_steps":         cfg.R_steps,
            "L_scales":        cfg.L_scales,
            "max_reach_tokens": cfg.T_steps * (2 ** (cfg.L_scales - 1)),
            "tbptt_chunk_size": cfg.tbptt_chunk_size,
            "vocab_size":      cfg.vocab_size,
            "d_emb":           cfg.d_emb,
            "d_kernel_h":      cfg.d_kernel_h,
            "perception":      "raw Moore neighbourhood concat (9×D)",
            "pos_encoding":    "RelativePositionEncoder (zero learned params, length-generalizing)",
            "efe_mode":        "dual (CE training / KL inference)",
            "action_masking":  True,
            "bptt":            f"TBPTT K={cfg.tbptt_chunk_size} + gradient checkpoint",
            "anytime_steps":   list(cfg.anytime_steps),
            "anytime_weights": list(cfg.anytime_weights),
            "fixes":           "A (gate bias -3.9 + layer scale), B (TBPTT), C (log cost + L_scales=18), D (relative pos enc)",
        }
