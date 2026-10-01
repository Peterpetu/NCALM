"""2D Reaction-Diffusion Morphogenetic Field for AENCA V15.

Core module implementing the dual-species Turing reaction-diffusion
NCA on a causal 2D grid. Replaces V14's 1D single-species morphogen.

Architecture per evolution step m (§9.1 of plan):
    1. D_a = P_{3×3}(a), D_b = P_{5×5}(b)           — causal 2D perception
    2. p = [a, b, D_a, D_b, c, π]                    — perception vector
    3. K parallel rules: δ_k = W_r2(tanh(W_r1(Norm(p))))
    4. State+step routing: w = softmax(W_route·s + W_step·embed(m))
    5. Content-conditioned bilinear gate per species
    6. Stochastic async update: a += mask · g_a ⊙ δa

Key properties:
  - Turing instability: k_b > k_a ⟹ D_b/D_a = 12/4 = 3.0
  - Edge of chaos: d_rule/d_s = 384/192 = 2.0
  - All NCA components SHARED across theta cycles
  - Content conditioning injected at every step (c-NCA)
  - Zero-init cascade on W_r2, W_route, W_fb → no dynamics at init
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v15.config import V15Config


# ═══════════════════════════════════════════════════════════════════════
# Causal Mask Construction
# ═══════════════════════════════════════════════════════════════════════

def build_causal_2d_mask(kernel_size: int) -> torch.Tensor:
    """Build a binary causal mask for a 2D kernel.

    An offset (Δr, Δc) is causal iff ΔrW + Δc < 0, where W ≫ k.
    For any practical W ≥ k:
      - Δr < 0: all columns causal (row above)
      - Δr = 0: only Δc < 0 (same row, left)
      - Δr > 0: none causal (row below)
    Self (0, 0) is included.

    Returns:
        mask: (1, 1, k, k) binary tensor (float)
    """
    k = kernel_size
    half = k // 2
    mask = torch.zeros(k, k)

    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            if dr < 0:
                mask[dr + half, dc + half] = 1.0  # row above: all causal
            elif dr == 0 and dc <= 0:
                mask[dr + half, dc + half] = 1.0  # same row: self + left
            # dr > 0: non-causal, leave as 0

    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)


# ═══════════════════════════════════════════════════════════════════════
# 2D Sinusoidal Position Encoding
# ═══════════════════════════════════════════════════════════════════════

def build_2d_sinusoidal_pe(
    H: int, W: int, d_pos: int, omega: float = 10000.0
) -> torch.Tensor:
    """Build 2D sinusoidal position encoding for an H×W grid.

    First d_pos/2 dims: row encoding. Last d_pos/2 dims: column encoding.

    Args:
        H: grid height
        W: grid width
        d_pos: total position encoding dimension (must be even)
        omega: frequency base

    Returns:
        pe: (H, W, d_pos) position encodings
    """
    assert d_pos % 2 == 0, f"d_pos must be even, got {d_pos}"
    d_half = d_pos // 2

    rows = torch.arange(H).float().unsqueeze(1)  # (H, 1)
    cols = torch.arange(W).float().unsqueeze(1)  # (W, 1)

    # Frequency terms
    div_term = torch.exp(
        torch.arange(0, d_half, 2).float() * -(math.log(omega) / d_half)
    )  # (d_half/2,)

    # Row encoding: (H, d_half)
    pe_row = torch.zeros(H, d_half)
    pe_row[:, 0::2] = torch.sin(rows * div_term)
    pe_row[:, 1::2] = torch.cos(rows * div_term)

    # Column encoding: (W, d_half)
    pe_col = torch.zeros(W, d_half)
    pe_col[:, 0::2] = torch.sin(cols * div_term)
    pe_col[:, 1::2] = torch.cos(cols * div_term)

    # Broadcast to (H, W, d_pos)
    pe = torch.zeros(H, W, d_pos)
    pe[:, :, :d_half] = pe_row.unsqueeze(1).expand(H, W, d_half)
    pe[:, :, d_half:] = pe_col.unsqueeze(0).expand(H, W, d_half)

    return pe


# ═══════════════════════════════════════════════════════════════════════
# Fold / Unfold Operations
# ═══════════════════════════════════════════════════════════════════════

def compute_grid_dims(N: int) -> tuple[int, int]:
    """Compute 2D grid dimensions for a sequence of length N.

    Returns (H, W) such that H * W >= N.
    """
    W = math.ceil(math.sqrt(N))
    H = math.ceil(N / W)
    return H, W


def fold_1d_to_2d(
    x: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    """Fold 1D sequence to 2D grid (row-major) + permute for Conv2d.

    Args:
        x: (B, N, d) 1D sequence tensor
        H: grid height
        W: grid width (H * W >= N)

    Returns:
        x_2d: (B, d, H, W) channels-first 2D tensor
    """
    B, N, d = x.shape
    grid_size = H * W

    # Pad if N < H*W
    if N < grid_size:
        pad = torch.zeros(B, grid_size - N, d, device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)

    # Reshape: (B, H*W, d) → (B, H, W, d) → (B, d, H, W)
    return x.view(B, H, W, d).permute(0, 3, 1, 2).contiguous()


def unfold_2d_to_1d(
    x_2d: torch.Tensor, N: int
) -> torch.Tensor:
    """Unfold 2D grid back to 1D sequence (row-major).

    Args:
        x_2d: (B, d, H, W) channels-first 2D tensor
        N: original sequence length (may be < H*W)

    Returns:
        x: (B, N, d) 1D sequence tensor
    """
    B, d, H, W = x_2d.shape
    # (B, d, H, W) → (B, H, W, d) → (B, H*W, d) → (B, N, d)
    x = x_2d.permute(0, 2, 3, 1).reshape(B, H * W, d)
    return x[:, :N, :]


# ═══════════════════════════════════════════════════════════════════════
# Causal 2D Perception Module
# ═══════════════════════════════════════════════════════════════════════

class CausalPerception2D(nn.Module):
    """Causal 2D neighborhood perception via masked depthwise Conv2d.

    Implements §8 of the plan: each channel has its own k×k kernel,
    masked to only allow causal neighbors. The mask is a fixed buffer;
    gradients flow only through unmasked entries.

    For k=3: 5 active entries (4 causal + self) of 9.
    For k=5: 13 active entries (12 causal + self) of 25.
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Depthwise kernel: (channels, 1, k, k)
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=0)

        # Fixed causal mask: (1, 1, k, k)
        self.register_buffer("causal_mask", build_causal_2d_mask(kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal depthwise 2D convolution.

        Args:
            x: (B, channels, H, W) input

        Returns:
            out: (B, channels, H, W) causally perceived output
        """
        masked_weight = self.weight * self.causal_mask
        return F.conv2d(
            x, masked_weight,
            padding=self.padding,
            groups=self.channels,
        )


# ═══════════════════════════════════════════════════════════════════════
# Reaction-Diffusion Field (Core Module)
# ═══════════════════════════════════════════════════════════════════════

class ReactionDiffusionField(nn.Module):
    """Complete dual-species RD-NCA morphogen field for V15.

    Manages:
      1. Seeding (a, b) from content z
      2. Evolving (a, b) via K parallel rules with causal 2D perception
      3. Content conditioning injection at every step
      4. Content → morphogen feedback
    """

    def __init__(self, config: V15Config):
        super().__init__()
        d = config.d_model
        da = config.d_a
        db = config.d_b
        ds = config.d_s
        dc = config.d_c
        dp = config.d_p
        dr = config.d_rule
        K = config.n_rules
        M = config.n_morph_steps

        self.config = config

        # ── Max grid dimensions (for position encoding buffer) ───────
        self.max_H = config.grid_height
        self.max_W = config.grid_width

        # ── Morphogen Seed ───────────────────────────────────────────
        self.W_seed_a = nn.Linear(d, da, bias=False)
        self.W_seed_b = nn.Linear(d, db, bias=False)

        # ── Content Conditioning ─────────────────────────────────────
        self.norm_cond = nn.RMSNorm(d)
        self.W_cond = nn.Linear(d, dc, bias=False)

        # ── Causal 2D Perception ─────────────────────────────────────
        self.perc_a = CausalPerception2D(da, config.k_a)
        self.perc_b = CausalPerception2D(db, config.k_b)

        # ── 2D Position Encoding (non-learnable) ────────────────────
        # Pre-compute for max grid size; slice as needed for actual N
        max_H = config.grid_height + 1  # +1 for L+1 training tokens
        max_W = config.grid_width + 1
        pe = build_2d_sinusoidal_pe(max_H, max_W, config.d_pos)
        # Shape: (1, d_pos, max_H, max_W) for broadcasting
        self.register_buffer(
            "pos_encoding",
            pe.permute(2, 0, 1).unsqueeze(0),  # (1, d_pos, max_H, max_W)
            persistent=False,
        )

        # ── K Parallel NCA Rules (batched via einsum) ────────────
        # Stacked weights for all K rules — single CUDA kernel via einsum.
        # W_r1: (K, d_p, d_rule), bias_r1: (K, d_rule)
        # W_r2: (K, d_rule, d_s) — ZERO INIT (cascade)
        self.rule_norm = nn.RMSNorm(dp)

        self.W_r1 = nn.Parameter(torch.empty(K, dp, dr))
        self.bias_r1 = nn.Parameter(torch.zeros(K, dr))
        self.W_r2 = nn.Parameter(torch.zeros(K, dr, ds))

        # ── State-Dependent Step-Dependent Routing ───────────────────
        self.W_route = nn.Linear(ds, K, bias=False)
        self.step_embed = nn.Embedding(M, config.step_embed_dim)
        self.W_step_route = nn.Linear(config.step_embed_dim, K, bias=False)

        # ── Content-Conditioned Bilinear Gate (per species) ──────────
        # Activator gate: input [a; c] → d_a
        self.W_ga1 = nn.Linear(da + dc, da, bias=False)
        self.W_ga2 = nn.Linear(da + dc, da, bias=False)
        self.bias_ga = nn.Parameter(torch.zeros(da))

        # Inhibitor gate: input [b; c] → d_b
        self.W_gb1 = nn.Linear(db + dc, db, bias=False)
        self.W_gb2 = nn.Linear(db + dc, db, bias=False)
        self.bias_gb = nn.Parameter(torch.zeros(db))

        # ── Content → Morphogen Feedback ─────────────────────────────
        self.norm_fb = nn.RMSNorm(d)
        self.W_fb_a = nn.Linear(d, da, bias=False)
        self.W_fb_b = nn.Linear(d, db, bias=False)

        # Store config references
        self.n_rules = K
        self.n_morph_steps = M
        self.d_a = da
        self.d_b = db
        self.d_s = ds
        self.p_async = config.p_async

        self._init_weights(config)

    def _init_weights(self, config: V15Config) -> None:
        """Initialize per §14 zero-init cascade."""
        # Seed: Xavier
        nn.init.xavier_normal_(self.W_seed_a.weight, gain=1.0)
        nn.init.xavier_normal_(self.W_seed_b.weight, gain=1.0)

        # Conditioning: Xavier
        nn.init.xavier_normal_(self.W_cond.weight, gain=1.0)

        # Rules: W_r1 Xavier per-rule slice, bias zero; W_r2 ZERO (cascade)
        for k in range(self.n_rules):
            nn.init.xavier_normal_(self.W_r1.data[k], gain=1.0)
        nn.init.zeros_(self.bias_r1.data)
        nn.init.zeros_(self.W_r2.data)  # ZERO-INIT cascade

        # Routing: ZERO-INIT → uniform 1/K at start
        nn.init.zeros_(self.W_route.weight)
        nn.init.normal_(self.step_embed.weight, std=0.02)
        nn.init.zeros_(self.W_step_route.weight)

        # Gates: Xavier branches, zero bias → σ(0)=0.5
        nn.init.xavier_normal_(self.W_ga1.weight, gain=1.0)
        nn.init.xavier_normal_(self.W_ga2.weight, gain=1.0)
        nn.init.xavier_normal_(self.W_gb1.weight, gain=1.0)
        nn.init.xavier_normal_(self.W_gb2.weight, gain=1.0)
        # bias_ga, bias_gb already init to 0 in constructor

        # Feedback: ZERO-INIT (no feedback at start)
        nn.init.zeros_(self.W_fb_a.weight)
        nn.init.zeros_(self.W_fb_b.weight)

    def seed(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create initial morphogen fields from content embedding.

        Args:
            z: (B, N, d_model) initial content state

        Returns:
            a: (B, d_a, H, W) initial activator on 2D grid
            b: (B, d_b, H, W) initial inhibitor on 2D grid
        """
        N = z.shape[1]
        H, W = compute_grid_dims(N)

        a_1d = torch.tanh(self.W_seed_a(z))  # (B, N, d_a)
        b_1d = torch.tanh(self.W_seed_b(z))  # (B, N, d_b)

        a = fold_1d_to_2d(a_1d, H, W)  # (B, d_a, H, W)
        b = fold_1d_to_2d(b_1d, H, W)  # (B, d_b, H, W)

        return a, b

    def compute_content_conditioning(
        self, z: torch.Tensor
    ) -> torch.Tensor:
        """Compute content conditioning signal and fold to 2D.

        Args:
            z: (B, N, d_model) content state

        Returns:
            c: (B, d_c, H, W) content conditioning on 2D grid
        """
        N = z.shape[1]
        H, W = compute_grid_dims(N)
        c_1d = self.W_cond(self.norm_cond(z))  # (B, N, d_c)
        return fold_1d_to_2d(c_1d, H, W)  # (B, d_c, H, W)

    def _single_step(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute one RD-NCA step (§9.1).

        Args:
            a: (B, d_a, H, W) activator
            b: (B, d_b, H, W) inhibitor
            c: (B, d_c, H, W) content conditioning
            step_idx: NCA step index m

        Returns:
            a: updated activator
            b: updated inhibitor
            routing_weights: (B, H*W, K) routing weights
            update_a: (B, d_a, H, W) gated activator update (for diagnostics)
            update_b: (B, d_b, H, W) gated inhibitor update (for diagnostics)
        """
        B, da, H, W = a.shape
        db = b.shape[1]
        ds = da + db

        # (A) Causal 2D Perception
        D_a = self.perc_a(a)  # (B, d_a, H, W)
        D_b = self.perc_b(b)  # (B, d_b, H, W)

        # (C) Perception Vector Assembly
        # Position encoding: (1, d_pos, H, W), broadcast over batch
        pi = self.pos_encoding[:, :, :H, :W]  # (1, d_pos, H, W)

        # Concatenate: [a, b, D_a, D_b, c, π] along channel dim
        p = torch.cat([a, b, D_a, D_b, c, pi.expand(B, -1, -1, -1)], dim=1)
        # p: (B, d_p, H, W)

        # Reshape for linear layers: (B, d_p, H, W) → (B, H*W, d_p)
        p_flat = p.permute(0, 2, 3, 1).reshape(B, H * W, -1)  # (B, H*W, d_p)

        # Normalize
        p_norm = self.rule_norm(p_flat)  # (B, H*W, d_p)

        # (D) K Parallel Rules (batched einsum)
        # W_r1: (K, d_p, d_rule), p_norm: (B, HW, d_p)
        # hidden = tanh(p_norm @ W_r1[k].T + bias_r1[k]) for each k, batched:
        hidden = torch.tanh(
            torch.einsum('bnd,kdf->bnkf', p_norm, self.W_r1) + self.bias_r1
        )  # (B, HW, K, d_rule)
        # delta_stack[b,n,k,:] = hidden[b,n,k,:] @ W_r2[k].T, batched:
        delta_stack = torch.einsum(
            'bnkf,kfs->bnks', hidden, self.W_r2
        )  # (B, HW, K, d_s)

        # (E) State-Dependent Step-Dependent Routing
        # State: [a; b] flattened → (B, H*W, d_s)
        state_flat = torch.cat([a, b], dim=1).permute(0, 2, 3, 1).reshape(B, H * W, ds)
        route_logits = self.W_route(state_flat)  # (B, H*W, K)

        # Step bias
        step_idx_t = torch.tensor(step_idx, device=a.device)
        step_bias = self.W_step_route(self.step_embed(step_idx_t))  # (K,)
        route_logits = route_logits + step_bias

        weights = F.softmax(route_logits, dim=-1)  # (B, H*W, K)

        # (F) Weighted Reaction
        delta = (weights.unsqueeze(-1) * delta_stack).sum(dim=2)  # (B, H*W, d_s)

        # Split into activator and inhibitor updates
        delta_a = delta[:, :, :da]  # (B, H*W, d_a)
        delta_b = delta[:, :, da:]  # (B, H*W, d_b)

        # Reshape back to 2D: (B, H*W, d) → (B, d, H, W)
        delta_a = delta_a.permute(0, 2, 1).reshape(B, da, H, W)
        delta_b = delta_b.permute(0, 2, 1).reshape(B, db, H, W)

        # (G) Content-Conditioned Bilinear Gate
        # Flatten a, b, c for gate computation: (B, d, H, W) → (B, H*W, d)
        a_flat = a.permute(0, 2, 3, 1).reshape(B, H * W, da)
        b_flat = b.permute(0, 2, 3, 1).reshape(B, H * W, db)
        c_flat = c.permute(0, 2, 3, 1).reshape(B, H * W, c.shape[1])

        # Activator gate: σ(SiLU(W_ga1([a;c])) ⊙ W_ga2([a;c]) + bias_ga)
        ac = torch.cat([a_flat, c_flat], dim=-1)  # (B, H*W, d_a+d_c)
        g_a = torch.sigmoid(
            F.silu(self.W_ga1(ac)) * self.W_ga2(ac) + self.bias_ga
        )  # (B, H*W, d_a)

        # Inhibitor gate: σ(SiLU(W_gb1([b;c])) ⊙ W_gb2([b;c]) + bias_gb)
        bc = torch.cat([b_flat, c_flat], dim=-1)  # (B, H*W, d_b+d_c)
        g_b = torch.sigmoid(
            F.silu(self.W_gb1(bc)) * self.W_gb2(bc) + self.bias_gb
        )  # (B, H*W, d_b)

        # Reshape gates to 2D: (B, H*W, d) → (B, d, H, W)
        g_a = g_a.permute(0, 2, 1).reshape(B, da, H, W)
        g_b = g_b.permute(0, 2, 1).reshape(B, db, H, W)

        # Gated updates
        update_a = g_a * delta_a  # (B, d_a, H, W)
        update_b = g_b * delta_b  # (B, d_b, H, W)

        # (H) Stochastic Asynchronous Update
        if self.training:
            mask = torch.bernoulli(
                torch.full((B, 1, H, W), self.p_async, device=a.device)
            )  # (B, 1, H, W) — broadcast over channels
            a = a + mask * update_a
            b = b + mask * update_b
        else:
            a = a + update_a
            b = b + update_b

        return a, b, weights, update_a, update_b

    def evolve(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        z: torch.Tensor,
        n_steps: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """Evolve morphogen for n_steps with content conditioning.

        Args:
            a: (B, d_a, H, W) activator
            b: (B, d_b, H, W) inhibitor
            z: (B, N, d_model) content state (for conditioning)
            n_steps: number of NCA steps (M)

        Returns:
            a: final activator
            b: final inhibitor
            routing_weights: list of (B, H*W, K) per step
            updates_a: list of (B, d_a, H, W) gated updates per step
            updates_b: list of (B, d_b, H, W) gated updates per step
        """
        routing_weights = []
        updates_a = []
        updates_b = []

        # Compute content conditioning once per theta cycle
        c = self.compute_content_conditioning(z)  # (B, d_c, H, W)

        for m in range(n_steps):
            a, b, rw, ua, ub = self._single_step(a, b, c, m)
            routing_weights.append(rw)
            updates_a.append(ua)
            updates_b.append(ub)

        return a, b, routing_weights, updates_a, updates_b

    def update_from_content(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Content → morphogen feedback (§12, step Θ-5).

        Args:
            a: (B, d_a, H, W) activator
            b: (B, d_b, H, W) inhibitor
            z: (B, N, d_model) content state

        Returns:
            a: updated activator
            b: updated inhibitor
        """
        H, W = a.shape[2], a.shape[3]
        z_norm = self.norm_fb(z)
        a_fb = torch.tanh(self.W_fb_a(z_norm))  # (B, N, d_a)
        b_fb = torch.tanh(self.W_fb_b(z_norm))  # (B, N, d_b)

        a = a + fold_1d_to_2d(a_fb, H, W)
        b = b + fold_1d_to_2d(b_fb, H, W)

        return a, b


# ═══════════════════════════════════════════════════════════════════════
# Content Readout
# ═══════════════════════════════════════════════════════════════════════

class ContentReadout(nn.Module):
    """Project evolved morphogen state back into content space (§11).

    One instance per theta cycle (unique weights). Reads the combined
    [a; b] state, normalizes, projects to d_model, unfolds to 1D.

    Phase 1 (L=1): single level, no scale weighting.

    At initialization:
      - W_readout = 0 (zero-init cascade → no morph→content)
    """

    def __init__(self, config: V15Config):
        super().__init__()
        ds = config.d_s
        d = config.d_model

        self.norm = nn.RMSNorm(ds)
        self.W_readout = nn.Linear(ds, d, bias=False)
        nn.init.zeros_(self.W_readout.weight)  # ZERO-INIT cascade

        # Scale logits for multi-step readout weighting (Phase 1: 1 level)
        self.scale_logits = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """Read content update from morphogen state.

        Args:
            a: (B, d_a, H, W) activator
            b: (B, d_b, H, W) inhibitor
            N: original sequence length

        Returns:
            z_update: (B, N, d_model) content update (residual)
        """
        B = a.shape[0]
        H, W = a.shape[2], a.shape[3]

        # Concatenate [a; b]: (B, d_s, H, W)
        h = torch.cat([a, b], dim=1)

        # Reshape to (B, H*W, d_s) for normalization and projection
        h_flat = h.permute(0, 2, 3, 1).reshape(B, H * W, -1)

        # Normalize and project
        z_update = self.W_readout(self.norm(h_flat))  # (B, H*W, d_model)

        # Truncate to original sequence length
        return z_update[:, :N, :]
