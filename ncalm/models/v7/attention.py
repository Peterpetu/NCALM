"""Theta attention blocks for AENCA V7.

Each theta block provides high-bandwidth global communication via full
causal self-attention followed by a Hebbian FFN (BDH-inspired). Unlike the
NCA kernel (shared weights across all gamma steps), theta blocks have
UNIQUE weights per theta cycle — this is the hybrid aspect of V7.

Architecture per ThetaBlock:
    Pre-attention norm → Q/K/V projections → RoPE → Flash Attention → Out proj
    HebbianFFN: ReLU(W_x(z_pre)) ⊙ ReLU(W_y(z_post)) → W_dec → residual
    Cross-modal gating: pre-attention × post-attention concept encoding

The hybrid design:
    - Within each theta cycle: shared kernel iterates (NCA paradigm)
    - At theta boundaries: unique attention layers (transformer paradigm)
    - HebbianFFN uses cross-modal gating: concepts active in the input (z_pre)
      are gated by concepts supported by attention context (z_post). Only
      concepts confirmed by BOTH paths contribute. (BDH: arXiv:2509.26507)

Reference: GPT-NeoX parallel residuals NOT used — sequential is cleaner
    for the hierarchical design where attention follows NCA refinement.
Reference: Full RoPE (not partial Pythia-style) — every dimension rotated.
Reference: Kosowski et al. (2025), "The Dragon Hatchling" — Hebbian gating.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v7.config import V7Config
from ncalm.models.v7.hebbian import HebbianFFN


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for theta attention.

    Full rotation (all dimensions), cached sin/cos tables.
    Dynamically extends cache if sequence length exceeds current capacity.
    """

    def __init__(self, d_head: int, max_seq_len: int = 8192):
        super().__init__()
        self.d_head = d_head
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, d_head, 2).float() / d_head)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int) -> None:
        """Build cos/sin cache for positions 0..max_seq_len-1."""
        t = torch.arange(max_seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)              # (L, d_head/2)
        emb = torch.cat([freqs, freqs], dim=-1)             # (L, d_head)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self._cached_len = max_seq_len

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cos/sin tables for sequence length.

        Returns:
            cos: (seq_len, d_head)
            sin: (seq_len, d_head)
        """
        if seq_len > self._cached_len:
            self._build_cache(seq_len * 2)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embedding to input tensor.

    Args:
        x:   (B, n_heads, L, d_head) query or key tensor
        cos: (L, d_head) cosine table
        sin: (L, d_head) sine table

    Returns:
        Rotated tensor of same shape.
    """
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]

    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, d_head)
    sin = sin.unsqueeze(0).unsqueeze(0)

    cos1 = cos[..., :d_half]
    sin1 = sin[..., :d_half]

    rotated = torch.cat([
        x1 * cos1 - x2 * sin1,
        x1 * sin1 + x2 * cos1,
    ], dim=-1)

    return rotated


class ThetaBlock(nn.Module):
    """Theta-level attention + HebbianFFN block with unique weights.

    Each theta cycle has one ThetaBlock. These provide the high-bandwidth
    global communication that connects distant positions, complementing
    the NCA kernel's local computation within gamma steps.

    Architecture:
        z_pre = z                                           [save pre-attention state]
        z = z + OutProj(Attention(PreNorm(z)))               [global mixing]
        z = z + HebbianFFN(z_pre, z)                         [cross-modal concept gating]

    HebbianFFN cross-modal gating (BDH-inspired):
        x = ReLU(W_enc_x(z_pre))    — "which concepts are in the input?"
        y = ReLU(W_enc_y(z_post))   — "which concepts are supported by context?"
        xy = x ⊙ y                  — "update only mutually confirmed concepts"
        output = W_dec(xy)

    Key design choices:
    - Separate Q/K/V projections (not fused) for clarity
    - Full RoPE on all dimensions
    - Flash attention via F.scaled_dot_product_attention
    - HebbianFFN replaces SwiGLU: same params/FLOPs, different inductive bias
    - Zero-init on output projections → blocks start silent
    """

    def __init__(self, config: V7Config, block_idx: int):
        """Initialize theta block.

        Args:
            config: V7 configuration
            block_idx: which theta cycle this block belongs to (for logging)
        """
        super().__init__()
        self.block_idx = block_idx
        d = config.d_model
        n_h = config.n_heads
        d_h = config.d_head
        d_ff = config.d_ff_theta

        # ── Attention ────────────────────────────────────────────────
        self.attn_norm = nn.RMSNorm(d)
        self.q_proj = nn.Linear(d, n_h * d_h, bias=False)
        self.k_proj = nn.Linear(d, n_h * d_h, bias=False)
        self.v_proj = nn.Linear(d, n_h * d_h, bias=False)
        self.out_proj = nn.Linear(n_h * d_h, d, bias=False)

        self.n_heads = n_h
        self.d_head = d_h

        # ── Hebbian FFN (BDH-inspired, replaces SwiGLU) ─────────────
        self.ffn = HebbianFFN(
            d_model=d,
            d_ff=d_ff,
            dropout=config.dropout,
        )

        # ── RoPE (shared across all theta blocks via buffer) ─────────
        self.rope = RotaryPositionEmbedding(d_h, config.max_seq_len * 2)

        # Initialize
        self._init_weights(config)

    def _init_weights(self, config: V7Config) -> None:
        """Initialize theta block weights.

        Q, K: Xavier normal, gain = 1/√d_head (attention scaling)
        V: Xavier normal, gain = 1.0
        Out proj: ZERO-INIT → attention starts silent
        HebbianFFN: initialized internally (zero-init on decoder)
        """
        d_h = config.d_head

        # Q, K: scaled Xavier
        nn.init.xavier_normal_(self.q_proj.weight, gain=1.0 / math.sqrt(d_h))
        nn.init.xavier_normal_(self.k_proj.weight, gain=1.0 / math.sqrt(d_h))

        # V: standard Xavier
        nn.init.xavier_normal_(self.v_proj.weight, gain=1.0)

        # Out projection: zero-init
        nn.init.zeros_(self.out_proj.weight)

        # HebbianFFN is initialized in its own __init__

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply theta block: attention + HebbianFFN with residual connections.

        Args:
            z: (B, L, d_model) cell states after gamma NCA steps

        Returns:
            z: (B, L, d_model) updated cell states
        """
        B, L, D = z.shape

        # Save pre-attention state for Hebbian cross-modal gating
        z_pre = z

        # ── Attention ────────────────────────────────────────────────
        z_norm = self.attn_norm(z)

        # Q, K, V projections → (B, n_heads, L, d_head)
        q = self.q_proj(z_norm).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(z_norm).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(z_norm).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(L)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        # Flash attention (causal)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=0.0,
        )  # (B, n_heads, L, d_head)

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).reshape(B, L, -1)  # (B, L, n_heads*d_head)
        z = z + self.out_proj(attn_out)

        # ── Hebbian FFN (BDH-inspired cross-modal gating) ───────────
        # x_path encodes z_pre: "which concepts are in the input?"
        # y_path encodes z (post-attn): "which concepts are supported by context?"
        # Hebbian product: only update mutually confirmed concepts
        z = z + self.ffn(z_pre, z)

        return z
