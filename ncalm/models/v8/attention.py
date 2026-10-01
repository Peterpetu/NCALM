"""Theta attention blocks for AENCA V8 — Settlement-Aware Global Integration.

V8 extends V7's ThetaBlock with settlement-aware computation:

Key V8 changes from V7 (Spec Section 13.3):
  1. **Full state input**: Receives z ∈ R^{d+1} (hidden + alpha), slices to
     z_hidden ∈ R^d for attention and HebbianFFN, then reattaches alpha.
  2. **Settlement masking**: Settled cells participate in attention (providing
     read-only context) but their hidden states are NOT modified by the
     attention or FFN outputs (Spec Section 13.3, "Organ scale").
  3. **Read-only context**: Settled cells attend and are attended to, providing
     global context. But the settlement mask zeros out their updates.
     Analogous to Levin's multi-scale hierarchy: organism-level process
     reads from all cells but only modifies the active ones.

Architecture per ThetaBlock (Spec Definition 13.2):
    z_hidden = z[..., :d]                              [slice off alpha]
    z_alpha  = z[..., d:]                              [preserve alpha]
    z_pre = z_hidden                                   [save for Hebbian FFN]
    z_hidden += OutProj(Attention(PreNorm(z_hidden)))  [global mixing]
    z_hidden += HebbianFFN(z_pre, z_hidden)             [cross-modal gating]
    z_hidden *= (1 − settled).unsqueeze(−1)            [mask settled updates]
    z_hidden += z_pre * settled.unsqueeze(−1)          [restore settled states]
    z = cat([z_hidden, z_alpha], dim=−1)               [reattach alpha]

Reference: model_documents/AENCA_V8_GROWING_CELLULAR_AUTOMATA.md
  Section 13 (Multi-Scale Architecture), Definition 13.2 (Theta Cycle),
  Section 13.3 (Growth at Each Scale — Organ scale description).

RotaryPositionEmbedding and _apply_rotary are preserved from V7 unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v8.config import V8Config
from ncalm.models.v8.hebbian import HebbianFFN


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for theta attention.

    Full rotation (all dimensions), cached sin/cos tables.
    Preserved from V7 unchanged.
    """

    def __init__(self, d_head: int, max_seq_len: int = 8192):
        super().__init__()
        self.d_head = d_head
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, d_head, 2).float() / d_head)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int) -> None:
        """Build cos/sin cache for positions 0..max_seq_len-1."""
        t = torch.arange(max_seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self._cached_len = max_seq_len

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cos/sin tables for sequence length."""
        if seq_len > self._cached_len:
            self._build_cache(seq_len * 2)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embedding to input tensor.

    Args:
        x:   (B, n_heads, L, d_head)
        cos: (L, d_head)
        sin: (L, d_head)

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
    """V8 theta-level attention + HebbianFFN with settlement awareness.

    V8 extension: the block receives the full cell state z ∈ R^{d+1}
    (including alpha channel) and a settlement mask. The attention and
    HebbianFFN operate on z_hidden ∈ R^d only (alpha channel is not
    involved in attention). After computing the residual updates, the
    settlement mask ensures that settled cells' hidden states remain
    unchanged (Spec Section 13.3).

    Settled cells serve as **read-only context**: they participate in
    attention (their Q, K, V are computed, they attend and are attended to)
    but the output is not written back to their states. This is analogous
    to Levin's multi-scale competency: the organism-level process (attention)
    reads from all cells but only modifies the active ones.

    Architecture:
        z_hidden = z[..., :d]                    # extract hidden state
        z_pre = z_hidden                         # save for cross-modal FFN
        z_hidden = z_hidden + Attn(Norm(z_hidden)) # attention residual
        z_hidden = z_hidden + FFN(z_pre, z_hidden)  # Hebbian FFN residual
        # Settlement masking: restore settled cells to their pre-theta state
        active_mask = (1 − settled).unsqueeze(−1)
        z_hidden = z_pre + active_mask · (z_hidden − z_pre)
        z = cat([z_hidden, z[..., d:]], dim=−1)  # reattach alpha

    Key design choices (preserved from V7):
    - Separate Q/K/V projections, full RoPE on all dimensions
    - Flash attention via F.scaled_dot_product_attention
    - HebbianFFN: cross-modal gating (pre-attn × post-attn)
    - Zero-init on output projections → blocks start silent

    Parameters: ~10.3M per block (unchanged from V7)
    """

    def __init__(self, config: V8Config, block_idx: int):
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
        self.d_model = d

        # ── Hebbian FFN (BDH-inspired, replaces SwiGLU) ─────────────
        self.ffn = HebbianFFN(
            d_model=d,
            d_ff=d_ff,
            dropout=config.dropout,
        )

        # ── RoPE ─────────────────────────────────────────────────────
        self.rope = RotaryPositionEmbedding(d_h, config.max_seq_len * 2)

        self._init_weights(config)

    def _init_weights(self, config: V8Config) -> None:
        """Initialize theta block weights.

        Q, K: scaled Xavier (1/√d_head)
        V: standard Xavier
        Out proj: ZERO-INIT → attention starts silent
        HebbianFFN: initialized internally (zero-init decoder)
        """
        d_h = config.d_head
        nn.init.xavier_normal_(self.q_proj.weight, gain=1.0 / math.sqrt(d_h))
        nn.init.xavier_normal_(self.k_proj.weight, gain=1.0 / math.sqrt(d_h))
        nn.init.xavier_normal_(self.v_proj.weight, gain=1.0)
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        z_full: torch.Tensor,
        settled: torch.Tensor,
    ) -> torch.Tensor:
        """Apply theta block with settlement masking.

        Settled cells participate in attention (read-only context) but
        their hidden states are not modified (Spec Section 13.3).

        Args:
            z_full:  (B, L, d+1) full cell states (hidden + alpha)
            settled: (B, L) soft settlement scores ∈ [0, 1]

        Returns:
            z_full:  (B, L, d+1) updated cell states.
                     Settled cells' hidden states preserved.
                     Alpha channel always preserved (not in attention).
        """
        d = self.d_model
        B, L, _ = z_full.shape

        # Slice hidden and alpha
        z_hidden = z_full[..., :d]   # (B, L, d)
        z_alpha = z_full[..., d:]    # (B, L, 1)

        # Save pre-attention hidden state for:
        #   1. Hebbian cross-modal gating (z_pre for FFN)
        #   2. Settlement restoration (settled cells get z_pre back)
        z_pre = z_hidden

        # ── Attention ────────────────────────────────────────────────
        z_norm = self.attn_norm(z_hidden)

        q = self.q_proj(z_norm).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(z_norm).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(z_norm).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        cos, sin = self.rope(L)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        # Flash attention (causal) — all cells participate
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, L, -1)
        z_hidden = z_hidden + self.out_proj(attn_out)

        # ── Hebbian FFN (cross-modal gating) ─────────────────────────
        z_hidden = z_hidden + self.ffn(z_pre, z_hidden)

        # ── Settlement masking (Spec Sec 13.3) ───────────────────────
        # Settled cells' hidden states are NOT modified by theta output.
        # active_mask = (1 - settled) → 1 for active cells, 0 for settled
        # z_hidden = z_pre + active_mask · (z_hidden - z_pre)
        #          = z_pre · settled + z_hidden · (1 - settled)
        # This smoothly interpolates: fully settled cells get z_pre back.
        active_mask = (1.0 - settled).unsqueeze(-1)  # (B, L, 1)
        z_hidden = z_pre + active_mask * (z_hidden - z_pre)

        # Reattach alpha channel (unchanged by theta block)
        z_out = torch.cat([z_hidden, z_alpha], dim=-1)  # (B, L, d+1)

        return z_out
