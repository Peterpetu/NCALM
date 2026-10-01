"""Morphogen-steered attention for AENCA V13.

Each theta cycle has a unique ThetaAttention module (own Q/K/V/Out projections).
The morphogen field generates Q/K gates that are applied in d_model space
BEFORE head reshape. RoPE is applied AFTER head reshape.

Architecture per ThetaAttention:
    z_norm = RMSNorm(z)
    Q = (q_proj(z_norm) * q_gate).view(B,L,H,d_h).transpose(1,2)
    K = (k_proj(z_norm) * k_gate).view(B,L,H,d_h).transpose(1,2)
    V = v_proj(z_norm).view(B,L,H,d_h).transpose(1,2)
    Q, K = apply_rotary(Q, K)
    attn_out = flash_attention(Q, K, V, is_causal=True)
    z = z + out_proj(attn_out)     ← out_proj is ZERO-INIT

Reference: V7.1 attention.py — RotaryPositionEmbedding, ThetaBlock attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v13.config import V13Config


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for theta attention.

    Full rotation (all dimensions), cached sin/cos tables.
    Dynamically extends cache if sequence length exceeds current capacity.
    Same implementation as V7.1.
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


class ThetaAttention(nn.Module):
    """Morphogen-steered attention block with unique weights per theta cycle.

    Q/K gates from the morphogen field modulate the query and key projections
    in d_model space, making attention patterns depend on the morphogen's
    local dynamics rather than purely content-based computation.

    Architecture:
        z_norm = RMSNorm(z)
        Q = (q_proj(z_norm) * q_gate).reshape_to_heads → RoPE
        K = (k_proj(z_norm) * k_gate).reshape_to_heads → RoPE
        V = v_proj(z_norm).reshape_to_heads
        attn_out = flash_attention(Q, K, V, is_causal=True)
        output = out_proj(attn_out)  ← ZERO-INIT

    Initialization:
        Q, K: xavier_normal_(gain=1/√d_head)
        V: xavier_normal_(gain=1.0)
        out_proj: zeros_()  ← zero-init cascade
    """

    def __init__(self, config: V13Config, block_idx: int):
        super().__init__()
        self.block_idx = block_idx
        d = config.d_model
        n_h = config.n_heads
        d_h = config.d_head

        self.attn_norm = nn.RMSNorm(d)
        self.q_proj = nn.Linear(d, n_h * d_h, bias=False)
        self.k_proj = nn.Linear(d, n_h * d_h, bias=False)
        self.v_proj = nn.Linear(d, n_h * d_h, bias=False)
        self.out_proj = nn.Linear(n_h * d_h, d, bias=False)

        self.n_heads = n_h
        self.d_head = d_h

        self.rope = RotaryPositionEmbedding(d_h, config.max_seq_len * 2)

        self._init_weights(config)

    def _init_weights(self, config: V13Config) -> None:
        """Initialize attention weights — same scheme as V7.1."""
        d_h = config.d_head
        nn.init.xavier_normal_(self.q_proj.weight, gain=1.0 / math.sqrt(d_h))
        nn.init.xavier_normal_(self.k_proj.weight, gain=1.0 / math.sqrt(d_h))
        nn.init.xavier_normal_(self.v_proj.weight, gain=1.0)
        nn.init.zeros_(self.out_proj.weight)  # zero-init cascade

    def forward(
        self,
        z: torch.Tensor,
        q_gate: torch.Tensor,
        k_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Apply morphogen-steered attention with residual.

        Args:
            z:      (B, L, d_model) content state
            q_gate: (B, L, d_model) query gate from morphogen [0, 2]
            k_gate: (B, L, d_model) key gate from morphogen [0, 2]

        Returns:
            z: (B, L, d_model) updated content state with attention residual
        """
        B, L, D = z.shape
        z_norm = self.attn_norm(z)

        # Project and apply gates in d_model space BEFORE head reshape
        q = (self.q_proj(z_norm) * q_gate).view(
            B, L, self.n_heads, self.d_head
        ).transpose(1, 2)                                     # (B, H, L, d_h)

        k = (self.k_proj(z_norm) * k_gate).view(
            B, L, self.n_heads, self.d_head
        ).transpose(1, 2)                                     # (B, H, L, d_h)

        v = self.v_proj(z_norm).view(
            B, L, self.n_heads, self.d_head
        ).transpose(1, 2)                                     # (B, H, L, d_h)

        # Apply RoPE AFTER head reshape
        cos, sin = self.rope(L)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        # Flash attention (causal)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=0.0,
        )                                                      # (B, H, L, d_h)

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).reshape(B, L, -1)  # (B, L, d_model)

        return z + self.out_proj(attn_out)
