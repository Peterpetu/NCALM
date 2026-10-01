"""Precision-Routed Message Passing (PRMP) for AENCA V5.

Implements the FEP-derived global communication mechanism (§4).
Each cell broadcasts a compressed message, computes precision weights via
dot-product (derived from free energy gradient), and receives weighted
aggregates. Uses RoPE for relative position encoding and causal masking.

Reference: model_documents/aenca_v5_architecture.md §4.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v5.config import V5Config


class RotaryPositionEmbedding(nn.Module):
    """RoPE (Su et al. 2021) for relative position encoding in PRMP.

    Rotates query and key vectors to encode relative positions (§3.2, Eq 3.3).
    Applied per-head with d_head dimensions.
    """

    def __init__(self, d_head: int, max_seq_len: int = 8192):
        super().__init__()
        assert d_head % 2 == 0, f"d_head must be even for RoPE, got {d_head}"

        # Compute rotation frequencies: θ_j = 10000^{-2j/d}
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for positions [0, max_seq_len)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int) -> None:
        """Precompute cos/sin tables."""
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (L, d_head/2)
        cos_cached = freqs.cos()  # (L, d_head/2)
        sin_cached = freqs.sin()  # (L, d_head/2)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        self._max_cached_len = max_seq_len

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE rotation to query and key tensors.

        Args:
            q: (B, H, L, d_head) query tensor
            k: (B, H, L, d_head) key tensor
            seq_len: sequence length L

        Returns:
            q_rot, k_rot: RoPE-rotated query and key tensors
        """
        if seq_len > self._max_cached_len:
            self._build_cache(seq_len * 2)

        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, L, d/2)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, L, d/2)

        q_rot = self._apply_rotation(q, cos, sin)
        k_rot = self._apply_rotation(k, cos, sin)
        return q_rot, k_rot

    @staticmethod
    def _apply_rotation(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply RoPE rotation: rotate adjacent dimension pairs."""
        # x: (B, H, L, d_head)
        d_half = x.shape[-1] // 2
        x1, x2 = x[..., :d_half], x[..., d_half:]
        # Rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class PrecisionRoutedMessagePassing(nn.Module):
    """PRMP: FEP-derived global communication for NCA cells (§4).

    Each cell broadcasts a compressed message and receives precision-weighted
    aggregates from causally-available cells. Mechanically this is multi-head
    causal attention with RoPE (the "precision weights" are the attention
    weights); the free-energy framing in the design notes (§4.3) is an
    interpretation, not a different computation.

    Multi-head (§4.4): H parallel heads with d_head = d_msg / H dimensions.
    RoPE provides relative position encoding (§4.2, Eq 4.4).
    Causal masking enforced (§3.3, Eq 4.5).
    W_O initialized to zero (§10.3, Eq 10.4) — starts as pure local NCA.
    """

    def __init__(self, config: V5Config):
        super().__init__()
        self.d_model = config.d_model
        self.d_msg = config.d_msg
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.dropout_p = config.dropout

        assert config.d_msg % config.n_heads == 0, (
            f"d_msg ({config.d_msg}) must be divisible by n_heads ({config.n_heads})"
        )

        # Pre-norm (§6.3, Eq 6.7)
        self.norm = nn.RMSNorm(config.d_model)

        # Projection matrices (§4.2, Eq 4.2) — fused QKV for efficiency
        self.W_qkv = nn.Linear(config.d_model, 3 * config.d_msg, bias=False)

        # Output projection (§4.2, Eq 4.8) — initialized to zero (§10.3)
        self.W_O = nn.Linear(config.d_msg, config.d_model, bias=False)

        # RoPE for relative position encoding
        self.rope = RotaryPositionEmbedding(config.d_head, config.max_seq_len * 2)

        # Optional dropout on output (§15.3)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Initialize (§10.3)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize PRMP weights per §10.3."""
        d_head = self.d_head

        # Split QKV init
        with torch.no_grad():
            qkv_weight = self.W_qkv.weight  # (3*d_msg, d_model)
            d_msg = self.d_msg

            # W_Q: Xavier normal, gain=1/√d_head (Eq 10.3)
            q_part = qkv_weight[:d_msg]
            nn.init.xavier_normal_(q_part, gain=1.0 / math.sqrt(d_head))

            # W_K: Xavier normal, gain=1/√d_head (Eq 10.3)
            k_part = qkv_weight[d_msg:2*d_msg]
            nn.init.xavier_normal_(k_part, gain=1.0 / math.sqrt(d_head))

            # W_V: Xavier normal, gain=1 (Eq 10.3)
            v_part = qkv_weight[2*d_msg:]
            nn.init.xavier_normal_(v_part, gain=1.0)

        # W_O: Zero initialization (Eq 10.4) — PRMP starts silent
        nn.init.zeros_(self.W_O.weight)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Compute precision-routed messages for all cells.

        Args:
            s: (B, L, d_model) cell states at current NCA step

        Returns:
            M_integrated: (B, L, d_model) precision-weighted message aggregates
        """
        B, L, D = s.shape

        # Pre-norm (§6.3)
        s_norm = self.norm(s)

        # Compute Q, K, V projections (§4.2, Eq 4.2)
        qkv = self.W_qkv(s_norm)  # (B, L, 3*d_msg)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, L, d_msg)

        # Reshape for multi-head: (B, L, d_msg) -> (B, H, L, d_head)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE (§4.2, Eq 4.4) — encodes relative position
        q, k = self.rope(q, k, L)

        # Use PyTorch's scaled_dot_product_attention with causal mask
        # This implements Eq 4.3-4.6 efficiently (including causal masking)
        M = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )  # (B, H, L, d_head)

        # Reshape back: (B, H, L, d_head) -> (B, L, d_msg)
        M = M.transpose(1, 2).contiguous().view(B, L, self.d_msg)

        # Output projection (§4.2, Eq 4.8) + dropout
        M_integrated = self.dropout(self.W_O(M))  # (B, L, d_model)

        return M_integrated
