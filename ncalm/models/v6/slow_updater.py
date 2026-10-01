"""Slow updater for AENCA V6 theta (slow) timescale.

The slow updater runs once per theta cycle, absorbing the fast loop's
result (z_L) into the slow state (z_H) via:
  1. Gated absorption of z_L information
  2. Global causal attention for inter-position communication
  3. SwiGLU MLP for nonlinear processing
  4. Gated output (NCA-style)

This replaces V5's per-step PRMP with per-cycle attention — same global
communication mechanism, but running N times instead of K times.

Reference: model_documents/AENCA_FIRST_PRINCIPLES_REFLECTION.md §5, §8.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v6.config import V6Config


class RotaryPositionEmbedding(nn.Module):
    """RoPE for relative position encoding in slow updater attention.

    Identical to V5's implementation — rotates query and key vectors
    to encode relative positions.
    """

    def __init__(self, d_head: int, max_seq_len: int = 8192):
        super().__init__()
        assert d_head % 2 == 0, f"d_head must be even for RoPE, got {d_head}"
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int) -> None:
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)
        self._max_cached_len = max_seq_len

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._max_cached_len:
            self._build_cache(seq_len * 2)
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        d_half = x.shape[-1] // 2
        x1, x2 = x[..., :d_half], x[..., d_half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class SlowUpdater(nn.Module):
    """Theta-timescale update module with global attention.

    Runs once per theta cycle. Architecture (pre-norm throughout):

        z_H' = z_H + absorb_proj(z_L - z_H)      ← inject fast result
        z_H' = z_H' + W_O(Attention(z_H'))         ← global communication
        z_H' = z_H' + SwiGLU_MLP(z_H')             ← nonlinear processing
        Δz_H = z_H' - z_H                          ← total proposed change
        gate_H = σ(W_gate(Δz_H))                   ← NCA-style gating
        z_H_next = z_H + gate_H ⊙ Δz_H             ← gated update

    All output-facing projections are zero-initialized so the model
    starts near-identity (same zero-init cascade as V5).
    """

    def __init__(self, config: V6Config):
        super().__init__()
        self.d_model = config.d_model
        self.d_msg = config.d_msg
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.dropout_p = config.dropout

        # ── z_L absorption ─────────────────────────────────────────────
        self.absorb_norm = nn.RMSNorm(config.d_model)
        self.absorb_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # ── Causal self-attention (global communication) ───────────────
        self.attn_norm = nn.RMSNorm(config.d_model)
        self.W_qkv = nn.Linear(config.d_model, 3 * config.d_msg, bias=False)
        self.W_O = nn.Linear(config.d_msg, config.d_model, bias=False)
        self.rope = RotaryPositionEmbedding(config.d_head, config.max_seq_len * 2)
        self.attn_dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

        # ── SwiGLU MLP ────────────────────────────────────────────────
        self.mlp_norm = nn.RMSNorm(config.d_model)
        self.W_mlp_gate = nn.Linear(config.d_model, config.d_slow_ff, bias=False)
        self.W_mlp_up = nn.Linear(config.d_model, config.d_slow_ff, bias=False)
        self.W_mlp_down = nn.Linear(config.d_slow_ff, config.d_model, bias=False)

        # ── Output gate (NCA-style per-dimension gating) ──────────────
        self.gate_norm = nn.RMSNorm(config.d_model)
        self.W_gate_H = nn.Linear(config.d_model, config.d_model, bias=True)

        # Initialize
        self._init_weights(config)

    def _init_weights(self, config: V6Config) -> None:
        """Initialize weights with zero-init cascade."""
        d_head = self.d_head

        # absorb_proj: zero-init → model starts without z_L influence
        nn.init.zeros_(self.absorb_proj.weight)

        # Attention QKV: Xavier normal with scaled gain
        with torch.no_grad():
            qkv_w = self.W_qkv.weight
            d_msg = config.d_msg
            # Q portion
            nn.init.xavier_normal_(qkv_w[:d_msg], gain=1.0 / math.sqrt(d_head))
            # K portion
            nn.init.xavier_normal_(qkv_w[d_msg:2*d_msg], gain=1.0 / math.sqrt(d_head))
            # V portion
            nn.init.xavier_normal_(qkv_w[2*d_msg:], gain=1.0)

        # W_O: zero-init → attention starts silent
        nn.init.zeros_(self.W_O.weight)

        # MLP: Xavier for gate/up, zero for down → MLP starts silent
        nn.init.xavier_normal_(self.W_mlp_gate.weight)
        nn.init.xavier_normal_(self.W_mlp_up.weight)
        nn.init.zeros_(self.W_mlp_down.weight)

        # Output gate: small weight + bias for 1/N initial opening
        nn.init.xavier_normal_(self.W_gate_H.weight, gain=0.1)
        gate_bias = config.gate_bias_init(config.n_theta_cycles)
        nn.init.constant_(self.W_gate_H.bias, gate_bias)

    def forward(
        self, z_H: torch.Tensor, z_L_final: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute theta-timescale update to slow state.

        Args:
            z_H:       (B, L, d_model) current slow state
            z_L_final: (B, L, d_model) halt-weighted fast loop result

        Returns:
            delta_H: (B, L, d_model) proposed update to z_H
            gate_H:  (B, L, d_model) per-dimension gate ∈ (0, 1)
        """
        B, L, D = z_H.shape

        # ── Step 1: Absorb z_L information ─────────────────────────────
        # Inject what the fast loop computed. zero-init → starts silent.
        diff_L = self.absorb_norm(z_L_final - z_H)
        h = z_H + self.absorb_proj(diff_L)

        # ── Step 2: Global communication (causal self-attention) ───────
        h_n = self.attn_norm(h)
        qkv = self.W_qkv(h_n)  # (B, L, 3*d_msg)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, L, d_msg)

        # Reshape to multi-head: (B, L, d_msg) → (B, H, L, d_head)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE for relative position encoding
        q, k = self.rope(q, k, L)

        # Causal self-attention via flash attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )  # (B, H, L, d_head)

        # Reshape back: (B, H, L, d_head) → (B, L, d_msg)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_msg)

        # Add attention residual (zero-init W_O → starts silent)
        h = h + self.attn_dropout(self.W_O(attn_out))

        # ── Step 3: SwiGLU MLP ─────────────────────────────────────────
        h_n = self.mlp_norm(h)
        gate_val = F.silu(self.W_mlp_gate(h_n))
        up_val = self.W_mlp_up(h_n)
        h = h + self.W_mlp_down(gate_val * up_val)  # zero-init → starts silent

        # ── Step 4: Gated output (NCA-style) ───────────────────────────
        delta_H = h - z_H  # total proposed change to z_H
        gate_H = torch.sigmoid(
            self.W_gate_H(self.gate_norm(delta_H))
        )  # (B, L, d_model) ∈ (0, 1)

        return delta_H, gate_H
