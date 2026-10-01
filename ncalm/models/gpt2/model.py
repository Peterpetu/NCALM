"""GPT-2/NeoX baseline model for academic comparison against AENCA v5.1.

Standard decoder-only transformer matching Pythia-160M architecture:
- Pre-norm (LayerNorm)
- Parallel attention + MLP residuals (GPT-NeoX/GPT-J formulation)
- Rotary position embeddings (partial, 25% of d_head by default)
- GELU activation in FFN
- Untied token embeddings and LM head

Designed for iso-parameter comparison: ~162M params at vocab=50304, d=768, L=12.

Reference: Black et al. (2022) "GPT-NeoX-20B: An Open-Source Autoregressive
Language Model"; Biderman et al. (2023) "Pythia".
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ncalm.model import NCALMModel, register_model
from ncalm.models.gpt2.config import GPT2Config


# ── Rotary Position Embedding ────────────────────────────────────────────


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Precomputes cos/sin tables for rotary position encoding.
    Applied to a configurable fraction of d_head dimensions (rotary_pct).
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Precompute full cache
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)  # (T, dim/2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin tables for positions [0, seq_len).

        Returns:
            cos: (T, dim/2)
            sin: (T, dim/2)
        """
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary PE to tensor x of shape (..., rotary_dim).

    Uses the 'split-halves' convention: first half rotates with second half.
    """
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ── Attention ────────────────────────────────────────────────────────────


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with partial rotary PE.

    Uses PyTorch's scaled_dot_product_attention for Flash Attention support.
    Only rotary_dim dimensions of each head receive RoPE; the rest are
    position-agnostic (matching Pythia's rotary_pct=0.25 default).
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.rotary_dim = config.rotary_dim

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.rotary_emb = RotaryEmbedding(self.rotary_dim, config.max_seq_len)
        self._dropout_p = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Apply RoPE to first rotary_dim dimensions of Q and K
        if self.rotary_dim > 0:
            cos, sin = self.rotary_emb(T)
            # Broadcast: (1, 1, T, rotary_dim/2)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)

            q_rot = _apply_rotary(q[..., : self.rotary_dim], cos, sin)
            k_rot = _apply_rotary(k[..., : self.rotary_dim], cos, sin)

            q = torch.cat([q_rot, q[..., self.rotary_dim :]], dim=-1)
            k = torch.cat([k_rot, k[..., self.rotary_dim :]], dim=-1)

        # Efficient causal attention (FlashAttention when available)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self._dropout_p if self.training else 0.0,
            is_causal=True,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(attn_out)


# ── Feed-Forward Network ────────────────────────────────────────────────


class MLP(nn.Module):
    """Two-layer FFN with GELU activation (matching Pythia/GPT-NeoX)."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, config.d_ff)
        self.down_proj = nn.Linear(config.d_ff, config.d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


# ── Transformer Block ───────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    """Single transformer layer with pre-norm.

    Supports both parallel (GPT-NeoX) and sequential (GPT-2) residuals:
    - Parallel:   h = x + Attn(LN₁(x)) + MLP(LN₂(x))
    - Sequential: h = x + Attn(LN₁(x));  h = h + MLP(LN₂(h))

    Default is parallel to match Pythia-160M.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, eps=1e-5)
        self.ln2 = nn.LayerNorm(config.d_model, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.use_parallel = config.use_parallel_residual
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_parallel:
            # GPT-NeoX formulation: attention and MLP see the SAME input
            h = (
                x
                + self.dropout(self.attn(self.ln1(x)))
                + self.dropout(self.mlp(self.ln2(x)))
            )
        else:
            # Standard sequential residuals
            h = x + self.dropout(self.attn(self.ln1(x)))
            h = h + self.dropout(self.mlp(self.ln2(h)))
        return h


# ── GPT-2 Model ─────────────────────────────────────────────────────────


@register_model("gpt2")
class GPT2Model(NCALMModel):
    """GPT-2/NeoX baseline transformer for academic comparison.

    Architecture matches Pythia-160M: pre-norm, parallel residuals, partial
    rotary PE, GELU FFN, untied embeddings. ~162M params at default config.

    Implements the NCALMModel interface for drop-in use with the existing
    NCALM training infrastructure (Trainer, scheduler, data pipeline).
    """

    def __init__(self, config_dict: dict) -> None:
        super().__init__()

        # Parse flat config dict into GPT2Config (same pattern as V5)
        gpt2_keys = {f.name for f in GPT2Config.__dataclass_fields__.values()}
        gpt2_params = {k: v for k, v in config_dict.items() if k in gpt2_keys}
        self.config = GPT2Config(**gpt2_params)
        cfg = self.config

        # ── Token Embedding (no positional — we use RoPE) ───────────────
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # ── Transformer Layers ──────────────────────────────────────────
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layer)]
        )

        # ── Final Layer Norm ────────────────────────────────────────────
        self.final_ln = nn.LayerNorm(cfg.d_model, eps=1e-5)

        # ── LM Head ────────────────────────────────────────────────────
        if cfg.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # ── Weight Initialization ───────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """GPT-2/NeoX style weight initialization.

        All weight matrices: N(0, 0.02).
        All biases: zero.
        Output projections (attn out_proj, MLP down_proj): scaled by
        1/√(2·n_layer) for stable residual accumulation.
        """
        cfg = self.config
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)

        # Scale output projections for residual depth
        residual_scale = 1.0 / math.sqrt(2 * cfg.n_layer)
        for layer in self.layers:
            with torch.no_grad():
                layer.attn.out_proj.weight.mul_(residual_scale)
                layer.mlp.down_proj.weight.mul_(residual_scale)

    def _get_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Convert hidden states to vocabulary logits."""
        if self.lm_head is not None:
            return self.lm_head(h)
        return F.linear(h, self.token_embedding.weight)

    # ── NCALMModel Interface ─────────────────────────────────────────────

    def forward_train(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Standard autoregressive training step.

        Args:
            batch: (B, L) token IDs.

        Returns:
            loss: Scalar cross-entropy loss.
            info: Dict with 'ce' key (detached CE for logging/PPL).
        """
        B, L = batch.shape
        cfg = self.config

        h = self.token_embedding(batch)

        # Run transformer layers with gradient checkpointing
        C = cfg.checkpoint_interval
        n = cfg.n_layer
        if self.training and C > 0:
            for seg_start in range(0, n, C):
                seg_layers = list(self.layers[seg_start : min(seg_start + C, n)])

                def run_segment(_h, _layers=seg_layers):
                    for layer in _layers:
                        _h = layer(_h)
                    return _h

                h = checkpoint(run_segment, h, use_reentrant=False)
        else:
            for layer in self.layers:
                h = layer(h)

        h = self.final_ln(h)
        logits = self._get_logits(h)

        # Cross-entropy loss: predict x_{i+1} from position i
        ce_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.vocab_size),
            batch[:, 1:].reshape(-1),
            reduction="mean",
        )

        return ce_loss, {"ce": ce_loss.detach()}

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Autoregressive inference: compute logits for all positions.

        Args:
            input_ids: (B, L) token IDs.

        Returns:
            logits: (B, L, vocab_size).
        """
        h = self.token_embedding(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.final_ln(h)
        return self._get_logits(h)

    def get_param_groups(self) -> list[dict]:
        """Parameter groups for AdamW optimizer.

        Groups:
        - decay: weight matrices (2D+ tensors, excluding norms/embeddings)
        - no_decay: biases, LayerNorm parameters, embeddings (weight_decay=0)
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # No weight decay for: biases, LayerNorm, embeddings
            if p.dim() < 2 or "ln" in name or "embedding" in name:
                no_decay.append(p)
            else:
                decay.append(p)

        return [
            {"params": decay, "label": "decay"},
            {"params": no_decay, "label": "no_decay", "weight_decay": 0.0},
        ]

    def describe(self) -> dict[str, Any]:
        """Architecture description for analysis.json and logging."""
        cfg = self.config
        total = sum(p.numel() for p in self.parameters())

        embed_params = self.token_embedding.weight.numel()
        head_params = self.lm_head.weight.numel() if self.lm_head else 0
        layer_params = sum(
            p.numel() for layer in self.layers for p in layer.parameters()
        )
        ln_params = sum(p.numel() for p in self.final_ln.parameters())

        return {
            "architecture": "GPT-2",
            "total_params": total,
            "embed_params": embed_params,
            "head_params": head_params,
            "layer_params": layer_params,
            "ln_params": ln_params,
            "d_model": cfg.d_model,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "d_ff": cfg.d_ff,
            "d_head": cfg.d_head,
            "rotary_dim": cfg.rotary_dim,
            "rotary_pct": cfg.rotary_pct,
            "vocab_size": cfg.vocab_size,
            "tie_embeddings": cfg.tie_embeddings,
            "use_parallel_residual": cfg.use_parallel_residual,
            "paradigm": "Standard Decoder-Only Transformer (GPT-NeoX style)",
        }

    def get_phases(self, step: int, max_steps: int) -> str:
        """Return active training phase for logging."""
        return "AR"
