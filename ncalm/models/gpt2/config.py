"""GPT-2/NeoX baseline configuration for academic comparison.

Matches Pythia-160M architecture choices for direct comparison:
- d_model=768, 12 layers, 12 heads, d_ff=3072
- Rotary position embeddings (partial, 25% of d_head)
- Parallel attention + MLP residuals (GPT-NeoX formulation)
- Untied embeddings, GPT-NeoX 50304 vocabulary

Reference: Biderman et al. (2023) "Pythia: A Suite for Analyzing Large
Language Models Across Training and Scaling"
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPT2Config:
    """Configuration for GPT-2/NeoX baseline transformer.

    Default values match Pythia-160M architecture for direct comparison.
    All fields have defaults; constructor filters from flat profile dict.
    """

    # ── Dimensions ──────────────────────────────────────────────────────
    d_model: int = 768          # Hidden dimension
    n_layer: int = 12           # Number of transformer layers
    n_head: int = 12            # Number of attention heads
    d_ff: int = 3072            # FFN intermediate dimension (4 × d_model)

    # ── Vocabulary and Sequence ─────────────────────────────────────────
    vocab_size: int = 50304     # GPT-NeoX vocabulary (Pythia tokenizer)
    max_seq_len: int = 2048     # Maximum sequence length for PE cache

    # ── Architecture Options ────────────────────────────────────────────
    tie_embeddings: bool = False        # Pythia-160M uses untied embeddings
    use_parallel_residual: bool = True  # GPT-NeoX parallel attn+MLP residuals
    rotary_pct: float = 0.25           # Fraction of d_head for rotary PE

    # ── Training ────────────────────────────────────────────────────────
    dropout: float = 0.0               # Dropout rate
    checkpoint_interval: int = 4       # Gradient checkpoint every N layers
    seq_len: int = 1024                # Training sequence length

    # ── Derived Properties ──────────────────────────────────────────────

    @property
    def d_head(self) -> int:
        """Per-head dimension."""
        return self.d_model // self.n_head

    @property
    def rotary_dim(self) -> int:
        """Number of dimensions for rotary PE per head (must be even)."""
        raw = int(self.d_head * self.rotary_pct)
        return raw + (raw % 2)  # Round up to even
