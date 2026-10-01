"""Causal State Stream (CSS) — Novel O(L) global communication for AENCA V7.

The CSS replaces V6's Kuramoto oscillator network as the global communication
mechanism. It provides O(L) sequential (or parallel-associative) state propagation
through N independent "streams" that flow causally through the sequence.

Design principles:
  1. **State-dependent decay** — each position learns its own forget rate from z
     content, not a fixed decay constant. This is the key differentiator from
     linear attention / S4 / Mamba: the forgetting is content-controlled.
  2. **Designed for repeated application** — same CSS module applied at every
     gamma NCA step. As z evolves through iteration, stream context enriches
     automatically ("Iterative Context Enrichment").
  3. **Multi-stream architecture** — N parallel streams capture different aspects
     (syntactic, semantic, positional) emergently.
  4. **Zero-init at start** — model begins as pure local NCA and discovers what
     global info to propagate during training (NCA zero-init cascade principle).

Architecture:
    Input: z (B, L, d_model)
    1. z_norm = RMSNorm(z)
    2. content = W_write(z_norm)                    → (B, L, N, d_state)
    3. gate_in = σ(W_gate_in(z_norm))               → (B, L, N)
    4. decay = σ(W_decay(z_norm))                    → (B, L, N)
    5. For each position i (causal scan):
         state[i] = decay[i] · state[i-1] + gate_in[i] · content[i]
    6. output = W_read(state.flatten(-2))            → (B, L, d_css)

Complexity: O(L × N × d_state) per call — linear in sequence length.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v7.config import V7Config


class CausalStateStream(nn.Module):
    """O(L) global communication via N parallel causal state streams.

    Each stream maintains a hidden state that flows causally through the
    sequence. At every position, the model decides:
    - WHAT to write (content, from z)
    - HOW MUCH to write (input gate, from z)
    - HOW MUCH to forget (decay gate, from z — state-dependent!)
    - WHAT to read (output projection of accumulated state)

    This is applied at every gamma NCA step, so the streams iteratively
    enrich as the cell states z improve through NCA iteration.

    Parameters:
        ~(d_model × css_state_dim + d_model × 2N + css_state_dim × d_css + norms)
        At d_model=768, N=8, d_state=32, d_css=256: ~230K params
    """

    def __init__(self, config: V7Config):
        super().__init__()
        self.d_model = config.d_model
        self.n_streams = config.n_streams
        self.d_state = config.d_state
        self.d_css = config.d_css
        self.css_state_dim = config.css_state_dim  # N × d_state

        # Pre-norm
        self.norm = nn.RMSNorm(config.d_model)

        # Write head: z → content to contribute to each stream
        # (B, L, d_model) → (B, L, N × d_state)
        self.W_write = nn.Linear(config.d_model, self.css_state_dim, bias=False)

        # Input gate: how much of content to write (per stream)
        # (B, L, d_model) → (B, L, N)
        self.W_gate_in = nn.Linear(config.d_model, config.n_streams, bias=True)

        # Decay gate: how much of previous state to keep (per stream)
        # (B, L, d_model) → (B, L, N)
        # KEY NOVELTY: decay is state-dependent (content-controlled forgetting)
        self.W_decay = nn.Linear(config.d_model, config.n_streams, bias=True)

        # Read head: project accumulated state back to model dimension
        # (B, L, N × d_state) → (B, L, d_css)
        self.W_read = nn.Linear(self.css_state_dim, config.d_css, bias=True)

        # Initialize
        self._init_weights(config)

    def _init_weights(self, config: V7Config) -> None:
        """Initialize CSS weights — zero-init cascade for read head.

        Write/gate projections: Xavier normal for stable initial dynamics.
        Read projection: ZERO-INIT → CSS starts producing zero context.
        The model begins as pure local NCA and learns to use CSS during training.

        Decay bias: initialized positive (σ(2) ≈ 0.88) → strong memory retention
        at start. Model learns to forget selectively.

        Input gate bias: initialized slightly negative (σ(-1) ≈ 0.27) → cautious
        initial writing. Model learns to write confidently.
        """
        # Write projection: Xavier normal, modest gain
        nn.init.xavier_normal_(self.W_write.weight, gain=0.5)

        # Input gate: Xavier, bias → σ(-1) ≈ 0.27 (cautious writing)
        nn.init.xavier_normal_(self.W_gate_in.weight, gain=1.0)
        nn.init.constant_(self.W_gate_in.bias, -1.0)

        # Decay gate: Xavier, bias → σ(2) ≈ 0.88 (strong retention)
        nn.init.xavier_normal_(self.W_decay.weight, gain=1.0)
        nn.init.constant_(self.W_decay.bias, 2.0)

        # Read projection: ZERO-INIT (NCA zero-init cascade)
        nn.init.zeros_(self.W_read.weight)
        nn.init.zeros_(self.W_read.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute CSS context via causal state scan.

        Args:
            z: (B, L, d_model) current cell states

        Returns:
            css_ctx: (B, L, d_css) global context from causal state streams
        """
        B, L, D = z.shape
        N = self.n_streams
        ds = self.d_state

        # Pre-norm
        z_norm = self.norm(z)

        # Compute write content, input gate, decay gate
        content = self.W_write(z_norm).view(B, L, N, ds)          # (B, L, N, ds)
        gate_in = torch.sigmoid(self.W_gate_in(z_norm))           # (B, L, N)
        decay = torch.sigmoid(self.W_decay(z_norm))               # (B, L, N)

        # Expand gates for broadcasting with d_state
        gate_in_exp = gate_in.unsqueeze(-1)                       # (B, L, N, 1)
        decay_exp = decay.unsqueeze(-1)                           # (B, L, N, 1)

        # Gated content: what to write at each position
        gated_content = gate_in_exp * content                     # (B, L, N, ds)

        # Causal scan: state[i] = decay[i] * state[i-1] + gated_content[i]
        # Sequential implementation — O(L) per stream, parallelized across B, N
        state = self._causal_scan(decay_exp, gated_content)       # (B, L, N, ds)

        # Flatten streams and project to output
        state_flat = state.reshape(B, L, N * ds)                  # (B, L, css_state_dim)
        css_ctx = self.W_read(state_flat)                         # (B, L, d_css)

        return css_ctx

    def _causal_scan(
        self,
        decay: torch.Tensor,
        content: torch.Tensor,
    ) -> torch.Tensor:
        """Parallel associative scan: s[i] = decay[i] * s[i-1] + content[i].

        Uses the Blelloch parallel prefix-sum algorithm with the associative
        operator for linear recurrences:
            (a₁, b₁) ∘ (a₂, b₂) = (a₁·a₂, a₂·b₁ + b₂)

        where a = cumulative decay, b = accumulated content.

        Complexity: O(L log L) work, O(log L) sequential depth — only 10 passes
        for L=1024 vs 1024 iterations in the sequential version.

        This is the critical performance fix: the sequential version launched
        ~1024 tiny CUDA kernels per call (which when multiplied by 8 gamma
        steps × checkpointing × grad accum = ~260K kernel launches per step).
        The parallel version does 10 large fused tensor operations per call.

        Args:
            decay:   (B, L, N, 1) or (B, L, N, d_state) — decay gates
            content: (B, L, N, d_state) — gated write content

        Returns:
            states: (B, L, N, d_state) accumulated state at each position
        """
        B, L, N, ds = content.shape

        # a[t] = decay factor, b[t] = accumulated value
        # Initialize: a[t] = decay[t], b[t] = content[t]
        a = decay.clone()    # (B, L, N, 1) or (B, L, N, ds)
        b = content.clone()  # (B, L, N, ds)

        # Parallel prefix scan: log₂(L) passes of tensor operations
        # Each pass doubles the "reach" of the scan.
        #
        # At pass k with stride s=2^k:
        #   For positions t ≥ s, combine element at (t-s) with element at t:
        #     b[t] = a[t] · b[t-s] + b[t]   (accumulate shifted content)
        #     a[t] = a[t] · a[t-s]           (accumulate decay product)
        #   Positions t < s are unchanged (identity element: a=1, b=0).
        #
        # After ceil(log₂(L)) passes, b[t] = s[t] (the scan result).
        num_passes = math.ceil(math.log2(L)) if L > 1 else 0

        for k in range(num_passes):
            s = 1 << k  # stride = 2^k

            # Shifted views: elements from s positions earlier
            # Pad left with identity: a=1 (no decay), b=0 (no content)
            a_shifted = F.pad(a[:, :L - s], (0, 0, 0, 0, s, 0), value=1.0)
            b_shifted = F.pad(b[:, :L - s], (0, 0, 0, 0, s, 0), value=0.0)

            # Combine: (a_shifted, b_shifted) ∘ (a, b) = (a·a_shifted, a·b_shifted + b)
            # IMPORTANT: read a before overwriting — b update uses old a
            b = a * b_shifted + b
            a = a * a_shifted

        return b  # b[t] now contains s[t] = scan result at position t

    def get_diagnostics(self, z: torch.Tensor) -> dict[str, float]:
        """Compute CSS diagnostics for monitoring.

        Args:
            z: (B, L, d_model) cell states

        Returns:
            Dict with gate statistics and stream activity metrics.
        """
        with torch.no_grad():
            z_norm = self.norm(z)
            gate_in = torch.sigmoid(self.W_gate_in(z_norm))
            decay = torch.sigmoid(self.W_decay(z_norm))

            return {
                "css_gate_in_mean": gate_in.mean().item(),
                "css_decay_mean": decay.mean().item(),
                "css_read_norm": self.W_read.weight.norm().item(),
                "css_write_norm": self.W_write.weight.norm().item(),
            }
