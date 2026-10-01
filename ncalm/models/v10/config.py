"""AENCA V10 configuration — Persistent Growth NCA Language Model.

V10 architecture: the fundamental CA inversion.

  Instead of FEW steps with a BIG kernel (V9: 3 steps × 59M kernel),
  use MANY steps with a TINY kernel (64 steps × 5M kernel).

Core principles (all PRIMARY CITIZENS):

  - Wolfram: Complex behavior from a single SIMPLE local rule applied many times.
    One shared SimpleHebbianRule (~5M params) applied 64 times — the genome.
    Rule 110 is 8 bits. Complexity comes from iteration, not rule complexity.

  - Permanence: Settlement IS the architecture. Cells that converge STOP updating.
    Their state becomes frozen read-only memory for neighbors. Classical CA
    still lifes are permanent. V10 enforces this structurally via (1-settled) gate.

  - Levin: Structure IS function. The grown z (with permanent settled regions)
    IS the language model representation. Growth happens through 64 iterative
    morphogenetic steps. Mental cognition IS developmental.

  - Sparsity: Only the active frontier computes. Settled cells contribute zero
    delta. As growth progresses, the frontier shrinks — later steps are cheaper.
    Game of Life still lifes don't compute. Neither do our settled cells.

  - Zenil: Edge of computability. L_CE prevents Class I (too simple).
    L_perm + L_spectral prevent Class III (too chaotic). The sweet spot IS
    Class IV — computational irreducibility, where short programs (small kernel)
    produce complex output given enough runtime (64 steps).

Forward pass (64 shared CA steps with cyclic dilations):
    z = Embed(x) + PE                                   # Seed
    settlement = SettlementState(...)                     # All cells active
    For t = 0..63 with dilation = 2^(t mod n_scales):
        perceived = DepthwiseCausalConv(z, dilation)     # Neighborhood
        p = ReLU(W_p · RMSNorm(perceived))               # Perception encoding
        s = ReLU(W_s · RMSNorm(z))                       # Self encoding
        delta = W_d · (p ⊙ s)                            # Hebbian update
        z = z + delta * (1 - settled)                    # PERMANENCE gate
        settlement.update(delta)                          # Track convergence
    logits = RMSNorm(z) · Embed^T / √d                  # Grown state IS readout
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class V10Config:
    """Complete configuration for AENCA V10: Persistent Growth NCA.

    Parameter budget (~110M at defaults):
        embed (tied)   = vocab_size × d_model            (~103M)
        pos_embed      = max_seq_len × d_model           (~2.1M)
        kernel         = 3 × d_model × d_rule            (~4.9M)
        conv (shared)  = d_model × kernel_size           (~6K)
        norms          = 3 × d_model                     (~6K)
        ──────────────────────────────────────────────────────
        Total          ≈ 110M

    Intentionally smaller than GPT-2's 162M — iteration replaces parameters.
    """

    # ── Core Dimensions ───────────────────────────────────────────────
    d_model: int = 2048         # Cell state width (same as V9)
    d_rule: int = 800           # Rule concept space (17× SMALLER than V9's 9255)
    n_steps: int = 64           # CA iterations (21× MORE than V9's 3)
    n_scales: int = 8           # Number of dilation scales per cycle

    # ── Causal Convolution ────────────────────────────────────────────
    conv_kernel_size: int = 3   # Depthwise conv kernel size (shared)

    # ── Settlement (PERMANENCE — PRIMARY CITIZEN) ─────────────────────
    settlement_beta: float = 0.9     # EMA decay for tracking convergence
    settlement_tau: float = 0.05     # Settlement threshold (small delta → settled)
    settlement_eta: float = 0.1      # Sigmoid temperature for soft transition

    # ── Vocabulary and Sequence ───────────────────────────────────────
    vocab_size: int = 50304     # |V| — vocabulary size
    max_seq_len: int = 1024     # L — max sequence length

    # ── Embeddings ────────────────────────────────────────────────────
    tie_embeddings: bool = True # Tie readout to embedding

    # ── Regularization ────────────────────────────────────────────────
    dropout: float = 0.0        # Dropout rate in Hebbian product
    lambda_perm: float = 0.01   # Permanence loss weight (drives delta² → 0)
    lambda_spectral: float = 0.001  # Spectral norm loss weight

    # ── Training ──────────────────────────────────────────────────────
    checkpoint_interval: int = 8    # Gradient checkpointing every N steps

    # ══════════════════════════════════════════════════════════════════
    # Derived Properties
    # ══════════════════════════════════════════════════════════════════

    @property
    def dilations(self) -> list[int]:
        """Cyclic multi-scale dilation schedule.

        Returns [1, 2, 4, 8, 16, 32, 64, 128] for n_scales=8.
        Every n_scales steps, each cell gets perception at ALL scales.
        After 64 steps = 8 complete cycles through all scales.
        """
        return [2 ** i for i in range(self.n_scales)]

    @property
    def a_bar_init(self) -> float:
        """Initial a_bar value — reachable settlement starting point.

        Set to tau + 3*eta so initial settled score ≈ 0.05 (clearly unsettled).
        V8 used 1e6 here and settlement NEVER activated. This is the fix.
        """
        return self.settlement_tau + 3 * self.settlement_eta

    @property
    def receptive_field(self) -> int:
        """Total receptive field after all steps.

        Each step adds (k-1) * dilation positions of reach.
        Over 64 steps (8 cycles of 8 scales), the total reach is:
        n_cycles × (k-1) × sum(dilations).
        """
        k = self.conv_kernel_size
        n_cycles = self.n_steps // self.n_scales
        return 1 + (k - 1) * n_cycles * sum(self.dilations)

    @property
    def total_params_estimate(self) -> int:
        """Estimated total parameter count."""
        d = self.d_model
        dr = self.d_rule
        V = self.vocab_size
        L = self.max_seq_len

        embed = V * d
        if not self.tie_embeddings:
            embed += V * d
        pe = L * d
        kernel = 3 * d * dr  # W_p + W_s + W_d
        conv = d * self.conv_kernel_size
        norms = 3 * d  # 2 kernel norms + final norm

        return embed + pe + kernel + conv + norms
