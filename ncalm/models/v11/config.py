"""AENCA V11 configuration — Morphogenetic Field NCA Language Model.

V11 architecture: two-field coupled CA with emergent information routing
and PERSISTENT MORPHOGEN SCAFFOLD for cross-step structural memory.

  V10 discovered that single-field local dynamics CANNOT produce selective
  information routing — every cell perceives neighbors identically. The fix
  is NOT to add attention. The fix is to add a SECOND COUPLED FIELD (the
  morphogen) that GATES content perception. Complex routing/hierarchy/
  attention-like behavior CAN emerge from the coupled local dynamics,
  without prescribing what that routing looks like.

  The Persistent Morphogen Scaffold adds EPIGENETIC MEMORY: beneficial
  morphogen configurations crystallize into a codebook of K templates
  via EMA across training steps. New forward passes seed from BOTH token
  content AND the accumulated scaffold — the organism inherits its
  lineage's epigenetic marks instead of rediscovering everything from
  scratch. Templates naturally prune (decay) and compress (merge).
  ZERO new learned parameters — templates evolve by simple EMA rule only.

Biological analogy:
  All cells share the same genome (shared rule). But cells DIFFERENTIATE
  because morphogen gradients in their environment control gene expression.
  V11 gives the NCA this same capability: morphogen h modulates how cells
  perceive each other's content z. The scaffold is the EPIGENOME: marks
  accumulated over the organism's life that bias how genes are expressed.

Physics analogy:
  Single-field systems tend toward trivial attractors. Coupled fields
  (reaction-diffusion, EM + matter) produce Turing patterns, solitons,
  self-organized criticality. The morphogen is the second field. The
  scaffold is the CONDENSATE: persistent order emerging from dynamics.

Forward pass (64 shared CA steps with two coupled fields + scaffold seeding):
    z = Embed(x) + PE                                   # Content seed (2048d)
    h_fresh = tanh(W_h_init · z)                         # Data-dependent seed (128d)
    scores = h_fresh @ M^T / τ                           # Affinity to K templates
    weights = softmax(scores, dim=-1)                    # Soft template selection
    h_scaffold = weights @ M                             # Blended persistent memory
    h = h_fresh + α · h_scaffold                         # Enriched seed
    settlement = SettlementState(...)                     # All cells active
    For t = 0..63 with dilation = 2^(t mod n_scales):
        gate = σ(SiLU(W_a · h) ⊙ W_b · h + bias)       # Bilinear morphogen gate
        perceived = DepthwiseCausalConv(z, dilation)     # Content neighborhood
        perceived = perceived ⊙ gate                     # GATED perception
        p = ReLU(W_p · RMSNorm(perceived))               # Perception encoding
        s = ReLU(W_s · RMSNorm(z))                       # Self encoding
        delta_z = W_d · (p ⊙ s)                          # Content update (Hebbian)
        h_nbr = DepthwiseCausalConv_h(h, dilation)       # Morphogen neighborhood
        delta_h = W_h2 · tanh(W_h1 · [h; h_nbr])        # Morphogen update (tiny MLP)
        growth = (1 - settled)                            # PERMANENCE gate
        z = z + delta_z · growth                         # Content grows
        h = h + delta_h · growth                         # Morphogen grows
        settlement.update(delta_z)                       # Track content convergence
    logits = RMSNorm(z) · Embed^T / √d                  # Content IS readout
    (after backward) EMA update: M ← decay·M + (1-decay)·settled_h

What CAN emerge (but is NOT designed):
  - Attention-like routing: morphogen states that match patterns, gating content
  - Hierarchical organization: morphogen gradients creating specialized regions
  - Oscillatory signaling: morphogens synchronizing distant cells
  - Content-based addressing: morphogen reflecting content type
  - Persistent cell types: scaffold templates encoding reusable morphogen configs
  - Something we haven't imagined: the whole point of not prescribing it
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class V11Config:
    """Complete configuration for AENCA V11: Morphogenetic Field NCA.

    Parameter budget (~110.6M at defaults):
        embed (tied)   = vocab_size × d_model            (~103M)
        pos_embed      = max_seq_len × d_model           (~2.1M)
        content kernel = 3 × d_model × d_rule            (~4.9M)
        content conv   = d_model × kernel_size           (~6K)
        content norms  = 3 × d_model                     (~6K)
        W_h_init       = d_model × d_h                   (~262K)   NEW
        W_gate_a/b     = 2 × d_h × d_model              (~1.05M)  BILINEAR
        gate_bias      = d_model                         (~2K)    BILINEAR
        W_h1           = 2·d_h × d_h + d_h (bias)        (~33K)   NEW
        W_h2           = d_h × d_h                       (~16K)   NEW
        morph conv     = d_h × kernel_size               (~384)    NEW
        morph norm     = 2·d_h                           (~256)    NEW
        ──────────────────────────────────────────────────────────
        Total          ≈ 110.6M (+575K over V10, 0.5%)

    The morphogen overhead is negligible. All emergence comes at inference cost
    in the coupled dynamics, not in parameter cost.
    """

    # ── Core Dimensions ───────────────────────────────────────────────
    d_model: int = 2048         # Content cell state width
    d_rule: int = 800           # Content rule concept space
    d_h: int = 128              # Morphogen field dimension (NEW)
    n_steps: int = 64           # CA iterations
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

    # ── Persistent Morphogen Scaffold (EPIGENETIC MEMORY) ─────────────
    # K morphogen templates crystallize from training via EMA. Each
    # forward pass seeds morphogen from BOTH token content AND the
    # accumulated scaffold. Templates prune (decay) and compress (merge).
    # ZERO new learned parameters — templates are register_buffer only.
    n_scaffold_templates: int = 64       # K — number of cell type prototypes
    scaffold_alpha: float = 0.3          # Blending weight: h = h_fresh + α·h_scaffold
    scaffold_temperature: float = 1.0    # Softmax temperature for template selection
    scaffold_ema_decay: float = 0.999    # EMA decay for template updates
    scaffold_warmup_steps: int = 200     # Steps before scaffold contributes to seeding
    scaffold_maintenance_interval: int = 500  # Steps between pruning/compression
    scaffold_min_norm: float = 0.01      # Templates below this norm are "dead"
    scaffold_merge_threshold: float = 0.95   # Cosine sim above this triggers merge

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
        """Estimated total parameter count including morphogen components."""
        d = self.d_model
        dr = self.d_rule
        dh = self.d_h
        V = self.vocab_size
        L = self.max_seq_len
        k = self.conv_kernel_size

        # ── V10 components ────────────────────────────────────────────
        embed = V * d
        if not self.tie_embeddings:
            embed += V * d
        pe = L * d
        content_kernel = 3 * d * dr  # W_p + W_s + W_d
        content_conv = d * k
        content_norms = 3 * d  # 2 kernel norms + final norm

        # ── Morphogen components (NEW) ────────────────────────────────
        w_h_init = d * dh              # Morphogen seed projection
        w_gate = 2 * dh * d + d        # Bilinear gate: 2 projections + bias
        w_h1 = 2 * dh * dh + dh       # MorphogenRule hidden: weight + bias
        w_h2 = dh * dh                 # MorphogenRule output: weight only
        morph_conv = dh * k            # Morphogen depthwise conv
        morph_norm = 2 * dh            # RMSNorm on concatenated morphogen input

        return (
            embed + pe + content_kernel + content_conv + content_norms
            + w_h_init + w_gate + w_h1 + w_h2 + morph_conv + morph_norm
        )
