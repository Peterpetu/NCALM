"""AENCA V12 configuration — Morphogenetic Mixture of Hebbian Rules (MMHR).

T-UoT Tier 3 redesign of V11. Challenged hidden assumption H1 ("one rule for all
cells") and H5 ("morphogen is secondary") to produce:

  1. K=8 parallel Hebbian rules with probabilistic per-cell routing (MNCA §3.1)
  2. d_h=512 morphogen as PRIMARY routing substrate — bilinear gate capacity
     d_h(d_h+1)/2 ≈ 131K independent patterns (vs 8K at d_h=128 in V11)
  3. Gumbel-softmax routing via W_router: d_h → K directed by morphogen
  4. Intrinsic noise per rule during training (MNCA §3.2)
  5. Step-annealed routing temperature: τ(t) = max(τ_min, τ_max − Δτ·t/T)
     → "attention growth": routing sharpens as morphogen matures across 64 steps
  6. Diversity loss −λ_div·H(weights) to prevent rule collapse (MNCA Appendix F)

MNCA paper mapping (arxiv 2506.20486):
  §3.1 Eq.4: z ~ Cat(π(s_i,η))  →  weights = Gumbel_softmax(W_router·h / τ)
  §3.1 Eq.4b: s_i^{t+1} = s_i^t + Σ_k φ_k^{z_k}  →  Δz = Σ_k w_k · δ_k
  §3.2 Eq.5b: x_k ~ N(0,1)  →  noise_sigma * randn added to Hebbian product
  Appendix F: mixture effective Lipschitz = Σ_k π_k · L_k ≤ max_k L_k

Parameter budget (~137M at defaults):
    embed (tied)         = vocab_size × d_model           (~103M)
    pos_embed            = max_seq_len × d_model           (~2.1M)
    shared W_p           = d_rule × d_model               (1.64M)
    K=8 binding pairs    = K × 2 × d_model × d_rule       (26.2M)
    bilinear gate        = 2 × d_h × d_model + d_model    ( 2.1M)
    morph rule (h1,h2)   = 2×d_h×d_h+d_h + d_h×d_h       ( 0.79M)
    morph seed W_h_init  = d_model × d_h                  ( 1.05M)
    router W_router      = d_h × K                        ( 4K)
    ──────────────────────────────────────────────────────────────────
    Non-embedding total  ≈ 31.8M    Grand total ≈ 137M
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class V12Config:
    """Complete configuration for AENCA V12: Morphogenetic Mixture of Hebbian Rules."""

    # ── Core Dimensions ───────────────────────────────────────────────
    d_model: int = 2048         # Content cell state width
    d_rule: int = 800           # Concept space (shared perception + per-rule binding)
    d_h: int = 512              # Morphogen dim — EXPANDED from V11's 128 to 512
    n_steps: int = 64           # CA iterations
    n_scales: int = 8           # Dilation scales per cycle

    # ── Causal Convolution ────────────────────────────────────────────
    conv_kernel_size: int = 3

    # ── Settlement (identical to V11) ─────────────────────────────────
    settlement_beta: float = 0.9
    settlement_tau: float = 0.05
    settlement_eta: float = 0.1

    # ── Vocabulary and Sequence ───────────────────────────────────────
    vocab_size: int = 50304
    max_seq_len: int = 1024
    tie_embeddings: bool = True

    # ── Regularization ────────────────────────────────────────────────
    dropout: float = 0.0
    lambda_perm: float = 0.01
    lambda_spectral: float = 0.001
    checkpoint_interval: int = 8

    # ── Persistent Morphogen Scaffold (identical logic to V11) ────────
    # Buffer sizes auto-scale with d_h — (K, d_h) = (64, 512) = 128KB
    n_scaffold_templates: int = 64
    scaffold_alpha: float = 0.3
    scaffold_temperature: float = 1.0
    scaffold_ema_decay: float = 0.999
    scaffold_warmup_steps: int = 200
    scaffold_maintenance_interval: int = 500
    scaffold_min_norm: float = 0.01
    scaffold_merge_threshold: float = 0.95

    # ── Mixture of Rules — NEW in V12 (MNCA §3.1 + §3.2) ────────────
    n_rules: int = 8
    # K parallel Hebbian rules. Shared perception W_p; K independent (W_s_k, W_d_k).
    # Routing: weights = Gumbel_softmax(W_router · h / τ(t))
    # where W_router: d_h→K is zero-initialised → uniform 1/K prior at start.

    noise_sigma: float = 0.02
    # Intrinsic Gaussian noise std added to Hebbian product per rule during training.
    # MNCA §3.2: x_k ~ N(0,1) fed into rule → captures rare stochastic events.
    # 0.02 is conservative: small relative to typical ||p ⊙ s|| after warm-up.

    route_temp_min: float = 0.1
    route_temp_max: float = 1.0
    # Gumbel-softmax temperature annealed over CA steps within each forward pass:
    #   τ(t) = max(route_temp_min, route_temp_max − (route_temp_max−route_temp_min)·t/T)
    # Step 0: τ=1.0 → nearly uniform weights (all rules contribute equally)
    # Step 63: τ=0.1 → sharp routing (specialised cells)
    # This IS the "attention growth" concept: routing sharpens as morphogen matures.

    lambda_div: float = 0.01
    # Weight for diversity loss: L_div = −mean(H(weights)) added to total loss.
    # H(w) = −Σ_k w_k log(w_k) ≥ 0. Maximising H → uniform routing → no rule collapse.
    # λ_div=0.01 same order as λ_perm — small enough not to dominate CE.

    # ══════════════════════════════════════════════════════════════════
    # Derived Properties
    # ══════════════════════════════════════════════════════════════════

    @property
    def dilations(self) -> list[int]:
        """Cyclic multi-scale dilation schedule: [1,2,4,8,16,32,64,128] for n_scales=8."""
        return [2 ** i for i in range(self.n_scales)]

    @property
    def a_bar_init(self) -> float:
        """Starting a_bar giving clearly-unsettled initial settlement score ≈ 0.047."""
        return self.settlement_tau + 3 * self.settlement_eta

    @property
    def receptive_field(self) -> int:
        """Total receptive field after all steps."""
        k = self.conv_kernel_size
        n_cycles = self.n_steps // self.n_scales
        return 1 + (k - 1) * n_cycles * sum(self.dilations)

    @property
    def total_params_estimate(self) -> int:
        """Estimated parameter count (excludes scaffold buffers, no-grad params)."""
        d = self.d_model
        dr = self.d_rule
        dh = self.d_h
        K_rules = self.n_rules
        V = self.vocab_size
        L = self.max_seq_len
        k = self.conv_kernel_size

        embed = V * d
        if not self.tie_embeddings:
            embed += V * d
        pe = L * d

        shared_wp = dr * d          # W_p: Linear(d_model, d_rule)
        binding = K_rules * 2 * d * dr  # K × (W_s_k + W_d_k), each (d_model×d_rule)
        content_conv = d * k
        content_norms = 3 * d       # norm_perceived, norm_z, final_norm

        router = dh * K_rules       # W_router: Linear(d_h, K_rules, bias=False)
        w_h_init = d * dh           # morphogen seed projection
        w_gate = 2 * dh * d + d     # bilinear gate: W_gate_a + W_gate_b + bias
        w_h1 = 2 * dh * dh + dh    # MorphogenRule hidden: weight + bias
        w_h2 = dh * dh             # MorphogenRule output (zero-init)
        morph_conv = dh * k
        morph_norm = 2 * dh

        return (
            embed + pe
            + shared_wp + binding + content_conv + content_norms
            + router
            + w_h_init + w_gate + w_h1 + w_h2 + morph_conv + morph_norm
        )
