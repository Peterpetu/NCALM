"""AENCA V11 model — Morphogenetic Field NCA Language Model.

Two-field coupled CA with emergent information routing and PERSISTENT
MORPHOGEN SCAFFOLD. V11 adds a morphogen field h that GATES content
perception, creating a substrate where global information routing CAN
emerge from local-only dynamics. The scaffold provides EPIGENETIC MEMORY:
beneficial morphogen configurations crystallize into persistent templates
that accumulate across training steps via EMA.

Architecture:
    SEED:     z = Embed(x) + PE                      ∈ R^{B × L × 2048}
              h_fresh = tanh(W_h_init · z)           ∈ R^{B × L × 128}
              scores = h_fresh @ M^T / τ             ∈ R^{B × L × K}
              weights = softmax(scores)              ∈ R^{B × L × K}
              h_scaffold = weights @ M               ∈ R^{B × L × 128}
              h = h_fresh + α · h_scaffold           ← persistent memory enrichment
              SettlementState initialized (all cells active)

    64 STEPS: For t = 0..63 with dilation = 2^(t mod 8):
                gate = σ(SiLU(W_a·h) ⊙ W_b·h + bias)  ← bilinear morphogen gate
                delta_z = HebbianRule(z, dilation, gate) ← GATED perception
                delta_h = MorphogenRule(h, dilation)    ← morphogen evolves
                growth = (1 - settled)
                z = z + delta_z * growth                ← content grows
                h = h + delta_h * growth                ← morphogen grows
                settlement.update(delta_z)              ← track content convergence

    READOUT:  logits = RMSNorm(z) · Embed^T / √d      ← content = readout

    AFTER BACKWARD (on_step_end):
              M ← decay · M + (1-decay) · settled_weighted_mean(h_final)
              Prune dead templates (norm < min_norm) → re-init from alive
              Merge similar templates (cosine > threshold) → free slots

Key properties:
  - TWO coupled fields: content z (2048d) + morphogen h (d_h)
  - BILINEAR morphogen gate → full-rank gating from compact dynamics
  - PERSISTENT SCAFFOLD: K=64 morphogen templates accumulate via EMA
  - One shared content rule + one shared morphogen rule
  - 64 CA steps with cyclic dilations (same as V10)
  - Settlement tracks content only; both fields gated by growth_gate
  - Gate bias = +3.0 → σ(3) ≈ 0.95 at init → V11 starts ≈ V10

Nine Axioms of True NCA (V11 extends V10):
  1. GROWTH: z and h develop from seed through 64 iterative steps
  2. PERMANENCE: settled cells ARE frozen permanent memory
  3. SPARSITY: (1-settled) gate → shrinking active frontier
  4. SIMPLICITY: TWO shared tiny rules — the coupled genome
  5. UNIVERSALITY: spectral norm bounds dynamics to Class IV
  6. STRUCTURE=FUNCTION: grown z with permanent regions IS the representation
  7. SELF-ORG: training shapes the coupled rules
  8. DIFFERENTIATION: morphogen field enables cells to SPECIALIZE
  9. INHERITANCE: scaffold passes beneficial configurations to future passes (NEW)

Loss function (3 terms, same as V10):
  L = L_CE + λ_perm · mean(delta_z²) + λ_spectral · σ_max(content_rule)
"""

from __future__ import annotations

import math
from dataclasses import fields as dataclass_fields
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from ncalm.model import NCALMModel, register_model
from ncalm.models.v11.config import V11Config
from ncalm.models.v11.kernel import SimpleHebbianRule, MorphogenRule
from ncalm.models.v11.settlement import SettlementState


@register_model("v11")
class AENCAv11Model(NCALMModel):
    """Morphogenetic Field NCA Language Model with Persistent Morphogen Scaffold.

    Two-field coupled CA: content z + morphogen h. The morphogen gates
    content perception, creating a substrate for emergent information routing.
    No attention. All routing emerges from coupled local dynamics.

    The Persistent Morphogen Scaffold provides EPIGENETIC MEMORY: K templates
    of successful morphogen configurations accumulate via EMA across training.
    Each forward pass seeds morphogen from BOTH token content AND scaffold.
    Templates prune (decay) and compress (merge). Zero new learned parameters.

    Implements NCALMModel interface for drop-in use with the NCALM
    training infrastructure (Trainer, scheduler, data pipeline).
    """

    def __init__(self, config_dict: dict) -> None:
        super().__init__()

        # ── Parse config ─────────────────────────────────────────────
        v11_keys = {f.name for f in dataclass_fields(V11Config)}
        v11_params = {k: v for k, v in config_dict.items() if k in v11_keys}
        self.config = V11Config(**v11_params)
        cfg = self.config

        # ── Token Embedding ──────────────────────────────────────────
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # ── Positional Embedding (learned, GPT-2 style) ─────────────
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        # ── Content Genome: ONE shared tiny Hebbian rule ─────────────
        self.rule = SimpleHebbianRule(cfg)

        # ── Morphogen Genome: ONE shared tiny MLP rule ───────────────
        self.morph_rule = MorphogenRule(cfg)

        # ── Morphogen Seed Projection: z → h₀ ───────────────────────
        # Seeds the morphogen from content, providing initial symmetry-
        # breaking. Different tokens → different initial morphogens →
        # different perception filters from step 1.
        self.W_h_init = nn.Linear(cfg.d_model, cfg.d_h, bias=False)

        # ── Bilinear Morphogen Gate: h → gate ∈ [0, 1]^d_model ──────
        # The KEY mechanism. Morphogen controls what each cell perceives.
        # BILINEAR PRODUCT: gate = σ(SiLU(h·W_a) ⊙ h·W_b + bias)
        # Two d_h→d_model projections with element-wise product create
        # QUADRATIC forms: d_h(d_h+1)/2 independent patterns >> d_model.
        # This gives FULL-RANK gating from a compact morphogen field,
        # while keeping dynamics MLP tiny (operates in d_h space only).
        # Bias=+3.0 → gate ≈ 0.95 at init (V11 ≈ V10 start).
        self.W_gate_a = nn.Linear(cfg.d_h, cfg.d_model, bias=False)
        self.W_gate_b = nn.Linear(cfg.d_h, cfg.d_model, bias=False)
        self.gate_bias = nn.Parameter(torch.empty(cfg.d_model))

        # ── Final RMSNorm (before readout, on content only) ──────────
        self.final_norm = nn.RMSNorm(cfg.d_model)

        # ── LM Head (tied or separate) ──────────────────────────────
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        else:
            self.lm_head = None

        # ── Spectral norm tracking (content rule only) ───────────────
        self._register_spectral_buffers()

        # ── Persistent Morphogen Scaffold (EPIGENETIC MEMORY) ────────
        # K templates of cell type prototypes. NOT learned parameters —
        # they evolve by EMA across training steps.
        # register_buffer(persistent=True) → saved in state_dict and
        # checkpoints, restored on load. This IS the persistence.
        self._register_scaffold_buffers()

        # ── Transient scaffold state (NOT persistent — per forward) ──
        # Accumulated across micro-batches during gradient accumulation,
        # consumed by on_step_end for EMA. Not saved in checkpoints.
        self._last_h_final: list[torch.Tensor] = []
        self._last_settled: list[torch.Tensor] = []

        # ── Cached scaffold step (avoid GPU→CPU sync in forward) ─────
        self._scaffold_step_val: int = 0

        # ── Pre-compute dilation schedule ────────────────────────────
        self._dilation_schedule = self._build_dilation_schedule()

        # ── Weight initialization ────────────────────────────────────
        self._init_weights()

    def _build_dilation_schedule(self) -> list[int]:
        """Build the full 64-step dilation schedule.

        Cyclic: [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, ...]
        Every n_scales steps covers all scales once.
        """
        dilations = self.config.dilations
        return [dilations[t % len(dilations)] for t in range(self.config.n_steps)]

    def _register_spectral_buffers(self) -> None:
        """Register power iteration vectors for spectral norm estimation.

        Only tracks the content rule weights (W_p, W_s, W_d). The morphogen
        rule is already bounded by tanh activation and zero-init, and its
        parameters are too small (~49K) for spectral norm to matter.
        """
        for name in ("W_p", "W_s", "W_d"):
            W = getattr(self.rule, name).weight
            u = torch.randn(W.shape[0])
            u = u / u.norm()
            self.register_buffer(f"_spectral_u_{name}", u, persistent=False)

    def _register_scaffold_buffers(self) -> None:
        """Register persistent scaffold buffers for morphogen template memory.

        Three buffers, ALL persistent=True (saved in checkpoints):

        1. scaffold_templates: (K, d_h) — morphogen cell type prototypes.
           Initialized near zero so scaffold has no effect at start.
           Updated by EMA in on_step_end, NOT by gradient descent.

        2. scaffold_usage: (K,) — tracks cumulative usage per template.
           Each step, usage for assigned templates increments by 1.
           Used for: pruning decisions, merge weighting, monitoring.

        3. scaffold_step: scalar int64 — counts training steps for
           warmup gating and maintenance scheduling.

        Memory overhead at K=64, d_h=128:
            Templates: 64 × 128 × 4B = 32KB
            Usage:     64 × 4B = 256B
            Step:      8B
            Total:     ~32KB persistent
        """
        cfg = self.config
        K = cfg.n_scaffold_templates
        d_h = cfg.d_h

        # Templates: near-zero init so scaffold has zero effect initially.
        # As training progresses, EMA fills templates with crystallized
        # morphogen configurations from settled cells.
        self.register_buffer(
            "scaffold_templates",
            torch.zeros(K, d_h),
            persistent=True,
        )

        # Usage: tracks how many cells are assigned to each template.
        # Accumulated across all on_step_end calls. Used for pruning
        # (zero-usage templates are dead) and merge weighting.
        self.register_buffer(
            "scaffold_usage",
            torch.zeros(K),
            persistent=True,
        )

        # Step counter: tracks training steps for warmup and maintenance.
        # Int64 to avoid float precision issues over long training runs.
        self.register_buffer(
            "scaffold_step",
            torch.zeros(1, dtype=torch.long),
            persistent=True,
        )

    def _init_weights(self) -> None:
        """Initialize weights for embeddings and morphogen-specific layers.

        Content rule and morphogen rule handle their own init internally.
        """
        # Embeddings (same as V10)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.01)
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        # Morphogen seed projection: Xavier for good gradient flow
        nn.init.xavier_normal_(self.W_h_init.weight, gain=1.0)

        # Bilinear morphogen gate: small random weights + bias=+3.0
        # At init: SiLU(small) × small ≈ 0, so gate ≈ σ(bias) = σ(3.0) ≈ 0.95.
        # Non-zero init ensures gradients flow through the product from step 1.
        nn.init.normal_(self.W_gate_a.weight, std=0.01)
        nn.init.normal_(self.W_gate_b.weight, std=0.01)
        nn.init.constant_(self.gate_bias, 3.0)

    def _get_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Convert final cell states to vocabulary logits.

        Tied embeddings with 1/√d scaling to prevent self-dot-product explosion.
        Only content z is used for readout — morphogen is internal signaling.
        """
        z = self.final_norm(z)
        if self.lm_head is not None:
            return self.lm_head(z)
        return F.linear(z, self.token_embedding.weight) * (1.0 / math.sqrt(self.config.d_model))

    def _spectral_norm_loss(self) -> torch.Tensor:
        """Spectral norms of content rule weights via power iteration.

        Forced to fp32 to avoid epsilon underflow in fp16.
        Bounds the content rule's Lipschitz constant → prevents Class III divergence.

        Only applied to content rule (W_p, W_s, W_d). Morphogen rule is
        bounded by tanh activation and tiny parameter count.
        """
        total = torch.tensor(0.0, device=self.token_embedding.weight.device)

        for name in ("W_p", "W_s", "W_d"):
            W = getattr(self.rule, name).weight.float()
            u = getattr(self, f"_spectral_u_{name}").float()

            with torch.no_grad():
                v = W.t() @ u
                v_norm = v.norm()
                if v_norm > 1e-8:
                    v = v / v_norm
                    u_new = W @ v
                    u_new = u_new / (u_new.norm() + 1e-8)
                    getattr(self, f"_spectral_u_{name}").copy_(u_new)
                else:
                    continue

            sigma = u_new @ (W @ v)
            total = total + sigma

        return total / 3.0

    def _ca_step(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        dilation: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single coupled CA step: content + morphogen update proposals.

        This is the checkpointable unit. All computation for one CA step
        happens here, returning the raw deltas before growth gating.

        The gate computation is the CORE of V11: morphogen h controls
        how cells perceive each other's content z. This is where emergent
        routing happens — but we don't prescribe what routing forms.

        Args:
            z: (B, L, d_model) current content state
            h: (B, L, d_h) current morphogen state
            dilation: Conv dilation for this step

        Returns:
            delta_z: (B, L, d_model) content update proposal
            delta_h: (B, L, d_h) morphogen update proposal
        """
        # ── Bilinear Morphogen → perception gate ────────────────────
        # SwiGLU-style factored projection: gate = σ(SiLU(h·W_a) ⊙ h·W_b + bias)
        # The element-wise product of two rank-d_h projections creates
        # d_h(d_h+1)/2 independent quadratic patterns — effective FULL RANK
        # over d_model from a compact d_h-dimensional morphogen.
        gate = torch.sigmoid(
            F.silu(self.W_gate_a(h)) * self.W_gate_b(h) + self.gate_bias
        )  # (B, L, d_model)

        # ── Content update with morphogen-gated perception ────────────
        delta_z = self.rule(z, dilation=dilation, gate=gate)  # (B, L, d_model)

        # ── Morphogen update via its own local rule ──────────────────
        delta_h = self.morph_rule(h, dilation=dilation)  # (B, L, d_h)

        return delta_z, delta_h

    def compute_gate(self, h: torch.Tensor) -> torch.Tensor:
        """Compute bilinear morphogen gate from morphogen state.

        gate = σ(SiLU(h·W_a) ⊙ h·W_b + bias)

        Exposed as a method so analysis scripts can call it without
        reimplementing the gate logic.
        """
        return torch.sigmoid(
            F.silu(self.W_gate_a(h)) * self.W_gate_b(h) + self.gate_bias
        )

    def _seed_morphogen(self, z: torch.Tensor) -> torch.Tensor:
        """Seed morphogen field from content AND persistent scaffold.

        The morphogen seed is a combination of:
          1. h_fresh = tanh(W_h_init · z) — data-dependent symmetry-breaking
          2. h_scaffold = softmax(h_fresh @ M^T / τ) @ M — persistent memory

        Combined: h = h_fresh + α · h_scaffold

        Before warmup completes (scaffold_step < scaffold_warmup_steps),
        α is ramped linearly from 0 to scaffold_alpha. When scaffold
        templates are near-zero (early training), h_scaffold ≈ 0 regardless.

        Gradient flows through h_fresh and the softmax weights (enabling
        W_h_init to learn to utilize scaffold). Templates M are buffers
        with no gradient — they evolve ONLY by EMA in on_step_end.

        Args:
            z: (B, L, d_model) content seed from embedding

        Returns:
            h: (B, L, d_h) scaffold-enriched morphogen seed
        """
        cfg = self.config

        # ── Data-dependent seed (always computed) ────────────────────
        h_fresh = torch.tanh(self.W_h_init(z))  # (B, L, d_h)

        # ── Check if scaffold should contribute ──────────────────────
        # Three conditions for scaffold to be inactive:
        # 1. α = 0 (scaffold disabled by config)
        # 2. Templates all zero (early training, nothing crystallized yet)
        # 3. Still in warmup
        scaffold_active = (
            cfg.scaffold_alpha > 0
            and cfg.n_scaffold_templates > 0
        )
        if not scaffold_active:
            return h_fresh

        # Check template norms — if all near zero, skip (no-op)
        template_norms = self.scaffold_templates.norm(dim=-1)  # (K,)
        if (template_norms.max() < 1e-8).item():
            return h_fresh

        # ── Warmup ramp: linearly increase α from 0 → scaffold_alpha ─
        step = int(self.scaffold_step.item())
        if step < cfg.scaffold_warmup_steps:
            alpha = cfg.scaffold_alpha * (step / max(cfg.scaffold_warmup_steps, 1))
        else:
            alpha = cfg.scaffold_alpha

        if alpha < 1e-8:
            return h_fresh

        # ── Soft template selection ──────────────────────────────────
        # Each cell computes affinity to every template, then selects
        # a blended combination. This is differentiable through h_fresh.
        #
        # Using COSINE similarity (not raw dot product) to prevent
        # high-norm templates from dominating selection. Temperature
        # scaled by √d_h for attention-like behavior.
        #
        # scores: (B, L, K) = normalize(h_fresh) @ normalize(M)^T / τ_scaled
        # weights: (B, L, K) = softmax(scores, dim=-1)
        # h_scaffold: (B, L, d_h) = weights @ M
        M = self.scaffold_templates  # (K, d_h)
        h_normed = F.normalize(h_fresh, dim=-1)       # (B, L, d_h)
        M_normed = F.normalize(M, dim=-1)              # (K, d_h)
        tau_scaled = cfg.scaffold_temperature * math.sqrt(cfg.d_h)
        scores = torch.matmul(h_normed, M_normed.t()) / tau_scaled  # (B, L, K)
        weights = F.softmax(scores, dim=-1)  # (B, L, K)
        h_scaffold = torch.matmul(weights, M)  # (B, L, d_h)

        # ── Combine: enriched seed ───────────────────────────────────
        return h_fresh + alpha * h_scaffold

    def _run_ca_steps(
        self,
        z: torch.Tensor,
        accumulate_perm: bool = False,
    ) -> tuple[torch.Tensor, SettlementState, torch.Tensor]:
        """Run the 64-step coupled CA growth process.

        Two fields evolve together:
          - Content z: Hebbian rule with morphogen-gated perception
          - Morphogen h: tiny MLP evolving the attention substrate

        Morphogen is seeded from BOTH token content AND the persistent
        scaffold (when available). Settlement tracks content convergence.
        Both fields are gated by the same growth_gate — when content
        converges, morphogen freezes too.

        After the CA loop, h_final and settled scores are saved (detached)
        for the on_step_end hook to perform scaffold EMA update.

        Permanence loss uses ONLY the last step's delta_z in the gradient graph.
        Earlier steps' delta_z² are accumulated detached (for monitoring).
        This prevents the perm loss from defeating gradient checkpointing.

        Args:
            z: (B, L, d_model) seed cell states (from embedding)
            accumulate_perm: If True, compute permanence loss

        Returns:
            z: (B, L, d_model) final grown content states
            settlement: Final SettlementState with settlement scores
            perm_loss: Scalar permanence loss (gradient only from last step)
        """
        B, L, _ = z.shape
        cfg = self.config

        # ── Seed morphogen from content + persistent scaffold ────────
        # h = h_fresh + α · h_scaffold (when scaffold is active)
        # h = h_fresh (when scaffold is inactive / warmup / templates zero)
        h = self._seed_morphogen(z)  # (B, L, d_h)

        # ── Initialize settlement state (all cells active) ───────────
        settlement = SettlementState.create(B, L, z.device, cfg)
        perm_accum = 0.0  # Detached running sum for monitoring
        last_delta_z = None
        use_ckpt = cfg.checkpoint_interval > 0 and self.training

        for t, dilation in enumerate(self._dilation_schedule):
            # ── Compute coupled update proposals ─────────────────────
            if use_ckpt and t > 0 and t % cfg.checkpoint_interval == 0:
                delta_z, delta_h = grad_checkpoint(
                    self._ca_step, z, h, dilation, use_reentrant=False
                )
            else:
                delta_z, delta_h = self._ca_step(z, h, dilation)

            # ── PERMANENCE: settled cells get zero update on BOTH fields ─
            growth = settlement.growth_gate().unsqueeze(-1)  # (B, L, 1)
            z = z + delta_z * growth
            h = h + delta_h * growth

            # ── Track content convergence (detached from gradient graph) ─
            settlement.update(delta_z, cfg)

            # ── Permanence loss: accumulate detached, keep last for grad ─
            if accumulate_perm:
                perm_accum += delta_z.detach().pow(2).mean().item()
                last_delta_z = delta_z

        # ── Save h_final and settled for scaffold EMA (on_step_end) ──
        # Detached: scaffold update is NOT part of the gradient graph.
        # Only saved during training — inference doesn't update scaffold.
        # ACCUMULATED across micro-batches (gradient accumulation): the
        # list grows with each forward_train call and is consumed/cleared
        # by on_step_end after the optimizer step.
        if self.training:
            self._last_h_final.append(h.detach())
            self._last_settled.append(settlement.settled.detach())
        else:
            self._last_h_final = []
            self._last_settled = []

        # Permanence loss: gradient from last step's content delta only (VRAM-safe)
        perm_loss = torch.tensor(0.0, device=z.device)
        if accumulate_perm and last_delta_z is not None:
            perm_loss = last_delta_z.pow(2).mean()

        return z, settlement, perm_loss

    # ═══════════════════════════════════════════════════════════════════
    # NCALMModel Interface
    # ═══════════════════════════════════════════════════════════════════

    def forward_train(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass with loss computation.

        Args:
            batch: (B, L) token IDs

        Returns:
            loss: Scalar — total training loss
            info: Dict with 'ce', 'perm', 'spectral', 'settled', 'active' keys
        """
        B, L = batch.shape
        cfg = self.config

        # ── Seed: embed tokens + positions ───────────────────────────
        positions = torch.arange(L, device=batch.device)
        z = self.token_embedding(batch) + self.pos_embedding(positions)

        # ── 64-step coupled CA growth with settlement ────────────────
        z, settlement, loss_perm = self._run_ca_steps(
            z, accumulate_perm=(cfg.lambda_perm > 0)
        )

        # ── Readout: grown content state → logits ────────────────────
        logits = self._get_logits(z)

        # ── Loss 1: Cross-Entropy (primary) ──────────────────────────
        loss_ce = F.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.vocab_size),
            batch[:, 1:].reshape(-1),
            reduction="mean",
        )

        # ── Loss 2: Permanence (accumulated online in _run_ca_steps) ─

        # ── Loss 3: Spectral norm (bounds content rule dynamics) ─────
        loss_spectral = torch.tensor(0.0, device=batch.device)
        if cfg.lambda_spectral > 0:
            loss_spectral = self._spectral_norm_loss()

        # ── Total loss ───────────────────────────────────────────────
        total_loss = (
            loss_ce
            + cfg.lambda_perm * loss_perm
            + cfg.lambda_spectral * loss_spectral
        )

        info = {
            "ce": loss_ce.detach(),
            "perm": loss_perm.detach() if isinstance(loss_perm, torch.Tensor) else loss_perm,
            "spectral": loss_spectral.detach() if isinstance(loss_spectral, torch.Tensor) else loss_spectral,
            "settled": settlement.mean_settled,
            "active": settlement.active_fraction,
        }

        return total_loss, info

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference/generation."""
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device)
        z = self.token_embedding(input_ids) + self.pos_embedding(positions)
        z, _, _ = self._run_ca_steps(z, accumulate_perm=False)
        return self._get_logits(z)

    def get_param_groups(self) -> list[dict]:
        """Parameter groups for AdamW optimizer.

        Decay: all 2D+ weight matrices (content rule, morphogen rule, gate, etc.)
        No decay: embeddings, norms, biases, 1D parameters
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() < 2 or "norm" in name or "embedding" in name:
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
        pe_params = self.pos_embedding.weight.numel()
        content_rule_params = sum(p.numel() for p in self.rule.parameters())
        morph_rule_params = sum(p.numel() for p in self.morph_rule.parameters())
        morph_gate_params = (
            sum(p.numel() for p in self.W_gate_a.parameters())
            + sum(p.numel() for p in self.W_gate_b.parameters())
            + self.gate_bias.numel()
        )
        morph_seed_params = self.W_h_init.weight.numel()
        norm_params = sum(p.numel() for p in self.final_norm.parameters())
        head_params = self.lm_head.weight.numel() if self.lm_head else 0

        morphogen_total = morph_rule_params + morph_gate_params + morph_seed_params

        # Scaffold buffer stats
        scaffold_template_count = self.scaffold_templates.numel()
        scaffold_alive = int((self.scaffold_templates.norm(dim=-1) >= cfg.scaffold_min_norm).sum().item())

        return {
            "architecture": "AENCA-V11-MorphogeneticField-Scaffold",
            "total_params": total,
            "embed_params": embed_params,
            "pe_params": pe_params,
            "content_rule_params": content_rule_params,
            "morphogen_params": morphogen_total,
            "morph_rule_params": morph_rule_params,
            "morph_gate_params": morph_gate_params,
            "morph_seed_params": morph_seed_params,
            "norm_params": norm_params,
            "head_params": head_params,
            "d_model": cfg.d_model,
            "d_rule": cfg.d_rule,
            "d_h": cfg.d_h,
            "n_steps": cfg.n_steps,
            "n_scales": cfg.n_scales,
            "conv_kernel_size": cfg.conv_kernel_size,
            "dilation_schedule": self._dilation_schedule[:cfg.n_scales],
            "receptive_field": cfg.receptive_field,
            "vocab_size": cfg.vocab_size,
            "max_seq_len": cfg.max_seq_len,
            "tie_embeddings": cfg.tie_embeddings,
            "settlement_beta": cfg.settlement_beta,
            "settlement_tau": cfg.settlement_tau,
            "settlement_eta": cfg.settlement_eta,
            "lambda_perm": cfg.lambda_perm,
            "lambda_spectral": cfg.lambda_spectral,
            "gate_bias_init": 3.0,
            # ── Scaffold info ────────────────────────────────────────
            "scaffold_templates_K": cfg.n_scaffold_templates,
            "scaffold_alpha": cfg.scaffold_alpha,
            "scaffold_temperature": cfg.scaffold_temperature,
            "scaffold_ema_decay": cfg.scaffold_ema_decay,
            "scaffold_warmup_steps": cfg.scaffold_warmup_steps,
            "scaffold_maintenance_interval": cfg.scaffold_maintenance_interval,
            "scaffold_buffer_elements": scaffold_template_count,
            "scaffold_alive_at_init": scaffold_alive,
            "paradigm": (
                "Morphogenetic Field NCA + Persistent Scaffold — TWO coupled rules "
                f"(content {content_rule_params / 1e6:.1f}M + "
                f"morphogen {morphogen_total / 1e3:.0f}K), "
                f"{cfg.n_steps} growth steps, d_h={cfg.d_h}, "
                f"K={cfg.n_scaffold_templates} scaffold templates (EMA), "
                "emergent routing via morphogen-gated perception, "
                "epigenetic memory, settlement-based permanence, no attention"
            ),
        }

    def get_phases(self, step: int, max_steps: int) -> str:
        """Single-phase coupled CA growth."""
        return "CA"

    # ═══════════════════════════════════════════════════════════════════
    # Scaffold Lifecycle Hooks (called by Trainer)
    # ═══════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def on_step_end(self, step: int, metrics: dict) -> None:
        """Update persistent morphogen scaffold after each training step.

        Called by the Trainer after optimizer.step(). All operations are
        detached — the scaffold evolves by EMA, NOT gradient descent.

        Three operations (in order):
          1. EMA UPDATE: crystallize observed morphogen states into templates
          2. PRUNING: recycle dead templates (norm < min_norm)
          3. COMPRESSION: merge duplicate templates (cosine > threshold)

        Settlement-weighted EMA with SOFT assignment: cells contribute to
        ALL templates proportional to cosine similarity, not just their
        argmax winner. This prevents winner-take-all template collapse.

        Morphogen states from ALL gradient accumulation micro-batches are
        concatenated before the update, ensuring the scaffold sees the full
        effective batch.

        Args:
            step: Current training step (0-indexed)
            metrics: Metrics dict from forward_train (not used here)
        """
        cfg = self.config

        # ── Increment scaffold step counter ──────────────────────────
        self.scaffold_step += 1
        self._scaffold_step_val = int(self.scaffold_step.item())

        # ── Skip if scaffold is disabled ─────────────────────────────
        if cfg.n_scaffold_templates == 0 or cfg.scaffold_alpha == 0:
            self._last_h_final = []
            self._last_settled = []
            return

        # ── Retrieve saved morphogen states from ALL micro-batches ───
        h_final_list = self._last_h_final    # list of (B, L, d_h)
        settled_list = self._last_settled     # list of (B, L)

        if not h_final_list or not settled_list:
            return

        # Concatenate all micro-batches → full effective batch
        h_final = torch.cat(h_final_list, dim=0)     # (B_total, L, d_h)
        settled = torch.cat(settled_list, dim=0)       # (B_total, L)

        # Free transient state immediately after retrieval
        self._last_h_final = []
        self._last_settled = []

        M = self.scaffold_templates  # (K, d_h)
        K = cfg.n_scaffold_templates
        decay = cfg.scaffold_ema_decay

        # ── 1. EMA UPDATE: crystallize morphogen into templates ──────
        # Flatten batch and sequence: (B_total*L, d_h)
        h_flat = h_final.reshape(-1, cfg.d_h)       # (N, d_h) where N = B_total*L
        w_flat = settled.reshape(-1)                 # (N,) settlement weights

        # First update: initialize templates with orthogonal directions
        # at the scale of observed h vectors (avoids 1D collapse)
        if (M.norm() < 1e-8).item():
            target_norm = h_flat.norm(dim=-1).mean()
            if target_norm < 1e-8:
                return
            templates = torch.randn(K, cfg.d_h, device=M.device)
            if K <= cfg.d_h:
                Q, _ = torch.linalg.qr(templates.t())
                templates = Q.t()[:K]
            templates = F.normalize(templates, dim=-1) * target_norm
            M.copy_(templates)
            self.scaffold_usage.zero_()
            return

        # Cosine-similarity soft assignment (prevents winner-take-all)
        # Each cell contributes to ALL templates proportionally to similarity
        h_normed = F.normalize(h_flat, dim=-1)         # (N, d_h)
        M_normed = F.normalize(M, dim=-1)              # (K, d_h)
        tau_scaled = cfg.scaffold_temperature * math.sqrt(cfg.d_h)
        affinities = torch.matmul(h_normed, M_normed.t()) / tau_scaled  # (N, K)
        soft_weights = F.softmax(affinities, dim=-1)   # (N, K)

        # Decay usage tracking (prevents stale counts from dominating)
        self.scaffold_usage *= decay

        # For each template: settlement-weighted soft update
        for k in range(K):
            # Soft weight for this template × settlement weight
            w_k = soft_weights[:, k] * w_flat         # (N,)
            w_sum = w_k.sum()

            if w_sum < 1e-8:
                # No effective contribution → natural decay
                M[k] *= decay
                continue

            # Settlement-weighted soft mean of all cells' morphogen states
            h_mean = (h_flat * w_k.unsqueeze(-1)).sum(dim=0) / w_sum  # (d_h,)

            # EMA update: blend old template with observed mean
            M[k] = decay * M[k] + (1.0 - decay) * h_mean

            # Update usage counter (soft count)
            self.scaffold_usage[k] += w_sum

        # ── 2. PRUNING: recycle dead templates ───────────────────────
        # Only run at maintenance intervals to avoid overhead
        current_step = self.scaffold_step.item()
        if current_step % cfg.scaffold_maintenance_interval == 0:
            self._scaffold_prune()
            self._scaffold_compress()

    @torch.no_grad()
    def _scaffold_prune(self) -> None:
        """Prune dead templates and re-initialize from alive ones.

        A template is "dead" when its norm falls below scaffold_min_norm.
        This happens naturally: templates that receive no cell assignments
        decay by factor scaffold_ema_decay each step. After enough steps
        without use, they shrink below the threshold.

        Dead templates are re-initialized by copying a random alive template
        and adding ORTHOGONAL perturbation at 30% of the donor's norm.
        This ensures recycled templates explore genuinely new directions
        rather than immediately collapsing back to the donor.
        """
        cfg = self.config
        M = self.scaffold_templates
        norms = M.norm(dim=-1)  # (K,)
        alive_mask = norms >= cfg.scaffold_min_norm  # (K,) bool
        dead_mask = ~alive_mask

        n_dead = dead_mask.sum().item()
        n_alive = alive_mask.sum().item()

        if n_dead == 0 or n_alive == 0:
            return

        # Get indices
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]

        # Re-initialize each dead template with orthogonal perturbation
        for dead_idx in dead_indices:
            donor_idx = alive_indices[torch.randint(n_alive, (1,), device=M.device)]
            donor = M[donor_idx].squeeze(0)
            # Orthogonal perturbation: project out the donor direction
            noise = torch.randn_like(donor)
            donor_norm_sq = donor.dot(donor) + 1e-8
            noise = noise - donor * (noise.dot(donor) / donor_norm_sq)
            noise = F.normalize(noise, dim=-1) * donor.norm() * 0.3
            M[dead_idx] = donor + noise
            self.scaffold_usage[dead_idx] = 0.0

    @torch.no_grad()
    def _scaffold_compress(self) -> None:
        """Merge similar templates to prevent redundancy and free capacity.

        Computes pairwise cosine similarity of all alive templates.
        When cos(m_i, m_j) > scaffold_merge_threshold, the two templates
        are merged into one via usage-weighted average. The freed slot is
        re-initialized from a random alive template + noise.

        This is O(K²) but K is small (64) so it's negligible.
        Run only at maintenance intervals (every scaffold_maintenance_interval steps).
        """
        cfg = self.config
        M = self.scaffold_templates
        K = cfg.n_scaffold_templates
        norms = M.norm(dim=-1)  # (K,)
        alive_mask = norms >= cfg.scaffold_min_norm

        # Need at least 2 alive templates to merge
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        n_alive = alive_indices.shape[0]
        if n_alive < 2:
            return

        # Compute pairwise cosine similarity among alive templates
        alive_templates = M[alive_indices]  # (n_alive, d_h)
        alive_norms = norms[alive_indices].unsqueeze(-1).clamp(min=1e-8)  # (n_alive, 1)
        normed = alive_templates / alive_norms  # (n_alive, d_h)
        cosine_sim = torch.matmul(normed, normed.t())  # (n_alive, n_alive)

        # Zero out diagonal and lower triangle (each pair checked once)
        cosine_sim.triu_(diagonal=1)

        # Find pairs above threshold
        merge_pairs = (cosine_sim > cfg.scaffold_merge_threshold).nonzero(as_tuple=False)

        if merge_pairs.shape[0] == 0:
            return

        # Track which templates have been merged (avoid double-merging)
        merged = set()

        for pair in merge_pairs:
            local_i, local_j = pair[0].item(), pair[1].item()
            global_i = alive_indices[local_i].item()
            global_j = alive_indices[local_j].item()

            if global_i in merged or global_j in merged:
                continue

            # Merge by usage-weighted average → keep in slot i, free slot j
            usage_i = self.scaffold_usage[global_i]
            usage_j = self.scaffold_usage[global_j]
            total_usage = usage_i + usage_j

            if total_usage < 1e-8:
                w_i = 0.5
            else:
                w_i = (usage_i / total_usage).item()

            M[global_i] = w_i * M[global_i] + (1.0 - w_i) * M[global_j]
            self.scaffold_usage[global_i] = total_usage

            # Re-initialize freed slot with orthogonal perturbation
            merged_template = M[global_i]
            noise = torch.randn_like(merged_template)
            mt_norm_sq = merged_template.dot(merged_template) + 1e-8
            noise = noise - merged_template * (noise.dot(merged_template) / mt_norm_sq)
            noise = F.normalize(noise, dim=-1) * merged_template.norm() * 0.3
            M[global_j] = merged_template + noise
            self.scaffold_usage[global_j] = 0.0

            merged.add(global_j)

    def extra_metrics(self) -> dict[str, float]:
        """Return scaffold health metrics for logging.

        Called after forward_train by the Trainer. Provides visibility
        into the scaffold's state without any gradient computation.

        Metrics:
            scaffold/mean_norm:     Average template vector magnitude.
                                    Low = templates decaying. High = healthy.
            scaffold/usage_entropy: Entropy of the usage distribution.
                                    High = diverse template utilization.
                                    Low = few templates dominate.
            scaffold/n_alive:       Number of templates with norm >= min_norm.
                                    Should stay near K after warmup.
            scaffold/mean_cosine:   Average pairwise cosine similarity.
                                    High = templates converging (need compression).
                                    Low = diverse template library.
            scaffold/step:          Current scaffold step counter.
        """
        cfg = self.config
        M = self.scaffold_templates
        K = cfg.n_scaffold_templates

        if K == 0:
            return {}

        norms = M.norm(dim=-1)  # (K,)
        alive_mask = norms >= cfg.scaffold_min_norm
        n_alive = int(alive_mask.sum().item())

        # Mean norm of all templates
        mean_norm = norms.mean().item()

        # Usage entropy: H = -Σ p_k log(p_k)
        usage = self.scaffold_usage
        usage_sum = usage.sum()
        if usage_sum > 0:
            p = usage / usage_sum
            # Avoid log(0) by masking zeros
            log_p = torch.where(p > 0, torch.log(p + 1e-10), torch.zeros_like(p))
            entropy = -(p * log_p).sum().item()
        else:
            entropy = 0.0

        # Mean pairwise cosine similarity (among alive templates)
        mean_cosine = 0.0
        if n_alive >= 2:
            alive_indices = alive_mask.nonzero(as_tuple=True)[0]
            alive_templates = M[alive_indices]
            alive_norms_clamped = norms[alive_indices].unsqueeze(-1).clamp(min=1e-8)
            normed = alive_templates / alive_norms_clamped
            cosine_matrix = torch.matmul(normed, normed.t())
            # Extract upper triangle (excluding diagonal)
            n = cosine_matrix.shape[0]
            n_pairs = n * (n - 1) // 2
            if n_pairs > 0:
                triu_sum = cosine_matrix.triu(diagonal=1).sum().item()
                mean_cosine = triu_sum / n_pairs

        return {
            "scaffold/mean_norm": mean_norm,
            "scaffold/usage_entropy": entropy,
            "scaffold/n_alive": float(n_alive),
            "scaffold/mean_cosine": mean_cosine,
            "scaffold/step": float(self._scaffold_step_val),
        }
