"""AENCA V12 model — Morphogenetic Mixture of Hebbian Rules (MMHR).

V12 extends V11's two-field coupled CA by replacing the single Hebbian rule
with a MIXTURE of K=8 parallel Hebbian rules, each specialised for a different
information binding pattern. The morphogen field h routes each cell to a soft
mixture of rules at each CA step.

V12 design nexus — MNCA paper (arxiv 2506.20486) meets NCA axioms:
  - MNCA §3.1 mixture model: K rules, per-cell probabilistic routing via morphogen
  - MNCA §3.2 intrinsic noise: ε_k ~ N(0,σ²) added per rule to Hebbian product
  - MNCA §4.3 Appendix F: mixture Lipschitz ≤ max_k L_k → stabler than single rule
  - d_h expanded 128→512: morphogen BECOMES the primary routing computation substrate

Architecture (extends V11):
    SEED:     z = Embed(x) + PE                      ∈ R^{B × L × 2048}
              h = tanh(W_h_init · z) + α · h_scaffold ∈ R^{B × L × 512}
              SettlementState initialized

    64 STEPS: For t = 0..63 with dilation = 2^(t mod 8):
                gate = σ(SiLU(W_a·h) ⊙ W_b·h + bias)    ← bilinear gate (d_h=512)
                τ(t) = max(0.1, 1.0 − 0.9·t/63)          ← temperature anneal
                weights = Gumbel_softmax(W_router·h/τ)   ← (B,L,K) rule weights
                delta_z = Σ_k w_k · φ_k(z,gate)          ← mixture update
                delta_h = W_h2(tanh(W_h1([h;h_nbr])))    ← morphogen evolves
                growth = (1 − settled)
                z = z + delta_z · growth
                h = h + delta_h · growth
                settlement.update(delta_z)

    READOUT:  logits = RMSNorm(z) · Embed^T / √d

Loss (4 terms):
    L = L_CE
      + λ_perm   · ‖Δz_last‖²    (content permanence)
      + λ_spec   · σ_max(W_p)    (spectral bound on shared perception)
      + λ_div    · (−H̄[weights]) (diversity: prevent rule collapse)

where H̄[weights] = mean over steps of Shannon entropy of routing distribution.
H̄ → log(K) when routing is uniform, 0 when routing is degenerate.
The diversity loss pulls routing AWAY from collapse toward specialised diversity.

Zero-init cascade (same as V11, extended):
    W_d_k = 0 for all k  → delta_all = 0 at init
    W_router = 0          → weights = 1/K (uniform)  
    W_h2 = 0              → delta_h = 0 at init
    → V12 starts as a pure copy-and-decay still-life (Class I), same as V11

Growth: Rules diverge from uniform mixture as gradients push W_router, W_s_k,
W_d_k to specialise. The morphogen field h differentiates into routing signals.
Routing sharpens (τ anneals) as morphogen matures through 64 coupled steps.

Parameter budget:
    Embeddings (tied):      2048 × 50304 = 103M
    PE:                     1024 × 2048  = 2.1M
    Shared W_p:             256 × 2048   = 0.52M
    K=8 × (W_s + W_d):     2 × 8 × 256 × 2048 = 8.39M
    Morphogen rule (d_h=512): ~0.79M
    Gate (d_h=512→2048×2):  2 × 512 × 2048 = 2.1M
    W_h_init (d_model→d_h): 2048 × 512 = 1.05M
    W_router (d_h=512→K=8): 512 × 8 = 0.004M
    Bilinear gate + conv:   ~0.02M
    Final norm:             2048 parms
    Non-embedding total: ~31.8M    Grand total: ~137M

Nine Axioms of True NCA (V12 extends V11 + V10):
  1. GROWTH:       Both fields develop from seed through 64 iterative steps
  2. PERMANENCE:   Settled cells ARE frozen permanent memory
  3. SPARSITY:     (1-settled) gating → shrinking active frontier
  4. SIMPLICITY:   K=8 SHARED tiny rules — the extended coupled genome
  5. UNIVERSALITY: Spectral norm bounds shared W_p Lipschitz → Class IV
  6. STRUCTURE=FUNCTION: Grown z IS the representation; h IS the routing
  7. SELF-ORG:     Training shapes the K rules + router from random → specialised
  8. DIFFERENTIATION: K rules enable ROLE-BASED cell differentiation (NEW dimension)
  9. INHERITANCE:  Scaffold passes beneficial h-configurations to future passes
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
from ncalm.models.v12.config import V12Config
from ncalm.models.v12.kernel import MixtureHebbianRules, MorphogenRule
from ncalm.models.v12.settlement import SettlementState


@register_model("v12")
class AENCAv12Model(NCALMModel):
    """Morphogenetic Mixture of Hebbian Rules NCA Language Model.

    V12 extends V11's two-field coupled CA with K=8 parallel Hebbian rules
    probabilistically routed per-cell by the expanded morphogen field (d_h=512).
    This implements MNCA §3.1 within the two-field NCA paradigm.

    The routing machinery is fully emergent: W_router zero-init ensures all
    rules are equally weighted at step 0. As the morphogen h differentiates
    through 64 coupled CA steps, routing sharpens. By step 63, cells have
    differentiated into linguistic roles — lexical, syntactic, semantic,
    boundary, and positional — each served by its specialist rule.

    No attention. No transformer blocks. All routing through dynamics.
    """

    def __init__(self, config_dict: dict) -> None:
        super().__init__()

        # ── Parse config ─────────────────────────────────────────────
        v12_keys = {f.name for f in dataclass_fields(V12Config)}
        v12_params = {k: v for k, v in config_dict.items() if k in v12_keys}
        self.config = V12Config(**v12_params)
        cfg = self.config

        # ── Token Embedding ──────────────────────────────────────────
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # ── Positional Embedding (learned, GPT-2 style) ─────────────
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        # ── Content Genome: K parallel Hebbian rules (mixture) ───────
        # All rules share one perception encoder W_p. Each rule has its
        # own specialised self-binding (W_s_k, W_d_k). The morphogen
        # routes each cell to a soft mixture of rules at each step.
        self.rules = MixtureHebbianRules(cfg)

        # ── Morphogen Genome: ONE shared tiny MLP rule ───────────────
        # Same structure as V11, parameterised at d_h=512 for 4× the
        # routing signal capacity vs V11's d_h=128.
        self.morph_rule = MorphogenRule(cfg)

        # ── Morphogen Seed Projection: z → h₀ ───────────────────────
        self.W_h_init = nn.Linear(cfg.d_model, cfg.d_h, bias=False)

        # ── Bilinear Morphogen Gate: h → gate ∈ [0, 1]^d_model ──────
        # gate = σ(SiLU(h·W_a) ⊙ h·W_b + bias)
        # At d_h=512: quadratic capacity d_h(d_h+1)/2 ≈ 131K >> d_model=2048
        # Bias=+3.0 → gate ≈ 0.95 at init (V12 ≈ V11 ≈ V10 start)
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

        # ── Spectral norm tracking (shared W_p only) ─────────────────
        # W_s and W_d start at zero/Xavier and are batched (K, ...).
        # Tracking only the shared perception matrix keeps monitoring
        # lightweight and avoids K separate power-iteration buffers.
        self._register_spectral_buffers()

        # ── Persistent Morphogen Scaffold (EPIGENETIC MEMORY) ────────
        self._register_scaffold_buffers()

        # ── Transient scaffold state ──────────────────────────────────
        self._last_h_final: list[torch.Tensor] = []
        self._last_settled: list[torch.Tensor] = []

        # ── Cached scaffold step (avoid GPU→CPU sync in forward) ─────
        self._scaffold_step_val: int = 0

        # ── Pre-compute dilation schedule ────────────────────────────
        self._dilation_schedule = self._build_dilation_schedule()

        # ── Weight initialization ────────────────────────────────────
        self._init_weights()

    def _build_dilation_schedule(self) -> list[int]:
        """Build the full 64-step dilation schedule (cyclic, same as V11)."""
        dilations = self.config.dilations
        return [dilations[t % len(dilations)] for t in range(self.config.n_steps)]

    def _register_spectral_buffers(self) -> None:
        """Register power-iteration vector for W_p (shared perception) spectral norm."""
        W = self.rules.W_p.weight   # (d_rule, d_model)
        u = torch.randn(W.shape[0])
        u = u / u.norm()
        self.register_buffer("_spectral_u_Wp", u, persistent=False)

    def _register_scaffold_buffers(self) -> None:
        """Register persistent scaffold buffers for morphogen template memory.

        At d_h=512: templates are (K=64, 512) = 128KB — still negligible.
        Buffers are persistent=True → saved in checkpoints.
        """
        cfg = self.config
        K = cfg.n_scaffold_templates
        d_h = cfg.d_h

        self.register_buffer(
            "scaffold_templates",
            torch.zeros(K, d_h),
            persistent=True,
        )
        self.register_buffer(
            "scaffold_usage",
            torch.zeros(K),
            persistent=True,
        )
        self.register_buffer(
            "scaffold_step",
            torch.zeros(1, dtype=torch.long),
            persistent=True,
        )

    def _init_weights(self) -> None:
        """Initialize weights for embeddings and morphogen-specific layers."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.01)
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        nn.init.xavier_normal_(self.W_h_init.weight, gain=1.0)

        # Small random init + bias=+3.0 → gate ≈ σ(3) ≈ 0.95 at start
        nn.init.normal_(self.W_gate_a.weight, std=0.01)
        nn.init.normal_(self.W_gate_b.weight, std=0.01)
        nn.init.constant_(self.gate_bias, 3.0)

    def _get_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Convert final cell states to vocabulary logits (readout content z only)."""
        z = self.final_norm(z)
        if self.lm_head is not None:
            return self.lm_head(z)
        return F.linear(z, self.token_embedding.weight) * (1.0 / math.sqrt(self.config.d_model))

    def _spectral_norm_loss(self) -> torch.Tensor:
        """Spectral norm of shared W_p via one power-iteration step.

        Bounds the shared perception encoder's Lipschitz constant, preventing
        runaway content amplification. The per-rule W_s/W_d start at zero
        (zero Lipschitz) and grow slowly; they don't need separate tracking.
        """
        W = self.rules.W_p.weight.float()              # (d_rule, d_model)
        u = self._spectral_u_Wp.float()                # (d_rule,)

        with torch.no_grad():
            v = W.t() @ u                              # (d_model,)
            v_norm = v.norm()
            if v_norm < 1e-8:
                return torch.tensor(0.0, device=W.device)
            v = v / v_norm                             # (d_model,) normalized
            u_new = W @ v                              # (d_rule,)
            u_new = u_new / (u_new.norm() + 1e-8)
            self._spectral_u_Wp.copy_(u_new)

        # sigma: gradient flows through W only (u_new, v are detached)
        sigma = u_new @ (W @ v)
        return sigma

    def _ca_step(
        self,
        z: torch.Tensor,      # (B, L, d_model)
        h: torch.Tensor,      # (B, L, d_h)
        dilation: int,
        t: int,               # CA step index (for temperature annealing)
        n_steps: int,         # total CA steps (for step_frac)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single coupled CA step: content mixture + morphogen update.

        The checkpointable unit. Computes rule routing from h, then the
        gated mixture Hebbian update for z, then the morphogen update.

        Args:
            z:       (B, L, d_model) current content state
            h:       (B, L, d_h) current morphogen state
            dilation: Conv dilation for this step
            t:       Step index for temperature τ(t) annealing
            n_steps: Total number of CA steps (63 for inference, cfg.n_steps)

        Returns:
            delta_z:  (B, L, d_model) content update proposal
            delta_h:  (B, L, d_h) morphogen update proposal
            weights:  (B, L, K) routing weights — in graph for diversity loss
        """
        # ── Bilinear gate: morphogen controls content perception ──────
        gate = torch.sigmoid(
            F.silu(self.W_gate_a(h)) * self.W_gate_b(h) + self.gate_bias
        )   # (B, L, d_model)

        # ── K-rule mixture content update ────────────────────────────
        step_frac = t / max(n_steps - 1, 1)    # ∈ [0, 1]
        delta_z, weights = self.rules(
            z, h, dilation, gate, self.training, step_frac
        )   # (B, L, d_model), (B, L, K)

        # ── Morphogen update via its own local rule ──────────────────
        delta_h = self.morph_rule(h, dilation=dilation)   # (B, L, d_h)

        return delta_z, delta_h, weights

    def compute_gate(self, h: torch.Tensor) -> torch.Tensor:
        """Compute bilinear morphogen gate (exposed for analysis scripts)."""
        return torch.sigmoid(
            F.silu(self.W_gate_a(h)) * self.W_gate_b(h) + self.gate_bias
        )

    def _seed_morphogen(self, z: torch.Tensor) -> torch.Tensor:
        """Seed morphogen from content + persistent scaffold.

        Identical logic to V11. The scaffold operates at d_h=512 now, so
        templates are (K=64, 512). The cosine similarity selection and
        EMA update are dimensionality-agnostic — they just use cfg.d_h.

        h = tanh(W_h_init · z) + α · softmax(scores) @ M
        where M ∈ R^{K × d_h} are the persistent scaffold templates.
        """
        cfg = self.config
        h_fresh = torch.tanh(self.W_h_init(z))   # (B, L, d_h)

        scaffold_active = (
            cfg.scaffold_alpha > 0
            and cfg.n_scaffold_templates > 0
        )
        if not scaffold_active:
            return h_fresh

        template_norms = self.scaffold_templates.norm(dim=-1)
        if (template_norms.max() < 1e-8).item():
            return h_fresh

        step = int(self.scaffold_step.item())
        if step < cfg.scaffold_warmup_steps:
            alpha = cfg.scaffold_alpha * (step / max(cfg.scaffold_warmup_steps, 1))
        else:
            alpha = cfg.scaffold_alpha

        if alpha < 1e-8:
            return h_fresh

        M = self.scaffold_templates                        # (K, d_h)
        h_normed = F.normalize(h_fresh, dim=-1)
        M_normed = F.normalize(M, dim=-1)
        tau_scaled = cfg.scaffold_temperature * math.sqrt(cfg.d_h)
        scores = torch.matmul(h_normed, M_normed.t()) / tau_scaled   # (B, L, K)
        weights = F.softmax(scores, dim=-1)
        h_scaffold = torch.matmul(weights, M)             # (B, L, d_h)

        return h_fresh + alpha * h_scaffold

    def _run_ca_steps(
        self,
        z: torch.Tensor,
        accumulate_perm: bool = False,
    ) -> tuple[torch.Tensor, SettlementState, torch.Tensor, torch.Tensor]:
        """Run the 64-step coupled CA growth process.

        Both content z and morphogen h evolve together. The diversity entropy
        H(weights) is accumulated in-graph across all steps and returned as a
        scalar tensor for the diversity loss.

        Permanence loss uses only the last step's delta_z (gradient-efficient).

        Args:
            z:               (B, L, d_model) seed from embedding
            accumulate_perm: If True, compute permanence loss

        Returns:
            z:            (B, L, d_model) final grown content states
            settlement:   SettlementState with final settlement info
            perm_loss:    Scalar permanence loss (gradient from last step only)
            div_entropy:  Mean Shannon entropy H̄(weights) over all steps (in graph)
        """
        B, L, _ = z.shape
        cfg = self.config

        h = self._seed_morphogen(z)               # (B, L, d_h)
        settlement = SettlementState.create(B, L, z.device, cfg)

        perm_accum = 0.0
        last_delta_z = None
        # Diversity entropy accumulator — must be a tensor in the graph
        # so gradients flow to W_router and W_s/W_d through the loss.
        div_entropy_accum = z.new_zeros(())

        use_ckpt = cfg.checkpoint_interval > 0 and self.training

        for t, dilation in enumerate(self._dilation_schedule):
            # ── Compute coupled update proposals ─────────────────────
            if use_ckpt and t > 0 and t % cfg.checkpoint_interval == 0:
                delta_z, delta_h, weights = grad_checkpoint(
                    self._ca_step, z, h, dilation, t, cfg.n_steps,
                    use_reentrant=False,
                )
            else:
                delta_z, delta_h, weights = self._ca_step(z, h, dilation, t, cfg.n_steps)

            # ── Growth gating (settled cells are frozen) ─────────────
            growth = settlement.growth_gate().unsqueeze(-1)   # (B, L, 1)
            z = z + delta_z * growth
            h = h + delta_h * growth

            # ── Settlement tracking (detached) ───────────────────────
            settlement.update(delta_z, cfg)

            # ── Permanence: accumulate detached, keep last for grad ──
            if accumulate_perm:
                perm_accum += delta_z.detach().pow(2).mean().item()
                last_delta_z = delta_z

            # ── Diversity entropy accumulation (IN GRAPH) ────────────
            # H = -Σ_k w_k log(w_k), averaged over (B, L) cells
            # weights: (B, L, K) ← from Gumbel_softmax (differentiable)
            # Gradient flows: H → weights → W_router, W_s, W_d
            H = -(weights * weights.clamp(min=1e-8).log()).sum(dim=-1).mean()
            div_entropy_accum = div_entropy_accum + H

        # Mean diversity entropy over all CA steps
        div_entropy = div_entropy_accum / cfg.n_steps

        # ── Save morphogen final states for scaffold EMA ─────────────
        if self.training:
            self._last_h_final.append(h.detach())
            self._last_settled.append(settlement.settled.detach())
        else:
            self._last_h_final = []
            self._last_settled = []

        # Permanence loss: gradient only from last step's content delta
        perm_loss = torch.tensor(0.0, device=z.device)
        if accumulate_perm and last_delta_z is not None:
            perm_loss = last_delta_z.pow(2).mean()

        return z, settlement, perm_loss, div_entropy

    # ═══════════════════════════════════════════════════════════════════
    # NCALMModel Interface
    # ═══════════════════════════════════════════════════════════════════

    def forward_train(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass with loss computation.

        Four loss terms:
          L = L_CE + λ_perm · ‖Δz_{last}‖² + λ_spec · σ(W_p) + λ_div · (−H̄)

        Args:
            batch: (B, L) token IDs

        Returns:
            loss: Scalar — total training loss
            info: Dict with ce, perm, spectral, div_loss, routing_entropy,
                  settled, active keys
        """
        B, L = batch.shape
        cfg = self.config

        # ── Seed ─────────────────────────────────────────────────────
        positions = torch.arange(L, device=batch.device)
        z = self.token_embedding(batch) + self.pos_embedding(positions)

        # ── 64-step coupled CA growth ─────────────────────────────────
        z, settlement, loss_perm, div_entropy = self._run_ca_steps(
            z, accumulate_perm=(cfg.lambda_perm > 0)
        )

        # ── Readout ───────────────────────────────────────────────────
        logits = self._get_logits(z)

        # ── Loss 1: Cross-Entropy ─────────────────────────────────────
        loss_ce = F.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.vocab_size),
            batch[:, 1:].reshape(-1),
            reduction="mean",
        )

        # ── Loss 2: Permanence ────────────────────────────────────────
        # (already computed in _run_ca_steps)

        # ── Loss 3: Spectral norm (W_p only) ─────────────────────────
        loss_spectral = torch.tensor(0.0, device=batch.device)
        if cfg.lambda_spectral > 0:
            loss_spectral = self._spectral_norm_loss()

        # ── Loss 4: Diversity (prevent routing collapse) ─────────────
        # div_entropy ≥ 0. Maximising entropy → minimising -entropy.
        # loss_div ≤ 0: as entropy → log(K), loss_div → -log(K).
        # Adding lambda_div * loss_div to total_loss: gradient pushes
        # routing TOWARD uniform (high entropy) away from degenerate.
        loss_div = -div_entropy   # ∈ [−log(K), 0]

        total_loss = (
            loss_ce
            + cfg.lambda_perm * loss_perm
            + cfg.lambda_spectral * loss_spectral
            + cfg.lambda_div * loss_div
        )

        info = {
            "ce": loss_ce.detach(),
            "perm": loss_perm.detach() if isinstance(loss_perm, torch.Tensor) else loss_perm,
            "spectral": loss_spectral.detach() if isinstance(loss_spectral, torch.Tensor) else loss_spectral,
            "routing_entropy": div_entropy.detach(),
            "div_loss": loss_div.detach(),
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
        z, _, _, _ = self._run_ca_steps(z, accumulate_perm=False)
        return self._get_logits(z)

    def get_param_groups(self) -> list[dict]:
        """Parameter groups for AdamW optimizer.

        Includes W_s, W_d (nn.Parameter, 3D) and W_router in decay group.
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
        shared_wp_params = self.rules.W_p.weight.numel()
        binding_rules_params = self.rules.W_s.numel() + self.rules.W_d.numel()
        router_params = self.rules.W_router.weight.numel()
        conv_params = sum(p.numel() for p in self.rules.conv.parameters())
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
        rules_total = shared_wp_params + binding_rules_params + router_params + conv_params

        scaffold_alive = int(
            (self.scaffold_templates.norm(dim=-1) >= cfg.scaffold_min_norm).sum().item()
        )

        return {
            "architecture": "AENCA-V12-MMHR",
            "total_params": total,
            "embed_params": embed_params,
            "pe_params": pe_params,
            # ── Rule params breakdown ────────────────────────────────
            "rules_total_params": rules_total,
            "shared_wp_params": shared_wp_params,
            "binding_rules_params": binding_rules_params,
            "router_params": router_params,
            "conv_params": conv_params,
            # ── Morphogen params breakdown ───────────────────────────
            "morphogen_params": morphogen_total,
            "morph_rule_params": morph_rule_params,
            "morph_gate_params": morph_gate_params,
            "morph_seed_params": morph_seed_params,
            # ── Other ────────────────────────────────────────────────
            "norm_params": norm_params,
            "head_params": head_params,
            # ── Architecture fields ──────────────────────────────────
            "d_model": cfg.d_model,
            "d_rule": cfg.d_rule,
            "d_h": cfg.d_h,
            "n_rules": cfg.n_rules,
            "n_steps": cfg.n_steps,
            "n_scales": cfg.n_scales,
            "conv_kernel_size": cfg.conv_kernel_size,
            "dilation_schedule": self._dilation_schedule[:cfg.n_scales],
            "receptive_field": cfg.receptive_field,
            "vocab_size": cfg.vocab_size,
            "max_seq_len": cfg.max_seq_len,
            "tie_embeddings": cfg.tie_embeddings,
            # ── V12-specific training fields ─────────────────────────
            "noise_sigma": cfg.noise_sigma,
            "route_temp_min": cfg.route_temp_min,
            "route_temp_max": cfg.route_temp_max,
            "lambda_div": cfg.lambda_div,
            "lambda_perm": cfg.lambda_perm,
            "lambda_spectral": cfg.lambda_spectral,
            # ── Settlement fields ────────────────────────────────────
            "settlement_beta": cfg.settlement_beta,
            "settlement_tau": cfg.settlement_tau,
            "settlement_eta": cfg.settlement_eta,
            "gate_bias_init": 3.0,
            # ── Scaffold fields ──────────────────────────────────────
            "scaffold_templates_K": cfg.n_scaffold_templates,
            "scaffold_alpha": cfg.scaffold_alpha,
            "scaffold_temperature": cfg.scaffold_temperature,
            "scaffold_ema_decay": cfg.scaffold_ema_decay,
            "scaffold_warmup_steps": cfg.scaffold_warmup_steps,
            "scaffold_maintenance_interval": cfg.scaffold_maintenance_interval,
            "scaffold_alive_at_init": scaffold_alive,
            "paradigm": (
                f"MMHR-NCA: K={cfg.n_rules} mixture Hebbian rules, "
                f"shared W_p ({shared_wp_params / 1e3:.0f}K), "
                f"per-rule W_s/W_d ({binding_rules_params / 1e6:.2f}M), "
                f"morphogen router ({router_params} params), "
                f"d_h={cfg.d_h}, {cfg.n_steps} growth steps, "
                f"diversity loss lam={cfg.lambda_div}, noise sig={cfg.noise_sigma}, "
                "epigenetic scaffold, no attention, all routing emergent"
            ),
        }

    def get_phases(self, step: int, max_steps: int) -> str:
        """Single-phase coupled CA growth."""
        return "CA"

    # ═══════════════════════════════════════════════════════════════════
    # Scaffold Lifecycle Hooks (identical to V11 — d_h-agnostic logic)
    # ═══════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def on_step_end(self, step: int, metrics: dict) -> None:
        """Update persistent morphogen scaffold after each training step.

        Identical logic to V11. The scaffold operates on h ∈ R^{d_h=512}.
        Cosine similarity, EMA, prune/compress are all d_h-agnostic.

        Three operations (in order):
          1. EMA UPDATE:   crystallize observed morphogen states into templates
          2. PRUNING:      recycle dead templates (norm < min_norm)
          3. COMPRESSION:  merge duplicate templates (cosine > threshold)
        """
        cfg = self.config

        self.scaffold_step += 1
        self._scaffold_step_val = int(self.scaffold_step.item())

        if cfg.n_scaffold_templates == 0 or cfg.scaffold_alpha == 0:
            self._last_h_final = []
            self._last_settled = []
            return

        h_final_list = self._last_h_final
        settled_list = self._last_settled

        if not h_final_list or not settled_list:
            return

        h_final = torch.cat(h_final_list, dim=0)     # (B_total, L, d_h)
        settled = torch.cat(settled_list, dim=0)       # (B_total, L)

        self._last_h_final = []
        self._last_settled = []

        M = self.scaffold_templates
        K = cfg.n_scaffold_templates
        decay = cfg.scaffold_ema_decay

        h_flat = h_final.reshape(-1, cfg.d_h)
        w_flat = settled.reshape(-1)

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

        h_normed = F.normalize(h_flat, dim=-1)
        M_normed = F.normalize(M, dim=-1)
        tau_scaled = cfg.scaffold_temperature * math.sqrt(cfg.d_h)
        affinities = torch.matmul(h_normed, M_normed.t()) / tau_scaled
        soft_weights = F.softmax(affinities, dim=-1)

        self.scaffold_usage *= decay

        for k in range(K):
            w_k = soft_weights[:, k] * w_flat
            w_sum = w_k.sum()

            if w_sum < 1e-8:
                M[k] *= decay
                continue

            h_mean = (h_flat * w_k.unsqueeze(-1)).sum(dim=0) / w_sum
            M[k] = decay * M[k] + (1.0 - decay) * h_mean
            self.scaffold_usage[k] += w_sum

        current_step = self.scaffold_step.item()
        if current_step % cfg.scaffold_maintenance_interval == 0:
            self._scaffold_prune()
            self._scaffold_compress()

    @torch.no_grad()
    def _scaffold_prune(self) -> None:
        """Prune dead templates and re-initialize with orthogonal perturbation."""
        cfg = self.config
        M = self.scaffold_templates
        norms = M.norm(dim=-1)
        alive_mask = norms >= cfg.scaffold_min_norm
        dead_mask = ~alive_mask

        n_dead = dead_mask.sum().item()
        n_alive = alive_mask.sum().item()

        if n_dead == 0 or n_alive == 0:
            return

        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]

        for dead_idx in dead_indices:
            donor_idx = alive_indices[torch.randint(n_alive, (1,), device=M.device)]
            donor = M[donor_idx].squeeze(0)
            noise = torch.randn_like(donor)
            donor_norm_sq = donor.dot(donor) + 1e-8
            noise = noise - donor * (noise.dot(donor) / donor_norm_sq)
            noise = F.normalize(noise, dim=-1) * donor.norm() * 0.3
            M[dead_idx] = donor + noise
            self.scaffold_usage[dead_idx] = 0.0

    @torch.no_grad()
    def _scaffold_compress(self) -> None:
        """Merge similar templates to free capacity for new patterns."""
        cfg = self.config
        M = self.scaffold_templates
        K = cfg.n_scaffold_templates
        norms = M.norm(dim=-1)
        alive_mask = norms >= cfg.scaffold_min_norm

        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        n_alive = alive_indices.shape[0]
        if n_alive < 2:
            return

        alive_templates = M[alive_indices]
        alive_norms = norms[alive_indices].unsqueeze(-1).clamp(min=1e-8)
        normed = alive_templates / alive_norms
        cosine_sim = torch.matmul(normed, normed.t())
        cosine_sim.triu_(diagonal=1)

        merge_pairs = (cosine_sim > cfg.scaffold_merge_threshold).nonzero(as_tuple=False)
        if merge_pairs.shape[0] == 0:
            return

        merged = set()

        for pair in merge_pairs:
            local_i, local_j = pair[0].item(), pair[1].item()
            global_i = alive_indices[local_i].item()
            global_j = alive_indices[local_j].item()

            if global_i in merged or global_j in merged:
                continue

            usage_i = self.scaffold_usage[global_i]
            usage_j = self.scaffold_usage[global_j]
            total_usage = usage_i + usage_j

            w_i = 0.5 if total_usage < 1e-8 else (usage_i / total_usage).item()

            M[global_i] = w_i * M[global_i] + (1.0 - w_i) * M[global_j]
            self.scaffold_usage[global_i] = total_usage

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

        Routing metrics (routing_entropy, div_loss) are already logged
        via the info dict from forward_train. This method surfaces the
        scaffold state for monitoring template health.
        """
        cfg = self.config
        M = self.scaffold_templates
        K = cfg.n_scaffold_templates

        if K == 0:
            return {}

        norms = M.norm(dim=-1)
        alive_mask = norms >= cfg.scaffold_min_norm
        n_alive = int(alive_mask.sum().item())
        mean_norm = norms.mean().item()

        usage = self.scaffold_usage
        usage_sum = usage.sum()
        if usage_sum > 0:
            p = usage / usage_sum
            log_p = torch.where(p > 0, torch.log(p + 1e-10), torch.zeros_like(p))
            entropy = -(p * log_p).sum().item()
        else:
            entropy = 0.0

        mean_cosine = 0.0
        if n_alive >= 2:
            alive_indices = alive_mask.nonzero(as_tuple=True)[0]
            alive_templates = M[alive_indices]
            alive_norms_clamped = norms[alive_indices].unsqueeze(-1).clamp(min=1e-8)
            normed = alive_templates / alive_norms_clamped
            cosine_matrix = torch.matmul(normed, normed.t())
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
