"""AENCA V10 model — Persistent Growth NCA Language Model.

The fundamental CA inversion: MANY cheap steps with a TINY shared rule,
not FEW expensive steps with a BIG kernel.

Architecture:
    SEED:     z = Embed(x) + PE                      ∈ R^{B × L × 2048}
              SettlementState initialized (all cells active)

    64 STEPS: For t = 0..63 with dilation = 2^(t mod 8):
                delta = SimpleHebbianRule(z, dilation)  ← 3 small GEMMs
                z = z + delta * (1 - settled)           ← PERMANENCE gate
                settlement.update(delta)                ← track convergence

    READOUT:  logits = RMSNorm(z) · Embed^T / √d      ← grown state = logits

Key properties:
  - ONE shared tiny rule (4.9M params) — the genome. 17× smaller than V9.
  - 64 CA steps — 21× more than V9. Complexity from iteration.
  - Settlement: converged cells FREEZE. Permanent structures persist.
  - Sparse execution: only active frontier computes meaningful updates.
  - ~110M total params — intentionally less than GPT-2's 162M.

Loss function (3 terms):
  L = L_CE + λ_perm · mean(delta²) + λ_spectral · σ_max(kernel)

  L_CE prevents Class I (too simple to model language).
  L_perm + L_spectral prevent Class III (too chaotic).
  The sweet spot IS Class IV — edge of computability (Zenil).

Seven Axioms of True NCA:
  1. GROWTH: z develops from seed embedding through 64 iterative steps
  2. PERMANENCE: settled cells ARE frozen permanent memory
  3. SPARSITY: (1-settled) gate → shrinking active frontier
  4. SIMPLICITY: ONE shared 4.9M rule — the genome
  5. UNIVERSALITY: spectral norm bounds dynamics to Class IV
  6. STRUCTURE=FUNCTION: grown z with permanent regions IS the representation
  7. SELF-ORG: training shapes the one shared rule
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
from ncalm.models.v10.config import V10Config
from ncalm.models.v10.kernel import SimpleHebbianRule
from ncalm.models.v10.settlement import SettlementState


@register_model("v10")
class AENCAv10Model(NCALMModel):
    """Persistent Growth NCA Language Model.

    True CA with permanence: one tiny shared rule, 64 growth steps,
    cells that converge become permanent memory. No attention.

    Implements NCALMModel interface for drop-in use with the NCALM
    training infrastructure (Trainer, scheduler, data pipeline).
    """

    def __init__(self, config_dict: dict) -> None:
        super().__init__()

        # ── Parse config ─────────────────────────────────────────────
        v10_keys = {f.name for f in dataclass_fields(V10Config)}
        v10_params = {k: v for k, v in config_dict.items() if k in v10_keys}
        self.config = V10Config(**v10_params)
        cfg = self.config

        # ── Token Embedding ──────────────────────────────────────────
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # ── Positional Embedding (learned, GPT-2 style) ─────────────
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        # ── The Genome: ONE shared tiny Hebbian rule ─────────────────
        self.rule = SimpleHebbianRule(cfg)

        # ── Final RMSNorm (before readout) ───────────────────────────
        self.final_norm = nn.RMSNorm(cfg.d_model)

        # ── LM Head (tied or separate) ──────────────────────────────
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        else:
            self.lm_head = None

        # ── Spectral norm tracking ───────────────────────────────────
        self._register_spectral_buffers()

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
        """Register power iteration vectors for spectral norm estimation."""
        for name in ("W_p", "W_s", "W_d"):
            W = getattr(self.rule, name).weight
            u = torch.randn(W.shape[0])
            u = u / u.norm()
            self.register_buffer(f"_spectral_u_{name}", u, persistent=False)

    def _init_weights(self) -> None:
        """Initialize embedding weights. Kernel handled by SimpleHebbianRule."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.01)
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def _get_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Convert final cell states to vocabulary logits.

        Tied embeddings with 1/√d scaling to prevent self-dot-product explosion.
        """
        z = self.final_norm(z)
        if self.lm_head is not None:
            return self.lm_head(z)
        return F.linear(z, self.token_embedding.weight) * (1.0 / math.sqrt(self.config.d_model))

    def _spectral_norm_loss(self) -> torch.Tensor:
        """Spectral norms of rule weights via power iteration.

        Forced to fp32 to avoid epsilon underflow in fp16.
        Bounds the rule's Lipschitz constant → prevents Class III divergence.
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

    def _run_ca_steps(
        self,
        z: torch.Tensor,
        accumulate_perm: bool = False,
    ) -> tuple[torch.Tensor, SettlementState, torch.Tensor]:
        """Run the 64-step CA growth process with settlement.

        Permanence loss uses ONLY the last step's delta in the gradient graph.
        Earlier steps' delta² are accumulated detached (for logging & monitoring).
        This prevents the perm loss from defeating gradient checkpointing by
        keeping all 64 delta tensors alive in the autograd graph.

        Args:
            z: (B, L, d_model) seed cell states
            accumulate_perm: If True, compute permanence loss

        Returns:
            z: (B, L, d_model) final grown cell states
            settlement: Final SettlementState with settlement scores
            perm_loss: Scalar permanence loss (gradient only from last step)
        """
        B, L, _ = z.shape
        cfg = self.config
        settlement = SettlementState.create(B, L, z.device, cfg)
        perm_accum = 0.0  # Detached running sum for monitoring
        last_delta = None
        use_ckpt = cfg.checkpoint_interval > 0 and self.training

        for t, dilation in enumerate(self._dilation_schedule):
            # Apply shared rule (3 small GEMMs)
            if use_ckpt and t > 0 and t % cfg.checkpoint_interval == 0:
                delta = grad_checkpoint(
                    self.rule, z, dilation, use_reentrant=False
                )
            else:
                delta = self.rule(z, dilation=dilation)

            # PERMANENCE: settled cells get zero update
            gate = settlement.growth_gate()  # (B, L)
            z = z + delta * gate.unsqueeze(-1)

            # Track convergence (detached from gradient graph)
            settlement.update(delta, cfg)

            # Accumulate detached perm for monitoring; keep last delta for gradient
            if accumulate_perm:
                perm_accum += delta.detach().pow(2).mean().item()
                last_delta = delta

        # Permanence loss: gradient from last step only (VRAM-safe)
        perm_loss = torch.tensor(0.0, device=z.device)
        if accumulate_perm and last_delta is not None:
            perm_loss = last_delta.pow(2).mean()

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
            info: Dict with 'ce', 'perm', 'spectral', 'settled' keys
        """
        B, L = batch.shape
        cfg = self.config

        # ── Seed: embed tokens + positions ───────────────────────────
        positions = torch.arange(L, device=batch.device)
        z = self.token_embedding(batch) + self.pos_embedding(positions)

        # ── 64-step CA growth with settlement ────────────────────────
        z, settlement, loss_perm = self._run_ca_steps(
            z, accumulate_perm=(cfg.lambda_perm > 0)
        )

        # ── Readout: grown state → logits ────────────────────────────
        logits = self._get_logits(z)

        # ── Loss 1: Cross-Entropy (primary) ──────────────────────────
        loss_ce = F.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.vocab_size),
            batch[:, 1:].reshape(-1),
            reduction="mean",
        )

        # ── Loss 2: Permanence (accumulated online in _run_ca_steps) ─

        # ── Loss 3: Spectral norm (bounds dynamics) ──────────────────
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
        """Parameter groups for AdamW optimizer."""
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
        rule_params = sum(p.numel() for p in self.rule.parameters())
        norm_params = sum(p.numel() for p in self.final_norm.parameters())
        head_params = self.lm_head.weight.numel() if self.lm_head else 0

        return {
            "architecture": "AENCA-V10-PersistentGrowth",
            "total_params": total,
            "embed_params": embed_params,
            "pe_params": pe_params,
            "rule_params": rule_params,
            "norm_params": norm_params,
            "head_params": head_params,
            "d_model": cfg.d_model,
            "d_rule": cfg.d_rule,
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
            "paradigm": (
                "Persistent Growth NCA — ONE shared tiny rule (4.9M), "
                f"{cfg.n_steps} growth steps, settlement-based permanence, "
                "no attention"
            ),
        }

    def get_phases(self, step: int, max_steps: int) -> str:
        """Single-phase CA growth."""
        return "CA"
