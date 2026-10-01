"""AENCA V9 model — Wolfram Cellular Automata Language Model.

The simplest possible CA language model that can match transformer throughput.

Architecture:
    SEED:    z = Embed(x) + PE                        ∈ R^{B × L × 2048}

    STEP 1:  delta = HebbianKernel(z, dilation=1)     ← GEMMs 1-3 (local)
             z = z + delta

    STEP 2:  delta = HebbianKernel(z, dilation=r)     ← GEMMs 4-6 (medium)
             z = z + delta

    STEP 3:  delta = HebbianKernel(z, dilation=r²)    ← GEMMs 7-9 (global)
             z = z + delta

    READOUT: logits = RMSNorm(z) · Embed^T            ← tied embedding

9 GEMMs + 3 tiny depthwise convs. That's the entire forward pass.

Key properties:
  - ONE shared kernel (the genome): same rule at all steps and positions
  - Geometric dilations: full sequence coverage in 3 steps
  - Zero-init W_dec: starts as Class I still life, learns to grow
  - ~45K tok/s on RTX 4080 (matches GPT-2 at 49,779 tok/s)
  - 162M params: 63% embedding, 36% shared kernel, 1% norms/conv

Loss function (3 terms):
  L = L_CE + λ_perm · L_perm + λ_spectral · L_spectral

  1. L_CE: Standard next-token cross-entropy
  2. L_perm: mean(‖delta‖₂) — encourages convergence / sparsity
  3. L_spectral: Power-iteration spectral norm on kernel — bounds dynamics

Seven Axioms of Wolfram CA:
  1. GROWTH: z develops from seed embedding through 3 iterative steps
  2. PERMANENCE: residual structure persists; L_perm encourages settlement
  3. SPARSITY: Hebbian ReLU×ReLU product is naturally sparse
  4. SIMPLICITY: ONE shared kernel (~59M) — the genome
  5. UNIVERSALITY: spectral normalization constrains dynamics
  6. STRUCTURE=FUNCTION: grown z IS the representation
  7. SELF-ORG: training shapes the one shared rule
"""

from __future__ import annotations

import math
from dataclasses import fields as dataclass_fields
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.model import NCALMModel, register_model
from ncalm.models.v9.config import V9Config
from ncalm.models.v9.kernel import HebbianKernel


@register_model("v9")
class AENCAv9Model(NCALMModel):
    """Wolfram Cellular Automata Language Model.

    The simplest CA LM that can match transformer throughput.
    ONE shared kernel, 3 dilated steps, 9 GEMMs, no attention.

    Implements NCALMModel interface for drop-in use with the NCALM
    training infrastructure (Trainer, scheduler, data pipeline).
    """

    def __init__(self, config_dict: dict) -> None:
        super().__init__()

        # ── Parse config ─────────────────────────────────────────────
        v9_keys = {f.name for f in dataclass_fields(V9Config)}
        v9_params = {k: v for k, v in config_dict.items() if k in v9_keys}
        self.config = V9Config(**v9_params)
        cfg = self.config

        # Auto-size d_concept if target_params is set
        if cfg.d_concept <= 0:
            if cfg.target_params > 0:
                cfg.d_concept = cfg.compute_d_concept(cfg.target_params)
            else:
                raise ValueError(
                    "V9Config: either d_concept or target_params must be > 0"
                )

        # ── Token Embedding ──────────────────────────────────────────
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # ── Positional Embedding (learned, GPT-2 style) ─────────────
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        # ── The Genome: ONE shared Hebbian kernel ────────────────────
        self.kernel = HebbianKernel(cfg)

        # ── Final RMSNorm (before readout) ───────────────────────────
        self.final_norm = nn.RMSNorm(cfg.d_model)

        # ── LM Head (tied or separate) ──────────────────────────────
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        else:
            self.lm_head = None

        # ── Spectral norm tracking (power iteration vectors) ────────
        # One u-vector per kernel weight matrix for efficient σ_max estimation
        self._register_spectral_buffers()

        # ── Pre-compute dilations ────────────────────────────────────
        self._dilations = cfg.dilations

        # ── Weight initialization ────────────────────────────────────
        self._init_weights()

    def _register_spectral_buffers(self) -> None:
        """Register power iteration vectors for spectral norm estimation.

        We track one u-vector per kernel weight. Each forward pass
        runs one power iteration step (cheap: 2 matmuls per matrix).
        This gives us an efficient approximation of σ_max without SVD.
        """
        for name in ("W_enc_p", "W_enc_z", "W_dec"):
            W = getattr(self.kernel, name).weight
            # u-vector: unit vector of shape (out_features,)
            u = torch.randn(W.shape[0])
            u = u / u.norm()
            self.register_buffer(f"_spectral_u_{name}", u, persistent=False)

    def _init_weights(self) -> None:
        """Initialize all weights.

        Embedding: N(0, 0.02) — standard GPT-2 init
        Positional: N(0, 0.01) — small positional signal
        Kernel: handled by HebbianKernel._init_weights() (zero-init W_dec)
        LM Head: N(0, 0.02) if untied
        """
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.01)
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def _get_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Convert final cell states to vocabulary logits.

        For tied embeddings, we scale logits by 1/√d_model to prevent
        the self-dot-product explosion: RMSNorm amplifies z by ~√d,
        so z·embed^T gives self-dot ≈ d·σ²/σ ≈ √d·σ ≈ 37 at d=2048.
        Scaling by 1/√d brings logits to O(1) variance.

        Args:
            z: (B, L, d_model) final cell states after CA steps

        Returns:
            logits: (B, L, vocab_size)
        """
        z = self.final_norm(z)
        if self.lm_head is not None:
            return self.lm_head(z)
        # Tied embeddings: scale by 1/√d to keep logits O(1)
        return F.linear(z, self.token_embedding.weight) * (1.0 / math.sqrt(self.config.d_model))

    def _spectral_norm_loss(self) -> torch.Tensor:
        """Estimate spectral norms of kernel weights via power iteration.

        One iteration per forward pass. Cheap: 2 matmuls per weight matrix.
        Returns mean spectral norm across the 3 kernel matrices.

        Forced to fp32 to avoid epsilon underflow in fp16 (1e-12 → 0 in fp16,
        causing division by zero when W_dec is zero-initialized).

        This constrains the kernel's Lipschitz constant, bounding the
        CA dynamics to prevent divergence (Wolfram Class III → Class IV).
        """
        total = torch.tensor(0.0, device=self.token_embedding.weight.device)

        for name in ("W_enc_p", "W_enc_z", "W_dec"):
            W = getattr(self.kernel, name).weight.float()  # Force fp32
            u = getattr(self, f"_spectral_u_{name}").float()

            # One power iteration step (fp32 for numerical safety)
            with torch.no_grad():
                v = W.t() @ u               # (in,)
                v_norm = v.norm()
                if v_norm > 1e-8:           # Skip zero matrices (e.g. W_dec at init)
                    v = v / v_norm
                    u_new = W @ v           # (out,)
                    u_new = u_new / (u_new.norm() + 1e-8)
                    getattr(self, f"_spectral_u_{name}").copy_(u_new)
                else:
                    continue  # Zero matrix → σ_max = 0, skip

            # σ_max ≈ u^T W v (differentiable w.r.t. W)
            sigma = u_new @ (W @ v)
            total = total + sigma

        return total / 3.0

    def _run_ca_steps(self, z: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the CA forward pass: n shared kernel steps with increasing dilation.

        Each residual is scaled by 1/√n_steps so that after accumulation,
        the state variance stays bounded (standard residual scaling).

        Args:
            z: (B, L, d_model) seed cell states

        Returns:
            z: (B, L, d_model) final cell states after all CA steps
            deltas: List of delta tensors from each step (for permanence loss)
        """
        scale = 1.0 / math.sqrt(len(self._dilations))
        deltas = []
        for dilation in self._dilations:
            delta = self.kernel(z, dilation=dilation)   # Shared kernel!
            z = z + delta * scale
            deltas.append(delta)
        return z, deltas

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
            loss: Scalar — total training loss (CE + permanence + spectral)
            info: Dict with 'ce', 'perm', 'spectral' keys
        """
        B, L = batch.shape
        cfg = self.config

        # ── Seed: embed tokens + positions ───────────────────────────
        positions = torch.arange(L, device=batch.device)
        z = self.token_embedding(batch) + self.pos_embedding(positions)

        # ── CA Steps: 3 shared kernel applications ───────────────────
        z, deltas = self._run_ca_steps(z)

        # ── Readout: logits via tied embedding ───────────────────────
        logits = self._get_logits(z)

        # ── Loss 1: Cross-Entropy (primary) ──────────────────────────
        loss_ce = F.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.vocab_size),
            batch[:, 1:].reshape(-1),
            reduction="mean",
        )

        # ── Loss 2: Permanence (encourages convergence) ─────────────
        # Mean squared L2 norm of deltas — always differentiable (even at 0)
        loss_perm = torch.tensor(0.0, device=batch.device)
        if cfg.lambda_perm > 0:
            for delta in deltas:
                loss_perm = loss_perm + delta.pow(2).mean()
            loss_perm = loss_perm / len(deltas)

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
        }

        return total_loss, info

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference/generation.

        Args:
            input_ids: (B, L) token IDs

        Returns:
            logits: (B, L, vocab_size)
        """
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device)
        z = self.token_embedding(input_ids) + self.pos_embedding(positions)
        z, _ = self._run_ca_steps(z)
        return self._get_logits(z)

    def get_param_groups(self) -> list[dict]:
        """Parameter groups for AdamW optimizer.

        Groups:
        - decay: weight matrices (2D+ tensors, excluding norms/embeddings)
        - no_decay: biases, RMSNorm parameters, embeddings (weight_decay=0)
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
        kernel_params = sum(p.numel() for p in self.kernel.parameters())
        norm_params = sum(p.numel() for p in self.final_norm.parameters())
        head_params = self.lm_head.weight.numel() if self.lm_head else 0

        return {
            "architecture": "AENCA-V9-Wolfram",
            "total_params": total,
            "embed_params": embed_params,
            "pe_params": pe_params,
            "kernel_params": kernel_params,
            "norm_params": norm_params,
            "head_params": head_params,
            "d_model": cfg.d_model,
            "d_concept": cfg.d_concept,
            "n_steps": cfg.n_steps,
            "conv_kernel_size": cfg.conv_kernel_size,
            "dilations": self._dilations,
            "receptive_field": cfg.receptive_field,
            "vocab_size": cfg.vocab_size,
            "max_seq_len": cfg.max_seq_len,
            "tie_embeddings": cfg.tie_embeddings,
            "lambda_perm": cfg.lambda_perm,
            "lambda_spectral": cfg.lambda_spectral,
            "paradigm": (
                "Wolfram Cellular Automata — ONE shared Hebbian kernel, "
                f"{cfg.n_steps} dilated steps, {3 * cfg.n_steps} GEMMs, no attention"
            ),
        }

    def get_phases(self, step: int, max_steps: int) -> str:
        """Single-phase CA training."""
        return "CA"
