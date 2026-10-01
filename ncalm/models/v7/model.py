"""AENCA V7.1 main model: BDH-inspired Hebbian NCA Language Model.

Architecture (V7.1 — BDH-inspired Hebbian gating):
  1. Token embedding + sinusoidal PE → initial state z₀ (B, L, d_model)
  2. Two-timescale loop:
     For θ in 1..n_theta:                          [THETA CYCLE]
       For γ in 1..t_gamma:                        [GAMMA STEPS — shared weights]
         P = CausalConv1d(RMSNorm(z))              [local perception]
         delta = HebbianKernel(P, z)                [ReLU concept encoding + Hebbian product]
         z = z + delta                              [NCA residual update]
       z = ThetaBlock_θ(z)                         [THETA STEP — unique weights]
         → Attention + HebbianFFN(z_pre, z_post)    [cross-modal concept gating]
       halt_prob = HaltHead(z)                      [readout accumulation]
       z_out += min(halt_prob, remaining) · z
  3. MoS Readout from accumulated z_out → logits

V7.1 changes from V7 (BDH-inspired, arXiv:2509.26507):
  - HebbianKernel replaces SlimKernel + CSS: ReLU-lowrank concept encoding
    with Hebbian product gating between local perception and state.
    "Only update concepts that are both perceived AND in current state."
  - HebbianFFN replaces SwiGLU in theta blocks: cross-modal Hebbian gating
    between pre-attention and post-attention states.
    "Only update concepts that are both in input AND supported by context."
  - CSS removed: global communication via theta attention only.
  - Same param count and FLOPs as V7.

NCA principles (maintained):
  - NCA purity in gamma loop: shared kernel, local conv, iterative refinement
  - Gated residual update: z ← z + delta
  - Per-position output with independent halting
  - Zero-init cascade (model starts as near-identity)
  - Positive sparse activations (BDH insight: ~5% nonzero from ReLU)
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ncalm.model import NCALMModel, register_model
from ncalm.models.v7.config import V7Config
from ncalm.models.v7.hebbian import HebbianKernel
from ncalm.models.v7.attention import ThetaBlock
from ncalm.models.v7.halting import AdaptiveHalting

# Reuse readout from v5 — needs d_model, vocab_size, mos_components, tie_embeddings
from ncalm.models.v5.readout import MixtureOfSoftmaxReadout


# ═══════════════════════════════════════════════════════════════════════
# Utility Modules
# ═══════════════════════════════════════════════════════════════════════

class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal position encoding (non-parametric).

    Produces d-dimensional sinusoidal features for position injection into
    the NCA state.
    """

    def __init__(self, d: int, max_seq_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_seq_len, d)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d, 2).float() * -(math.log(10000.0) / d)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return position encodings for positions 0..seq_len-1."""
        return self.pe[:seq_len]


class CausalConv1D(nn.Module):
    """Causal 1D convolution for local spatial perception.

    Left-padded convolution: position i sees only positions i-r, ..., i.
    Shared across all gamma steps (NCA principle).
    """

    def __init__(self, d_model: int, kernel_radius: int):
        super().__init__()
        kernel_size = kernel_radius + 1
        self.padding = kernel_radius
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=0,
            bias=False,
        )
        self.norm = nn.RMSNorm(d_model)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Apply causal conv with pre-norm.

        Args:
            s: (B, L, d_model) cell states

        Returns:
            P: (B, L, d_model) local perception
        """
        s_norm = self.norm(s)
        x = s_norm.transpose(1, 2)             # (B, d_model, L)
        x = F.pad(x, (self.padding, 0))         # left-pad for causal
        P = self.conv(x)                         # (B, d_model, L)
        return P.transpose(1, 2)                 # (B, L, d_model)


# ═══════════════════════════════════════════════════════════════════════
# Main Model
# ═══════════════════════════════════════════════════════════════════════

@register_model("v7")
class AENCAv7Model(NCALMModel):
    """AENCA V7.1: BDH-inspired Hebbian NCA Language Model.

    A two-timescale Neural Cellular Automaton with:
    - Gamma level (fast, shared weights): CausalConv → HebbianKernel → residual
    - Theta level (slow, unique weights): full causal attention + HebbianFFN
    - Adaptive halting at theta boundaries (PonderNet readout accumulation)

    NCA Purity (maintained in gamma loop):
        ✓ Shared kernel weights applied to all cells
        ✓ Local computation (causal conv, radius r)
        ✓ Iterative refinement (t_gamma steps per theta cycle)
        ✓ Residual update: z ← z + delta (Hebbian product is implicit gate)
        ✓ Per-position output with independent halting
        ✓ Zero-init cascade (model starts as near-identity)
        ✓ Positive sparse activations (BDH-inspired ReLU concept encoding)

    Hybrid extension (theta level):
        ✓ Unique attention + HebbianFFN per theta cycle
        ✓ Full causal self-attention for global mixing
        ✓ HebbianFFN: cross-modal gating (pre-attn × post-attn concepts)
    """

    def __init__(self, config_dict: dict) -> None:
        super().__init__()

        # Parse flat config dict into V7Config
        v7_keys = {f.name for f in V7Config.__dataclass_fields__.values()}
        v7_params = {k: v for k, v in config_dict.items() if k in v7_keys}
        self.config = V7Config(**v7_params)
        cfg = self.config

        # Auto-size theta FFN if not specified
        if cfg.d_ff_theta == 0:
            target = config_dict.get("target_params", 162_000_000)
            cfg.d_ff_theta = cfg.compute_d_ff_theta(target)

        # ── Token Embedding ──────────────────────────────────────────
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.token_embedding.weight, std=1.0 / math.sqrt(cfg.d_model))

        # ── Position Encoding ────────────────────────────────────────
        # State PE: added to initial embedding (d_model dim)
        self.state_pe = SinusoidalPositionEncoding(cfg.d_model, cfg.max_seq_len * 2)

        # ── Shared NCA Components (gamma level) ─────────────────────
        self.conv = CausalConv1D(cfg.d_model, cfg.kernel_radius)
        self.hebbian_kernel = HebbianKernel(cfg)

        # ── Unique Theta Blocks (one per theta cycle) ────────────────
        self.theta_blocks = nn.ModuleList([
            ThetaBlock(cfg, block_idx=i) for i in range(cfg.n_theta)
        ])

        # ── Adaptive Halting ─────────────────────────────────────────
        if cfg.enable_halting:
            self.halting = AdaptiveHalting(cfg)

        # ── Readout (reused from V5) ─────────────────────────────────
        emb_weight = self.token_embedding.weight if cfg.tie_embeddings else None
        self.readout = MixtureOfSoftmaxReadout(cfg, emb_weight)

    # ══════════════════════════════════════════════════════════════════
    # Training Forward Pass
    # ══════════════════════════════════════════════════════════════════

    def forward_train(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Complete V7.1 training step with autoregressive loss.

        Runs the two-timescale hierarchical loop:
        - Inner (gamma): shared kernel iterates NCA state
        - Outer (theta): unique attention + HebbianFFN + readout accumulation

        With gradient checkpointing per theta cycle.

        Args:
            batch: (B, L) token IDs, dtype long.

        Returns:
            loss:  Scalar total training loss.
            info:  Dict with 'ce' key plus auxiliary components.
        """
        B, L = batch.shape
        cfg = self.config

        # Step 1: Embed tokens + position encoding
        s = self.token_embedding(batch)
        s = s + self.state_pe(L).unsqueeze(0)

        # Step 2: Run hierarchical loop
        z_out, loop_info = self._run_hierarchical_loop(s, batch)

        # Step 3: Readout
        logits = self.readout(z_out)

        # Step 4: Compute losses
        logits_flat = logits[:, :-1].reshape(-1, cfg.vocab_size)
        targets_flat = batch[:, 1:].reshape(-1)

        if self.readout.returns_log_probs:
            ce_loss = F.nll_loss(logits_flat, targets_flat, reduction="mean")
        else:
            ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")

        total_loss = ce_loss
        info: dict[str, Any] = {"ce": ce_loss.detach()}

        # State regularization: ‖z_out - s0‖² / d_model
        s0 = s.detach()
        if cfg.lambda_reg > 0:
            reg_loss = (z_out - s0).pow(2).mean()
            total_loss = total_loss + cfg.lambda_reg * reg_loss
            info["reg"] = reg_loss.detach()

        # Deep supervision auxiliary losses
        if cfg.lambda_aux > 0 and loop_info["aux_loss"] is not None:
            total_loss = total_loss + cfg.lambda_aux * loop_info["aux_loss"]
            info["aux"] = loop_info["aux_loss"].detach()

        # PonderNet halting KL (if halting enabled)
        if cfg.enable_halting and cfg.lambda_halt > 0 and loop_info["halt_kl"] is not None:
            total_loss = total_loss + cfg.lambda_halt * loop_info["halt_kl"]
            info["halt_kl"] = loop_info["halt_kl"].detach()

        # Monitoring metrics
        info["expected_steps"] = loop_info["expected_steps"]
        info["pct_halted_early"] = loop_info["pct_halted_early"]

        return total_loss, info

    # ══════════════════════════════════════════════════════════════════
    # Hierarchical NCA Loop
    # ══════════════════════════════════════════════════════════════════

    def _gamma_step(self, z: torch.Tensor) -> torch.Tensor:
        """Single NCA gamma step: conv → HebbianKernel → residual update.

        This is the atomic NCA computation unit. All components use
        shared weights (NCA principle). The HebbianKernel provides:
        - ReLU-lowrank encoding to concept space (positive sparse activations)
        - Hebbian product gating (perception × state → confirmed concepts)

        Args:
            z:  (B, L, d_model) current cell states

        Returns:
            z_new: (B, L, d_model) updated cell states
        """
        # Local perception via causal conv
        P = self.conv(z)                                       # (B, L, d_model)

        # Hebbian kernel: concept encoding + Hebbian product
        delta = self.hebbian_kernel(P, z)                      # (B, L, d_model)

        # NCA residual update (Hebbian product acts as implicit gate)
        return z + delta

    def _run_theta_cycle(
        self,
        z: torch.Tensor,
        theta_block: ThetaBlock,
    ) -> torch.Tensor:
        """Run one theta cycle: t_gamma gamma steps + theta attention/FFN.

        Args:
            z:           (B, L, d_model) cell states at cycle start
            theta_block: the unique ThetaBlock for this cycle

        Returns:
            z: (B, L, d_model) cell states after gamma steps + theta attention
        """
        # Gamma steps (shared kernel, iterative refinement)
        for _ in range(self.config.t_gamma):
            z = self._gamma_step(z)

        # Theta step (unique attention + HebbianFFN)
        z = theta_block(z)

        return z

    def _run_hierarchical_loop(
        self,
        z: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run the full two-timescale hierarchical loop.

        Outer loop: n_theta theta cycles with gradient checkpointing.
        Inner loop: t_gamma gamma steps (inside _run_theta_cycle).
        Halting: per-position readout accumulation at theta boundaries.

        Args:
            z:     (B, L, d_model) initial cell states
            batch: (B, L) token IDs for aux loss computation (None during inference)

        Returns:
            z_out:     (B, L, d_model) accumulated state
            loop_info: dict with diagnostics and loss components
        """
        cfg = self.config
        B, L, D = z.shape
        device = z.device
        n_theta = cfg.n_theta
        use_halting = cfg.enable_halting

        # ── Readout accumulators ─────────────────────────────────────
        z_out = torch.zeros_like(z)
        remaining = torch.ones(B, L, device=device)
        halt_probs: list[torch.Tensor] = []

        # ── Deep supervision substrates ──────────────────────────────
        aux_substrates: list[tuple[int, torch.Tensor, torch.Tensor]] = []

        # ── Main theta loop with gradient checkpointing ──────────────
        n_active_cycles = 0

        for theta_idx in range(n_theta):
            theta_block = self.theta_blocks[theta_idx]
            is_last = (theta_idx == n_theta - 1)

            # Early exit if all positions halted
            if use_halting and (remaining <= cfg.halt_eps).all():
                break

            # Run theta cycle (gamma steps + theta attention)
            if self.training:
                z = checkpoint(
                    self._run_theta_cycle,
                    z, theta_block,
                    use_reentrant=False,
                )
            else:
                z = self._run_theta_cycle(z, theta_block)

            n_active_cycles += 1

            # ── Halting / readout accumulation ───────────────────────
            if use_halting:
                z_out, remaining, hp = self.halting.accumulate(
                    z, z_out, remaining, is_last,
                )
                halt_probs.append(hp)
            else:
                # No halting: accumulate uniformly (1/n_theta per cycle)
                if is_last:
                    z_out = z_out + remaining.unsqueeze(-1) * z
                    remaining = torch.zeros_like(remaining)
                else:
                    weight = 1.0 / n_theta
                    z_out = z_out + weight * z
                    remaining = remaining - weight

            # Deep supervision snapshot (for all cycles except the last)
            if not is_last and batch is not None and cfg.lambda_aux > 0:
                aux_substrates.append(
                    (theta_idx + 1, z_out.clone(), remaining.clone())
                )

        # ── Absorb any remaining mass (safety, shouldn't be needed) ──
        if remaining.sum() > 0.001:
            z_out = z_out + remaining.unsqueeze(-1) * z

        # ── Compute deep supervision auxiliary losses ────────────────
        aux_loss: torch.Tensor | None = None
        if batch is not None and cfg.lambda_aux > 0 and aux_substrates:
            targets_flat = batch[:, 1:].reshape(-1)
            aux_total = torch.tensor(0.0, device=device)
            n_aux = 0

            for step_idx, z_out_snap, remaining_snap in aux_substrates:
                # Weight: (θ/n_theta)² — later boundaries weighted higher
                w = (step_idx / n_theta) ** 2

                # Only compute aux if meaningful mass accumulated
                snap_mass = (1.0 - remaining_snap).mean()
                if snap_mass < 0.01:
                    continue

                aux_logits = self.readout(z_out_snap)
                aux_flat = aux_logits[:, :-1].reshape(-1, cfg.vocab_size)

                if self.readout.returns_log_probs:
                    aux_ce = F.nll_loss(aux_flat, targets_flat, reduction="mean")
                else:
                    aux_ce = F.cross_entropy(aux_flat, targets_flat, reduction="mean")

                aux_total = aux_total + w * aux_ce
                n_aux += 1

            if n_aux > 0:
                aux_loss = aux_total / n_aux

        # ── Halting KL regularization ────────────────────────────────
        halt_kl: torch.Tensor | None = None
        if use_halting and halt_probs and cfg.lambda_halt > 0:
            halt_kl = self.halting.compute_kl_loss(halt_probs)

        # ── Compute monitoring metrics ───────────────────────────────
        with torch.no_grad():
            halted_mask = (remaining <= cfg.halt_eps)
            pct_halted = halted_mask.float().mean().item()
            expected_steps = n_active_cycles * (cfg.t_gamma + 1)

        loop_info: dict[str, Any] = {
            "aux_loss": aux_loss,
            "halt_kl": halt_kl,
            "expected_steps": expected_steps,
            "pct_halted_early": pct_halted,
            "n_active_cycles": n_active_cycles,
        }

        return z_out, loop_info

    # ══════════════════════════════════════════════════════════════════
    # Inference
    # ══════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Autoregressive inference with adaptive theta halting.

        Per-position halting is active: easy tokens converge in fewer
        theta cycles; complex tokens use all n_theta cycles.

        Args:
            input_ids: (B, L) token IDs.

        Returns:
            logits: (B, L, vocab_size) predictions at all positions.
        """
        B, L = input_ids.shape
        cfg = self.config
        device = input_ids.device

        # Embed tokens
        s = self.token_embedding(input_ids)
        s = s + self.state_pe(L).unsqueeze(0)

        # Initialize state
        z = s
        z_out = torch.zeros_like(z)
        remaining = torch.ones(B, L, device=device)

        for theta_idx in range(cfg.n_theta):
            theta_block = self.theta_blocks[theta_idx]
            is_last = (theta_idx == cfg.n_theta - 1)

            # Early exit if all halted
            if cfg.enable_halting and (remaining <= cfg.halt_eps).all():
                break

            # Run theta cycle
            z = self._run_theta_cycle(z, theta_block)

            # Halting / accumulation
            if cfg.enable_halting:
                z_out, remaining, hp = self.halting.accumulate(
                    z, z_out, remaining, is_last,
                )
            else:
                if is_last:
                    z_out = z_out + remaining.unsqueeze(-1) * z
                else:
                    weight = 1.0 / cfg.n_theta
                    z_out = z_out + weight * z
                    remaining = remaining - weight

        # Safety: absorb remaining
        if remaining.sum() > 0.001:
            z_out = z_out + remaining.unsqueeze(-1) * z

        return self.readout(z_out)

    # ══════════════════════════════════════════════════════════════════
    # Parameter Groups
    # ══════════════════════════════════════════════════════════════════

    def get_param_groups(self) -> list[dict]:
        """Parameter groups for optimizer.

        decay: 2D+ weight matrices (kernels, attention, FFN)
        no_decay: biases (dim < 2), norms, embeddings
        """
        decay = []
        no_decay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2 or "norm" in name or "token_embedding" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {"params": decay, "label": "decay"},
            {"params": no_decay, "label": "no_decay", "weight_decay": 0.0},
        ]

    # ══════════════════════════════════════════════════════════════════
    # Description
    # ══════════════════════════════════════════════════════════════════

    def describe(self) -> dict[str, Any]:
        """V7.1 architecture description for analysis.json and logging."""
        cfg = self.config
        total = sum(p.numel() for p in self.parameters())

        kernel_params = sum(p.numel() for p in self.hebbian_kernel.parameters())
        conv_params = sum(p.numel() for p in self.conv.parameters())
        embed_params = sum(p.numel() for p in self.token_embedding.parameters())
        readout_params = sum(p.numel() for p in self.readout.parameters())

        theta_params = sum(p.numel() for p in self.theta_blocks.parameters())
        halt_params = (
            sum(p.numel() for p in self.halting.parameters())
            if cfg.enable_halting else 0
        )

        return {
            "architecture": "AENCA v7.1 (BDH-inspired Hebbian NCA)",
            "total_params": total,
            "hebbian_kernel_params": kernel_params,
            "conv_params": conv_params,
            "embed_params": embed_params,
            "readout_params": readout_params,
            "theta_block_params": theta_params,
            "halt_params": halt_params,
            "d_model": cfg.d_model,
            "d_concept": cfg.d_concept,
            "d_ff_theta": cfg.d_ff_theta,
            "n_theta": cfg.n_theta,
            "t_gamma": cfg.t_gamma,
            "total_gamma_steps": cfg.total_gamma_steps,
            "n_heads": cfg.n_heads,
            "d_head": cfg.d_head,
            "kernel_radius": cfg.kernel_radius,
            "enable_halting": cfg.enable_halting,
            "mos_components": cfg.mos_components,
            "tie_embeddings": cfg.tie_embeddings,
            "vocab_size": cfg.vocab_size,
            "paradigm": "BDH-inspired Hebbian NCA (ReLU-lowrank + cross-modal gating)",
        }

    # ══════════════════════════════════════════════════════════════════
    # Lifecycle Hooks
    # ══════════════════════════════════════════════════════════════════

    def get_phases(self, step: int, max_steps: int) -> str:
        """Return active training phases for logging."""
        return "HNCA"  # Hebbian NCA

    def extra_metrics(self) -> dict[str, float]:
        """V7.1-specific monitoring metrics."""
        metrics: dict[str, float] = {}

        # HebbianKernel: decoder norm (how much state update kernel proposes)
        metrics["kernel_dec_norm"] = self.hebbian_kernel.W_dec.weight.norm().item()

        # HebbianKernel: encoder norms (concept encoding strength)
        metrics["kernel_enc_p_norm"] = self.hebbian_kernel.W_enc_p.weight.norm().item()
        metrics["kernel_enc_z_norm"] = self.hebbian_kernel.W_enc_z.weight.norm().item()

        # Theta block activity: out_proj and HebbianFFN norms per block
        for i, block in enumerate(self.theta_blocks):
            metrics[f"theta{i}_attn_norm"] = block.out_proj.weight.norm().item()
            metrics[f"theta{i}_ffn_dec_norm"] = block.ffn.W_dec.weight.norm().item()

        # Halting: halt bias
        if self.config.enable_halting:
            metrics["halt_bias"] = self.halting.W_halt.bias.item()

        return metrics
