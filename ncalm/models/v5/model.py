"""AENCA V5 main model: Precision-Routed NCA Language Model.

Implements the complete V5 training and inference pipeline:
1. Token embedding + sinusoidal position encoding (§3.2)
2. K NCA steps across 3 phases (§6):
   - Causal Conv1D local perception
   - PRMP global message passing
   - Phase-specialized kernel with SwiGLU trunk
   - Per-dimension gated update
3. MoS readout (§7)
4. Autoregressive CE loss (§8, §9)

This model replaces V4/V4.1's MDLM diffusion training with standard
autoregressive next-token prediction, uses PRMP instead of sensors for
global communication, and uses 3 phase-specialized kernels instead of 1.

Reference: model_documents/aenca_v5_architecture.md (full document).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ncalm.model import NCALMModel, register_model
from ncalm.models.v5.config import V5Config
from ncalm.models.v5.kernel import PhaseKernel
from ncalm.models.v5.prmp import PrecisionRoutedMessagePassing
from ncalm.models.v5.readout import MixtureOfSoftmaxReadout


class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal position encoding (§3.2, Eq 3.4).

    Produces d-dimensional sinusoidal position features.
    Used for:
        - d_model-dimensional PE added to initial cell state (Eq 3.4)
        - d_pe-dimensional PE passed as kernel input for position awareness
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
        """Return position encodings for positions [0, seq_len).

        Returns:
            pe: (L, d) position encoding tensor
        """
        return self.pe[:seq_len]


class CausalConv1D(nn.Module):
    """Causal 1D convolution for local perception (§6.1, Eq 6.1-6.2).

    Left-padded convolution: cell i sees only s_{i-r}, ..., s_i.
    kernel_size = r + 1 for causal (no future context).
    Each phase has its own conv weights.
    """

    def __init__(self, d_model: int, kernel_radius: int):
        super().__init__()
        kernel_size = kernel_radius + 1  # Causal: r+1
        self.padding = kernel_radius  # Left-pad by r
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=0,  # Manual padding for causal
            bias=False,
        )
        self.norm = nn.RMSNorm(d_model)

        # Initialize: Xavier uniform (§10.2, Eq 10.2)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Compute causal local perception.

        Args:
            s: (B, L, d_model) cell states

        Returns:
            P: (B, L, d_model) local perception output
        """
        # Pre-norm (§6.3)
        s_norm = self.norm(s)

        # (B, L, D) -> (B, D, L) for Conv1d
        x = s_norm.transpose(1, 2)

        # Left-pad for causal: pad (left, right) = (r, 0)
        x = F.pad(x, (self.padding, 0))

        # Conv1d
        P = self.conv(x)  # (B, D, L)

        # (B, D, L) -> (B, L, D)
        return P.transpose(1, 2)


@register_model("v5")
class AENCAv5Model(NCALMModel):
    """AENCA V5: Precision-Routed NCA Language Model.

    A Neural Cellular Automaton with PRMP global communication, phase-specialized
    kernel bank, and autoregressive training. Solves V4/V4.1's bandwidth
    starvation, readout bottleneck, and gradient instability.

    Architecture (§2.1):
        init → (CausalConv + PRMP → PhaseKernel → GatedUpdate) × K → MoS Readout

    NCA Purity (§13):
        - Uniform kernel within each phase
        - Local computation (cells process own state + received messages)
        - Iterative K-step loop
        - Per-position output
    """

    def __init__(self, config_dict: dict) -> None:
        super().__init__()

        # Parse flat config dict into V5Config (same pattern as V4)
        v5_keys = {f.name for f in V5Config.__dataclass_fields__.values()}
        v5_params = {k: v for k, v in config_dict.items() if k in v5_keys}
        self.config = V5Config(**v5_params)
        cfg = self.config

        # ── Token Embedding (§3.2, Eq 3.2) ──────────────────────────────
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.token_embedding.weight, std=1.0 / math.sqrt(cfg.d_model))

        # ── Position Encoding (§3.2) ─────────────────────────────────────
        # d_model-dimensional PE added to initial cell state (Eq 3.4)
        self.state_pe = SinusoidalPositionEncoding(cfg.d_model, cfg.max_seq_len * 2)
        # d_pe-dimensional PE for kernel input (separate positional signal)
        self.pos_encoder = SinusoidalPositionEncoding(cfg.d_pe, cfg.max_seq_len * 2)

        # ── Causal Convolutions — one per phase (§6.1) ───────────────────
        self.convs = nn.ModuleList([
            CausalConv1D(cfg.d_model, cfg.kernel_radius)
            for _ in range(3)
        ])

        # ── PRMP — shared across all steps (§4) ─────────────────────────
        self.prmp = PrecisionRoutedMessagePassing(cfg)

        # ── Phase-Specialized Kernel Bank (§5.2) ────────────────────────
        self.kernels = nn.ModuleList([
            PhaseKernel(cfg, phase_steps)
            for phase_steps in cfg.phase_steps
        ])

        # ── Readout (§7) ─────────────────────────────────────────────────
        emb_weight = self.token_embedding.weight if cfg.tie_embeddings else None
        self.readout = MixtureOfSoftmaxReadout(cfg, emb_weight)

    # ── Training Forward Pass (§8.1, §9.1) ───────────────────────────────

    def forward_train(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Complete V5 training step with autoregressive loss.

        Args:
            batch: (B, L) token IDs, dtype long.

        Returns:
            loss:  Scalar total training loss for backward.
            info:  Dict with 'ce' key plus auxiliary loss components.
        """
        B, L = batch.shape
        cfg = self.config

        # Step 1: Token embedding (§3.2, Eq 3.2)
        s = self.token_embedding(batch)  # (B, L, d_model)

        # Step 1b: Add positional encoding to initial state (Eq 3.4)
        # s_i^(0) = W_E[x_i] + SinPE(i)
        s = s + self.state_pe(L).unsqueeze(0)  # broadcast (1, L, d_model) over batch

        # Step 2: Position encoding for kernel input
        pe = self.pos_encoder(L)  # (L, d_pe)
        pe = pe.unsqueeze(0).expand(B, -1, -1)  # (B, L, d_pe)

        # Step 3: Save initial state for regularization (§9.1, Eq 9.3)
        s0 = s.detach().clone()

        # Step 4: Run K NCA steps across 3 phases
        s, aux_logits = self._run_nca_steps(s, pe)

        # Step 5: Final readout (§7)
        logits = self.readout(s)  # (B, L, V)

        # Step 6: Compute losses (§9.1)
        # AR loss: predict x_{i+1} from position i (Eq 9.2)
        # logits[:, :-1] predicts tokens batch[:, 1:]
        logits_flat = logits[:, :-1].reshape(-1, cfg.vocab_size)
        targets_flat = batch[:, 1:].reshape(-1)

        if self.readout.returns_log_probs:
            # MoS readout returns log-probabilities → use nll_loss (no double log_softmax)
            ce_loss = F.nll_loss(logits_flat, targets_flat, reduction="mean")
        else:
            # Single readout returns raw logits → use cross_entropy
            ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")

        total_loss = ce_loss

        info: dict[str, Any] = {"ce": ce_loss.detach()}

        # State regularization (Eq 9.3)
        # ℒ_reg = (1/L) Σ_i ||s_i^(K) - s_i^(0)||² / d_model
        # = mean over (B, L) of [sum over d_model / d_model]
        # = mean over all dims (B, L, d_model) of squared diff
        if cfg.lambda_reg > 0:
            reg_loss = (s - s0).pow(2).mean()
            total_loss = total_loss + cfg.lambda_reg * reg_loss
            info["reg"] = reg_loss.detach()

        # Auxiliary early-exit loss at phase boundaries (Eq 9.4)
        if cfg.lambda_aux > 0 and aux_logits:
            aux_total = torch.tensor(0.0, device=batch.device)
            K = cfg.total_steps
            for step_k, aux_log in aux_logits.items():
                w_k = (step_k / K) ** 2  # Quadratic weighting
                aux_flat = aux_log[:, :-1].reshape(-1, cfg.vocab_size)
                if self.readout.returns_log_probs:
                    aux_ce = F.nll_loss(aux_flat, targets_flat, reduction="mean")
                else:
                    aux_ce = F.cross_entropy(aux_flat, targets_flat, reduction="mean")
                aux_total = aux_total + w_k * aux_ce
            aux_total = aux_total / len(aux_logits)
            total_loss = total_loss + cfg.lambda_aux * aux_total
            info["aux"] = aux_total.detach()

        return total_loss, info

    def _run_nca_steps(
        self,
        s: torch.Tensor,
        pe: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Run K NCA steps across 3 phases with gradient checkpointing.

        Args:
            s:  (B, L, d_model) initial cell states
            pe: (B, L, d_pe) position encodings

        Returns:
            s:          (B, L, d_model) final cell states
            aux_logits: {step: (B, L, V)} logits at phase boundaries for aux loss
        """
        cfg = self.config
        C = cfg.checkpoint_interval
        K = cfg.total_steps
        phase_steps = cfg.phase_steps  # [K₁, K₂, K₃]

        # Determine phase boundaries for aux loss
        # Aux readout at end of phase 1 and phase 2 (Eq 9.4)
        aux_steps = set()
        cumulative = 0
        for p_idx in range(len(phase_steps) - 1):
            cumulative += phase_steps[p_idx]
            aux_steps.add(cumulative - 1)  # Last step of each phase (except final)

        # Side-channel for aux logits (populated inside checkpoint)
        aux_substrates: dict[int, torch.Tensor] = {}
        aux_logits: dict[int, torch.Tensor] = {}

        # Build step-to-phase mapping
        phase_map = []
        for p_idx, p_steps in enumerate(phase_steps):
            phase_map.extend([p_idx] * p_steps)

        # Run in checkpoint segments
        global_step = 0
        for seg_start in range(0, K, C):
            seg_end = min(seg_start + C, K)

            def run_segment(
                _s,
                _seg_start=seg_start,
                _seg_end=seg_end,
            ):
                """Run one checkpoint segment of NCA steps."""
                for k in range(_seg_start, _seg_end):
                    p_idx = phase_map[k]

                    # Step 1: Local perception (§6.1)
                    P = self.convs[p_idx](_s)

                    # Step 2: PRMP messages (§6.1, Eq 6.3)
                    M = self.prmp(_s)

                    # Step 3: Kernel application (§6.1, Eq 6.4)
                    delta_s, gate = self.kernels[p_idx](P, M, pe)

                    # Step 4: Gated update (§6.1, Eq 6.5)
                    _s = _s + gate * delta_s

                    # Save substrate at aux steps for later readout
                    if k in aux_steps:
                        aux_substrates[k] = _s

                return _s

            if self.training:
                s = checkpoint(
                    run_segment,
                    s,
                    use_reentrant=False,
                )
            else:
                s = run_segment(s)

            # Compute aux logits from saved substrates (outside checkpoint for proper grad flow)
            for k in list(aux_substrates.keys()):
                if seg_start <= k < seg_end:
                    aux_logits[k] = self.readout(aux_substrates.pop(k))

        return s, aux_logits

    # ── Inference (§8.1) ──────────────────────────────────────────────────

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Autoregressive inference: compute logits for all positions.

        Args:
            input_ids: (B, L) token IDs.

        Returns:
            logits: (B, L, vocab_size) predictions at ALL positions.
        """
        B, L = input_ids.shape
        cfg = self.config

        # Embed tokens
        s = self.token_embedding(input_ids)  # (B, L, d_model)

        # Add positional encoding to initial state (Eq 3.4)
        s = s + self.state_pe(L).unsqueeze(0)

        # Position encoding for kernel
        pe = self.pos_encoder(L).unsqueeze(0).expand(B, -1, -1)

        # Build phase map
        phase_map = []
        for p_idx, p_steps in enumerate(cfg.phase_steps):
            phase_map.extend([p_idx] * p_steps)

        # Run K NCA steps (no checkpointing during inference)
        K = cfg.total_steps
        for k in range(K):
            p_idx = phase_map[k]
            P = self.convs[p_idx](s)
            M = self.prmp(s)
            delta_s, gate = self.kernels[p_idx](P, M, pe)
            s = s + gate * delta_s

        # Final readout
        logits = self.readout(s)
        return logits

    # ── Parameter Groups ──────────────────────────────────────────────────

    def get_param_groups(self) -> list[dict]:
        """Parameter groups for optimizer (§8.3).

        Uses type-based decay/no-decay assignment (matching GPT-2 baseline):
        - decay: 2D+ weight matrices in kernels, prmp, convs, readout projection
        - no_decay: all biases (dim < 2), all norms, token embeddings

        This ensures symmetric regularization treatment with the GPT-2
        baseline for fair academic comparison.
        """
        decay = []
        no_decay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # No weight decay for: biases/scalars (dim < 2), norms, embeddings
            if param.dim() < 2 or "norm" in name or "token_embedding" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {"params": decay, "label": "decay"},
            {"params": no_decay, "label": "no_decay", "weight_decay": 0.0},
        ]

    # ── Description ───────────────────────────────────────────────────────

    def describe(self) -> dict[str, Any]:
        """V5 architecture description for analysis.json and logging."""
        cfg = self.config
        total = sum(p.numel() for p in self.parameters())

        kernel_params = sum(
            p.numel() for kernel in self.kernels for p in kernel.parameters()
        )
        prmp_params = sum(p.numel() for p in self.prmp.parameters())
        conv_params = sum(
            p.numel() for conv in self.convs for p in conv.parameters()
        )
        embed_params = sum(p.numel() for p in self.token_embedding.parameters())
        readout_params = sum(p.numel() for p in self.readout.parameters())

        return {
            "architecture": "AENCA v5.1",
            "total_params": total,
            "kernel_params": kernel_params,
            "prmp_params": prmp_params,
            "conv_params": conv_params,
            "embed_params": embed_params,
            "readout_params": readout_params,
            "d_model": cfg.d_model,
            "d_msg": cfg.d_msg,
            "d_ff": cfg.d_ff,
            "d_pe": cfg.d_pe,
            "n_heads": cfg.n_heads,
            "n_trunk": cfg.n_trunk,
            "K": cfg.total_steps,
            "phases": f"{cfg.k_phase1}+{cfg.k_phase2}+{cfg.k_phase3}",
            "kernel_radius": cfg.kernel_radius,
            "mos_components": cfg.mos_components,
            "tie_embeddings": cfg.tie_embeddings,
            "vocab_size": cfg.vocab_size,
            "paradigm": "Autoregressive + Precision-Routed Message Passing (NCA)",
        }

    # ── Lifecycle Hooks ───────────────────────────────────────────────────

    def get_phases(self, step: int, max_steps: int) -> str:
        """Return active training phases for logging."""
        return "AR"

    def extra_metrics(self) -> dict[str, float]:
        """V5-specific metrics for logging."""
        # Monitor W_O norm to track PRMP activation (§15.3)
        w_o_norm = 0.0
        for name, param in self.prmp.named_parameters():
            if "W_O" in name:
                w_o_norm = param.data.norm().item()
                break
        return {"w_o_norm": w_o_norm}
