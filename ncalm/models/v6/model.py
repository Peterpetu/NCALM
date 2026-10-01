"""AENCA V6 main model: Oscillator-Controlled Adaptive NCA Language Model.

Architecture:
  1. Token embedding + sinusoidal position encoding → single state z
  2. T_max NCA steps (up to 256), each step:
     a. Oscillator tick: Kuramoto phase update → temporal gates + slow context
     b. Causal conv → local perception P
     c. Fast kernel([P; slow_ctx; pe]) → (Δz, kernel_gate)
     d. z ← z + pos_active · kernel_gate · osc_gate · Δz
     e. Per-position readout accumulation (tokens halt independently)
  3. MoS readout from accumulated z_out

Temporal dynamics are fully controlled by a per-cell coupled Kuramoto
oscillator network with three coupling mechanisms:
  - Inter-channel (K×K): oscillator channels synchronize within each cell
  - Local spatial (depthwise Conv1d): neighboring cells' phases couple
  - Global mean-field (causal cumsum): long-range phase coherence

Key properties:
  - Single state vector z (no z_H/z_L split)
  - Per-position halting (individual tokens converge at different steps)
  - Self-regulated effective timescales (K=8 is capacity, not fixed count)
  - Global coordination via oscillator mean-field (replaces attention)
  - Segment-based gradient checkpointing for T_max=256

Reference: Kuramoto (1984), "Chemical Oscillations, Waves, and Turbulence."
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ncalm.model import NCALMModel, register_model
from ncalm.models.v6.config import V6Config
from ncalm.models.v6.kernel import FastKernel
from ncalm.models.v6.oscillator import CoupledOscillatorNetwork

# Reuse readout from v5 — it only depends on config having
# d_model, vocab_size, mos_components, tie_embeddings attributes.
from ncalm.models.v5.readout import MixtureOfSoftmaxReadout


class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal position encoding (same as V5).

    Produces d-dimensional sinusoidal position features.
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
        return self.pe[:seq_len]


class CausalConv1D(nn.Module):
    """Causal 1D convolution for local perception (same as V5).

    Left-padded convolution: cell i sees only s_{i-r}, ..., s_i.
    V6 uses ONE shared conv (not per-phase like V5).
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
        s_norm = self.norm(s)
        x = s_norm.transpose(1, 2)
        x = F.pad(x, (self.padding, 0))
        P = self.conv(x)
        return P.transpose(1, 2)


@register_model("v6")
class AENCAv6Model(NCALMModel):
    """AENCA V6: Oscillator-Controlled Adaptive NCA Language Model.

    A Neural Cellular Automaton with per-cell coupled Kuramoto oscillators
    controlling all temporal dynamics. Each token has K=8 oscillator channels
    that self-organize into emergent timescales via synchronization.

    Architecture:
        embed → T_max NCA steps of:
            oscillator.step(z, φ) → osc_gate, readout_prob, slow_ctx
            conv(z) → P (local perception)
            kernel([P; slow_ctx; pe]) → (Δz, kernel_gate)
            z ← z + pos_active · kernel_gate · osc_gate · Δz
            readout accumulation (per-position)
        → MoS Readout(z_out)

    NCA Purity:
        - Shared kernel for all steps
        - Local spatial computation (causal conv, radius r)
        - Global coordination via oscillator mean-field (not attention)
        - Per-position output with independent halting
        - Iterative T-step loop with oscillator-controlled temporal structure
    """

    def __init__(self, config_dict: dict) -> None:
        super().__init__()

        # Parse flat config dict into V6Config
        v6_keys = {f.name for f in V6Config.__dataclass_fields__.values()}
        v6_params = {k: v for k, v in config_dict.items() if k in v6_keys}
        self.config = V6Config(**v6_params)
        cfg = self.config

        # ── Token Embedding ──────────────────────────────────────────────
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.token_embedding.weight, std=1.0 / math.sqrt(cfg.d_model))

        # ── Position Encoding ────────────────────────────────────────────
        self.state_pe = SinusoidalPositionEncoding(cfg.d_model, cfg.max_seq_len * 2)
        self.pos_encoder = SinusoidalPositionEncoding(cfg.d_pe, cfg.max_seq_len * 2)

        # ── Local Perception (ONE shared causal conv) ────────────────────
        self.conv = CausalConv1D(cfg.d_model, cfg.kernel_radius)

        # ── Fast Kernel (ONE, shared across all NCA steps) ───────────────
        self.fast_kernel = FastKernel(cfg)

        # ── Coupled Oscillator Network ───────────────────────────────────
        self.oscillator = CoupledOscillatorNetwork(cfg)

        # ── Attention Hook (dormant by default) ──────────────────────────
        if cfg.enable_attention:
            from ncalm.models.v6.slow_updater import SlowUpdater
            self.slow_updater = SlowUpdater(cfg)

        # ── Readout (same as V5) ─────────────────────────────────────────
        emb_weight = self.token_embedding.weight if cfg.tie_embeddings else None
        self.readout = MixtureOfSoftmaxReadout(cfg, emb_weight)

        # ── Cache deep supervision step set for fast lookup ──────────────
        self._ds_steps = set(cfg.deep_supervision_steps)

    # ══════════════════════════════════════════════════════════════════════
    # Training Forward Pass
    # ══════════════════════════════════════════════════════════════════════

    def forward_train(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Complete V6 training step with autoregressive loss.

        Runs the oscillator-controlled NCA loop with:
        - Per-position readout accumulation (tokens halt independently)
        - Oscillator-derived temporal gates and slow context
        - Segment-based gradient checkpointing
        - Deep supervision at powers-of-2 step indices
        - Frequency diversity regularization

        Args:
            batch: (B, L) token IDs, dtype long.

        Returns:
            loss:  Scalar total training loss for backward.
            info:  Dict with 'ce' key plus auxiliary loss components.
        """
        B, L = batch.shape
        cfg = self.config

        # Step 1: Token embedding + position encoding
        s = self.token_embedding(batch)           # (B, L, d_model)
        s = s + self.state_pe(L).unsqueeze(0)

        # Step 2: Position encoding for kernel input
        pe = self.pos_encoder(L).unsqueeze(0).expand(B, -1, -1)  # (B, L, d_pe)

        # Step 3: Initialize single state (no z_H/z_L split)
        z = s

        # Step 4: Save initial state for regularization
        s0 = s.detach().clone()

        # Step 5: Run oscillator-controlled NCA loop
        z_out, loop_info = self._run_oscillator_loop(z, pe, batch)

        # Step 6: Final readout from accumulated z_out
        logits = self.readout(z_out)              # (B, L, V)

        # Step 7: Compute losses
        logits_flat = logits[:, :-1].reshape(-1, cfg.vocab_size)
        targets_flat = batch[:, 1:].reshape(-1)

        if self.readout.returns_log_probs:
            ce_loss = F.nll_loss(logits_flat, targets_flat, reduction="mean")
        else:
            ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")

        total_loss = ce_loss
        info: dict[str, Any] = {"ce": ce_loss.detach()}

        # State regularization: ‖z_out - s0‖² / d_model
        if cfg.lambda_reg > 0:
            reg_loss = (z_out - s0).pow(2).mean()
            total_loss = total_loss + cfg.lambda_reg * reg_loss
            info["reg"] = reg_loss.detach()

        # Frequency diversity regularization
        if cfg.lambda_freq_reg > 0 and loop_info["freq_div_loss"] is not None:
            total_loss = total_loss + cfg.lambda_freq_reg * loop_info["freq_div_loss"]
            info["freq_div"] = loop_info["freq_div_loss"].detach()

        # Deep supervision auxiliary losses
        if cfg.lambda_aux > 0 and loop_info["aux_loss"] is not None:
            total_loss = total_loss + cfg.lambda_aux * loop_info["aux_loss"]
            info["aux"] = loop_info["aux_loss"].detach()

        # Monitoring metrics (detached, not in loss)
        info["expected_steps"] = loop_info["expected_steps"]
        info["pct_halted_early"] = loop_info["pct_halted_early"]

        return total_loss, info

    # ══════════════════════════════════════════════════════════════════════
    # Oscillator-Controlled NCA Loop
    # ══════════════════════════════════════════════════════════════════════

    def _nca_step(
        self,
        z: torch.Tensor,
        phi: torch.Tensor,
        pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single NCA step: oscillator tick → conv → kernel → gated update.

        Factored out for gradient checkpointing. This is the atomic unit
        of computation that gets repeated T_max times.

        Args:
            z:   (B, L, d_model) current cell states
            phi: (B, L, K) current oscillator phases
            pe:  (B, L, d_pe) position encoding

        Returns:
            z_new:       (B, L, d_model) updated cell states (before active masking)
            phi_new:     (B, L, K) updated oscillator phases
            osc_gate:    (B, L, d_model) oscillator temporal gate
            readout_prob: (B, L) readout accumulation weight
        """
        # 1. Oscillator tick
        phi_new, osc_gate, readout_prob, slow_ctx = self.oscillator.step(z, phi)

        # 2. Local perception
        P = self.conv(z)                          # (B, L, d_model)

        # 3. Kernel proposal
        delta, kernel_gate = self.fast_kernel(P, slow_ctx, pe)

        # 4. Gated state update (triple gating: kernel × oscillator × delta)
        z_new = z + kernel_gate * osc_gate * delta

        return z_new, phi_new, osc_gate, readout_prob

    def _run_segment(
        self,
        z: torch.Tensor,
        phi: torch.Tensor,
        z_out: torch.Tensor,
        remaining: torch.Tensor,
        pe: torch.Tensor,
        t_start: int,
        t_end: int,
        freq_div_accum: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Run a segment of NCA steps (for gradient checkpointing).

        Processes steps [t_start, t_end) with per-position halting and
        readout accumulation. This function is designed to be wrapped in
        torch.utils.checkpoint.checkpoint during training.

        Args:
            z:              (B, L, d_model) cell states at segment start
            phi:            (B, L, K) oscillator phases at segment start
            z_out:          (B, L, d_model) accumulated readout state
            remaining:      (B, L) remaining readout mass per position
            pe:             (B, L, d_pe) position encoding
            t_start:        first step index in this segment
            t_end:          one past last step index
            freq_div_accum: (scalar) accumulated frequency diversity loss

        Returns:
            z:              updated cell states
            phi:            updated oscillator phases
            z_out:          updated readout accumulator
            remaining:      updated remaining mass
            freq_div_accum: updated freq diversity loss accumulator
        """
        cfg = self.config
        eps = cfg.readout_halt_eps
        t_min = cfg.t_min
        T_max = cfg.t_max

        for t in range(t_start, t_end):
            # Per-position active mask: positions with remaining mass
            pos_active = (remaining > eps).float()           # (B, L)

            # Skip if all positions in all samples are halted
            if pos_active.sum() == 0:
                break

            # NCA step: oscillator → conv → kernel → gated update
            z_candidate, phi, osc_gate, readout_prob = self._nca_step(z, phi, pe)

            # Apply active mask: halted positions get zero update
            # but their frozen state persists for neighbor conv reads
            z = z + pos_active.unsqueeze(-1) * (z_candidate - z)

            # Readout accumulation (per-position)
            can_readout = (t >= t_min)
            is_last_step = (t == T_max - 1)

            if can_readout:
                if is_last_step:
                    # Last step: absorb all remaining mass
                    effective_readout = remaining * pos_active    # (B, L)
                else:
                    # Normal step: readout_prob fraction of remaining mass
                    effective_readout = readout_prob * pos_active  # (B, L)
                    # Clamp to remaining mass (can't readout more than what's left)
                    effective_readout = torch.min(effective_readout, remaining)

                # Accumulate readout-weighted state
                z_out = z_out + effective_readout.unsqueeze(-1) * z
                remaining = remaining - effective_readout

            # Accumulate frequency diversity loss (sampled, not every step)
            if t % 8 == 0:  # sample every 8th step to reduce overhead
                freq_div_accum = freq_div_accum + self.oscillator.compute_freq_diversity(z)

        return z, phi, z_out, remaining, freq_div_accum

    def _run_oscillator_loop(
        self,
        z: torch.Tensor,
        pe: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run the full oscillator-controlled NCA loop.

        The main NCA computation: iterates up to T_max steps, each controlled
        by coupled Kuramoto oscillators. Per-position readout accumulation
        allows different tokens to halt at different steps.

        Gradient checkpointing: the loop is split into segments of
        checkpoint_segment_size steps. Each segment is wrapped in
        torch.utils.checkpoint.checkpoint during training.

        Args:
            z:     (B, L, d_model) initial cell states (from embedding)
            pe:    (B, L, d_pe) position encoding
            batch: (B, L) token IDs for deep supervision (optional)

        Returns:
            z_out:     (B, L, d_model) readout-accumulated state
            loop_info: dict with diagnostics and loss components
        """
        cfg = self.config
        B, L, D = z.shape
        device = z.device
        T_max = cfg.t_max
        seg_size = cfg.checkpoint_segment_size

        # ── Initialize oscillator phases ─────────────────────────────
        phi = self.oscillator.init_phase(B, L, device, deterministic=not self.training)

        # ── Readout accumulators (per-position) ──────────────────────
        z_out = torch.zeros_like(z)                          # (B, L, D)
        remaining = torch.ones(B, L, device=device)          # (B, L)

        # ── Frequency diversity loss accumulator ─────────────────────
        freq_div_accum = torch.tensor(0.0, device=device)

        # ── Deep supervision substrates ──────────────────────────────
        # Collect (z_out, remaining) snapshots at power-of-2 steps for aux losses
        # Both are snapshotted so the mass guard uses snapshot-time remaining
        aux_substrates: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        # ── Run loop in segments for gradient checkpointing ──────────
        n_segments = (T_max + seg_size - 1) // seg_size
        actual_last_step = 0

        for seg_idx in range(n_segments):
            t_start = seg_idx * seg_size
            t_end = min(t_start + seg_size, T_max)

            # Early exit: all positions halted
            if (remaining <= cfg.readout_halt_eps).all():
                break

            if self.training:
                z, phi, z_out, remaining, freq_div_accum = checkpoint(
                    self._run_segment,
                    z, phi, z_out, remaining, pe,
                    t_start, t_end, freq_div_accum,
                    use_reentrant=False,
                )
            else:
                z, phi, z_out, remaining, freq_div_accum = self._run_segment(
                    z, phi, z_out, remaining, pe,
                    t_start, t_end, freq_div_accum,
                )

            actual_last_step = t_end

            # Collect deep supervision snapshot at segment boundaries
            # Check if any power-of-2 step falls in this segment
            for ds_step in self._ds_steps:
                if t_start < ds_step <= t_end and ds_step < T_max:
                    aux_substrates[ds_step] = (z_out.clone(), remaining.clone())

        # ── Compute deep supervision losses ──────────────────────────
        aux_loss: torch.Tensor | None = None
        if batch is not None and cfg.lambda_aux > 0 and aux_substrates:
            targets_flat = batch[:, 1:].reshape(-1)
            aux_total = torch.tensor(0.0, device=device)
            n_aux = 0

            for step_t, (z_out_snap, remaining_snap) in aux_substrates.items():
                # Weight: (t/T_max)² — later steps weighted more
                w = (step_t / T_max) ** 2

                # Only compute aux loss if snapshot had accumulated mass
                # Uses snapshot-time remaining, not end-of-loop remaining
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

        # ── Normalize freq diversity loss ────────────────────────────
        n_freq_samples = max(actual_last_step // 8, 1)
        freq_div_loss = freq_div_accum / n_freq_samples if n_freq_samples > 0 else None

        # ── Compute monitoring metrics ───────────────────────────────
        with torch.no_grad():
            # Expected steps: where remaining mass drops to near-zero
            halted_mask = (remaining <= cfg.readout_halt_eps)  # (B, L)
            pct_halted = halted_mask.float().mean().item()

            # Rough expected steps from how much mass was accumulated
            # (1 - remaining) gives fraction of readout mass assigned
            mass_used = 1.0 - remaining.mean().item()

            # Estimate mean step from segment boundaries
            expected_steps = actual_last_step * mass_used

        loop_info: dict[str, Any] = {
            "freq_div_loss": freq_div_loss,
            "aux_loss": aux_loss,
            "expected_steps": expected_steps,
            "pct_halted_early": pct_halted,
        }

        return z_out, loop_info

    # ══════════════════════════════════════════════════════════════════════
    # Inference
    # ══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Autoregressive inference with oscillator-controlled adaptive compute.

        Per-position halting is active: easy tokens converge fast and halt
        early, complex tokens continue iterating. A sample exits when all
        its positions have halted.

        Args:
            input_ids: (B, L) token IDs.

        Returns:
            logits: (B, L, vocab_size) predictions at ALL positions.
        """
        B, L = input_ids.shape
        cfg = self.config
        device = input_ids.device

        # Embed tokens
        s = self.token_embedding(input_ids)
        s = s + self.state_pe(L).unsqueeze(0)
        pe = self.pos_encoder(L).unsqueeze(0).expand(B, -1, -1)

        # Initialize state and oscillator phases
        z = s
        phi = self.oscillator.init_phase(B, L, device, deterministic=True)

        # Per-position readout accumulators
        z_out = torch.zeros_like(z)
        remaining = torch.ones(B, L, device=device)
        eps = cfg.readout_halt_eps

        for t in range(cfg.t_max):
            pos_active = (remaining > eps).float()

            # Sample-level early exit
            if pos_active.sum() == 0:
                break

            # NCA step
            z_candidate, phi, osc_gate, readout_prob = self._nca_step(z, phi, pe)
            z = z + pos_active.unsqueeze(-1) * (z_candidate - z)

            # Readout accumulation
            if t >= cfg.t_min:
                is_last = (t == cfg.t_max - 1)
                if is_last:
                    effective_readout = remaining * pos_active
                else:
                    effective_readout = torch.min(readout_prob * pos_active, remaining)

                z_out = z_out + effective_readout.unsqueeze(-1) * z
                remaining = remaining - effective_readout

        # Final readout
        logits = self.readout(z_out)
        return logits

    # ══════════════════════════════════════════════════════════════════════
    # Parameter Groups
    # ══════════════════════════════════════════════════════════════════════

    def get_param_groups(self) -> list[dict]:
        """Parameter groups for optimizer.

        Uses type-based decay/no-decay assignment (matching V5 and GPT-2):
        - decay: 2D+ weight matrices in kernel, oscillator, readout
        - no_decay: all biases (dim < 2), all norms, token embeddings,
                    oscillator coupling parameters (small, shouldn't decay)
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

    # ══════════════════════════════════════════════════════════════════════
    # Description
    # ══════════════════════════════════════════════════════════════════════

    def describe(self) -> dict[str, Any]:
        """V6 architecture description for analysis.json and logging."""
        cfg = self.config
        total = sum(p.numel() for p in self.parameters())

        kernel_params = sum(p.numel() for p in self.fast_kernel.parameters())
        osc_params = sum(p.numel() for p in self.oscillator.parameters())
        conv_params = sum(p.numel() for p in self.conv.parameters())
        embed_params = sum(p.numel() for p in self.token_embedding.parameters())
        readout_params = sum(p.numel() for p in self.readout.parameters())

        desc = {
            "architecture": "AENCA v6 (Oscillator-Controlled Adaptive NCA)",
            "total_params": total,
            "fast_kernel_params": kernel_params,
            "oscillator_params": osc_params,
            "conv_params": conv_params,
            "embed_params": embed_params,
            "readout_params": readout_params,
            "d_model": cfg.d_model,
            "d_ff": cfg.d_ff,
            "d_pe": cfg.d_pe,
            "n_trunk": cfg.n_trunk,
            "kernel_radius": cfg.kernel_radius,
            "n_osc_channels": cfg.n_osc_channels,
            "t_max": cfg.t_max,
            "t_min": cfg.t_min,
            "enable_spatial_coupling": cfg.enable_spatial_coupling,
            "enable_mean_field": cfg.enable_mean_field,
            "enable_attention": cfg.enable_attention,
            "mos_components": cfg.mos_components,
            "tie_embeddings": cfg.tie_embeddings,
            "vocab_size": cfg.vocab_size,
            "checkpoint_segment_size": cfg.checkpoint_segment_size,
            "paradigm": "Oscillator-Controlled Adaptive NCA (Kuramoto)",
        }

        if cfg.enable_attention:
            desc["slow_updater_params"] = sum(
                p.numel() for p in self.slow_updater.parameters()
            )

        return desc

    # ══════════════════════════════════════════════════════════════════════
    # Lifecycle Hooks
    # ══════════════════════════════════════════════════════════════════════

    def get_phases(self, step: int, max_steps: int) -> str:
        """Return active training phases for logging."""
        return "ONCA"  # Oscillator NCA

    def extra_metrics(self) -> dict[str, float]:
        """V6 oscillator-specific metrics for logging."""
        metrics: dict[str, float] = {}

        # Oscillator base coupling matrix norm
        C = self.oscillator._base_coupling()
        metrics["coupling_base_norm"] = C.norm().item()

        # Coupling strengths
        if self.oscillator.enable_spatial:
            metrics["kappa_local"] = self.oscillator.kappa_local.item()
        if self.oscillator.enable_mean_field:
            metrics["kappa_global"] = self.oscillator.kappa_global.item()

        # Fast kernel W_delta norm (kernel activity)
        for name, param in self.fast_kernel.named_parameters():
            if "W_delta" in name:
                metrics["fast_delta_norm"] = param.data.norm().item()
                break

        # Slow context projection norm (output layer of slow_ctx MLP)
        for name, param in self.oscillator.named_parameters():
            if "slow_ctx_down" in name and "weight" in name:
                metrics["slow_ctx_norm"] = param.data.norm().item()
                break

        # Readout head bias (monitors readout probability evolution)
        metrics["readout_bias"] = self.oscillator.readout_down.bias.item()

        return metrics
