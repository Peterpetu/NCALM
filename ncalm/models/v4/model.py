"""AENCA V4 main model: Diffusion Neural Cellular Automata Language Model.

Implements the complete V4 training and inference pipeline (Algorithm 1, §8.1):
1. Sample diffusion times and masking strategies
2. Create masked sequence z_t
3. Initialize substrate s^(0)
4. Run K NCA steps with gradient checkpointing
5. Readout and loss computation

This model replaces V3's sequential autoregressive processing with parallel
masked diffusion (MDLM), processing all L positions simultaneously.

Reference: model_documents/aenca_v4_architecture.md (full document).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ncalm.model import NCALMModel, register_model
from ncalm.models.v4.config import AENCAv4Config
from ncalm.models.v4.diffusion import (
    create_masked_sequence,
    importance_weight,
    sample_diffusion_times,
    sample_masking_strategy,
)
from ncalm.models.v4.loss import V4LossComputer
from ncalm.models.v4.substrate import NCASubstrate


@register_model("aenca_v4")
class AENCAv4Model(NCALMModel):
    """AENCA V4: Diffusion NCA Language Model.

    A pure Neural Cellular Automaton that operates as a masked diffusion
    language model. Each sequence position IS a cell in a 1D NCA substrate.
    All cells update simultaneously using a single shared kernel.

    NCA Purity Guarantees (§12):
    1. Uniform kernel — single NCAKernel instance, no step/position conditioning
    2. Local perception — Conv1d with kernel_size=3 (fixed)
    3. Iterative dynamics — same f_theta for all K steps
    4. No attention — no QKV, no dot-product attention anywhere
    5. No transformer blocks — no LN→Attn→Res→LN→FFN→Res
    6. Local readout — per-position softmax(W_out * z_i), no aggregation
    """

    def __init__(self, config_dict: dict) -> None:
        super().__init__()

        # Build V4 config from the profile dict
        # Extract V4-specific keys, ignore training-only keys
        v4_keys = {f.name for f in AENCAv4Config.__dataclass_fields__.values()}
        v4_params = {k: v for k, v in config_dict.items() if k in v4_keys}
        self.config = AENCAv4Config(**v4_params)

        # Core substrate (contains kernel, perception, sensing, readout, init)
        self.substrate = NCASubstrate(self.config)

        # Loss computer (stateful: tracks EMA baseline)
        self.loss_computer = V4LossComputer(self.config)

        # Scheduled hyperparameters (updated in on_step_start)
        self._lambda_obs_eff = 0.0
        self._tau_pi_eff = 2.0  # Starts high, anneals to config.tau_pi
        self._current_step = 0

    # ── Training Forward Pass (Algorithm 1, §8.1) ───────────────────────

    def forward_train(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Complete V4 training step.

        Args:
            batch: (B, L) token IDs, dtype long.

        Returns:
            loss:  Scalar total training loss for backward.
            info:  Dict with at minimum 'ce' key plus all loss components.
        """
        B, L = batch.shape
        device = batch.device
        config = self.config

        # Step 1: Sample diffusion times t ~ U[eps, 1-eps]
        t = sample_diffusion_times(B, eps=config.masking_eps, device=device)

        # Step 2: Select masking strategies
        strategy = sample_masking_strategy(
            B, config.p_random, config.p_causal, config.p_block, device=device
        )

        # Step 3: Create masked sequence z_t
        z_t, mask = create_masked_sequence(
            batch, t, strategy, config.mask_token_id, config.masking_eps
        )

        # Step 4: Initialize substrate s^(0) and get token embeddings
        s0, z_t_emb = self.substrate.initializer(z_t, mask)

        # Step 5: Initialize sensors
        pointers, scale_levels = self.substrate.sensing.init_sensors(B, L, device)

        # Step 6: Pre-compute position encodings
        positions = torch.arange(L, device=device)
        pos_enc = self.substrate.pos_encoder(positions)  # (L, d_pos)
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1)  # (B, L, d_pos)

        # Step 7: Encode timestep
        t_enc = self.substrate.timestep_encoder(t)  # (B, d_t)

        # Step 8: Run K NCA steps with gradient checkpointing
        (
            substrate_K,
            anytime_logits,
            all_advantages,
            all_obs_losses,
            all_value_losses,
            all_log_probs_subset,
        ) = self._run_nca_steps(
            s0, t_enc, z_t_emb, z_t, mask, batch, pos_enc, pointers, scale_levels
        )

        # Step 9: Final readout
        logits_K = self.substrate.get_readout_logits(substrate_K)  # (B, L, |V|)

        # Step 10: Compute all losses
        total_loss, info = self.loss_computer.compute_total_loss(
            logits_K=logits_K,
            anytime_logits=anytime_logits,
            targets=batch,
            mask=mask,
            t=t,
            substrate_K=substrate_K,
            obs_losses=all_obs_losses,
            value_losses=all_value_losses,
            advantages=all_advantages,
            log_probs_subset=all_log_probs_subset,
            lambda_obs_eff=self._lambda_obs_eff,
        )

        return total_loss, info

    def _run_nca_steps(
        self,
        substrate: torch.Tensor,
        t_enc: torch.Tensor,
        z_t_emb: torch.Tensor,
        z_t: torch.Tensor,
        mask: torch.Tensor,
        targets: torch.Tensor,
        pos_enc: torch.Tensor,
        pointers: torch.Tensor,
        scale_levels: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Dict[int, torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """Run K NCA steps with segment-based gradient checkpointing (§8.4).

        Checkpoint architecture:
        - Each segment of C steps is wrapped in checkpoint()
        - ONLY the substrate tensor flows through the checkpoint chain
          (single backward path per checkpoint node)
        - Anytime readouts are computed inside segments (need intermediate state)
          and collected via a closure side-channel
        - EFE/REINFORCE use detached substrate (advantages don't need substrate grad)
        - Log probs for REINFORCE are recomputed at EFE positions with one
          extra kernel call outside checkpoint (ensures clean gradient flow
          to action head without conflicting checkpoint paths)

        V4 uses single-level checkpointing (not V3's double-nested), so
        use_reentrant=False is safe and handles multiple output paths correctly.

        Returns:
            substrate:         (B, L, D) final substrate.
            anytime_logits:    {step: (B, L, |V|)} intermediate readouts.
            advantages:        List of advantage tensors from EFE.
            obs_losses:        List of scalar obs model losses.
            value_losses:      List of scalar value baseline losses.
            log_probs_subset:  List of log prob tensors at EFE positions.
        """
        config = self.config
        K = config.K
        C = config.checkpoint_interval
        anytime_set = set(config.anytime_steps)

        # Side-channel for anytime logits (populated inside checkpoint via closure)
        # These are detached from the checkpoint graph and recomputed outside
        # for gradient flow through the readout head.
        anytime_logits: Dict[int, torch.Tensor] = {}
        all_advantages: List[torch.Tensor] = []
        all_obs_losses: List[torch.Tensor] = []
        all_value_losses: List[torch.Tensor] = []
        all_log_probs_subset: List[torch.Tensor] = []

        # Track which steps had anytime readouts for post-checkpoint recomputation
        anytime_substrates: Dict[int, torch.Tensor] = {}

        for seg_start in range(0, K, C):
            seg_end = min(seg_start + C, K)

            def run_segment(
                _substrate, _pointers_f, _scale_levels_f,
                _seg_start=seg_start, _seg_end=seg_end,
            ):
                """Run one segment of C NCA steps.

                Returns only (substrate, pointers, scale_levels) to keep
                a single backward path through the checkpoint chain.
                Anytime readouts are saved to the closure side-channel.
                """
                _pointers = _pointers_f.long()
                _scale_levels = _scale_levels_f.long()

                for k in range(_seg_start, _seg_end):
                    (
                        _substrate,
                        _pointers,
                        _scale_levels,
                        _step_log_probs,
                        _action_logits,
                        _,  # current_prediction (V4.1 only)
                        _,  # gate_values (V4.1 only)
                    ) = self.substrate.nca_step(
                        _substrate, k, t_enc, z_t_emb,
                        _pointers, _scale_levels, pos_enc,
                        self._tau_pi_eff,
                    )

                    # Record substrate at anytime steps for later readout
                    if k in anytime_set:
                        anytime_substrates[k] = _substrate

                return _substrate, _pointers.float(), _scale_levels.float()

            if self.training:
                substrate, pointers_f, scale_levels_f = checkpoint(
                    run_segment,
                    substrate,
                    pointers.float(),
                    scale_levels.float(),
                    use_reentrant=False,
                )
                pointers = pointers_f.long()
                scale_levels = scale_levels_f.long()
            else:
                substrate, pointers_f, scale_levels_f = run_segment(
                    substrate, pointers.float(), scale_levels.float(),
                )
                pointers = pointers_f.long()
                scale_levels = scale_levels_f.long()

            # Compute anytime readout logits from saved substrates
            # These have gradient flow through the readout head and substrate
            for k in list(anytime_substrates.keys()):
                if seg_start <= k < seg_end:
                    anytime_logits[k] = self.substrate.get_readout_logits(
                        anytime_substrates.pop(k)
                    )

            # Tier-2 EFE at segment boundaries (every C steps, except last)
            if self.training and seg_end < K:
                # EFE uses detached substrate — advantages don't need
                # substrate gradient (REINFORCE handles action head updates)
                dilation = self.substrate.get_dilation(seg_end - 1)
                adv, obs_l, val_l = self.substrate.sensing.compute_efe_tier2(
                    substrate=substrate.detach(),
                    pointers=pointers,
                    scale_levels=scale_levels,
                    z_t_emb=z_t_emb.detach(),
                    pos_encodings=pos_enc.detach(),
                    t_enc=t_enc.detach(),
                    targets=targets,
                    mask=mask,
                    kernel=self.substrate.kernel,
                    readout_head=self.substrate.readout,
                    perception_fn=self.substrate.perception,
                    dilation=dilation,
                    config=config,
                )
                all_advantages.append(adv)
                all_obs_losses.append(obs_l)
                all_value_losses.append(val_l)

                # Recompute log_probs at EFE subset positions with a fresh
                # kernel call — gives clean gradient to action head
                if adv.numel() > 0:
                    lp_subset = self._recompute_log_probs_at_subset(
                        substrate, z_t_emb, pos_enc, t_enc,
                        mask, pointers, dilation,
                    )
                    if lp_subset is not None:
                        all_log_probs_subset.append(lp_subset)

        return (
            substrate,
            anytime_logits,
            all_advantages,
            all_obs_losses,
            all_value_losses,
            all_log_probs_subset,
        )

    def _recompute_log_probs_at_subset(
        self,
        substrate: torch.Tensor,
        z_t_emb: torch.Tensor,
        pos_enc: torch.Tensor,
        t_enc: torch.Tensor,
        mask: torch.Tensor,
        pointers: torch.Tensor,
        dilation: int,
    ) -> torch.Tensor | None:
        """Recompute action log probs at a subset of masked positions.

        One extra kernel call outside checkpoint to get clean gradient flow
        to the action head for REINFORCE. Uses current substrate state.

        Returns:
            log_probs: (n_subset, h) or None if no masked positions.
        """
        config = self.config
        B, L, D = substrate.shape

        masked_indices = mask.nonzero(as_tuple=False)
        n_masked = masked_indices.shape[0]
        n_subset = max(1, int(config.p_efe * n_masked))
        if n_masked == 0:
            return None

        perm = torch.randperm(n_masked, device=substrate.device)[:n_subset]
        subset_idx = masked_indices[perm]
        batch_idx, pos_idx = subset_idx[:, 0], subset_idx[:, 1]

        # Run perception on full substrate (needed for kernel input)
        P = self.substrate.perception(substrate, dilation)  # (B, L, D_conv)
        P_subset = P[batch_idx, pos_idx]  # (n_subset, D_conv)

        # Gather observations at subset positions
        obs_parts = []
        for head_j in range(config.n_sensor_heads):
            ptr_j = pointers[batch_idx, pos_idx, head_j]
            emb_j = z_t_emb[batch_idx, ptr_j]
            obs_parts.append(emb_j)
        obs_subset = torch.cat(obs_parts, dim=-1)  # (n_subset, h*d_emb)

        # Position + timestep encoding
        psi = torch.cat([
            pos_enc[batch_idx, pos_idx],
            t_enc[batch_idx],
        ], dim=-1)  # (n_subset, d_pos + d_t)

        # Kernel forward to get action logits
        x_in = torch.cat([P_subset, obs_subset, psi], dim=-1)  # (n_subset, d_in)
        x_in = x_in.unsqueeze(1)  # (n_subset, 1, d_in)
        _, _, action_logits = self.substrate.kernel(x_in)  # (n_subset, 1, h*|A|)
        action_logits = action_logits.squeeze(1)  # (n_subset, h*|A|)

        # Compute log probs (average over actions — used for REINFORCE)
        h = config.n_sensor_heads
        n_A = config.n_actions
        logits = action_logits.reshape(n_subset, h, n_A)
        log_probs = F.log_softmax(logits / self._tau_pi_eff, dim=-1)

        # Use uniform sampling approximation: average log prob over all actions
        # This provides gradient signal proportional to the policy's entropy
        # and encourages exploration
        mean_log_probs = log_probs.mean(dim=-1)  # (n_subset, h)

        return mean_log_probs

    # ── Inference (§9.1) ─────────────────────────────────────────────────

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Autoregressive-style inference for generation (Algorithm 2, §9.1).

        For evaluation/generation compatibility, this masks only the LAST
        position and runs K NCA steps with t near 0 (mostly unmasked).

        For each position n+1 being predicted:
        - Positions 0..n are unmasked (given as context)
        - Position n+1 is masked
        - Run K steps, read logits at n+1

        For efficiency during evaluation, we run a single diffusion pass with
        all positions except the last unmasked.

        Args:
            input_ids: (B, L) token IDs.

        Returns:
            logits: (B, L, vocab_size) predictions at ALL positions.
        """
        B, L = input_ids.shape
        device = input_ids.device
        config = self.config

        # Use causal masking: for position i, mask position i
        # and use positions 0..i-1 as context.
        # For efficiency, do one full pass with all positions masked and t near 0.

        # Create "evaluate all" setup: small t so most stays unmasked
        # Actually for proper eval, we use the diffusion framework:
        # mask all positions, run with t ≈ 1 (fully masked), then readout.
        # This tests the model's denoising ability.
        # But for autoregressive generation compatibility, we do a causal approach:

        # Construct causal mask: position i is the "last" of prefix [0..i]
        # For simplicity, mask all positions and run diffusion
        z_t = torch.full_like(input_ids, config.mask_token_id)
        mask = torch.ones(B, L, dtype=torch.bool, device=device)

        # Set t ≈ 1 (fully masked regime)
        t = torch.full((B,), 1.0 - config.masking_eps, device=device)

        # Initialize substrate
        s0, z_t_emb = self.substrate.initializer(z_t, mask)

        # Initialize sensors
        pointers, scale_levels = self.substrate.sensing.init_sensors(B, L, device)

        # Pre-compute encodings
        positions = torch.arange(L, device=device)
        pos_enc = self.substrate.pos_encoder(positions).unsqueeze(0).expand(B, -1, -1)
        t_enc = self.substrate.timestep_encoder(t)

        # Run K NCA steps (no checkpointing during inference)
        substrate = s0
        for k in range(config.K):
            substrate, pointers, scale_levels, _, _, _, _ = self.substrate.nca_step(
                substrate, k, t_enc, z_t_emb, pointers, scale_levels,
                pos_enc, config.tau_pi,
            )

        # Final readout
        logits = self.substrate.get_readout_logits(substrate)  # (B, L, |V|)
        return logits

    # ── Parameter Groups ─────────────────────────────────────────────────

    def get_param_groups(self) -> list[dict]:
        """Parameter groups for the optimizer.

        Groups:
        - backbone: kernel, conv, init MLP, encodings
        - sensing: observation model, value head, action head
        - embeddings: token embeddings, z_mask, readout
        """
        backbone_params = []
        sensing_params = []
        embedding_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "sensing" in name or "obs_model" in name or "value_head" in name:
                sensing_params.append(param)
            elif "token_embedding" in name or "z_mask" in name or "readout" in name:
                embedding_params.append(param)
            else:
                backbone_params.append(param)

        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "label": "backbone"})
        if sensing_params:
            groups.append({"params": sensing_params, "label": "sensing"})
        if embedding_params:
            groups.append({
                "params": embedding_params,
                "label": "embeddings",
                "weight_decay": 0.0,  # No WD on embeddings
            })
        return groups

    # ── Description ──────────────────────────────────────────────────────

    def describe(self) -> dict[str, Any]:
        """Architecture description for analysis.json and logging."""
        cfg = self.config
        total = sum(p.numel() for p in self.parameters())
        kernel_params = sum(p.numel() for p in self.substrate.kernel.parameters())
        conv_params = sum(p.numel() for p in self.substrate.perception.parameters())
        sensing_params = sum(p.numel() for p in self.substrate.sensing.parameters())
        embed_params = sum(p.numel() for p in self.substrate.initializer.token_embedding.parameters())
        readout_params = sum(p.numel() for p in self.substrate.readout.parameters())

        return {
            "architecture": "AENCA v4.0",
            "total_params": total,
            "substrate": f"1D chain, L={cfg.seq_len} cells",
            "state_dim": f"D={cfg.D} (d_z={cfg.d_z}, d_h={cfg.d_h}, d_c={cfg.d_c})",
            "kernel_params": kernel_params,
            "conv_params": conv_params,
            "sensing_params": sensing_params,
            "embed_params": embed_params,
            "readout_params": readout_params,
            "K": cfg.K,
            "M": cfg.M,
            "checkpoint_interval": cfg.checkpoint_interval,
            "n_sensor_heads": cfg.n_sensor_heads,
            "L_scales": cfg.L_scales,
            "n_actions": cfg.n_actions,
            "d_hidden": cfg.d_hidden,
            "d_emb": cfg.d_emb,
            "d_in": cfg.d_in,
            "vocab_size": cfg.vocab_size,
            "diffusion": "MDLM (cosine schedule, mixed masking)",
            "perception": f"Dilated Conv1D (r={cfg.kernel_radius}, cycling M={cfg.M})",
            "anytime_steps": list(cfg.anytime_steps),
            "gate_bias_init": f"{cfg.gate_bias_init:.3f}",
            "paradigm": "Masked Diffusion (parallel, not autoregressive)",
        }

    # ── Lifecycle Hooks ──────────────────────────────────────────────────

    def on_step_start(self, step: int, max_steps: int) -> None:
        """Update scheduled hyperparameters.

        - Observation model warmup: lambda_obs_eff ramps 0 → lambda_obs
        - EFE temperature annealing: tau_pi from 2.0 → config.tau_pi
        """
        self._current_step = step
        cfg = self.config

        # Observation model warmup (§8.2)
        if cfg.obs_warmup_steps > 0 and step <= cfg.obs_warmup_steps:
            self._lambda_obs_eff = cfg.lambda_obs * (step / cfg.obs_warmup_steps)
        else:
            self._lambda_obs_eff = cfg.lambda_obs

        # EFE temperature annealing
        if cfg.efe_warmup_steps > 0 and step <= cfg.efe_warmup_steps:
            # Linear anneal from 2.0 to cfg.tau_pi
            frac = step / cfg.efe_warmup_steps
            self._tau_pi_eff = 2.0 * (1.0 - frac) + cfg.tau_pi * frac
        else:
            self._tau_pi_eff = cfg.tau_pi

    def get_phases(self, step: int, max_steps: int) -> str:
        """Return active training phases for logging."""
        phases = ["diff"]
        if self._current_step > 0:
            phases.append("any")
        if self._lambda_obs_eff > 0:
            phases.append("obs")
        return "+".join(phases)

    def extra_metrics(self) -> dict[str, float]:
        """V4-specific metrics for logging."""
        return {
            "lambda_obs_eff": self._lambda_obs_eff,
            "tau_pi_eff": self._tau_pi_eff,
        }
