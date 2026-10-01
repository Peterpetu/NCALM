"""AENCA V4.1 main model: Targeted Evolution with Substantial Reforms.

Implements the complete V4.1 training and inference pipeline (§8–§10):
1. Curriculum diffusion time sampling (§6.3, Eq 6.11-6.13)
2. Rebalanced masking strategies (§6.1, Eq 6.2)
3. Initialize substrate with d_z=192, d_h=192 (§3.1)
4. Run K NCA steps with:
   - Prediction-error perception (§5, Eq 5.1-5.6)
   - Content-gated perception (§2, Eq 2.1-2.3)
   - SwiGLU kernel with pre-norm residual (§4.2, Eq 4.8-4.17)
   - Stochastic dynamics (§8, Eq 8.2)
5. Contextual readout (§3, Eq 3.3-3.5)
6. 9-term VFE loss with GradNorm (§9.1, §10.4)

This model replaces V4's sequential processing of perception with
prediction-error dynamics, making it a TRUE Active Inference model:
perception operates on precision-weighted prediction errors (§5.4, Eq 5.13).

Reference: model_documents/aenca_v4_1_architecture.md (full document).
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
    sample_diffusion_times_curriculum,
    sample_masking_strategy,
)
from ncalm.models.v4.loss import V4LossComputer
from ncalm.models.v4.substrate import NCASubstrate


@register_model("aenca_v41")
class AENCAv41Model(NCALMModel):
    """AENCA V4.1: Targeted Reform of Diffusion NCA Language Model.

    A pure Neural Cellular Automaton operating as a masked diffusion
    language model with true Active Inference dynamics.

    V4.1 Reforms (§1–§8 of spec):
        A1. Wider convolution r=2 (§1): 67% more perception bandwidth
        A2. Content-gated perception (§2): precision-weighted channel selection
        B.  Contextual readout (§3): 3-position window [z_{i-1}, z_i, z_{i+1}]
        C.  Gradient stabilization (§4): AGC, SwiGLU kernel, gate init fix
        D.  Prediction-error perception (§5): true Active Inference dynamics
        E.  MDLM training refinements (§6): masking rebalance, curriculum
        F.  Sensor scaling h=4 (§7): 2× non-local bandwidth + diversity loss
        Stochastic NCA dynamics (§8): learned noise for exploration

    NCA Purity (§14, Theorem 14.1): All 6 axioms preserved.
    Active Inference (§9, Theorem 9.1): NCA dynamics implement amortized
        stochastic gradient descent on the variational free energy F_V4.1.
    """

    def __init__(self, config_dict: dict) -> None:
        super().__init__()

        # Build V4 config from the profile dict
        v4_keys = {f.name for f in AENCAv4Config.__dataclass_fields__.values()}
        v4_params = {k: v for k, v in config_dict.items() if k in v4_keys}
        self.config = AENCAv4Config(**v4_params)

        # Core substrate (contains kernel, perception, sensing, readout, init,
        # V4.1 content_gate, temporal_predictor, context_proj, log_sigma_noise)
        self.substrate = NCASubstrate(self.config)

        # Loss computer (stateful: EMA baseline, GradNorm weights)
        self.loss_computer = V4LossComputer(self.config)

        # Scheduled hyperparameters (updated in on_step_start)
        self._lambda_obs_eff = 0.0
        self._tau_pi_eff = 2.0  # Starts high, anneals to config.tau_pi
        self._current_step = 0

    # ── Training Forward Pass (V4.1 §8.1 + §9) ──────────────────────

    def forward_train(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Complete V4.1 training step with all reforms.

        Implements Algorithm 1 (V4 spec §8.1) with V4.1 modifications:
        - Curriculum time sampling replaces uniform sampling
        - Prediction-error perception threaded through NCA steps
        - 9-term VFE loss (§9.1, Eq 9.12)

        Args:
            batch: (B, L) token IDs, dtype long.

        Returns:
            loss:  Scalar total training loss for backward.
            info:  Dict with at minimum 'ce' key plus all loss components.
        """
        B, L = batch.shape
        device = batch.device
        config = self.config

        # Step 1: Curriculum diffusion time sampling (§6.3, Eq 6.11-6.13)
        t = sample_diffusion_times_curriculum(
            B,
            step=self._current_step,
            easy_steps=config.curriculum_easy_steps,
            mid_steps=config.curriculum_mid_steps,
            eps=config.masking_eps,
            device=device,
        )

        # Step 2: Rebalanced masking strategies (§6.1, Eq 6.2)
        strategy = sample_masking_strategy(
            B, config.p_random, config.p_causal, config.p_block, device=device
        )

        # Step 3: Create masked sequence z_t
        z_t, mask = create_masked_sequence(
            batch, t, strategy, config.mask_token_id, config.masking_eps
        )

        # Step 4: Initialize substrate s^(0) (§4.4, Eq 4.6-4.9)
        s0, z_t_emb = self.substrate.initializer(z_t, mask)

        # Step 5: Initialize sensors (h=4, §7.4, Eq 7.7-7.8)
        pointers, scale_levels = self.substrate.sensing.init_sensors(B, L, device)

        # Step 6: Pre-compute position encodings
        positions = torch.arange(L, device=device)
        pos_enc = self.substrate.pos_encoder(positions)  # (L, d_pos)
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1)  # (B, L, d_pos)

        # Step 7: Encode timestep
        t_enc = self.substrate.timestep_encoder(t)  # (B, d_t)

        # Step 8: Run K NCA steps with prediction-error flow (§5)
        (
            substrate_K,
            final_pointers,
            anytime_logits,
            all_advantages,
            all_obs_losses,
            all_value_losses,
            all_log_probs_subset,
            pred_loss_live,
            all_pred_errors,
            all_gate_values,
        ) = self._run_nca_steps(
            s0, t_enc, z_t_emb, z_t, mask, batch, pos_enc, pointers, scale_levels
        )

        # Step 9: Final readout (contextual, §3, Eq 3.3-3.5)
        logits_K = self.substrate.get_readout_logits(substrate_K)  # (B, L, |V|)

        # Step 10: Gather per-head observations for diversity loss (§7.3, Eq 7.4)
        # Uses FINAL pointers (after NCA steps) per spec o_{i,j}^(K)
        observations_per_head = self._gather_observations_per_head(
            z_t_emb, final_pointers
        )

        # Step 10b: Recompute precision prior from live substrate (Eq 9.1g)
        # Recomputing from substrate_K (in computation graph) instead of
        # detached gate_values ensures L_prec trains W_sel.
        final_gate_live = torch.sigmoid(
            self.substrate.content_gate.gate_proj(substrate_K)
        )
        log_gate = torch.log(final_gate_live.clamp(min=1e-8))
        prec_loss_live = (log_gate ** 2).mean()

        # Step 11: Compute all losses (9-term VFE, §9.1, Eq 9.12)
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
            prediction_errors=all_pred_errors,
            gate_values=all_gate_values,
            observations_per_head=observations_per_head,
            pred_loss_live=pred_loss_live,
            prec_loss_live=prec_loss_live,
        )

        return total_loss, info

    def _gather_observations_per_head(
        self, z_t_emb: torch.Tensor, pointers: torch.Tensor
    ) -> torch.Tensor:
        """Gather per-head observations for diversity loss (§7.3, Eq 7.4-7.5).

        Args:
            z_t_emb:  (B, L, d_emb) token embeddings.
            pointers: (B, L, h) current pointer positions.

        Returns:
            observations_per_head: (B, L, h, d_emb).
        """
        B, L, d_emb = z_t_emb.shape
        h = pointers.shape[2]

        # gather_observations returns (B, L, h*d_emb) concatenated
        obs_concat = self.substrate.sensing.gather_observations(z_t_emb, pointers)
        return obs_concat.reshape(B, L, h, d_emb)

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
        torch.Tensor,
        Dict[int, torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        torch.Tensor,
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """Run K NCA steps with segment-based checkpointing and V4.1 reforms.

        Checkpoint architecture (§8.4, adapted for V4.1):
        - Each segment of C steps is wrapped in checkpoint()
        - Substrate, prev_prediction, AND prev_error flow through checkpoint chain
        - L_pred accumulated INSIDE checkpoints (graph-connected, Eq 9.12 §1895)
        - Detached copies of pred errors/gates kept for monitoring only
        - Anytime readouts use contextual readout (§10.2, Eq 10.4)
        - EFE/REINFORCE use detached substrate (same as V4)

        V4.1 additions over V4:
        - prev_prediction threaded through checkpoint boundary
        - prev_error threaded for cross-boundary pred loss pairing (§9.1b)
        - Bootstrap at k=0: raw substrate for conv (§5.6, Eq 5.6)
        - k≥1: prediction error ε = s - μ for conv (§5.2, Eq 5.2-5.3)
        - L_pred computed inside checkpoint for proper gradient flow to W_pred, W_sel
        - nca_step returns 7 values (added current_prediction, gate_values)

        Returns:
            substrate:         (B, L, D) final substrate s^(K).
            pointers:          (B, L, h) final pointer positions.
            anytime_logits:    {step: (B, L, |V|)} intermediate readouts.
            advantages:        List of advantage tensors from EFE.
            obs_losses:        List of scalar obs model losses.
            value_losses:      List of scalar value baseline losses.
            log_probs_subset:  List of log prob tensors at EFE positions.
            pred_loss_live:    Scalar — graph-connected prediction loss (Eq 9.1b).
            pred_errors:       List of (B, L, D) detached prediction errors (monitoring).
            gate_values:       List of (B, L, D) detached gate values (monitoring).
        """
        config = self.config
        K = config.K
        C = config.checkpoint_interval
        anytime_set = set(config.anytime_steps)

        # Side-channels (populated inside checkpoint via closure)
        anytime_logits: Dict[int, torch.Tensor] = {}
        all_advantages: List[torch.Tensor] = []
        all_obs_losses: List[torch.Tensor] = []
        all_value_losses: List[torch.Tensor] = []
        all_log_probs_subset: List[torch.Tensor] = []
        anytime_substrates: Dict[int, torch.Tensor] = {}

        # V4.1 side-channels: detached copies for monitoring only
        # Keyed by step index to avoid duplicates during checkpoint recomputation
        pred_errors_by_step: Dict[int, torch.Tensor] = {}
        gate_values_by_step: Dict[int, torch.Tensor] = {}

        # V4.1: prev_prediction starts as None (bootstrap, §5.6)
        # For checkpoint tensor compatibility, use zeros; track None-state via flag
        prev_prediction = torch.zeros_like(substrate)
        prev_pred_is_bootstrap = True  # True → use raw substrate for conv at k=0

        # V4.1: prev_error for cross-boundary pred loss pairing (§9.1b)
        # ε^(k-1) paired with Π^(k) = gate^(k) at each step
        prev_error = torch.zeros_like(substrate)
        prev_error_is_none = True  # True → no previous error (first step)

        # Accumulate live prediction loss across segments (graph-connected)
        total_pred_loss = torch.zeros(1, device=substrate.device)
        total_pred_n_pairs = 0

        for seg_start in range(0, K, C):
            seg_end = min(seg_start + C, K)

            def run_segment(
                _substrate, _pointers_f, _scale_levels_f, _prev_pred, _prev_error,
                _seg_start=seg_start, _seg_end=seg_end,
                _is_bootstrap=prev_pred_is_bootstrap,
                _prev_error_is_none=prev_error_is_none,
            ):
                """Run one segment of C NCA steps.

                Returns (substrate, pointers, scale_levels, prev_prediction,
                         prev_error, seg_pred_loss).
                L_pred accumulated INSIDE checkpoint for proper gradient flow
                to W_pred and W_sel (spec §9.1, Eq 9.12: "all terms differentiable").
                Detached copies saved to closure side-channels for monitoring.
                """
                _pointers = _pointers_f.long()
                _scale_levels = _scale_levels_f.long()
                _bootstrap = _is_bootstrap
                _no_prev_error = _prev_error_is_none

                # Accumulate prediction loss inside checkpoint (graph-connected)
                _seg_pred_loss = torch.zeros(1, device=_substrate.device)

                for k in range(_seg_start, _seg_end):
                    # Determine prev_prediction for this step
                    _actual_prev = None if _bootstrap else _prev_pred

                    (
                        _substrate,
                        _pointers,
                        _scale_levels,
                        _step_log_probs,
                        _action_logits,
                        _current_prediction,
                        _gate_values,
                    ) = self.substrate.nca_step(
                        _substrate, k, t_enc, z_t_emb,
                        _pointers, _scale_levels, pos_enc,
                        self._tau_pi_eff,
                        prev_prediction=_actual_prev,
                    )

                    # V4.1: Prediction error ε^(k) = s^(k+1) - μ^(k)
                    # (§5.2, Eq 5.2: ε = s_new - W_pred·s_old)
                    _pred_error = _substrate - _current_prediction

                    # V4.1: Accumulate precision-weighted pred loss (Eq 9.1b)
                    # Pair: ε^(k-1) with Π^(k) = gate^(k) (§5.4, Eq 5.4)
                    # gate^(k) = σ(W_sel · s^(k)) computed by content_gate
                    if not _no_prev_error:
                        _weighted = _gate_values * (_prev_error ** 2)  # (B, L, D)
                        _seg_pred_loss = _seg_pred_loss + _weighted.mean()

                    # Save detached copies for monitoring metrics
                    pred_errors_by_step[k] = _pred_error.detach()
                    gate_values_by_step[k] = _gate_values.detach()

                    # Thread error to next step for pairing
                    _prev_error = _pred_error
                    _no_prev_error = False

                    # Update prev_prediction for next step
                    _prev_pred = _current_prediction
                    _bootstrap = False

                    # Record substrate at anytime steps for later readout
                    if k in anytime_set:
                        anytime_substrates[k] = _substrate

                return (
                    _substrate, _pointers.float(), _scale_levels.float(),
                    _prev_pred, _prev_error, _seg_pred_loss,
                )

            if self.training:
                (substrate, pointers_f, scale_levels_f,
                 prev_prediction, prev_error, seg_pred_loss) = checkpoint(
                    run_segment,
                    substrate,
                    pointers.float(),
                    scale_levels.float(),
                    prev_prediction,
                    prev_error,
                    use_reentrant=False,
                )
                pointers = pointers_f.long()
                scale_levels = scale_levels_f.long()
            else:
                (substrate, pointers_f, scale_levels_f,
                 prev_prediction, prev_error, seg_pred_loss) = run_segment(
                    substrate, pointers.float(), scale_levels.float(),
                    prev_prediction, prev_error,
                )
                pointers = pointers_f.long()
                scale_levels = scale_levels_f.long()

            # Accumulate live prediction loss across segments
            total_pred_loss = total_pred_loss + seg_pred_loss
            # Count pairs: first segment loses 1 (bootstrap k=0), rest are full
            seg_len = seg_end - seg_start
            total_pred_n_pairs += seg_len - (1 if prev_error_is_none else 0)

            # After first segment, bootstrap is over
            prev_pred_is_bootstrap = False
            prev_error_is_none = False

            # Compute anytime readout logits from saved substrates
            # Uses contextual readout (§10.2, Eq 10.4): shares W_ctx and W_D
            for k in list(anytime_substrates.keys()):
                if seg_start <= k < seg_end:
                    anytime_logits[k] = self.substrate.get_readout_logits(
                        anytime_substrates.pop(k)
                    )

            # Tier-2 EFE at segment boundaries (every C steps, except last)
            if self.training and seg_end < K:
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

                # Recompute log_probs at EFE positions for REINFORCE
                if adv.numel() > 0:
                    lp_subset = self._recompute_log_probs_at_subset(
                        substrate, z_t_emb, pos_enc, t_enc,
                        mask, pointers, dilation,
                    )
                    if lp_subset is not None:
                        all_log_probs_subset.append(lp_subset)

        # Normalize live prediction loss by number of pairings (K-1 total)
        if total_pred_n_pairs > 0:
            pred_loss_live = total_pred_loss / total_pred_n_pairs
        else:
            pred_loss_live = total_pred_loss  # 0

        # Convert step-keyed dicts to sorted lists (detached, monitoring only)
        sorted_steps = sorted(pred_errors_by_step.keys())
        all_pred_errors = [pred_errors_by_step[k] for k in sorted_steps]
        all_gate_values = [gate_values_by_step[k] for k in sorted_steps]

        return (
            substrate,
            pointers,
            anytime_logits,
            all_advantages,
            all_obs_losses,
            all_value_losses,
            all_log_probs_subset,
            pred_loss_live,
            all_pred_errors,
            all_gate_values,
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
        to the action head for REINFORCE. Uses raw perception (no content
        gate or prediction error) since this is just for policy gradient.

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

        # Raw perception at subset positions (no content gate for efficiency)
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

        # Compute log probs
        h = config.n_sensor_heads
        n_A = config.n_actions
        logits = action_logits.reshape(n_subset, h, n_A)
        log_probs = F.log_softmax(logits / self._tau_pi_eff, dim=-1)
        mean_log_probs = log_probs.mean(dim=-1)  # (n_subset, h)

        return mean_log_probs

    # ── Inference (§9.1 from V4 spec) ─────────────────────────────────

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Autoregressive-style inference with V4.1 reforms.

        Fully masks all positions (t ≈ 1), runs K NCA steps with
        prediction-error perception, no stochastic noise (§8.5).

        Args:
            input_ids: (B, L) token IDs.

        Returns:
            logits: (B, L, vocab_size) predictions at ALL positions.
        """
        B, L = input_ids.shape
        device = input_ids.device
        config = self.config

        # Fully masked setup
        z_t = torch.full_like(input_ids, config.mask_token_id)
        mask = torch.ones(B, L, dtype=torch.bool, device=device)
        t = torch.full((B,), 1.0 - config.masking_eps, device=device)

        # Initialize substrate
        s0, z_t_emb = self.substrate.initializer(z_t, mask)

        # Initialize sensors (h=4, §7.4)
        pointers, scale_levels = self.substrate.sensing.init_sensors(B, L, device)

        # Pre-compute encodings
        positions = torch.arange(L, device=device)
        pos_enc = self.substrate.pos_encoder(positions).unsqueeze(0).expand(B, -1, -1)
        t_enc = self.substrate.timestep_encoder(t)

        # Run K NCA steps with prediction-error flow (no noise at inference)
        substrate = s0
        prev_prediction = None
        for k in range(config.K):
            (
                substrate, pointers, scale_levels,
                _, _,  # log_probs, action_logits
                current_prediction, _,  # gate_values
            ) = self.substrate.nca_step(
                substrate, k, t_enc, z_t_emb, pointers, scale_levels,
                pos_enc, config.tau_pi,
                prev_prediction=prev_prediction,
            )
            prev_prediction = current_prediction

        # Contextual readout (§3, Eq 3.3-3.5)
        logits = self.substrate.get_readout_logits(substrate)  # (B, L, |V|)
        return logits

    # ── Parameter Groups ─────────────────────────────────────────────

    def get_param_groups(self) -> list[dict]:
        """Parameter groups for the optimizer.

        V4.1 additions (content_gate, temporal_predictor, context_proj,
        log_sigma_noise) go into the backbone group.

        Groups:
        - backbone: kernel, conv, init MLP, encodings, content_gate,
                     temporal_predictor, context_proj, log_sigma_noise
        - sensing: observation model, value head, action head
        - embeddings: token embeddings, z_mask, readout (no WD)
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

    # ── Description ──────────────────────────────────────────────────

    def describe(self) -> dict[str, Any]:
        """V4.1 architecture description for analysis.json and logging."""
        cfg = self.config
        total = sum(p.numel() for p in self.parameters())
        kernel_params = sum(p.numel() for p in self.substrate.kernel.parameters())
        conv_params = sum(p.numel() for p in self.substrate.perception.parameters())
        sensing_params = sum(p.numel() for p in self.substrate.sensing.parameters())
        embed_params = sum(
            p.numel() for p in self.substrate.initializer.token_embedding.parameters()
        )
        readout_params = sum(p.numel() for p in self.substrate.readout.parameters())
        gate_params = sum(p.numel() for p in self.substrate.content_gate.parameters())
        pred_params = sum(
            p.numel() for p in self.substrate.temporal_predictor.parameters()
        )
        ctx_params = sum(p.numel() for p in self.substrate.context_proj.parameters())

        return {
            "architecture": "AENCA v4.1",
            "total_params": total,
            "substrate": f"1D chain, L={cfg.seq_len} cells",
            "state_dim": f"D={cfg.D} (d_z={cfg.d_z}, d_h={cfg.d_h}, d_c={cfg.d_c})",
            "kernel_params": kernel_params,
            "kernel_trunk": (
                f"{cfg.n_trunk_layers}x SwiGLU"
                f"({cfg.d_hidden}->{cfg.d_ff}->{cfg.d_hidden}),"
                f" RMSNorm pre-norm, alpha_res={cfg.alpha_res}"
            ),
            "conv_params": conv_params,
            "sensing_params": sensing_params,
            "embed_params": embed_params,
            "readout_params": readout_params,
            "content_gate_params": gate_params,
            "temporal_predictor_params": pred_params,
            "context_readout_params": ctx_params,
            "K": cfg.K,
            "M": cfg.M,
            "checkpoint_interval": cfg.checkpoint_interval,
            "n_sensor_heads": cfg.n_sensor_heads,
            "L_scales": cfg.L_scales,
            "n_actions": cfg.n_actions,
            "d_hidden": cfg.d_hidden,
            "d_ff": cfg.d_ff,
            "d_emb": cfg.d_emb,
            "d_in": cfg.d_in,
            "vocab_size": cfg.vocab_size,
            "diffusion": "MDLM (cosine schedule, rebalanced masking, curriculum)",
            "perception": (
                f"Dilated Conv1D (r={cfg.kernel_radius}, cycling M={cfg.M}),"
                f" content-gated, prediction-error"
            ),
            "readout": (
                f"Contextual 3-position window:"
                f" [{3*cfg.d_z}]->[{cfg.d_z}]->[{cfg.vocab_size}]"
            ),
            "anytime_steps": list(cfg.anytime_steps),
            "gate_bias_init": f"{cfg.gate_bias_init:.3f}",
            "stochastic_noise": f"sigma_init={cfg.sigma_noise_init}",
            "paradigm": (
                "Masked Diffusion + Active Inference"
                " (precision-weighted prediction-error perception)"
            ),
        }

    # ── Lifecycle Hooks ──────────────────────────────────────────────

    def on_step_start(self, step: int, max_steps: int) -> None:
        """Update scheduled hyperparameters.

        - Observation model warmup: lambda_obs_eff ramps 0 -> lambda_obs
        - EFE temperature annealing: tau_pi from 2.0 -> config.tau_pi
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
        # V4.1 curriculum phase indicator
        if self._current_step < self.config.curriculum_easy_steps:
            phases.append("easy")
        elif self._current_step < self.config.curriculum_mid_steps:
            phases.append("std")
        else:
            phases.append("hard")
        return "+".join(phases)

    def extra_metrics(self) -> dict[str, float]:
        """V4.1-specific metrics for logging."""
        metrics = {
            "lambda_obs_eff": self._lambda_obs_eff,
            "tau_pi_eff": self._tau_pi_eff,
        }
        # GradNorm task weights (§10.4)
        for k, v in self.loss_computer.task_weights.items():
            metrics[f"gradnorm_w_{k}"] = v
        return metrics
