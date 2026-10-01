"""Complete loss function for AENCA V4/V4.1.

Implements the unified loss from the variational free energy functional F (V4.1 §9):
    L_total = L_diff + λ_any * L_any + λ_obs * L_obs + λ_pg * L_pg
              + Ω(s^(K)) + value_loss
              + λ_pred * L_pred + λ_prec * L_prec + λ_div * L_div    (V4.1 new)

All loss components are computed ONLY at masked positions with importance
weighting w(t) = π / tan(πt/2).

V4 reference: model_documents/aenca_v4_architecture.md §7, §8.2.
V4.1 reference: model_documents/aenca_v4_1_architecture.md §9, §10.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ncalm.models.v4.config import AENCAv4Config
from ncalm.models.v4.diffusion import importance_weight


class V4LossComputer:
    """Computes all V4/V4.1 loss components (V4.1 Eq 9.1, 9.12).

    Stateful:
        - EMA baseline for REINFORCE (Eq 6.31)
        - GradNorm dynamic task weights (V4.1 §10.4, Eq 10.6-10.8)
    """

    def __init__(self, config: AENCAv4Config) -> None:
        self.config = config
        self.pg_baseline_ema: float = 0.0  # Running EMA baseline

        # GradNorm state (V4.1 §10.4)
        self.grad_norms_t0: Dict[str, float] = {}  # Initial gradient norms
        self.task_weights: Dict[str, float] = {
            "diff": 1.0,
            "any": config.lambda_any,
            "pred": config.lambda_pred,
            "obs": config.lambda_obs,
            "pg": config.lambda_pg,
        }

    def compute_diffusion_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Diffusion loss L_diff (Eq 9.1a): weighted masked cross-entropy.

        Args:
            logits:  (B, L, |V|) — final readout predictions.
            targets: (B, L) — true token IDs.
            mask:    (B, L) — bool, True at masked positions.
            t:       (B,) — diffusion times.

        Returns:
            loss:    scalar — weighted masked CE loss.
            ce_mean: scalar — unweighted mean CE (for PPL tracking).
        """
        B, L, V = logits.shape

        # Importance weights per example (Eq 8.8)
        w_t = importance_weight(t)  # (B,)
        # V4.1: clamp at 100 (was 200 in V4) (Eq 6.8)
        w_t = w_t.clamp(max=100.0)

        # Per-position cross-entropy (no reduction)
        ce_flat = F.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1), reduction="none"
        )  # (B*L,)
        ce = ce_flat.reshape(B, L)  # (B, L)

        # Masked CE: zero out unmasked positions
        ce_masked = ce * mask.float()  # (B, L)

        # Per-example mean over masked positions
        n_masked = mask.float().sum(dim=1).clamp(min=1)  # (B,)
        ce_per_example = ce_masked.sum(dim=1) / n_masked  # (B,)

        # Weighted average over batch
        loss = (w_t * ce_per_example).mean()

        # Unweighted CE for PPL tracking
        ce_mean = ce_per_example.mean()

        return loss, ce_mean

    def compute_anytime_loss(
        self,
        anytime_logits: Dict[int, torch.Tensor],
        targets: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Anytime supervision loss L_any (V4.1 Eq 10.2).

        V4.1 change: w_k = (k/K)² (quadratic, was k/K linear in V4).
        Quadratic weighting reduces early-step penalty — early NCA steps
        are primarily doing information PROPAGATION, not prediction.

        Args:
            anytime_logits: {step_k: (B, L, |V|)} for k in anytime_steps.
            targets:        (B, L) — true token IDs.
            mask:           (B, L) — bool, True at masked positions.
            t:              (B,) — diffusion times.

        Returns:
            loss: scalar.
        """
        if not anytime_logits:
            return torch.tensor(0.0, device=targets.device, requires_grad=True)

        B, L = targets.shape
        K = self.config.K
        n_masked = mask.float().sum(dim=1).clamp(min=1)  # (B,)

        # Importance weights (V4.1: clamp at 100)
        w_t = importance_weight(t).clamp(max=100.0)  # (B,)

        total = torch.tensor(0.0, device=targets.device)
        for step_k, logits_k in anytime_logits.items():
            V = logits_k.shape[-1]
            # V4.1: Quadratic ramp (Eq 10.2)
            w_k = (step_k / K) ** 2

            ce_flat = F.cross_entropy(
                logits_k.reshape(-1, V), targets.reshape(-1), reduction="none"
            ).reshape(B, L)

            ce_masked = (ce_flat * mask.float()).sum(dim=1) / n_masked
            total = total + w_k * (w_t * ce_masked).mean()

        return total / len(anytime_logits)

    def compute_obs_loss(
        self, obs_losses: List[torch.Tensor]
    ) -> torch.Tensor:
        """Observation model loss L_obs (Eq 9.1e).

        Average MSE from Tier-2 EFE computations.

        Args:
            obs_losses: List of scalar MSE losses from EFE computations.

        Returns:
            loss: scalar.
        """
        if not obs_losses:
            return torch.tensor(0.0, requires_grad=True)
        return torch.stack(obs_losses).mean()

    def compute_pg_loss(
        self,
        advantages: List[torch.Tensor],
        log_probs: List[torch.Tensor],
    ) -> torch.Tensor:
        """REINFORCE policy gradient loss L_pg (Eq 8.5).

        L_pg = -(1/|S|) * sum advantage_i * log pi_1(a_i | s_i)

        Args:
            advantages: List of (n_subset,) advantage tensors.
            log_probs:  List of (n_subset, h) log probability tensors.

        Returns:
            loss: scalar.
        """
        if not advantages or not log_probs:
            device = advantages[0].device if advantages else "cpu"
            return torch.tensor(0.0, device=device, requires_grad=True)

        total_loss = torch.tensor(0.0, device=advantages[0].device)
        total_count = 0

        for adv, lp in zip(advantages, log_probs):
            if adv.numel() == 0:
                continue
            # adv: (n_subset,), lp: (n_subset, h)
            pg = adv.detach().unsqueeze(-1) * lp  # (n_subset, h)
            total_loss = total_loss - pg.sum()
            total_count += adv.numel() * lp.shape[-1]

        if total_count == 0:
            return torch.tensor(0.0, device=total_loss.device, requires_grad=True)

        return total_loss / total_count

    def compute_regularization(
        self, substrate: torch.Tensor
    ) -> torch.Tensor:
        """Regularization Ω(s^(K)) (Eq 9.1c-9.1d).

        Ω = λ_h * (1/L) * Σ_i ||h_i||² + λ_c * (1/L) * Σ_i ||h_i - h_{i+1}||²

        Args:
            substrate: (B, L, D) — final substrate state.

        Returns:
            reg: scalar regularization loss.
        """
        d_z = self.config.d_z
        d_h = self.config.d_h

        # Extract hidden channels
        h = substrate[:, :, d_z : d_z + d_h]  # (B, L, d_h)

        # State complexity: λ_h * mean(||h||²)
        state_penalty = self.config.lambda_h * (h ** 2).mean()

        # Consensus: λ_c * mean(||h_i - h_{i+1}||²)
        diff_right = h[:, 1:, :] - h[:, :-1, :]  # (B, L-1, d_h)
        consensus_penalty = self.config.lambda_c * (diff_right ** 2).mean()

        return state_penalty + consensus_penalty

    # ── V4.1 New Loss Terms ──────────────────────────────────────────────

    def compute_prediction_loss(
        self,
        prediction_errors: List[torch.Tensor],
        gate_values: List[torch.Tensor],
    ) -> torch.Tensor:
        """Prediction error loss L_pred (V4.1 Eq 9.1b, Term 2).

        L_pred = (1/K) * Σ_k (1/L) * Σ_i ||ε_i^(k)||²_{Π_i^(k)}

        where ||ε||²_Π = ε^T · Π · ε = Σ_d (g_sel_d · ε_d²)
        is the precision-weighted squared error.

        Args:
            prediction_errors: List of (B, L, D) prediction errors per step.
            gate_values:       List of (B, L, D) gate values (precision) per step.

        Returns:
            loss: scalar.
        """
        if not prediction_errors:
            return torch.tensor(0.0, requires_grad=True)

        total = torch.tensor(0.0, device=prediction_errors[0].device)
        for eps, gate in zip(prediction_errors, gate_values):
            # Precision-weighted squared error (Eq 9.4)
            weighted = gate * (eps ** 2)  # (B, L, D)
            total = total + weighted.mean()

        return total / len(prediction_errors)

    def compute_precision_prior_loss(
        self,
        gate_values: List[torch.Tensor],
    ) -> torch.Tensor:
        """Precision prior loss L_prec (V4.1 Eq 9.1g, Term 7).

        L_prec = (1/L) * Σ_i ||ln Π_i^(K)||²

        Penalizes extreme precision values (near 0 or 1) to prevent
        degenerate solutions where all gates close to minimize prediction error.

        Uses final step gate values only.

        Args:
            gate_values: List of (B, L, D) gate values per step.

        Returns:
            loss: scalar.
        """
        if not gate_values:
            return torch.tensor(0.0, requires_grad=True)

        # Use final step only (Eq 9.5)
        final_gate = gate_values[-1]  # (B, L, D)
        log_gate = torch.log(final_gate.clamp(min=1e-8))
        return (log_gate ** 2).mean()

    def compute_diversity_loss(
        self,
        observations_per_head: torch.Tensor,
    ) -> torch.Tensor:
        """Sensor diversity loss L_div (V4.1 Eq 9.1f, Term 6).

        L_div = (1/L) * Σ_i Σ_{j₁<j₂} max(0, cos(o_{i,j₁}, o_{i,j₂}) - τ_div)²

        Penalizes sensor heads that observe similar content, encouraging
        information diversity.

        Args:
            observations_per_head: (B, L, h, d_emb) per-head observations.

        Returns:
            loss: scalar.
        """
        if observations_per_head is None or observations_per_head.numel() == 0:
            return torch.tensor(0.0, requires_grad=True)

        # Normalize observations for cosine similarity
        obs_norm = F.normalize(observations_per_head, dim=-1)  # (B, L, h, d_emb)

        # Pairwise cosine similarity: (B, L, h, h)
        cos_sim = torch.einsum("blhd,blkd->blhk", obs_norm, obs_norm)

        h = obs_norm.shape[2]
        # Upper triangular mask for unique pairs (h*(h-1)/2 pairs)
        triu_mask = torch.triu(
            torch.ones(h, h, device=obs_norm.device, dtype=torch.bool),
            diagonal=1,
        )

        # Extract upper-triangular pair similarities
        # Expand triu_mask to match batch dims
        pair_sims = cos_sim[:, :, triu_mask]  # (B, L, n_pairs)

        # Hinge loss: only penalize similarity above threshold τ_div
        violations = torch.clamp(pair_sims - self.config.tau_div, min=0)
        return (violations ** 2).mean()

    # ── GradNorm (V4.1 §10.4) ───────────────────────────────────────────

    def update_gradnorm(
        self,
        model: torch.nn.Module,
        individual_losses: Dict[str, torch.Tensor],
        alpha: float = 1.5,
    ) -> None:
        """Update task weights via GradNorm (V4.1 §10.4, Eq 10.6-10.8).

        Computes gradient norms of each task loss w.r.t. kernel parameters
        and adjusts weights to equalize relative training rates.

        Args:
            model:             The model (to access kernel parameters).
            individual_losses: Dict of {task_name: loss_tensor} for task losses.
            alpha:             GradNorm strength parameter (default 1.5).
        """
        # Collect kernel parameters (shared weights)
        kernel_params = [
            p for n, p in model.named_parameters()
            if "kernel" in n and p.requires_grad
        ]
        if not kernel_params:
            return

        # Compute per-task gradient norms on kernel params
        current_norms: Dict[str, float] = {}
        for name, loss_val in individual_losses.items():
            if name not in self.task_weights:
                continue
            if not loss_val.requires_grad:
                current_norms[name] = 0.0
                continue
            grads = torch.autograd.grad(
                loss_val, kernel_params,
                retain_graph=True, allow_unused=True,
            )
            grad_norms = [g.norm() ** 2 for g in grads if g is not None]
            if not grad_norms:
                current_norms[name] = 0.0
                continue
            norm_sq = sum(grad_norms)
            current_norms[name] = norm_sq.sqrt().item()

        if not current_norms:
            return

        # Record t0 norms on first call
        if not self.grad_norms_t0:
            self.grad_norms_t0 = dict(current_norms)
            return

        # Relative training rates: r_i(t) = ||∇ℒ_i(t)|| / ||∇ℒ_i(t₀)|| (Eq 10.6)
        rates: Dict[str, float] = {}
        for k in current_norms:
            t0_norm = self.grad_norms_t0.get(k, 1e-8)
            rates[k] = current_norms[k] / max(t0_norm, 1e-8)

        if not rates:
            return

        # Mean rate (Eq 10.7)
        r_bar = sum(rates.values()) / len(rates)

        # Update weights (Eq 10.8)
        for k in list(self.task_weights.keys()):
            if k in rates:
                self.task_weights[k] *= (rates[k] / max(r_bar, 1e-8)) ** alpha

        # Normalize to keep total weight budget constant
        N = len(self.task_weights)
        total_w = sum(self.task_weights.values())
        if total_w > 0:
            for k in self.task_weights:
                self.task_weights[k] = N * self.task_weights[k] / total_w

    # ── Total Loss ───────────────────────────────────────────────────────

    def compute_total_loss(
        self,
        logits_K: torch.Tensor,
        anytime_logits: Dict[int, torch.Tensor],
        targets: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        substrate_K: torch.Tensor,
        obs_losses: List[torch.Tensor],
        value_losses: List[torch.Tensor],
        advantages: List[torch.Tensor],
        log_probs_subset: List[torch.Tensor],
        lambda_obs_eff: float,
        prediction_errors: List[torch.Tensor] | None = None,
        gate_values: List[torch.Tensor] | None = None,
        observations_per_head: torch.Tensor | None = None,
        pred_loss_live: torch.Tensor | None = None,
        prec_loss_live: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Total loss (V4.1 Eq 9.12):

        L_total = w_diff * L_diff + w_any * L_any + w_obs * L_obs
                  + w_pg * L_pg + Ω(s^(K)) + value_loss
                  + w_pred * L_pred + λ_prec * L_prec + λ_div * L_div

        Task weights (w_*) are dynamically adjusted by GradNorm.
        Regularizer weights (λ_prec, λ_div) remain fixed.

        Args:
            logits_K:             (B, L, |V|) — final readout.
            anytime_logits:       {step: (B, L, |V|)} — intermediate readouts.
            targets:              (B, L) — true token IDs.
            mask:                 (B, L) — masked positions.
            t:                    (B,) — diffusion times.
            substrate_K:          (B, L, D) — final substrate state.
            obs_losses:           List[scalar] — obs model MSE losses.
            value_losses:         List[scalar] — value baseline losses.
            advantages:           List[(n,)] — per-position advantages.
            log_probs_subset:     List[(n, h)] — log probs at EFE subset.
            lambda_obs_eff:       float — effective obs weight (with warmup).
            prediction_errors:    List[(B,L,D)] — detached pred errors (monitoring).
            gate_values:          List[(B,L,D)] — detached gate values (monitoring).
            observations_per_head: (B,L,h,d_emb) — V4.1 per-head obs (optional).
            pred_loss_live:       Scalar — graph-connected L_pred from checkpoint
                                  (Eq 9.1b). When provided, used instead of computing
                                  from detached lists.
            prec_loss_live:       Scalar — graph-connected L_prec recomputed from
                                  live substrate (Eq 9.1g). When provided, used instead
                                  of computing from detached gate_values.

        Returns:
            total_loss: scalar for backward.
            info:       dict with component losses, CE for PPL, and individual
                        loss tensors for GradNorm.
        """
        device = targets.device

        # (a) Diffusion loss
        diff_loss, ce_mean = self.compute_diffusion_loss(logits_K, targets, mask, t)

        # (b) Anytime loss
        anytime_loss = self.compute_anytime_loss(anytime_logits, targets, mask, t)

        # (c) Observation model loss
        obs_loss = self.compute_obs_loss(obs_losses) if obs_losses else torch.tensor(
            0.0, device=device
        )

        # (d) Policy gradient loss
        pg_loss = self.compute_pg_loss(advantages, log_probs_subset)

        # (e) Regularization (fixed weights, not GradNorm-controlled)
        reg_loss = self.compute_regularization(substrate_K)

        # (f) Value baseline loss (auxiliary, not weighted by lambda)
        val_loss = torch.stack(value_losses).mean() if value_losses else torch.tensor(
            0.0, device=device
        )

        # (g) V4.1: Prediction error loss (Term 2, §9.1b)
        # Use live (graph-connected) tensor when available — this ensures
        # gradients flow to W_pred and W_sel (spec: "all terms differentiable").
        # Falls back to detached computation for V4 compatibility.
        if pred_loss_live is not None:
            pred_loss = pred_loss_live
        else:
            pe = prediction_errors or []
            gv = gate_values or []
            pred_errors_paired = pe[:-1] if len(pe) > 1 else []
            pred_gates_paired = gv[1:] if len(gv) > 1 else []
            pred_loss = self.compute_prediction_loss(
                pred_errors_paired, pred_gates_paired
            )

        # (h) V4.1: Precision prior loss (Term 7, §9.1g, fixed weight)
        # Use live tensor when available — recomputed from substrate_K so
        # gradients flow to W_sel. Falls back to detached gate_values.
        if prec_loss_live is not None:
            prec_loss = prec_loss_live
        else:
            gv = gate_values or []
            prec_loss = self.compute_precision_prior_loss(gv)

        # (i) V4.1: Sensor diversity loss (Term 6, fixed weight)
        div_loss = self.compute_diversity_loss(observations_per_head)

        # Obs weight: respect warmup ramp even when GradNorm is active.
        # During warmup (lambda_obs_eff < lambda_obs), scale down the GradNorm
        # task weight proportionally. After warmup, warmup_scale=1.0 → no effect.
        obs_base_w = self.task_weights.get("obs", lambda_obs_eff)
        if self.config.lambda_obs > 0:
            warmup_scale = lambda_obs_eff / self.config.lambda_obs
            obs_w = obs_base_w * warmup_scale
        else:
            obs_w = 0.0

        # Total (V4.1 Eq 9.12): task weights from GradNorm, regularizer weights fixed
        total = (
            self.task_weights["diff"] * diff_loss
            + self.task_weights["any"] * anytime_loss
            + obs_w * obs_loss
            + self.task_weights["pg"] * pg_loss
            + reg_loss
            + val_loss
            + self.task_weights["pred"] * pred_loss
            + self.config.lambda_prec * prec_loss
            + self.config.lambda_div * div_loss
        )

        info = {
            "ce": ce_mean.detach(),
            "diff_loss": diff_loss.detach(),
            "anytime_loss": anytime_loss.detach(),
            "obs_loss": obs_loss.detach() if torch.is_tensor(obs_loss) else obs_loss,
            "pg_loss": pg_loss.detach() if torch.is_tensor(pg_loss) else pg_loss,
            "reg_loss": reg_loss.detach(),
            "value_loss": val_loss.detach() if torch.is_tensor(val_loss) else val_loss,
            "pred_loss": pred_loss.detach() if torch.is_tensor(pred_loss) else pred_loss,
            "prec_loss": prec_loss.detach() if torch.is_tensor(prec_loss) else prec_loss,
            "div_loss": div_loss.detach() if torch.is_tensor(div_loss) else div_loss,
        }

        # Store individual losses for GradNorm (needs grad for gradient computation)
        # pred_loss is now graph-connected when pred_loss_live is provided
        info["_individual_losses"] = {
            "diff": diff_loss,
            "any": anytime_loss,
            "pred": pred_loss,
            "obs": obs_loss,
            "pg": pg_loss,
        }

        return total, info
