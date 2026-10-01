"""AENCA V14 main model: Emergent Dynamics Language Model.

Architecture (V14 — Emergent Dynamics):
  1. Token embedding + sinusoidal PE → initial state z₀ (B, L, d_model)
  2. Morphogen seed: h = tanh(W_h_init(z₀))             (B, L, d_h)
  3. Theta loop (θ = 0..n_theta-1):
     a. Content-enriched NCA communication:
        For m = 0..n_morph_steps-1:
          - h_nbr = morph_conv(h, dilation=d_m)           [MORPHOGEN NEIGHBORHOOD]
          - c_m   = content_conv(W_c(norm(z)), dilation)  [CONTENT PERCEPTION]
          - h = h + gate * Σ_k w_k · rule_k([h, h_nbr, c_m])  [NCA RULES]
          - h_multi += α_m · h                            [MULTI-SCALE AGGREGATE]
     b. Content readout: z = z + W_readout_θ(norm(h_multi)) [MORPH → CONTENT]
     c. Integration FFN: z = z + SwiGLU(z)               [POST-NCA PROCESSING]
     d. Content→morphogen feedback: h = h + tanh(W_fb(z)) [CONTENT → MORPH]
     e. PonderNet halting: halt_prob → z_out accumulation [ADAPTIVE HALT]
  4. Readout from accumulated z_out → logits

V14 thesis:
  Information routing EMERGES from NCA dynamics on morphogen h enriched with
  content perception at every step. No explicit Q/K/V, no dot-product
  attention, no RoPE — just local NCA rules with exponentially growing
  dilations. Position info from sinusoidal PE + causal conv structure.

  At initialization, zero-init cascade ensures V14 starts as a "bag of
  embeddings" model (near-zero residuals), identical to V13 behavior.

Key differences from V13:
  - ThetaAttention REMOVED (Q/K/V dot-product attention, RoPE, flash attention)
  - Content perception ADDED to morphogen evolution (z→h at every NCA step)
  - ContentReadout replaces attention output (h→z per theta, unique weights)
  - IntegrationFFN replaces HebbianFFN (SwiGLU-like, single input)
  - d_h: 128 → 256 (wider morphogen carries content information)
  - n_rules: 4 → 8 (more diverse routing behaviors)
  - n_morph_steps: 4 → 12 (full-sequence coverage via exponential dilation)
  - Dynamics diversity loss ADDED (encourages varied NCA behavior)

Reference: ANCA paper (Kvalsund et al., 2026) — emergent attention from NCA.
Reference: NCA dynamics paper (Xu et al., 2024) — channel/neuron ratio.
Reference: V13 model.py — theta-loop structure, loss computation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ncalm.model import NCALMModel, register_model
from ncalm.models.v14.config import V14Config
from ncalm.models.v14.morphogen import ContentEnrichedMorphogenField, ContentReadout
from ncalm.models.v14.integration_ffn import IntegrationFFN
from ncalm.models.v14.halting import AdaptiveHalting

# Reuse readout from v5 — with mos_components=1 it's a plain linear head
from ncalm.models.v5.readout import MixtureOfSoftmaxReadout


# ═══════════════════════════════════════════════════════════════════════
# Utility Modules
# ═══════════════════════════════════════════════════════════════════════

class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal position encoding (non-parametric).

    Same as V13 — provides absolute position information to compensate
    for the removal of RoPE (which was rotation applied to Q/K in V13).
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


# ═══════════════════════════════════════════════════════════════════════
# Main Model
# ═══════════════════════════════════════════════════════════════════════

@register_model("v14")
class AENCAv14Model(NCALMModel):
    """AENCA V14: Emergent Dynamics Language Model.

    A fully emergent information routing architecture where:
    - A content-enriched morphogen field (d_h=256) evolves via local NCA dynamics
    - Content perception at every NCA step creates content-dependent routing
    - Exponential dilation schedule covers full sequence in O(log N) steps
    - NO explicit Q/K/V, NO dot-product attention, NO RoPE
    - IntegrationFFN (SwiGLU-like) for post-NCA content processing
    - Zero-init cascade ensures V14 = bag-of-embeddings at initialization
    """

    def __init__(self, config_dict: dict) -> None:
        super().__init__()

        # Parse flat config dict into V14Config
        v14_keys = {f.name for f in V14Config.__dataclass_fields__.values()}
        v14_params = {k: v for k, v in config_dict.items() if k in v14_keys}

        # Handle morph_dilations: may come as list from JSON
        if "morph_dilations" in v14_params and isinstance(v14_params["morph_dilations"], list):
            v14_params["morph_dilations"] = tuple(v14_params["morph_dilations"])

        self.config = V14Config(**v14_params)
        cfg = self.config
        # __post_init__ already calls compute_d_ff() and validate()
        assert cfg.d_ff > 0, f"d_ff must be positive, got {cfg.d_ff}. Reduce target_params or model dimensions."

        # ── Token Embedding ──────────────────────────────────────────
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.token_embedding.weight, std=1.0 / math.sqrt(cfg.d_model))

        # ── Position Encoding ────────────────────────────────────────
        self.state_pe = SinusoidalPositionEncoding(cfg.d_model, cfg.max_seq_len * 2)

        # ── Content-Enriched Morphogen Field (SHARED across theta) ───
        self.morphogen = ContentEnrichedMorphogenField(cfg)

        # ── Content Readout (UNIQUE per theta cycle) ─────────────────
        self.theta_readouts = nn.ModuleList([
            ContentReadout(cfg.d_h, cfg.d_model, cfg.n_morph_steps)
            for _ in range(cfg.n_theta)
        ])

        # ── Integration FFN (UNIQUE per theta cycle) ─────────────────
        self.theta_ffns = nn.ModuleList([
            IntegrationFFN(cfg.d_model, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_theta)
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
        """Complete V14 training step with autoregressive loss.

        Runs the emergent dynamics loop:
        - Content-enriched morphogen evolution per theta cycle (shared rules)
        - Content readout per theta (unique projections)
        - Integration FFN per theta (unique weights)
        - Content→morphogen feedback (shared)
        - PonderNet halting accumulation

        Args:
            batch: (B, L+1) token IDs, dtype long.

        Returns:
            loss:  Scalar total training loss.
            info:  Dict with 'ce' key plus auxiliary components.
        """
        B, L_full = batch.shape
        cfg = self.config

        # ── 1. EMBED ─────────────────────────────────────────────────
        s = self.token_embedding(batch)
        s = s + self.state_pe(L_full).unsqueeze(0)
        z = s
        z0 = z.detach().clone()  # save for drift regularization

        # ── 2. MORPHOGEN SEED ────────────────────────────────────────
        h = self.morphogen.seed(z)                          # (B, L, d_h)

        # ── 3. THETA LOOP ───────────────────────────────────────────
        z_out = torch.zeros_like(z)
        remaining = torch.ones(B, L_full, device=z.device)
        halt_probs: list[torch.Tensor] = []
        all_routing_weights: list[torch.Tensor] = []
        all_step_deltas: list[torch.Tensor] = []

        # Deep supervision substrates
        aux_substrates: list[tuple[int, torch.Tensor, torch.Tensor]] = []

        n_active_cycles = 0

        for theta_idx in range(cfg.n_theta):
            is_last = (theta_idx == cfg.n_theta - 1)

            # Early exit if all positions halted
            if cfg.enable_halting and (remaining <= cfg.halt_eps).all():
                break

            # Possibly gradient checkpoint
            if self.training and cfg.checkpoint_interval > 0:
                z, h, cycle_routing, cycle_deltas = checkpoint(
                    self._run_theta_cycle,
                    z, h, theta_idx,
                    use_reentrant=False,
                )
            else:
                z, h, cycle_routing, cycle_deltas = self._run_theta_cycle(
                    z, h, theta_idx,
                )

            all_routing_weights.extend(cycle_routing)
            all_step_deltas.extend(cycle_deltas)
            n_active_cycles += 1

            # ── Halting / readout accumulation ───────────────────────
            if cfg.enable_halting:
                z_out, remaining, hp = self.halting.accumulate(
                    z, z_out, remaining, is_last,
                )
                halt_probs.append(hp)
            else:
                if is_last:
                    z_out = z_out + remaining.unsqueeze(-1) * z
                    remaining = torch.zeros_like(remaining)
                else:
                    weight = 1.0 / cfg.n_theta
                    z_out = z_out + weight * z
                    remaining = remaining - weight

            # Deep supervision snapshot (all but last θ)
            if not is_last and cfg.lambda_aux > 0:
                aux_substrates.append(
                    (theta_idx + 1, z_out.clone(), remaining.clone())
                )

        # Absorb remaining mass (safety)
        if remaining.sum() > 0.001:
            z_out = z_out + remaining.unsqueeze(-1) * z

        # ── 4. READOUT ───────────────────────────────────────────────
        logits = self.readout(z_out)

        # ── 5. CE LOSS ───────────────────────────────────────────────
        logits_flat = logits[:, :-1].reshape(-1, cfg.vocab_size)
        targets_flat = batch[:, 1:].reshape(-1)

        if self.readout.returns_log_probs:
            ce_loss = F.nll_loss(logits_flat, targets_flat, reduction="mean")
        else:
            ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")

        total_loss = ce_loss
        info: dict[str, Any] = {"ce": ce_loss.detach()}

        # ── 6. DRIFT REGULARIZATION ──────────────────────────────────
        if cfg.lambda_reg > 0:
            reg_loss = (z_out - z0).pow(2).mean()
            total_loss = total_loss + cfg.lambda_reg * reg_loss
            info["reg"] = reg_loss.detach()

        # ── 7. DEEP SUPERVISION ──────────────────────────────────────
        if cfg.lambda_aux > 0 and aux_substrates:
            targets_flat_aux = batch[:, 1:].reshape(-1)
            aux_total = torch.tensor(0.0, device=z.device)
            n_aux = 0

            for step_idx, z_out_snap, remaining_snap in aux_substrates:
                w = (step_idx / cfg.n_theta) ** 2

                snap_mass = (1.0 - remaining_snap).mean()
                if snap_mass < 0.01:
                    continue

                aux_logits = self.readout(z_out_snap)
                aux_flat = aux_logits[:, :-1].reshape(-1, cfg.vocab_size)

                if self.readout.returns_log_probs:
                    aux_ce = F.nll_loss(aux_flat, targets_flat_aux, reduction="mean")
                else:
                    aux_ce = F.cross_entropy(aux_flat, targets_flat_aux, reduction="mean")

                aux_total = aux_total + w * aux_ce
                n_aux += 1

            if n_aux > 0:
                aux_loss = aux_total / n_aux
                total_loss = total_loss + cfg.lambda_aux * aux_loss
                info["aux"] = aux_loss.detach()

        # ── 8. HALT KL ──────────────────────────────────────────────
        if cfg.enable_halting and halt_probs and cfg.lambda_halt > 0:
            halt_kl = self.halting.compute_kl_loss(halt_probs)
            total_loss = total_loss + cfg.lambda_halt * halt_kl
            info["halt_kl"] = halt_kl.detach()

        # ── 9. ROUTING DIVERSITY LOSS (entropy + load-balance) ────────
        if all_routing_weights:
            # Stack all routing weights: list of (B, L, K)
            all_w = torch.stack(all_routing_weights, dim=0)  # (S, B, L, K)
            S_steps = all_w.shape[0]

            # 9a. Negative entropy → maximize routing entropy
            if cfg.lambda_div > 0:
                div_loss = (all_w * (all_w + 1e-8).log()).sum(dim=-1).mean()
                total_loss = total_loss + cfg.lambda_div * div_loss
                info["div"] = div_loss.detach()

            # 9b. Switch Transformer load-balance loss
            # Prevents winner-take-all routing collapse.
            # L_bal = K * sum_k(f_k * P_k) where:
            #   f_k = fraction of tokens routed to rule k
            #   P_k = mean routing probability for rule k
            if cfg.lambda_bal > 0:
                # f_k: fraction of tokens where rule k is the argmax
                top1 = all_w.argmax(dim=-1)                      # (S, B, L)
                f = torch.zeros(cfg.n_rules, device=z.device)
                for k in range(cfg.n_rules):
                    f[k] = (top1 == k).float().mean()
                # P_k: mean routing probability assigned to rule k
                P = all_w.mean(dim=(0, 1, 2))                   # (K,)
                bal_loss = cfg.n_rules * (f * P).sum()
                total_loss = total_loss + cfg.lambda_bal * bal_loss
                info["bal"] = bal_loss.detach()

        # ── 10. DYNAMICS DIVERSITY LOSS ──────────────────────────────
        if cfg.lambda_dyn > 0 and all_step_deltas:
            # all_step_deltas: list of (B, L) delta norms per step
            delta_stack = torch.stack(all_step_deltas, dim=0)  # (M*n_theta, B, L)
            # Penalize collapse to zero dynamics: negative mean delta norm
            dyn_loss = -delta_stack.mean()
            total_loss = total_loss + cfg.lambda_dyn * dyn_loss
            info["dyn"] = dyn_loss.detach()

        # ── 11. STEP COSINE-DIVERSITY LOSS ───────────────────────────
        # Penalize consecutive NCA steps that produce identical updates.
        # Without this, the morphogen applies the same update 12 times.
        if cfg.lambda_step_div > 0 and len(all_step_deltas) >= 2:
            # We need the actual update vectors, not just norms.
            # Approximate: use routing weights as a proxy for step behavior.
            # Consecutive routing should differ (different dilations → different rules).
            if len(all_routing_weights) >= 2:
                rw_stack = torch.stack(all_routing_weights, dim=0)  # (S, B, L, K)
                # Cosine similarity between consecutive steps' routing
                rw_norm = F.normalize(rw_stack, dim=-1)
                cos_sim = (rw_norm[:-1] * rw_norm[1:]).sum(dim=-1)  # (S-1, B, L)
                # Penalize high similarity (push toward 0 mean cos)
                step_div_loss = cos_sim.mean()
                total_loss = total_loss + cfg.lambda_step_div * step_div_loss
                info["step_div"] = step_div_loss.detach()

        # ── Monitoring Metrics ───────────────────────────────────────
        with torch.no_grad():
            if cfg.enable_halting:
                halted_mask = (remaining <= cfg.halt_eps)
                info["pct_halted_early"] = float(halted_mask.float().mean().item())
            else:
                info["pct_halted_early"] = 0.0
            info["expected_steps"] = n_active_cycles * (cfg.n_morph_steps + 1)

        return total_loss, info

    # ══════════════════════════════════════════════════════════════════
    # Theta Cycle
    # ══════════════════════════════════════════════════════════════════

    def _run_theta_cycle(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        theta_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Run one theta cycle: NCA comm → readout → FFN → feedback.

        Args:
            z:         (B, L, d_model) content state
            h:         (B, L, d_h) morphogen state
            theta_idx: which theta cycle (for selecting unique readout/FFN)

        Returns:
            z:                updated content state
            h:                updated morphogen state
            routing_weights:  list of (B, L, K) routing weights from evolution
            step_deltas:      list of (B, L) delta norms per step
        """
        cfg = self.config

        # ── 3a. CONTENT-ENRICHED NCA COMMUNICATION ───────────────────
        # Multi-scale readout: get per-step aggregation weights from readout
        scale_weights = self.theta_readouts[theta_idx].get_scale_weights()
        h, h_multi, routing_weights, step_deltas = self.morphogen.evolve(
            h, z, cfg.n_morph_steps, scale_weights,
        )

        # ── 3b. CONTENT READOUT (morph→content, unique per θ) ────────
        # Reads from multi-scale aggregated h_multi (weighted across steps)
        # Feedback (3d) still uses final h for temporal continuity
        z = z + self.theta_readouts[theta_idx](h_multi)

        # ── 3c. INTEGRATION FFN (unique per θ) ──────────────────────
        z = z + self.theta_ffns[theta_idx](z)

        # ── 3d. CONTENT → MORPHOGEN FEEDBACK (shared) ────────────────
        h = self.morphogen.update_from_content(h, z)

        return z, h, routing_weights, step_deltas

    # ══════════════════════════════════════════════════════════════════
    # Inference
    # ══════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Autoregressive inference with adaptive theta halting.

        Args:
            input_ids: (B, L) token IDs.

        Returns:
            logits: (B, L, vocab_size) predictions at all positions.
        """
        B, L = input_ids.shape
        cfg = self.config
        device = input_ids.device

        # Embed
        z = self.token_embedding(input_ids)
        z = z + self.state_pe(L).unsqueeze(0)

        # Morphogen seed
        h = self.morphogen.seed(z)

        # Halting accumulators
        z_out = torch.zeros_like(z)
        remaining = torch.ones(B, L, device=device)

        for theta_idx in range(cfg.n_theta):
            is_last = (theta_idx == cfg.n_theta - 1)

            if cfg.enable_halting and (remaining <= cfg.halt_eps).all():
                break

            z, h, _, _ = self._run_theta_cycle(z, h, theta_idx)

            if cfg.enable_halting:
                z_out, remaining, _ = self.halting.accumulate(
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

        decay: 2D+ weight matrices (morphogen, FFN, readout projections)
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
        """V14 architecture description for analysis.json and logging."""
        cfg = self.config
        total = sum(p.numel() for p in self.parameters())

        morph_params = sum(p.numel() for p in self.morphogen.parameters())
        embed_params = sum(p.numel() for p in self.token_embedding.parameters())
        readout_head_params = sum(p.numel() for p in self.readout.parameters())
        content_readout_params = sum(
            p.numel() for p in self.theta_readouts.parameters()
        )
        ffn_params = sum(p.numel() for p in self.theta_ffns.parameters())
        halt_params = (
            sum(p.numel() for p in self.halting.parameters())
            if cfg.enable_halting else 0
        )

        return {
            "architecture": "AENCA v14 (Emergent Dynamics)",
            "total_params": total,
            "morphogen_params": morph_params,
            "embed_params": embed_params,
            "readout_head_params": readout_head_params,
            "content_readout_params": content_readout_params,
            "theta_ffn_params": ffn_params,
            "halt_params": halt_params,
            "d_model": cfg.d_model,
            "d_h": cfg.d_h,
            "d_ff": cfg.d_ff,
            "n_theta": cfg.n_theta,
            "n_morph_steps": cfg.n_morph_steps,
            "n_rules": cfg.n_rules,
            "effective_rule_hidden": cfg.effective_rule_hidden,
            "morph_dilations": list(cfg.morph_dilations),
            "enable_halting": cfg.enable_halting,
            "mos_components": cfg.mos_components,
            "tie_embeddings": cfg.tie_embeddings,
            "vocab_size": cfg.vocab_size,
            "target_params": cfg.target_params,
            "paradigm": (
                "Emergent Dynamics (content-enriched NCA communication, "
                "multi-scale readout, no explicit attention)"
            ),
        }

    # ══════════════════════════════════════════════════════════════════
    # Lifecycle Hooks
    # ══════════════════════════════════════════════════════════════════

    def get_phases(self, step: int, max_steps: int) -> str:
        """Return active training phases for logging."""
        return "EmergentDynamics"

    def extra_metrics(self) -> dict[str, float]:
        """V14-specific monitoring metrics.

        Includes Phase C emergent dynamics monitoring:
          - Per-rule W_h2 norms (zero-init → should grow)
          - Content perception coupling strength
          - Router/feedback weight norms
          - Per-theta readout/FFN norms
          - Multi-scale readout weights (which morph steps are read from)
          - Scale concentration (entropy of scale weights — high = diverse, low = focused)
        """
        metrics: dict[str, float] = {}

        # Morphogen rule activity: W_h2 norms per rule (zero-init, should grow)
        for k in range(self.config.n_rules):
            metrics[f"morph_rule{k}_norm"] = (
                self.morphogen.W_h2[k].weight.norm().item()
            )

        # Content perception weight norm (should grow as content→morph coupling develops)
        metrics["content_proj_norm"] = (
            self.morphogen.W_content_proj.weight.norm().item()
        )

        # Router weight norm (zero-init, should grow)
        metrics["morph_router_norm"] = self.morphogen.W_router.weight.norm().item()

        # Feedback weight norm (zero-init, should grow)
        metrics["morph_feedback_norm"] = (
            self.morphogen.W_h_feedback.weight.norm().item()
        )

        # Content readout norms per theta (zero-init, should grow)
        for i in range(self.config.n_theta):
            metrics[f"theta{i}_readout_norm"] = (
                self.theta_readouts[i].W_readout.weight.norm().item()
            )
            metrics[f"theta{i}_ffn_down_norm"] = (
                self.theta_ffns[i].W_down.weight.norm().item()
            )

            # ── Phase C: Multi-scale readout monitoring ──────────────
            # Which morphogen evolution steps does each theta prefer?
            scale_w = self.theta_readouts[i].get_scale_weights()  # (M,)
            # Peak step (argmax): which dilation scale dominates
            metrics[f"theta{i}_scale_peak"] = float(scale_w.argmax().item())
            # Scale entropy: high = reads from many steps, low = focused on few
            scale_entropy = -(scale_w * (scale_w + 1e-8).log()).sum().item()
            metrics[f"theta{i}_scale_entropy"] = scale_entropy
            # Max scale weight: how concentrated is the readout
            metrics[f"theta{i}_scale_max_w"] = float(scale_w.max().item())

        # Halting bias
        if self.config.enable_halting:
            metrics["halt_bias"] = self.halting.W_halt.bias.item()

        return metrics
