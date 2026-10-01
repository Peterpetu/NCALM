"""AENCA V15 main model: 2D Latent Reaction-Diffusion Morphogenetic Network.

Architecture (V15 — Turing RD-NCA):
  1. Token embedding + sinusoidal PE → initial state z₀ (B, N, d_model)
  2. Morphogen seed: a, b = seed(z₀) → (B, d_a, H, W), (B, d_b, H, W)
  3. Theta loop (θ = 0..n_theta-1):
     a. Content conditioning: c = Fold(W_cond(Norm(z)))
     b. RD-NCA evolution (M steps on 2D grid):
        For m = 0..M-1:
          - D_a = P_{3×3}(a), D_b = P_{5×5}(b)      [CAUSAL 2D PERCEPTION]
          - p = [a, b, D_a, D_b, c, π]               [PERCEPTION VECTOR]
          - δ = Σ_k w_k · rule_k(Norm(p))             [K PARALLEL RULES]
          - g_a, g_b = BilinearGate(a, b, c)          [CONTENT-COND GATE]
          - a += mask · g_a ⊙ δ_a                     [STOCHASTIC UPDATE]
          - b += mask · g_b ⊙ δ_b
     c. Content readout: z += W_read(Norm([a;b]))     [MORPH → CONTENT]
     d. Integration FFN: z += SwiGLU(z)               [PER-POSITION FFN]
     e. Content→morph feedback: a+=Fold(tanh(W_fb_a)), etc.
     f. PonderNet halting                             [ADAPTIVE HALT]
  4. Readout from accumulated z_out → logits

V15 thesis:
  Information routing EMERGES from dual-species Turing reaction-diffusion
  dynamics on a 2D latent grid. Position differentiation is GUARANTEED by
  the differential diffusion architecture (k_b > k_a → D_b/D_a > 1).
  The model starts as bag-of-embeddings (zero-init cascade) and gradually
  activates NCA dynamics during training.

Loss functions (§15):
  L_CE: cross-entropy
  L_spatial: spatial diversity (targets position cos_sim < 0.99)
  L_temporal: temporal diversity (targets step cos_sim < 0.99)
  L_div: routing entropy
  L_bal: load balance
  L_dyn: dynamics magnitude
  L_drift: drift regularization
  L_aux: deep supervision
  L_halt: PonderNet KL
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ncalm.model import NCALMModel, register_model
from ncalm.models.v15.config import V15Config
from ncalm.models.v15.reaction_diffusion import (
    ReactionDiffusionField,
    ContentReadout,
    unfold_2d_to_1d,
)
from ncalm.models.v15.integration_ffn import IntegrationFFN
from ncalm.models.v15.halting import AdaptiveHalting

# Reuse readout from v5
from ncalm.models.v5.readout import MixtureOfSoftmaxReadout


# ═══════════════════════════════════════════════════════════════════════
# Utility Modules
# ═══════════════════════════════════════════════════════════════════════

class SinusoidalPositionEncoding(nn.Module):
    """Fixed 1D sinusoidal position encoding (non-parametric).

    Applied to the initial token embeddings in the 1D sequence domain
    before folding to 2D. Provides absolute sequence position information.
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

@register_model("v15")
class AENCAv15Model(NCALMModel):
    """AENCA V15: 2D Latent Reaction-Diffusion Morphogenetic Network.

    Dual-species Turing reaction-diffusion NCA on a folded 2D grid:
    - Activator a (d_a=128, 3×3 causal) + Inhibitor b (d_b=64, 5×5 causal)
    - Turing instability: D_b/D_a = 12/4 = 3.0 guarantees spatial patterns
    - Content conditioning at every NCA step (c-NCA)
    - Stochastic asynchronous updates (p=0.5 training)
    - Zero-init cascade: starts as bag-of-embeddings
    - O(N) compute, no quadratic attention
    """

    def __init__(self, config_dict: dict) -> None:
        super().__init__()

        # Parse flat config dict into V15Config
        v15_keys = {f.name for f in V15Config.__dataclass_fields__.values()}
        v15_params = {k: v for k, v in config_dict.items() if k in v15_keys}
        self.config = V15Config(**v15_params)
        cfg = self.config
        assert cfg.d_ff > 0, (
            f"d_ff must be positive, got {cfg.d_ff}. "
            f"Reduce target_params or model dimensions."
        )

        # ── Token Embedding ──────────────────────────────────────────
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.token_embedding.weight, std=0.02)

        # ── Position Encoding (1D, for initial embeddings) ───────────
        self.state_pe = SinusoidalPositionEncoding(cfg.d_model, cfg.max_seq_len * 2)

        # ── Reaction-Diffusion Field (SHARED across theta) ───────────
        self.rd_field = ReactionDiffusionField(cfg)

        # ── Content Readout (UNIQUE per theta cycle) ─────────────────
        self.theta_readouts = nn.ModuleList([
            ContentReadout(cfg) for _ in range(cfg.n_theta)
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
        """Complete V15 training step with autoregressive loss.

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
        z0 = z.detach().clone()  # for drift regularization

        # ── 2. MORPHOGEN SEED ────────────────────────────────────────
        a, b = self.rd_field.seed(z)  # (B, d_a, H, W), (B, d_b, H, W)

        # ── 3. THETA LOOP ───────────────────────────────────────────
        z_out = torch.zeros_like(z)
        remaining = torch.ones(B, L_full, device=z.device)
        halt_probs: list[torch.Tensor] = []
        all_routing_weights: list[torch.Tensor] = []
        all_updates_a: list[torch.Tensor] = []
        all_updates_b: list[torch.Tensor] = []

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
                z, a, b, cycle_routing, cycle_updates_a, cycle_updates_b = checkpoint(
                    self._run_theta_cycle,
                    z, a, b, theta_idx,
                    use_reentrant=False,
                )
            else:
                z, a, b, cycle_routing, cycle_updates_a, cycle_updates_b = (
                    self._run_theta_cycle(z, a, b, theta_idx)
                )

            all_routing_weights.extend(cycle_routing)
            all_updates_a.extend(cycle_updates_a)
            all_updates_b.extend(cycle_updates_b)
            n_active_cycles += 1

            # ── Halting / readout accumulation ───────────────────────
            if cfg.enable_halting:
                z_out, remaining, hp = self.halting.accumulate(
                    z, a, L_full, z_out, remaining, is_last,
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

            # Deep supervision snapshot: use z^(θ) directly (plan §15.9)
            if not is_last and cfg.lambda_aux > 0:
                aux_substrates.append((theta_idx + 1, z.clone()))

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

        # ── 7. DEEP SUPERVISION (§15.9) ────────────────────────────────
        # L_aux = (1/n_θ) Σ_θ CE(Readout(z^(θ)), targets)
        # Uses content state z directly — NOT the partial PonderNet z_out
        if cfg.lambda_aux > 0 and aux_substrates:
            targets_flat_aux = batch[:, 1:].reshape(-1)
            aux_total = torch.tensor(0.0, device=z.device)
            n_aux = 0

            for step_idx, z_snap in aux_substrates:
                w = (step_idx / cfg.n_theta) ** 2

                aux_logits = self.readout(z_snap)
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

        # ── 9. ROUTING DIVERSITY LOSS (entropy + load-balance) ───────
        if all_routing_weights:
            all_w = torch.stack(all_routing_weights, dim=0)  # (S, B, HW, K)

            # 9a. Negative entropy → maximize routing entropy
            if cfg.lambda_div > 0:
                div_loss = (all_w * (all_w + 1e-8).log()).sum(dim=-1).mean()
                total_loss = total_loss + cfg.lambda_div * div_loss
                info["div"] = div_loss.detach()

            # 9b. Switch Transformer load-balance loss
            if cfg.lambda_bal > 0:
                top1 = all_w.argmax(dim=-1)  # (S, B, HW)
                f = torch.zeros(cfg.n_rules, device=z.device)
                for k in range(cfg.n_rules):
                    f[k] = (top1 == k).float().mean()
                P = all_w.mean(dim=(0, 1, 2))  # (K,)
                bal_loss = cfg.n_rules * (f * P).sum()
                total_loss = total_loss + cfg.lambda_bal * bal_loss
                info["bal"] = bal_loss.detach()

        # ── 10. DYNAMICS MAGNITUDE LOSS ──────────────────────────────
        if cfg.lambda_dyn > 0 and (all_updates_a or all_updates_b):
            # Compute smooth norms of gated updates
            dyn_norms = []
            for ua, ub in zip(all_updates_a, all_updates_b):
                # ua: (B, d_a, H, W), ub: (B, d_b, H, W)
                norm_a = ua.pow(2).sum(dim=1)  # (B, H, W)
                norm_b = ub.pow(2).sum(dim=1)  # (B, H, W)
                smooth = (norm_a + norm_b + 1e-8).sqrt()  # (B, H, W)
                dyn_norms.append(smooth)

            dyn_stack = torch.stack(dyn_norms, dim=0)  # (S, B, H, W)
            dyn_loss = -dyn_stack.mean()
            total_loss = total_loss + cfg.lambda_dyn * dyn_loss
            info["dyn"] = dyn_loss.detach()

        # ── 11. SPATIAL DIVERSITY LOSS (§15.2) ───────────────────────
        if cfg.lambda_spatial > 0 and n_active_cycles > 0:
            spatial_loss = self._compute_spatial_diversity(a, L_full)
            total_loss = total_loss + cfg.lambda_spatial * spatial_loss
            info["spatial"] = spatial_loss.detach()

        # ── 12. TEMPORAL DIVERSITY LOSS (§15.3) ──────────────────────
        if cfg.lambda_temporal > 0 and len(all_updates_a) >= 2:
            temporal_loss = self._compute_temporal_diversity(
                all_updates_a, all_updates_b,
            )
            total_loss = total_loss + cfg.lambda_temporal * temporal_loss
            info["temporal"] = temporal_loss.detach()

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
    # Spatial Diversity Loss (§15.2)
    # ══════════════════════════════════════════════════════════════════

    def _compute_spatial_diversity(
        self, a: torch.Tensor, N: int
    ) -> torch.Tensor:
        """Penalize position-identical activator morphogen.

        Samples 64 random position pairs, computes mean cosine similarity.

        Args:
            a: (B, d_a, H, W) final activator
            N: sequence length

        Returns:
            spatial_loss: scalar (high = bad, positions too similar)
        """
        # Unfold to 1D: (B, N, d_a)
        a_1d = unfold_2d_to_1d(a, N)
        B, N_actual, da = a_1d.shape

        n_pairs = min(64, N_actual * (N_actual - 1) // 2)
        if n_pairs < 1:
            return torch.tensor(0.0, device=a.device)

        # Sample random pairs
        idx = torch.randint(0, N_actual, (n_pairs, 2), device=a.device)
        # Ensure i != j
        mask = idx[:, 0] == idx[:, 1]
        idx[mask, 1] = (idx[mask, 1] + 1) % N_actual

        # Gather: (B, n_pairs, d_a)
        ai = a_1d[:, idx[:, 0], :]  # (B, n_pairs, d_a)
        aj = a_1d[:, idx[:, 1], :]  # (B, n_pairs, d_a)

        # Cosine similarity
        cos_sim = F.cosine_similarity(ai, aj, dim=-1)  # (B, n_pairs)
        return cos_sim.mean()

    # ══════════════════════════════════════════════════════════════════
    # Temporal Diversity Loss (§15.3)
    # ══════════════════════════════════════════════════════════════════

    def _compute_temporal_diversity(
        self,
        updates_a: list[torch.Tensor],
        updates_b: list[torch.Tensor],
    ) -> torch.Tensor:
        """Penalize consecutive NCA steps producing identical updates.

        Computes mean cosine similarity of spatially-averaged consecutive
        update vectors [u_a; u_b].

        Args:
            updates_a: list of (B, d_a, H, W) gated updates per step
            updates_b: list of (B, d_b, H, W) gated updates per step

        Returns:
            temporal_loss: scalar (high = bad, steps too similar)
        """
        M = len(updates_a)
        if M < 2:
            return torch.tensor(0.0, device=updates_a[0].device)

        # Compute spatially-averaged update vectors per step
        mean_updates = []
        for ua, ub in zip(updates_a, updates_b):
            # Mean over spatial dims: (B, d, H, W) → (B, d)
            ua_mean = ua.mean(dim=(2, 3))  # (B, d_a)
            ub_mean = ub.mean(dim=(2, 3))  # (B, d_b)
            mean_updates.append(torch.cat([ua_mean, ub_mean], dim=-1))  # (B, d_s)

        # Stack: (M, B, d_s)
        update_stack = torch.stack(mean_updates, dim=0)

        # Cosine similarity between consecutive steps
        cos_sim = F.cosine_similarity(
            update_stack[:-1], update_stack[1:], dim=-1
        )  # (M-1, B)

        return cos_sim.mean()

    # ══════════════════════════════════════════════════════════════════
    # Theta Cycle
    # ══════════════════════════════════════════════════════════════════

    def _run_theta_cycle(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        theta_idx: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """Run one theta cycle: RD-NCA → readout → FFN → feedback.

        Args:
            z:         (B, N, d_model) content state
            a:         (B, d_a, H, W) activator
            b:         (B, d_b, H, W) inhibitor
            theta_idx: theta cycle index

        Returns:
            z:              updated content state
            a:              updated activator
            b:              updated inhibitor
            routing_weights: list of (B, HW, K) per NCA step
            updates_a:      list of (B, d_a, H, W) gated updates
            updates_b:      list of (B, d_b, H, W) gated updates
        """
        cfg = self.config
        N = z.shape[1]

        # ── Θ-2: RD-NCA Evolution ────────────────────────────────────
        a, b, routing_weights, updates_a, updates_b = self.rd_field.evolve(
            a, b, z, cfg.n_morph_steps,
        )

        # ── Θ-3: Content Readout (morph→content, unique per θ) ───────
        z = z + self.theta_readouts[theta_idx](a, b, N)

        # ── Θ-4: Integration FFN (unique per θ) ─────────────────────
        z = z + self.theta_ffns[theta_idx](z)

        # ── Θ-5: Content → Morphogen Feedback (shared) ──────────────
        a, b = self.rd_field.update_from_content(a, b, z)

        return z, a, b, routing_weights, updates_a, updates_b

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
        a, b = self.rd_field.seed(z)

        # Halting accumulators
        z_out = torch.zeros_like(z)
        remaining = torch.ones(B, L, device=device)

        for theta_idx in range(cfg.n_theta):
            is_last = (theta_idx == cfg.n_theta - 1)

            if cfg.enable_halting and (remaining <= cfg.halt_eps).all():
                break

            z, a, b, _, _, _ = self._run_theta_cycle(z, a, b, theta_idx)

            if cfg.enable_halting:
                z_out, remaining, _ = self.halting.accumulate(
                    z, a, L, z_out, remaining, is_last,
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

        decay: 2D+ weight matrices
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
        """V15 architecture description for analysis.json and logging."""
        cfg = self.config
        total = sum(p.numel() for p in self.parameters())

        rd_params = sum(p.numel() for p in self.rd_field.parameters())
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
            "architecture": "AENCA v15 (2D RD-NCA)",
            "total_params": total,
            "rd_field_params": rd_params,
            "embed_params": embed_params,
            "readout_head_params": readout_head_params,
            "content_readout_params": content_readout_params,
            "theta_ffn_params": ffn_params,
            "halt_params": halt_params,
            "d_model": cfg.d_model,
            "d_a": cfg.d_a,
            "d_b": cfg.d_b,
            "d_s": cfg.d_s,
            "d_c": cfg.d_c,
            "d_pos": cfg.d_pos,
            "d_rule": cfg.d_rule,
            "d_p": cfg.d_p,
            "d_ff": cfg.d_ff,
            "n_theta": cfg.n_theta,
            "n_morph_steps": cfg.n_morph_steps,
            "n_rules": cfg.n_rules,
            "k_a": cfg.k_a,
            "k_b": cfg.k_b,
            "grid_size": f"{cfg.grid_height}x{cfg.grid_width}",
            "p_async": cfg.p_async,
            "enable_halting": cfg.enable_halting,
            "mos_components": cfg.mos_components,
            "tie_embeddings": cfg.tie_embeddings,
            "vocab_size": cfg.vocab_size,
            "target_params": cfg.target_params,
            "n_levels": cfg.n_levels,
            "turing_ratio": (
                f"D_b/D_a = "
                f"{int(self.rd_field.perc_b.causal_mask.sum().item()) - 1}/"
                f"{int(self.rd_field.perc_a.causal_mask.sum().item()) - 1} = "
                f"{(self.rd_field.perc_b.causal_mask.sum().item() - 1) / max(self.rd_field.perc_a.causal_mask.sum().item() - 1, 1):.1f}"
            ),
            "edge_of_chaos": f"ρ = {cfg.d_rule}/{cfg.d_s} = {cfg.d_rule/cfg.d_s:.1f}",
            "paradigm": (
                "2D Latent Reaction-Diffusion NCA "
                "(dual-species Turing, causal 2D, no attention)"
            ),
        }

    # ══════════════════════════════════════════════════════════════════
    # Lifecycle Hooks
    # ══════════════════════════════════════════════════════════════════

    def get_phases(self, step: int, max_steps: int) -> str:
        """Return active training phases for logging."""
        return "TuringRD-NCA"

    def extra_metrics(self) -> dict[str, float]:
        """V15-specific monitoring metrics."""
        metrics: dict[str, float] = {}

        # Rule W_r2 norms (should grow from zero)
        for k in range(self.config.n_rules):
            w_norm = self.rd_field.W_r2.data[k].norm().item()
            metrics[f"rule_{k}_W_r2_norm"] = w_norm

        # Feedback norms (should grow from zero)
        metrics["W_fb_a_norm"] = self.rd_field.W_fb_a.weight.data.norm().item()
        metrics["W_fb_b_norm"] = self.rd_field.W_fb_b.weight.data.norm().item()

        # Readout norms (should grow from zero)
        for theta_idx in range(self.config.n_theta):
            w_norm = self.theta_readouts[theta_idx].W_readout.weight.data.norm().item()
            metrics[f"theta_{theta_idx}_readout_norm"] = w_norm

        # Perception kernel norms
        metrics["perc_a_weight_norm"] = self.rd_field.perc_a.weight.data.norm().item()
        metrics["perc_b_weight_norm"] = self.rd_field.perc_b.weight.data.norm().item()

        # Router norm (should grow from zero)
        metrics["W_route_norm"] = self.rd_field.W_route.weight.data.norm().item()

        return metrics
