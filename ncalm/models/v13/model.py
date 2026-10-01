"""AENCA V13 main model: Morphogenetic Attention Language Model.

Architecture (V13 — Morphogenetic Attention):
  1. Token embedding + sinusoidal PE → initial state z₀ (B, L, d_model)
  2. Morphogen seed: h = tanh(W_h_init(z₀)) (B, L, d_h)
  3. Theta loop (θ = 0..n_theta-1):
     a. Morphogen evolution: M steps of conv → route → gate → update
     b. Generate Q/K gates: 1 + tanh(W_h(norm(h)))
     c. Morphogen-steered attention: Q*gate, K*gate → flash attention
     d. HebbianFFN: cross-modal pre-attn × post-attn concept gating
     e. Content→morphogen feedback: h = h + tanh(W_fb(norm(z)))
     f. PonderNet halting: halt_prob → z_out accumulation
  4. Readout from accumulated z_out → logits

V13 thesis:
  Morphogen field h steers HOW attention operates (via Q/K gates) without
  replacing attention itself. Local NCA dynamics on h create emergent
  attention patterns that cannot be achieved by content-only Q/K computation.
  At initialization, zero-init cascade ensures V13 = V7.1 exactly.

Key differences from V7.1:
  - HebbianKernel + CausalConv1d REMOVED (subsumed by morphogen, saves ~6.5M)
  - Gamma loop REMOVED (morphogen evolution replaces it)
  - Morphogen field h ADDED (~0.6M params)
  - Q/K gates from morphogen ADDED
  - d_ff wider (~8100 vs 7464) from freed params

Reference: ANCA paper (Kvalsund et al., 2026) — emergent attention from NCA.
Reference: V7.1 — proven theta-block architecture.
Reference: V11 — morphogen field dynamics.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ncalm.model import NCALMModel, register_model
from ncalm.models.v13.config import V13Config
from ncalm.models.v13.morphogen import MorphogenField
from ncalm.models.v13.attention import ThetaAttention
from ncalm.models.v13.hebbian_ffn import HebbianFFN
from ncalm.models.v13.halting import AdaptiveHalting

# Reuse readout from v5 — with mos_components=1 it's a plain linear head
from ncalm.models.v5.readout import MixtureOfSoftmaxReadout


# ═══════════════════════════════════════════════════════════════════════
# Utility Modules
# ═══════════════════════════════════════════════════════════════════════

class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal position encoding (non-parametric).

    Same as V7.1 — produces d-dimensional sinusoidal features.
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

@register_model("v13")
class AENCAv13Model(NCALMModel):
    """AENCA V13: Morphogenetic Attention Language Model.

    A morphogen-steered attention architecture where:
    - A lightweight morphogen field (d_h=128) evolves via local NCA dynamics
    - The morphogen generates Q/K gates that modulate attention patterns
    - Attention + HebbianFFN are unique per theta cycle (proven V7.1 pattern)
    - Zero-init cascade ensures V13 = V7.1 at initialization
    """

    def __init__(self, config_dict: dict) -> None:
        super().__init__()

        # Parse flat config dict into V13Config
        v13_keys = {f.name for f in V13Config.__dataclass_fields__.values()}
        v13_params = {k: v for k, v in config_dict.items() if k in v13_keys}

        # Handle morph_dilations: may come as list from JSON
        if "morph_dilations" in v13_params and isinstance(v13_params["morph_dilations"], list):
            v13_params["morph_dilations"] = tuple(v13_params["morph_dilations"])

        self.config = V13Config(**v13_params)
        cfg = self.config

        # ── Token Embedding ──────────────────────────────────────────
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.token_embedding.weight, std=1.0 / math.sqrt(cfg.d_model))

        # ── Position Encoding ────────────────────────────────────────
        self.state_pe = SinusoidalPositionEncoding(cfg.d_model, cfg.max_seq_len * 2)

        # ── Morphogen Field (SHARED across theta cycles) ─────────────
        self.morphogen = MorphogenField(cfg)

        # ── Unique Theta Attention Blocks (one per theta cycle) ──────
        self.theta_attns = nn.ModuleList([
            ThetaAttention(cfg, block_idx=i) for i in range(cfg.n_theta)
        ])

        # ── Unique Hebbian FFN Blocks (one per theta cycle) ──────────
        self.theta_ffns = nn.ModuleList([
            HebbianFFN(cfg.d_model, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_theta)
        ])

        # ── Pre-FFN norm per theta block ─────────────────────────────
        # Not part of HebbianFFN (which has its own internal norm_x, norm_y)
        # This is the block-level norm for the FFN sub-block — NOT NEEDED
        # because HebbianFFN handles its own normalization internally.

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
        """Complete V13 training step with autoregressive loss.

        Runs the morphogenetic attention loop:
        - Morphogen evolution per theta cycle (shared weights)
        - Morphogen-steered attention (unique weights per theta)
        - HebbianFFN (unique weights per theta)
        - Content→morphogen feedback (shared weights)
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
                z, h, cycle_routing = checkpoint(
                    self._run_theta_cycle,
                    z, h, theta_idx,
                    use_reentrant=False,
                )
            else:
                z, h, cycle_routing = self._run_theta_cycle(z, h, theta_idx)

            all_routing_weights.extend(cycle_routing)
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

        # ── 9. DIVERSITY LOSS ────────────────────────────────────────
        if cfg.lambda_div > 0 and all_routing_weights:
            # Stack all routing weights: list of (B, L, K) → (N, L, K)
            all_w = torch.cat(all_routing_weights, dim=0)   # (B*M*n_theta, L, K)
            # Negative entropy (Σ w·log(w) ≤ 0) → minimize = push most negative
            # = maximize entropy = encourage uniform/diverse routing
            div_loss = (all_w * (all_w + 1e-8).log()).sum(dim=-1).mean()
            total_loss = total_loss + cfg.lambda_div * div_loss
            info["div"] = div_loss.detach()

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
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Run one theta cycle: morph evolve → gated attention → FFN → feedback.

        Args:
            z:         (B, L, d_model) content state
            h:         (B, L, d_h) morphogen state
            theta_idx: which theta cycle (for selecting unique attention/FFN)

        Returns:
            z:                updated content state
            h:                updated morphogen state
            routing_weights:  list of (B, L, K) routing weights from morph evolution
        """
        cfg = self.config

        # ── 3a. MORPHOGEN EVOLUTION ──────────────────────────────────
        h, routing_weights = self.morphogen.evolve(h, cfg.n_morph_steps)

        # ── 3b. MORPHOGEN → ATTENTION GATES ──────────────────────────
        q_gate, k_gate = self.morphogen.generate_gates(h)

        # ── 3c. MORPHOGEN-STEERED ATTENTION ──────────────────────────
        z_pre = z  # save pre-attention state for HebbianFFN
        z = self.theta_attns[theta_idx](z, q_gate, k_gate)

        # ── 3d. HEBBIAN FFN ──────────────────────────────────────────
        z = z + self.theta_ffns[theta_idx](z_pre, z)

        # ── 3e. CONTENT → MORPHOGEN FEEDBACK ─────────────────────────
        h = self.morphogen.update_from_content(h, z)

        return z, h, routing_weights

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

            z, h, _ = self._run_theta_cycle(z, h, theta_idx)

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

        decay: 2D+ weight matrices (attention, FFN, morphogen projections)
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
        """V13 architecture description for analysis.json and logging."""
        cfg = self.config
        total = sum(p.numel() for p in self.parameters())

        morph_params = sum(p.numel() for p in self.morphogen.parameters())
        embed_params = sum(p.numel() for p in self.token_embedding.parameters())
        readout_params = sum(p.numel() for p in self.readout.parameters())
        attn_params = sum(p.numel() for p in self.theta_attns.parameters())
        ffn_params = sum(p.numel() for p in self.theta_ffns.parameters())
        halt_params = (
            sum(p.numel() for p in self.halting.parameters())
            if cfg.enable_halting else 0
        )

        return {
            "architecture": "AENCA v13 (Morphogenetic Attention)",
            "total_params": total,
            "morphogen_params": morph_params,
            "embed_params": embed_params,
            "readout_params": readout_params,
            "theta_attention_params": attn_params,
            "theta_ffn_params": ffn_params,
            "halt_params": halt_params,
            "d_model": cfg.d_model,
            "d_h": cfg.d_h,
            "d_ff": cfg.d_ff,
            "n_theta": cfg.n_theta,
            "n_morph_steps": cfg.n_morph_steps,
            "n_rules": cfg.n_rules,
            "n_heads": cfg.n_heads,
            "d_head": cfg.d_head,
            "morph_dilations": list(cfg.morph_dilations),
            "enable_halting": cfg.enable_halting,
            "mos_components": cfg.mos_components,
            "tie_embeddings": cfg.tie_embeddings,
            "vocab_size": cfg.vocab_size,
            "paradigm": "Morphogenetic Attention (NCA morphogen → Q/K gate steering)",
        }

    # ══════════════════════════════════════════════════════════════════
    # Lifecycle Hooks
    # ══════════════════════════════════════════════════════════════════

    def get_phases(self, step: int, max_steps: int) -> str:
        """Return active training phases for logging."""
        return "MorphAttn"

    def extra_metrics(self) -> dict[str, float]:
        """V13-specific monitoring metrics."""
        metrics: dict[str, float] = {}

        # Morphogen rule activity: W_h2 norms per rule (zero-init, should grow)
        for k in range(self.config.n_rules):
            metrics[f"morph_rule{k}_norm"] = (
                self.morphogen.W_h2[k].weight.norm().item()
            )

        # Q/K gate weight norms (zero-init, should grow)
        metrics["morph_Wqh_norm"] = self.morphogen.W_Q_h.weight.norm().item()
        metrics["morph_Wkh_norm"] = self.morphogen.W_K_h.weight.norm().item()

        # Router weight norm (zero-init)
        metrics["morph_router_norm"] = self.morphogen.W_router.weight.norm().item()

        # Feedback weight norm (zero-init)
        metrics["morph_feedback_norm"] = self.morphogen.W_h_feedback.weight.norm().item()

        # Theta block activity: out_proj and FFN decoder norms
        for i in range(self.config.n_theta):
            metrics[f"theta{i}_attn_norm"] = (
                self.theta_attns[i].out_proj.weight.norm().item()
            )
            metrics[f"theta{i}_ffn_dec_norm"] = (
                self.theta_ffns[i].W_dec.weight.norm().item()
            )

        # Halting bias
        if self.config.enable_halting:
            metrics["halt_bias"] = self.halting.W_halt.bias.item()

        return metrics

    def on_step_start(self, step: int, max_steps: int) -> None:
        """Hook called before each training step."""
        pass

    def on_step_end(self, step: int, metrics: dict) -> None:
        """Hook called after each training step."""
        pass
