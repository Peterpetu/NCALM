#!/usr/bin/env python3
"""Morphogenetic field analysis for AENCA V11.

Instruments the coupled CA loop to capture per-step dynamics of both
the content field z and morphogen field h, then produces a comprehensive
diagnostic report answering:

  1. GATE HETEROGENEITY: Do morphogen gates develop spatial diversity?
     (uniform gates ≈ V10, diverse gates ≈ emergent routing)

  2. MORPHOGEN EVOLUTION: Does h evolve non-trivially over 64 CA steps?
     (frozen h ≈ wasted params, dynamic h ≈ signaling)

  3. FIELD COUPLING STRENGTH: Does morphogen actually modulate content?
     (measuring gate variance, content-gated vs ungated divergence)

  4. SPATIAL DIFFERENTIATION: Do different positions develop different
     morphogen signatures? (Turing patterns, gradients, specialization)

  5. TEMPORAL DYNAMICS: Reaction-diffusion patterns, oscillation, convergence

Usage:
    python scripts/analyze_morphogen.py <checkpoint_path> [--device cuda] [--n_samples 5]
    python scripts/analyze_morphogen.py experiments/AENCA-V11-MorphogeneticField/smoke_v11_.../best_model.pt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np

# ── Project imports ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ncalm.model import get_model_class
from ncalm.data.dataset import (
    TinyStoriesData, ThePileData, SmolTalk2Data, create_data_loader,
)
from ncalm.data.tokenizer import create_tokenizer


# ═══════════════════════════════════════════════════════════════════════
# Data Structures for per-step recordings
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StepRecord:
    """Per-step statistics captured during instrumented CA loop."""
    step: int
    dilation: int
    # ── Content field z ──
    z_mean: float
    z_std: float
    z_norm: float
    delta_z_norm: float
    # ── Morphogen field h ──
    h_mean: float
    h_std: float
    h_norm: float
    delta_h_norm: float
    # ── Gate statistics (THE key diagnostic) ──
    gate_mean: float          # Mean across (B, L, d_model). Should be ≈0.95 at init
    gate_std: float           # Std across positions. Higher = more routing diversity
    gate_min: float
    gate_max: float
    gate_spatial_std: float   # Avg std across positions (per batch/dim), collapsed to scalar
    gate_dim_std: float       # Avg std across dimensions (per batch/pos), collapsed to scalar
    # ── Growth/Settlement ──
    growth_frac: float        # Fraction of cells still active
    settled_frac: float       # Fraction of cells that have settled


@dataclass
class MorphogenAnalysis:
    """Complete morphogen analysis results."""
    step_records: list[StepRecord]
    # ── Global statistics ──
    n_steps: int
    n_params_content: int
    n_params_morphogen: int
    # ── Gate weight analysis (learned W_gate) ──
    w_gate_weight_norm: float
    w_gate_bias_mean: float
    w_gate_weight_rank: int      # Effective rank of W_gate
    w_gate_weight_svd_top5: list[float]  # Top 5 singular values
    # ── Morphogen init analysis (W_h_init) ──
    w_h_init_norm: float
    w_h_init_rank: int
    # ── Content rule analysis (W_d learned from zero) ──
    w_d_norm: float
    # ── Morphogen rule analysis (W_h2 learned from zero) ──
    w_h2_norm: float
    # ── Morphogen dimension specialization ──
    h_dim_variance: list[float]          # Per-dimension variance across positions
    h_dim_correlation_matrix: np.ndarray  # d_h × d_h correlation of morphogen dims
    # ── Gate spatial patterns (per position) ──
    gate_position_profile: np.ndarray     # (L,) avg gate value per position
    # ── Morphogen position profile ──
    h_position_norms: np.ndarray          # (L,) avg h norm per position


# ═══════════════════════════════════════════════════════════════════════
# Instrumented Forward Pass
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def instrumented_ca_forward(
    model,
    batch: torch.Tensor,
) -> tuple[list[StepRecord], dict]:
    """Run V11 CA loop with full per-step instrumentation.

    Instead of calling model.forward_train(), we replicate the CA loop
    here with recording hooks at every step. This captures the full
    morphogenetic dynamics that normal training never logs.

    Returns:
        records: List of StepRecord for all 64 steps
        final_state: Dict with final z, h, settlement, gate tensors
    """
    cfg = model.config
    B, L = batch.shape
    device = batch.device

    # ── Seed (same as model.forward_train) ────────────────────────────
    positions = torch.arange(L, device=device)
    z = model.token_embedding(batch) + model.pos_embedding(positions)

    # ── Seed morphogen (via scaffold if available) ─────────────────
    h = model._seed_morphogen(z)  # (B, L, d_h) — includes scaffold contribution

    # ── Also compute pure h_fresh for scaffold comparison ─────────────
    h_fresh = torch.tanh(model.W_h_init(z))  # (B, L, d_h)
    scaffold_contribution = (h - h_fresh).norm(dim=-1).mean().item()  # scalar

    # ── Scaffold template selection weights (if scaffold active) ──────
    scaffold_weights = None  # (B, L, K) or None
    if (hasattr(model, 'scaffold_templates')
            and model.config.n_scaffold_templates > 0
            and model.scaffold_templates.norm(dim=-1).max().item() > 1e-8):
        M = model.scaffold_templates  # (K, d_h)
        h_normed = F.normalize(h_fresh, dim=-1)
        M_normed = F.normalize(M, dim=-1)
        tau_scaled = model.config.scaffold_temperature * math.sqrt(model.config.d_h)
        scores = torch.matmul(h_normed, M_normed.t()) / tau_scaled
        scaffold_weights = F.softmax(scores, dim=-1)  # (B, L, K)

    # ── Settlement ────────────────────────────────────────────────────
    from ncalm.models.v11.settlement import SettlementState
    settlement = SettlementState.create(B, L, device, cfg)

    records = []
    all_h_snapshots = []   # Store h at each step for temporal analysis
    all_gate_snapshots = []

    for t, dilation in enumerate(model._dilation_schedule):
        # ── Compute gate ──────────────────────────────────────────────
        gate = model.compute_gate(h)  # (B, L, d_model)

        # ── Content update (with gating) ──────────────────────────────
        delta_z = model.rule(z, dilation=dilation, gate=gate)

        # ── Morphogen update ──────────────────────────────────────────
        delta_h = model.morph_rule(h, dilation=dilation)

        # ── Growth gating ─────────────────────────────────────────────
        growth = settlement.growth_gate().unsqueeze(-1)  # (B, L, 1)
        growth_frac = growth.squeeze(-1).mean().item()

        # ── Record BEFORE applying updates ────────────────────────────
        gate_per_pos = gate.mean(dim=-1)    # (B, L) avg gate across dims
        gate_per_dim = gate.mean(dim=1)     # (B, d_model) avg gate across positions

        record = StepRecord(
            step=t,
            dilation=dilation,
            # Content
            z_mean=z.mean().item(),
            z_std=z.std().item(),
            z_norm=z.norm(dim=-1).mean().item(),
            delta_z_norm=delta_z.norm(dim=-1).mean().item(),
            # Morphogen
            h_mean=h.mean().item(),
            h_std=h.std().item(),
            h_norm=h.norm(dim=-1).mean().item(),
            delta_h_norm=delta_h.norm(dim=-1).mean().item(),
            # Gate
            gate_mean=gate.mean().item(),
            gate_std=gate.std().item(),
            gate_min=gate.min().item(),
            gate_max=gate.max().item(),
            gate_spatial_std=gate_per_pos.std(dim=-1).mean().item(),  # diversity across positions
            gate_dim_std=gate_per_dim.std(dim=-1).mean().item(),      # diversity across dimensions
            # Settlement
            growth_frac=growth_frac,
            settled_frac=1.0 - growth_frac,
        )
        records.append(record)

        # Save snapshots for deeper analysis (keep on CPU to save VRAM)
        if t % max(1, cfg.n_steps // 16) == 0 or t == cfg.n_steps - 1:
            all_h_snapshots.append((t, h.cpu()))
            all_gate_snapshots.append((t, gate.cpu()))

        # ── Apply updates ─────────────────────────────────────────────
        z = z + delta_z * growth
        h = h + delta_h * growth
        settlement.update(delta_z, cfg)

    final_state = {
        "z": z.cpu(),
        "h": h.cpu(),
        "h_fresh": h_fresh.cpu(),
        "scaffold_contribution": scaffold_contribution,
        "scaffold_weights": scaffold_weights.cpu() if scaffold_weights is not None else None,
        "gate": model.compute_gate(h).cpu(),
        "h_snapshots": all_h_snapshots,
        "gate_snapshots": all_gate_snapshots,
        "settlement_settled": settlement.settled.cpu(),
        "settlement_a_bar": settlement.a_bar.cpu(),
    }

    return records, final_state


# ═══════════════════════════════════════════════════════════════════════
# Weight Analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze_weights(model) -> dict:
    """Analyze learned weight matrices for morphogenetic significance."""
    results = {}

    # ── Bilinear gate: analyze the combined projection structure ─────
    # W_gate_a and W_gate_b produce gate = σ(SiLU(h·W_a) ⊙ h·W_b + bias)
    # We analyze the effective combined structure via W_a^T W_b
    W_a = model.W_gate_a.weight.detach().float().cpu()  # (d_model, d_h)
    W_b = model.W_gate_b.weight.detach().float().cpu()  # (d_model, d_h)
    # Combined norm indicates how much gate can modulate
    results["w_gate_weight_norm"] = (W_a.norm() + W_b.norm()).item() / 2.0
    results["w_gate_bias_mean"] = model.gate_bias.detach().float().mean().item()

    # SVD of element-wise combined projection (cat along columns)
    W_combined = torch.cat([W_a, W_b], dim=1)  # (d_model, 2*d_h)
    U, S, Vh = torch.linalg.svd(W_combined, full_matrices=False)
    results["w_gate_weight_svd_top5"] = S[:5].tolist()

    # Effective rank via Shannon entropy
    s_normalized = S / S.sum()
    s_pos = s_normalized[s_normalized > 1e-10]
    entropy = -(s_pos * s_pos.log()).sum().item()
    results["w_gate_weight_rank"] = int(math.exp(entropy))

    # ── W_h_init: morphogen seeding structure ─────────────────────────
    W_h_init = model.W_h_init.weight.detach().float().cpu()  # (d_h, d_model)
    results["w_h_init_norm"] = W_h_init.norm().item()
    U2, S2, _ = torch.linalg.svd(W_h_init, full_matrices=False)
    s2_norm = S2 / S2.sum()
    s2_pos = s2_norm[s2_norm > 1e-10]
    e2 = -(s2_pos * s2_pos.log()).sum().item()
    results["w_h_init_rank"] = int(math.exp(e2))

    # ── W_d: content rule decoder (learned from zero) ─────────────────
    results["w_d_norm"] = model.rule.W_d.weight.detach().float().norm().item()

    # ── W_h2: morphogen rule output (learned from zero) ───────────────
    results["w_h2_norm"] = model.morph_rule.W_h2.weight.detach().float().norm().item()

    return results


# ═══════════════════════════════════════════════════════════════════════
# Morphogen Spatial Analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze_spatial_patterns(final_state: dict) -> dict:
    """Analyze spatial structure in morphogen and gate fields."""
    results = {}

    h = final_state["h"]  # (B, L, d_h)
    gate = final_state["gate"]  # (B, L, d_model)

    # ── Per-position gate profile ─────────────────────────────────────
    # If morphogen creates meaningful structure, different positions
    # should have different average gate values
    gate_per_pos = gate.mean(dim=(0, 2))  # (L,) average across batch & dims
    results["gate_position_profile"] = gate_per_pos.numpy()
    results["gate_position_range"] = (gate_per_pos.max() - gate_per_pos.min()).item()

    # ── Per-position morphogen norms ──────────────────────────────────
    h_pos_norms = h.norm(dim=-1).mean(dim=0)  # (L,)
    results["h_position_norms"] = h_pos_norms.numpy()
    results["h_position_norm_range"] = (h_pos_norms.max() - h_pos_norms.min()).item()

    # ── Morphogen dimension specialization ────────────────────────────
    # If different d_h dims encode different spatial patterns,
    # they should have low inter-dimension correlation
    h_flat = h.reshape(-1, h.shape[-1])  # (B*L, d_h)
    h_centered = h_flat - h_flat.mean(dim=0, keepdim=True)
    h_std = h_centered.std(dim=0, keepdim=True).clamp(min=1e-8)
    h_normalized = h_centered / h_std
    corr_matrix = (h_normalized.T @ h_normalized) / h_normalized.shape[0]  # (d_h, d_h)
    results["h_dim_correlation"] = corr_matrix.numpy()

    # Off-diagonal correlation: lower = more independent dimensions
    d_h = corr_matrix.shape[0]
    off_diag_mask = ~torch.eye(d_h, dtype=torch.bool)
    off_diag_corr = corr_matrix[off_diag_mask].abs().mean().item()
    results["h_dim_off_diag_corr"] = off_diag_corr

    # Per-dimension variance across positions (are dims specialized?)
    h_dim_var = h.var(dim=1).mean(dim=0)  # (d_h,) variance across L, avg across B
    results["h_dim_variance"] = h_dim_var.numpy()
    results["h_dim_var_range"] = (h_dim_var.max() - h_dim_var.min()).item()

    return results


# ═══════════════════════════════════════════════════════════════════════
# Scaffold Analysis (Persistent Morphogen Templates)
# ═══════════════════════════════════════════════════════════════════════

def analyze_scaffold(model, final_states: list[dict]) -> dict:
    """Analyze persistent scaffold templates and their influence on seeding.

    This addresses the epigenetic memory question: do templates crystallize
    meaningful structure, and does that structure actually affect inference?
    """
    results = {"active": False}

    cfg = model.config
    K = cfg.n_scaffold_templates
    if K == 0 or not hasattr(model, "scaffold_templates"):
        return results

    M = model.scaffold_templates.detach().cpu().float()  # (K, d_h)
    usage = model.scaffold_usage.detach().cpu().float()   # (K,)
    step = model.scaffold_step.item()

    # ── Template health ───────────────────────────────────────────────
    norms = M.norm(dim=-1)  # (K,)
    alive_mask = norms >= cfg.scaffold_min_norm
    n_alive = alive_mask.sum().item()

    results["active"] = n_alive > 0
    results["K"] = K
    results["n_alive"] = n_alive
    results["step"] = step
    results["alpha"] = cfg.scaffold_alpha
    results["ema_decay"] = cfg.scaffold_ema_decay
    results["warmup_steps"] = cfg.scaffold_warmup_steps

    results["template_norms"] = norms.tolist()
    results["template_norm_mean"] = norms[alive_mask].mean().item() if n_alive > 0 else 0.0
    results["template_norm_std"] = norms[alive_mask].std().item() if n_alive > 1 else 0.0
    results["template_norm_min"] = norms[alive_mask].min().item() if n_alive > 0 else 0.0
    results["template_norm_max"] = norms[alive_mask].max().item() if n_alive > 0 else 0.0

    # ── Usage distribution ────────────────────────────────────────────
    results["usage"] = usage.tolist()
    results["usage_total"] = usage.sum().item()
    if usage.sum() > 0:
        p = usage / usage.sum()
        p_pos = p[p > 0]
        results["usage_entropy"] = -(p_pos * p_pos.log()).sum().item()
        results["usage_max_entropy"] = math.log(max(1, n_alive))
        results["usage_uniformity"] = (
            results["usage_entropy"] / max(results["usage_max_entropy"], 1e-8)
        )
    else:
        results["usage_entropy"] = 0.0
        results["usage_max_entropy"] = 0.0
        results["usage_uniformity"] = 0.0

    # ── Template diversity (inter-template cosine similarity) ─────────
    if n_alive >= 2:
        alive_M = M[alive_mask]  # (n_alive, d_h)
        alive_M_norm = F.normalize(alive_M, dim=-1)
        cosine_matrix = alive_M_norm @ alive_M_norm.T  # (n_alive, n_alive)
        # Off-diagonal mean cosine
        mask = ~torch.eye(n_alive, dtype=torch.bool)
        off_diag = cosine_matrix[mask]
        results["mean_cosine"] = off_diag.mean().item()
        results["max_cosine"] = off_diag.max().item()
        results["min_cosine"] = off_diag.min().item()
        results["cosine_matrix"] = cosine_matrix.numpy()
    else:
        results["mean_cosine"] = 0.0
        results["max_cosine"] = 0.0
        results["min_cosine"] = 0.0

    # ── Template PCA (what dimensions do templates span?) ─────────────
    if n_alive >= 2:
        alive_M = M[alive_mask]
        centered = alive_M - alive_M.mean(dim=0, keepdim=True)
        try:
            _, S_vals, _ = torch.linalg.svd(centered, full_matrices=False)
            total_var = (S_vals ** 2).sum().item()
            cum_var = (S_vals ** 2).cumsum(dim=0) / max(total_var, 1e-10)
            results["template_pca_explained_90"] = int((cum_var < 0.9).sum().item()) + 1
            results["template_pca_variance"] = (S_vals ** 2 / max(total_var, 1e-10)).tolist()[:8]
        except Exception:
            results["template_pca_explained_90"] = n_alive

    # ── Scaffold effect on seeding (from instrumented forward states) ─
    scaffold_contributions = []
    template_activation_histograms = []
    for fs in final_states:
        scaffold_contributions.append(fs["scaffold_contribution"])
        sw = fs.get("scaffold_weights")
        if sw is not None:
            # sw: (B, L, K) — average across batch and positions → (K,)
            avg_weights = sw.mean(dim=(0, 1)).numpy()
            template_activation_histograms.append(avg_weights)

    results["seeding_contribution_mean"] = float(np.mean(scaffold_contributions))
    results["seeding_contribution_std"] = float(np.std(scaffold_contributions))

    if template_activation_histograms:
        avg_activation = np.mean(template_activation_histograms, axis=0)
        results["template_activation_profile"] = avg_activation.tolist()
        # Entropy of activation distribution
        avg_activation = avg_activation / (avg_activation.sum() + 1e-10)
        a_pos = avg_activation[avg_activation > 1e-10]
        results["activation_entropy"] = -float(np.sum(a_pos * np.log(a_pos)))

    return results


# ═══════════════════════════════════════════════════════════════════════
# Temporal Dynamics Analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze_temporal_dynamics(records: list[StepRecord], final_state: dict) -> dict:
    """Analyze how morphogen evolves over the 64 CA steps."""
    results = {}

    n = len(records)

    # ── Morphogen evolution rate ──────────────────────────────────────
    delta_h_norms = [r.delta_h_norm for r in records]
    results["delta_h_norm_trajectory"] = delta_h_norms
    results["delta_h_norm_first"] = delta_h_norms[0]
    results["delta_h_norm_last"] = delta_h_norms[-1]
    results["delta_h_norm_peak"] = max(delta_h_norms)
    results["delta_h_total_movement"] = sum(delta_h_norms)

    # ── Gate evolution over steps ─────────────────────────────────────
    gate_means = [r.gate_mean for r in records]
    gate_stds = [r.gate_std for r in records]
    gate_spatial_stds = [r.gate_spatial_std for r in records]
    results["gate_mean_trajectory"] = gate_means
    results["gate_std_trajectory"] = gate_stds
    results["gate_spatial_std_trajectory"] = gate_spatial_stds
    results["gate_diversity_growth"] = gate_spatial_stds[-1] - gate_spatial_stds[0]

    # ── Morphogen snapshot divergence ─────────────────────────────────
    # How much does h change between snapshot timepoints?
    snapshots = final_state["h_snapshots"]
    if len(snapshots) >= 2:
        divergences = []
        for i in range(1, len(snapshots)):
            t_prev, h_prev = snapshots[i - 1]
            t_curr, h_curr = snapshots[i]
            div = (h_curr - h_prev).norm(dim=-1).mean().item()
            divergences.append((t_prev, t_curr, div))
        results["h_snapshot_divergences"] = divergences

    # ── Settlement progression ────────────────────────────────────────
    settled_fracs = [r.settled_frac for r in records]
    results["settlement_trajectory"] = settled_fracs
    results["final_settled_frac"] = settled_fracs[-1]

    return results


# ═══════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════

def format_report(
    records: list[StepRecord],
    weight_analysis: dict,
    spatial_analysis: dict,
    temporal_analysis: dict,
    scaffold_analysis: dict,
    config_info: dict,
) -> str:
    """Generate comprehensive text report of morphogen analysis."""
    lines = []

    def section(title: str):
        lines.append("")
        lines.append("=" * 74)
        lines.append(f"  {title}")
        lines.append("=" * 74)

    def subsection(title: str):
        lines.append("")
        lines.append(f"  ── {title} " + "─" * max(0, 60 - len(title)))

    # ═══════════════════════════════════════════════════════════════════
    section("MORPHOGENETIC FIELD ANALYSIS — AENCA V11")
    # ═══════════════════════════════════════════════════════════════════
    lines.append(f"  Model: d_model={config_info['d_model']}, d_h={config_info['d_h']}, "
                 f"n_steps={config_info['n_steps']}, n_scales={config_info['n_scales']}")
    lines.append(f"  Content rule: ~{config_info.get('n_params_content', '?')} params")
    lines.append(f"  Morphogen system: ~{config_info.get('n_params_morphogen', '?')} params")

    # ── 1. ZERO-INIT BARRIER STATUS ───────────────────────────────────
    section("1. ZERO-INIT BARRIER STATUS (Did growth happen?)")
    lines.append("")
    w_d = weight_analysis["w_d_norm"]
    w_h2 = weight_analysis["w_h2_norm"]
    w_gate_w = weight_analysis["w_gate_weight_norm"]
    lines.append(f"  ||W_d||   = {w_d:>10.4f}  {'✓ LEARNED' if w_d > 0.01 else '✗ STILL ZERO (no content growth)'}")
    lines.append(f"  ||W_h2||  = {w_h2:>10.4f}  {'✓ LEARNED' if w_h2 > 0.01 else '✗ STILL ZERO (no morphogen growth)'}")
    lines.append(f"  ||W_gate||= {w_gate_w:>10.4f}  {'✓ LEARNED' if w_gate_w > 0.01 else '✗ STILL ZERO (gate is constant)'}")
    lines.append(f"  Gate bias = {weight_analysis['w_gate_bias_mean']:>10.4f}  (init=3.0, σ(3)≈0.95)")

    barrier_status = "ALL THREE BARRIERS BROKEN" if all(v > 0.01 for v in [w_d, w_h2, w_gate_w]) else "BARRIERS REMAIN"
    lines.append(f"")
    lines.append(f"  >>> {barrier_status}")
    if w_d > 0.01 and w_h2 <= 0.01:
        lines.append(f"      Content rule learned BUT morphogen still frozen.")
        lines.append(f"      V11 is effectively V10 — morphogen is decoration.")
    if w_d > 0.01 and w_h2 > 0.01 and w_gate_w <= 0.01:
        lines.append(f"      Both fields evolve BUT gate is constant — no coupling!")
        lines.append(f"      Morphogen runs independently without routing effect.")

    # ── 2. W_GATE STRUCTURE (Emergent routing complexity) ─────────────
    section("2. W_GATE STRUCTURE (Morphogen → Content routing)")
    lines.append("")
    lines.append(f"  W_gate: ({config_info['d_h']},) → ({config_info['d_model']},)")
    lines.append(f"  ||W_gate.weight|| = {w_gate_w:.4f}")
    lines.append(f"  Effective rank    = {weight_analysis['w_gate_weight_rank']} / {config_info['d_h']}")
    lines.append(f"  Top-5 singular values: {', '.join(f'{s:.3f}' for s in weight_analysis['w_gate_weight_svd_top5'])}")
    lines.append("")

    rank = weight_analysis["w_gate_weight_rank"]
    d_h = config_info["d_h"]
    if rank <= 3:
        lines.append(f"  >>> LOW RANK ({rank}): Only {rank} morphogen dimensions affect gating.")
        lines.append(f"      Simple routing: the morphogen acts as a {rank}D control signal.")
    elif rank <= d_h // 4:
        lines.append(f"  >>> MODERATE RANK ({rank}/{d_h}): Morphogen uses ~{rank} dims for routing.")
        lines.append(f"      Structured routing: moderate-complexity gating patterns.")
    else:
        lines.append(f"  >>> HIGH RANK ({rank}/{d_h}): Most morphogen dims affect routing.")
        lines.append(f"      Rich routing: complex, high-dimensional gating structure.")

    # ── 3. GATE DYNAMICS OVER CA STEPS ────────────────────────────────
    section("3. GATE DYNAMICS OVER 64 CA STEPS")
    lines.append("")
    lines.append(f"  {'Step':>4} {'Dil':>4} {'Gate μ':>8} {'Gate σ':>8} "
                 f"{'Spat σ':>8} {'Dim σ':>8} {'Δh norm':>8} {'Growth%':>8}")
    lines.append(f"  {'':─>4} {'':─>4} {'':─>8} {'':─>8} {'':─>8} {'':─>8} {'':─>8} {'':─>8}")

    # Show subset of steps
    step_indices = list(range(0, 8)) + list(range(8, len(records), max(1, len(records) // 12)))
    step_indices = sorted(set(step_indices))
    for i in step_indices:
        if i >= len(records):
            continue
        r = records[i]
        lines.append(f"  {r.step:>4} {r.dilation:>4} {r.gate_mean:>8.4f} {r.gate_std:>8.4f} "
                     f"{r.gate_spatial_std:>8.4f} {r.gate_dim_std:>8.4f} {r.delta_h_norm:>8.4f} "
                     f"{r.growth_frac * 100:>7.1f}%")

    # ── Gate diversity assessment ─────────────────────────────────────
    subsection("Gate Diversity Assessment")
    first = records[0]
    last = records[-1]

    lines.append(f"  Gate spatial σ: {first.gate_spatial_std:.5f} → {last.gate_spatial_std:.5f} "
                 f"(Δ = {last.gate_spatial_std - first.gate_spatial_std:+.5f})")
    lines.append(f"  Gate dim σ:     {first.gate_dim_std:.5f} → {last.gate_dim_std:.5f} "
                 f"(Δ = {last.gate_dim_std - first.gate_dim_std:+.5f})")
    lines.append(f"  Gate range:     [{last.gate_min:.4f}, {last.gate_max:.4f}] "
                 f"(span = {last.gate_max - last.gate_min:.4f})")
    lines.append("")

    diversity_growth = temporal_analysis.get("gate_diversity_growth", 0)
    if diversity_growth > 0.01:
        lines.append(f"  >>> GATE DIVERSITY IS GROWING (+{diversity_growth:.4f})")
        lines.append(f"      Morphogen is creating DIFFERENTIATED routing over CA steps.")
        lines.append(f"      This is emergent spatial specialization — the signal we want!")
    elif diversity_growth > 0.001:
        lines.append(f"  >>> GATE DIVERSITY WEAKLY GROWING (+{diversity_growth:.5f})")
        lines.append(f"      Morphogen shows early signs of differentiation.")
    else:
        lines.append(f"  >>> GATE DIVERSITY STATIC OR DECLINING ({diversity_growth:+.5f})")
        lines.append(f"      Morphogen is not creating spatial routing patterns.")
        if w_gate_w < 0.01:
            lines.append(f"      (W_gate.weight ≈ 0: gate is constant σ(bias) everywhere)")

    # ── 4. MORPHOGEN FIELD EVOLUTION ──────────────────────────────────
    section("4. MORPHOGEN FIELD EVOLUTION")
    lines.append("")
    delta_h_first = temporal_analysis["delta_h_norm_first"]
    delta_h_last = temporal_analysis["delta_h_norm_last"]
    delta_h_peak = temporal_analysis["delta_h_norm_peak"]
    delta_h_total = temporal_analysis["delta_h_total_movement"]

    lines.append(f"  ||Δh|| first step:  {delta_h_first:.6f}")
    lines.append(f"  ||Δh|| last step:   {delta_h_last:.6f}")
    lines.append(f"  ||Δh|| peak:        {delta_h_peak:.6f}")
    lines.append(f"  Total h movement:   {delta_h_total:.4f}")
    lines.append("")

    if delta_h_total < 0.001:
        lines.append(f"  >>> MORPHOGEN FIELD ESSENTIALLY STATIC")
        lines.append(f"      h ≈ tanh(W_h_init · z) throughout. No dynamics.")
        lines.append(f"      W_h2 ≈ 0: morphogen rule not producing updates.")
    elif delta_h_peak > delta_h_first * 5:
        lines.append(f"  >>> MORPHOGEN SHOWS AMPLIFICATION (peak/first = {delta_h_peak/max(delta_h_first,1e-10):.1f}x)")
        lines.append(f"      Possible reaction-diffusion dynamics — morphogen amplifies over steps.")
    else:
        lines.append(f"  >>> MORPHOGEN EVOLVES MODERATELY")
        lines.append(f"      Consistent activity. No extreme amplification.")

    # ── Snapshot divergences ──────────────────────────────────────────
    divergences = temporal_analysis.get("h_snapshot_divergences", [])
    if divergences:
        subsection("Morphogen Snapshot Divergences")
        for t_prev, t_curr, div in divergences:
            bar = "█" * min(60, int(div * 100))
            lines.append(f"  Step {t_prev:>3}→{t_curr:>3}: {div:>8.4f} |{bar}")

    # ── 5. SPATIAL DIFFERENTIATION ────────────────────────────────────
    section("5. SPATIAL DIFFERENTIATION")
    lines.append("")

    gate_pos_range = spatial_analysis["gate_position_range"]
    h_pos_range = spatial_analysis["h_position_norm_range"]
    h_off_diag = spatial_analysis["h_dim_off_diag_corr"]

    lines.append(f"  Gate position range:  {gate_pos_range:.6f}  (max - min avg gate across positions)")
    lines.append(f"  h norm range:         {h_pos_range:.4f}  (max - min avg ||h|| across positions)")
    lines.append(f"  h dim off-diag corr:  {h_off_diag:.4f}  (lower = more independent dim specialization)")
    lines.append("")

    if gate_pos_range > 0.01:
        lines.append(f"  >>> SPATIAL GATE DIFFERENTIATION DETECTED (range={gate_pos_range:.4f})")
        lines.append(f"      Different positions have different gate strengths.")
        lines.append(f"      The morphogen is creating positional specialization!")
    elif gate_pos_range > 0.001:
        lines.append(f"  >>> WEAK SPATIAL DIFFERENTIATION (range={gate_pos_range:.5f})")
    else:
        lines.append(f"  >>> NO SIGNIFICANT SPATIAL DIFFERENTIATION (range={gate_pos_range:.6f})")

    lines.append("")
    if h_off_diag < 0.3:
        lines.append(f"  >>> MORPHOGEN DIMS WELL-SPECIALIZED (corr={h_off_diag:.3f})")
        lines.append(f"      Different h dimensions encode INDEPENDENT spatial patterns.")
    elif h_off_diag < 0.6:
        lines.append(f"  >>> MODERATE DIM SPECIALIZATION (corr={h_off_diag:.3f})")
    else:
        lines.append(f"  >>> LOW DIM SPECIALIZATION (corr={h_off_diag:.3f})")
        lines.append(f"      Morphogen dimensions are redundant — encoding similar info.")

    # ── Gate position profile (first 32 and last 32 positions) ────────
    gate_profile = spatial_analysis["gate_position_profile"]
    L = len(gate_profile)
    if L > 0:
        subsection(f"Gate Position Profile (L={L})")
        # Show a compact ASCII heatmap
        n_show = min(L, 64)
        profile_subset = gate_profile[:n_show]
        vmin, vmax = profile_subset.min(), profile_subset.max()

        # Normalize to 0-8 for block chars
        blocks = " ▁▂▃▄▅▆▇█"
        if vmax - vmin > 1e-8:
            normalized = (profile_subset - vmin) / (vmax - vmin) * 8
        else:
            normalized = np.full_like(profile_subset, 4.0)

        bar_line = "".join(blocks[min(8, int(v))] for v in normalized)
        lines.append(f"  Pos 0→{n_show-1}: |{bar_line}|")
        lines.append(f"  Range: [{vmin:.5f}, {vmax:.5f}]")

    # ── 6. SETTLEMENT DYNAMICS ────────────────────────────────────────
    section("6. SETTLEMENT DYNAMICS")
    lines.append("")
    final_settled = temporal_analysis["final_settled_frac"]
    settled_traj = temporal_analysis["settlement_trajectory"]
    mid_settled = settled_traj[len(settled_traj) // 2] if settled_traj else 0

    lines.append(f"  Settled at step 0:  {settled_traj[0]*100:.1f}%")
    lines.append(f"  Settled at midpoint: {mid_settled*100:.1f}%")
    lines.append(f"  Settled at final:   {final_settled*100:.1f}%")
    lines.append("")

    if final_settled > 0.5:
        lines.append(f"  >>> STRONG SETTLEMENT ({final_settled*100:.0f}%)")
        lines.append(f"      More than half the cells have frozen into permanent memory.")
    elif final_settled > 0.1:
        lines.append(f"  >>> MODERATE SETTLEMENT ({final_settled*100:.0f}%)")
    else:
        lines.append(f"  >>> MINIMAL SETTLEMENT ({final_settled*100:.1f}%)")
        lines.append(f"      Very few cells reaching permanence in 64 steps.")

    # ── 7. SCAFFOLD (PERSISTENT MORPHOGEN TEMPLATES) ─────────────────
    section("7. PERSISTENT SCAFFOLD (Epigenetic Memory)")
    lines.append("")

    if scaffold_analysis.get("active", False):
        K = scaffold_analysis["K"]
        n_alive = scaffold_analysis["n_alive"]
        step = scaffold_analysis["step"]
        alpha = scaffold_analysis["alpha"]

        lines.append(f"  Templates: {n_alive}/{K} alive (trained for {step} steps, α={alpha})")
        lines.append(f"  EMA decay: {scaffold_analysis['ema_decay']}")
        lines.append(f"")

        # Template norms
        subsection(f"Template Norms (K={K})")
        norms = scaffold_analysis["template_norms"]
        blocks = " ▁▂▃▄▅▆▇█"
        vmax = max(norms) if norms else 1.0
        for i, n_val in enumerate(norms):
            bar_idx = min(8, int(n_val / max(vmax, 1e-8) * 8))
            alive_tag = "✓" if n_val >= config_info.get("scaffold_min_norm", 0.01) else "✗"
            lines.append(f"  T{i:>2d}: {blocks[bar_idx]} {n_val:.4f}  {alive_tag}")
        lines.append(f"")
        lines.append(f"  Norm range: [{scaffold_analysis['template_norm_min']:.4f}, "
                     f"{scaffold_analysis['template_norm_max']:.4f}]")
        lines.append(f"  Norm mean:  {scaffold_analysis['template_norm_mean']:.4f} "
                     f"± {scaffold_analysis['template_norm_std']:.4f}")

        # Usage distribution
        subsection("Usage Distribution")
        usage = scaffold_analysis["usage"]
        u_total = scaffold_analysis["usage_total"]
        for i, u in enumerate(usage):
            pct = u / max(u_total, 1e-8) * 100
            bar = "█" * min(40, int(pct * 0.8))
            lines.append(f"  T{i:>2d}: {u:>8.0f} ({pct:>5.1f}%) |{bar}")
        lines.append(f"")
        lines.append(f"  Usage entropy: {scaffold_analysis['usage_entropy']:.3f} / "
                     f"{scaffold_analysis['usage_max_entropy']:.3f} "
                     f"(uniformity={scaffold_analysis['usage_uniformity']:.2f})")

        if scaffold_analysis["usage_uniformity"] > 0.7:
            lines.append(f"  >>> WELL-DISTRIBUTED: Most templates participate equally.")
        elif scaffold_analysis["usage_uniformity"] > 0.3:
            lines.append(f"  >>> MODERATE SKEW: Some templates dominate.")
        else:
            lines.append(f"  >>> HIGHLY SKEWED: Few templates get most usage — possible collapse.")

        # Template diversity
        subsection("Template Diversity (Inter-Template Cosine Similarity)")
        lines.append(f"  Mean cosine: {scaffold_analysis['mean_cosine']:.4f}")
        lines.append(f"  Range: [{scaffold_analysis['min_cosine']:.4f}, "
                     f"{scaffold_analysis['max_cosine']:.4f}]")
        lines.append(f"")

        if scaffold_analysis["max_cosine"] > 0.95:
            lines.append(f"  >>> REDUNDANT TEMPLATES: Max cosine {scaffold_analysis['max_cosine']:.3f} > 0.95")
            lines.append(f"      Some templates are near-identical. compression should merge them.")
        elif scaffold_analysis["mean_cosine"] < 0.3:
            lines.append(f"  >>> DIVERSE TEMPLATES: Mean cosine {scaffold_analysis['mean_cosine']:.3f}")
            lines.append(f"      Templates encode distinct morphogen archetypes. Good!")
        else:
            lines.append(f"  >>> MODERATE DIVERSITY: Templates partially overlapping.")

        # PCA structure
        if "template_pca_explained_90" in scaffold_analysis:
            subsection("Template PCA Structure")
            dims_90 = scaffold_analysis["template_pca_explained_90"]
            lines.append(f"  Dims for 90% variance: {dims_90} / {scaffold_analysis.get('K', '?')}")
            pca_var = scaffold_analysis.get("template_pca_variance", [])
            if pca_var:
                lines.append(f"  Top-8 PCA explained var: {', '.join(f'{v:.3f}' for v in pca_var)}")

        # Scaffold seeding effect
        subsection("Scaffold Seeding Effect (h_scaffold contribution to h)")
        sc_mean = scaffold_analysis["seeding_contribution_mean"]
        sc_std = scaffold_analysis["seeding_contribution_std"]
        lines.append(f"  ||h - h_fresh|| mean: {sc_mean:.6f} ± {sc_std:.6f}")
        lines.append(f"")

        if sc_mean > 0.01:
            lines.append(f"  >>> SCAFFOLD ACTIVELY MODIFIES SEEDING")
            lines.append(f"      Persistent templates measurably change the morphogen seed.")
            lines.append(f"      This is the epigenetic memory at work — the organism")
            lines.append(f"      is born with learned predispositions, not a blank slate.")
        elif sc_mean > 0.001:
            lines.append(f"  >>> WEAK SCAFFOLD EFFECT on seeding ({sc_mean:.5f})")
        else:
            lines.append(f"  >>> SCAFFOLD HAS NEGLIGIBLE EFFECT on seeding ({sc_mean:.6f})")
            if step < scaffold_analysis["warmup_steps"]:
                lines.append(f"      (Still in warmup: step {step} < {scaffold_analysis['warmup_steps']})")

        # Template activation profile
        if "template_activation_profile" in scaffold_analysis:
            subsection("Template Activation Profile (which templates fire?)")
            profile = scaffold_analysis["template_activation_profile"]
            act_entropy = scaffold_analysis.get("activation_entropy", 0)
            for i, a in enumerate(profile):
                bar = "█" * min(40, int(a * K * 40))
                lines.append(f"  T{i:>2d}: {a:.4f} |{bar}")
            lines.append(f"")
            lines.append(f"  Activation entropy: {act_entropy:.3f} / {math.log(max(1, K)):.3f}")
            if act_entropy > math.log(max(1, K)) * 0.7:
                lines.append(f"  >>> DIVERSE ACTIVATION: Tokens select templates broadly.")
            else:
                lines.append(f"  >>> CONCENTRATED ACTIVATION: Few templates dominate selection.")

    else:
        lines.append(f"  Scaffold is INACTIVE (K=0, or templates all zero, or not trained).")
        if hasattr(model, 'scaffold_templates'):
            step = model.scaffold_step.item() if hasattr(model, 'scaffold_step') else 0
            lines.append(f"  scaffold_step={step}, K={model.config.n_scaffold_templates}")
            if step == 0:
                lines.append(f"  (Model has never run on_step_end — fresh checkpoint?)")

    # ── 8. OVERALL ASSESSMENT ─────────────────────────────────────────
    section("8. MORPHOGENETIC EMERGENCE ASSESSMENT")
    lines.append("")

    # Score the morphogen's contribution
    scores = {}
    scores["zero_init_broken"] = 1.0 if all(v > 0.01 for v in [w_d, w_h2, w_gate_w]) else 0.0
    scores["gate_diversity"] = min(1.0, (last.gate_spatial_std - first.gate_spatial_std) * 50)
    scores["h_dynamics"] = min(1.0, delta_h_total * 10)
    scores["spatial_diff"] = min(1.0, gate_pos_range * 100)
    scores["dim_independence"] = 1.0 - min(1.0, h_off_diag)
    scores["gate_rank_util"] = min(1.0, rank / max(d_h, 1))

    # Scaffold scores
    if scaffold_analysis.get("active", False):
        sc_mean = scaffold_analysis["seeding_contribution_mean"]
        scores["scaffold_seeding"] = min(1.0, sc_mean * 100)
        scores["scaffold_diversity"] = 1.0 - min(1.0, scaffold_analysis.get("mean_cosine", 1.0))
        scores["scaffold_utilization"] = scaffold_analysis.get("usage_uniformity", 0.0)

    total = sum(scores.values()) / len(scores)

    for name, score in scores.items():
        bar = "█" * int(score * 20)
        pad = "░" * (20 - int(score * 20))
        lines.append(f"  {name:<20s} {score:.2f}  |{bar}{pad}|")

    lines.append("")
    lines.append(f"  >>> OVERALL MORPHOGEN SCORE: {total:.2f} / 1.00")
    lines.append("")

    if total > 0.7:
        lines.append(f"  STRONG MORPHOGENETIC EMERGENCE")
        lines.append(f"  The morphogen field creates structured, diverse routing patterns.")
        lines.append(f"  V11's two-field CA shows clear emergent complexity beyond V10.")
    elif total > 0.4:
        lines.append(f"  MODERATE MORPHOGENETIC ACTIVITY")
        lines.append(f"  The morphogen shows early signs of useful structure but hasn't")
        lines.append(f"  fully differentiated yet. Longer training or larger scale may help.")
    elif total > 0.1:
        lines.append(f"  WEAK MORPHOGENETIC ACTIVITY")
        lines.append(f"  The morphogen field exists but adds minimal routing diversity.")
        lines.append(f"  Consider: is the gate bias too high? is d_h too small?")
    else:
        lines.append(f"  NO MEANINGFUL MORPHOGENETIC EMERGENCE")
        lines.append(f"  The morphogen field is essentially dormant. V11 ≈ V10.")
        lines.append(f"  Zero-init barriers may not have been broken.")

    lines.append("")
    lines.append("=" * 74)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyze morphogenetic field dynamics in V11 checkpoints"
    )
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of sample batches to analyze")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load checkpoint ───────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config_dict = ckpt["config"]

    # ── Build model ───────────────────────────────────────────────────
    model_name = config_dict.get("model_name", "v11")
    ModelClass = get_model_class(model_name)
    inf_config = dict(config_dict)
    inf_config["dropout"] = 0.0
    model = ModelClass(inf_config)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()

    print(f"Model loaded: {model.describe().get('architecture', model_name)}")
    print(f"  d_model={model.config.d_model}, d_h={model.config.d_h}, n_steps={model.config.n_steps}")
    print(f"  Training step: {ckpt.get('step', '?')}, Val PPL: {ckpt.get('best_val_ppl', '?')}")

    # ── Load sample data ──────────────────────────────────────────────
    print(f"\nLoading data ({config_dict.get('dataset', 'unknown')})...")
    B = config_dict.get("batch_size", 2)
    L = config_dict.get("seq_len", 128)

    tokenizer_name = config_dict.get("tokenizer_name", "bpe")
    tokenizer = create_tokenizer(
        tokenizer_name,
        vocab_size=config_dict.get("tokenizer_vocab_size", config_dict.get("vocab_size", 10000)),
        cache_dir=config_dict.get("tokenizer_cache_dir"),
        cache_stem=config_dict.get("tokenizer_cache_stem"),
        dataset_name=config_dict.get("tokenizer_dataset_name", config_dict.get("dataset_hf_name")),
        dataset_config=config_dict.get("tokenizer_dataset_config", config_dict.get("dataset_config")),
        dataset_splits=config_dict.get("tokenizer_dataset_splits"),
        train_max_examples=config_dict.get("tokenizer_train_max_examples"),
        pretrained_name=config_dict.get("tokenizer_pretrained_name"),
    )

    dataset_name = config_dict.get("dataset", "tinystories")
    if dataset_name == "smoltalk2":
        data = SmolTalk2Data(
            seq_len=L, tokenizer=tokenizer,
            cache_name=config_dict.get("dataset_cache_name", "smoltalk2_sft_tokens_5b.pt"),
            dataset_config=config_dict.get("dataset_config", "SFT"),
            dataset_splits=config_dict.get("dataset_splits"),
            dataset_sources=config_dict.get("dataset_sources"),
            max_tokens=config_dict.get("max_dataset_tokens", 5_000_000_000),
            cache_dir=config_dict.get("dataset_cache_dir"),
        )
    elif dataset_name == "the_pile":
        data = ThePileData(
            seq_len=L, tokenizer=tokenizer,
            cache_name=config_dict.get("dataset_cache_name", "pile_tokens_2b.bin"),
            dataset_name=config_dict.get("dataset_hf_name"),
            max_tokens=config_dict.get("max_dataset_tokens", 2_000_000_000),
            cache_dir=config_dict.get("dataset_cache_dir"),
        )
    else:
        data = TinyStoriesData(
            seq_len=L, tokenizer=tokenizer,
            cache_name=config_dict.get("dataset_cache_name", "tinystories_tokens_ncalm.pt"),
        )
    val_loader = create_data_loader(data, B, device, "val")

    # ── Run instrumented forward passes ───────────────────────────────
    print(f"Running {args.n_samples} instrumented forward passes (B={B}, L={L})...")
    all_records = []
    final_states = []

    for i in range(args.n_samples):
        batch = next(val_loader)
        records, final_state = instrumented_ca_forward(model, batch)
        all_records.append(records)
        final_states.append(final_state)
        print(f"  Sample {i+1}/{args.n_samples}: gate_mean={records[-1].gate_mean:.4f}, "
              f"h_norm={records[-1].h_norm:.4f}, settled={records[-1].settled_frac:.1%}")

    # ── Aggregate records across samples (average stats) ──────────────
    n_steps = len(all_records[0])
    avg_records = []
    for t in range(n_steps):
        step_records = [all_records[s][t] for s in range(len(all_records))]
        n = len(step_records)
        avg_records.append(StepRecord(
            step=step_records[0].step,
            dilation=step_records[0].dilation,
            z_mean=sum(r.z_mean for r in step_records) / n,
            z_std=sum(r.z_std for r in step_records) / n,
            z_norm=sum(r.z_norm for r in step_records) / n,
            delta_z_norm=sum(r.delta_z_norm for r in step_records) / n,
            h_mean=sum(r.h_mean for r in step_records) / n,
            h_std=sum(r.h_std for r in step_records) / n,
            h_norm=sum(r.h_norm for r in step_records) / n,
            delta_h_norm=sum(r.delta_h_norm for r in step_records) / n,
            gate_mean=sum(r.gate_mean for r in step_records) / n,
            gate_std=sum(r.gate_std for r in step_records) / n,
            gate_min=min(r.gate_min for r in step_records),
            gate_max=max(r.gate_max for r in step_records),
            gate_spatial_std=sum(r.gate_spatial_std for r in step_records) / n,
            gate_dim_std=sum(r.gate_dim_std for r in step_records) / n,
            growth_frac=sum(r.growth_frac for r in step_records) / n,
            settled_frac=sum(r.settled_frac for r in step_records) / n,
        ))

    # Use last sample's final state for spatial analysis (representative)
    last_final = final_states[-1]

    # ── Weight analysis ───────────────────────────────────────────────
    print("\nAnalyzing weight matrices...")
    weight_analysis = analyze_weights(model)

    # ── Spatial analysis ──────────────────────────────────────────────
    print("Analyzing spatial patterns...")
    spatial_analysis = analyze_spatial_patterns(last_final)

    # ── Temporal analysis ─────────────────────────────────────────────
    print("Analyzing temporal dynamics...")
    temporal_analysis = analyze_temporal_dynamics(avg_records, last_final)

    # ── Scaffold analysis ─────────────────────────────────────────────
    print("Analyzing scaffold (persistent morphogen templates)...")
    scaffold_analysis_result = analyze_scaffold(model, final_states)

    # ── Config info ───────────────────────────────────────────────────
    cfg = model.config
    content_params = sum(p.numel() for p in model.rule.parameters())
    morph_params = (sum(p.numel() for p in model.morph_rule.parameters())
                    + model.W_h_init.weight.numel()
                    + model.W_gate.weight.numel()
                    + model.W_gate.bias.numel())

    config_info = {
        "d_model": cfg.d_model,
        "d_h": cfg.d_h,
        "n_steps": cfg.n_steps,
        "n_scales": cfg.n_scales,
        "n_params_content": f"{content_params:,}",
        "n_params_morphogen": f"{morph_params:,}",
        "scaffold_min_norm": cfg.scaffold_min_norm if hasattr(cfg, 'scaffold_min_norm') else 0.01,
    }

    # ── Generate report ───────────────────────────────────────────────
    report = format_report(
        avg_records,
        weight_analysis,
        spatial_analysis,
        temporal_analysis,
        scaffold_analysis_result,
        config_info,
    )
    print(report)

    # ── Save report ───────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint).parent
    report_path = ckpt_dir / "morphogen_analysis.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
