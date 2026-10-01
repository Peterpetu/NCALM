#!/usr/bin/env python3
"""Morphogenetic structure visualizer for AENCA V11.

Generates an interactive HTML report showing the ACTUAL STRUCTURES
that the morphogenetic field builds during a forward pass — not just 
summary statistics, but the spatial patterns, token-level morphogen 
signatures, gate heatmaps, and temporal field evolution.

Key insight: Morphogenesis happens at BOTH training and inference.
Every forward pass re-grows h and z from the token embedding (the seed).
Training learns the GENOME (shared rules). Inference RE-RUNS morphogenesis.
The structures are EPHEMERAL — grown anew for each input sequence.

This tool lets you SEE those ephemeral structures for any input text:
  - Gate heatmap: (positions × CA steps) showing how routing evolves
  - Morphogen field: PCA-projected h at each step → spatial patterns
  - Per-token morphogen signatures: what "cell type" each token develops
  - Content update magnitude: where is the CA actively building?
  - Settlement progression: which cells freeze first?
  - Gated vs ungated perception: what does morphogen filtering look like?

Usage:
    python scripts/visualize_morphogen.py <checkpoint.pt> --text "Once upon a time"
    python scripts/visualize_morphogen.py <checkpoint.pt> --device cuda --text "The cat sat on the mat"
    python scripts/visualize_morphogen.py <checkpoint.pt>  # uses validation data
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ncalm.model import get_model_class
from ncalm.data.tokenizer import create_tokenizer


# ═══════════════════════════════════════════════════════════════════════
# Instrumented CA Forward Pass — full per-step recording
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def instrumented_forward(model, input_ids: torch.Tensor):
    """Run V11 CA loop capturing FULL per-step field states.

    Returns a dict with all the raw tensors needed for visualization.
    All tensors returned on CPU as numpy arrays.
    """
    cfg = model.config
    B, L = input_ids.shape
    device = input_ids.device

    positions = torch.arange(L, device=device)
    z = model.token_embedding(input_ids) + model.pos_embedding(positions)

    h = model._seed_morphogen(z)  # (B, L, d_h) — includes scaffold contribution
    h_fresh = torch.tanh(model.W_h_init(z))  # for comparison

    from ncalm.models.v11.settlement import SettlementState
    settlement = SettlementState.create(B, L, device, cfg)

    # ── Storage for per-step recordings ───────────────────────────────
    n_steps = cfg.n_steps
    # Full field snapshots (keep manageable by storing PCA projections and norms)
    gate_heatmap = np.zeros((n_steps, L), dtype=np.float32)      # mean gate per pos
    gate_full = []        # (n_steps,) list of (L, d_model_subset) for detailed view
    h_norms = np.zeros((n_steps, L), dtype=np.float32)           # ||h|| per pos
    h_pca = np.zeros((n_steps, L, 3), dtype=np.float32)          # PCA(h) → RGB
    delta_z_norms = np.zeros((n_steps, L), dtype=np.float32)     # ||Δz|| per pos  
    delta_h_norms = np.zeros((n_steps, L), dtype=np.float32)     # ||Δh|| per pos
    settlement_map = np.zeros((n_steps, L), dtype=np.float32)    # settled score
    growth_map = np.zeros((n_steps, L), dtype=np.float32)        # growth gate
    gate_variance_per_pos = np.zeros((n_steps, L), dtype=np.float32)  # gate dim variance
    
    # For ungated comparison
    ungated_delta_z_norms = np.zeros((n_steps, L), dtype=np.float32)

    # Morphogen trajectories for select positions
    h_trajectories = np.zeros((n_steps, min(L, 16), cfg.d_h), dtype=np.float32)

    # Compute PCA basis from initial h (for consistent projection)
    h_flat = h[0].cpu().float()  # (L, d_h), first batch item
    h_centered = h_flat - h_flat.mean(dim=0, keepdim=True)
    try:
        U, S, Vh = torch.linalg.svd(h_centered, full_matrices=False)
        pca_basis = Vh[:3]  # (3, d_h)
    except Exception:
        pca_basis = torch.eye(3, cfg.d_h)

    for t, dilation in enumerate(model._dilation_schedule):
        gate = model.compute_gate(h)  # (B, L, d_model)
        delta_z = model.rule(z, dilation=dilation, gate=gate)
        delta_h = model.morph_rule(h, dilation=dilation)
        
        # Also compute ungated delta_z for comparison
        delta_z_ungated = model.rule(z, dilation=dilation, gate=None)

        growth = settlement.growth_gate().unsqueeze(-1)  # (B, L, 1)

        # ── Record (batch item 0) ─────────────────────────────────────
        b = 0
        gate_b = gate[b].cpu().float()          # (L, d_model)
        h_b = h[b].cpu().float()                # (L, d_h)
        dz_b = delta_z[b].cpu().float()         # (L, d_model)
        dh_b = delta_h[b].cpu().float()         # (L, d_h)
        dz_ug = delta_z_ungated[b].cpu().float()

        gate_heatmap[t] = gate_b.mean(dim=-1).numpy()
        gate_variance_per_pos[t] = gate_b.var(dim=-1).numpy()
        h_norms[t] = h_b.norm(dim=-1).numpy()
        delta_z_norms[t] = dz_b.norm(dim=-1).numpy()
        delta_h_norms[t] = dh_b.norm(dim=-1).numpy()
        ungated_delta_z_norms[t] = dz_ug.norm(dim=-1).numpy()
        settlement_map[t] = settlement.settled[b].cpu().numpy()
        growth_map[t] = settlement.growth_gate()[b].cpu().numpy()

        # PCA projection of h → 3D (for RGB visualization)
        h_proj = (h_b - h_b.mean(dim=0, keepdim=True)) @ pca_basis.T
        h_pca[t] = h_proj.numpy()

        # Store full trajectories for subset of positions
        n_track = min(L, 16)
        h_trajectories[t] = h_b[:n_track].numpy()

        # Store subset of gate values for detailed view (every 8th dim)
        d_sub = min(gate_b.shape[-1], 64)
        step_size = max(1, gate_b.shape[-1] // d_sub)
        gate_full.append(gate_b[:, ::step_size].numpy())

        # ── Apply updates ─────────────────────────────────────────────
        z = z + delta_z * growth
        h = h + delta_h * growth
        settlement.update(delta_z, cfg)

    # ── Final logits for PPL ──────────────────────────────────────────
    logits = model._get_logits(z)  # (B, L, V)

    # ── Morphogen-to-gate sensitivity: bilinear gate structure ───────
    # W_gate_a and W_gate_b: (d_model, d_h) each — combined importance
    W_a_np = model.W_gate_a.weight.detach().cpu().float().numpy()  # (d_model, d_h)
    W_b_np = model.W_gate_b.weight.detach().cpu().float().numpy()  # (d_model, d_h)
    W_gate_np = (W_a_np + W_b_np) / 2.0  # Average for visualization
    W_gate_bias = model.gate_bias.detach().cpu().float().numpy()  # (d_model,)

    # Per-dim contribution: combined ||W_a[:, j]|| + ||W_b[:, j]|| influence
    gate_dim_importance = (
        np.linalg.norm(W_a_np, axis=0) + np.linalg.norm(W_b_np, axis=0)
    ) / 2.0  # (d_h,)

    # ── Scaffold data ─────────────────────────────────────────────────
    scaffold_info = {}
    if hasattr(model, 'scaffold_templates') and cfg.n_scaffold_templates > 0:
        M = model.scaffold_templates.detach().cpu().float()
        norms = M.norm(dim=-1)
        scaffold_info["template_norms"] = norms.numpy().tolist()
        scaffold_info["n_alive"] = int((norms >= cfg.scaffold_min_norm).sum().item())
        scaffold_info["K"] = cfg.n_scaffold_templates
        scaffold_info["alpha"] = cfg.scaffold_alpha
        scaffold_info["step"] = int(model.scaffold_step.item())

        # h_fresh vs h_scaffold comparison
        h_fresh_b = h_fresh[0].cpu().float()  # (L, d_h)
        h_b = model._seed_morphogen(
            model.token_embedding(input_ids) + model.pos_embedding(positions)
        )[0].cpu().float()
        scaffold_info["seeding_diff_per_pos"] = (h_b - h_fresh_b).norm(dim=-1).numpy().tolist()

        # Template selection weights (cosine similarity + sqrt(d_h) scaling)
        if norms.max().item() > 1e-8:
            h_normed = F.normalize(h_fresh_b, dim=-1)
            M_normed = F.normalize(M.cpu().float(), dim=-1)
            tau_scaled = cfg.scaffold_temperature * math.sqrt(cfg.d_h)
            scores = torch.matmul(h_normed, M_normed.t()) / tau_scaled
            weights = torch.softmax(scores, dim=-1)  # (L, K)
            scaffold_info["template_weights"] = weights.numpy().tolist()

    return {
        "gate_heatmap": gate_heatmap,           # (n_steps, L)
        "gate_variance": gate_variance_per_pos,  # (n_steps, L)
        "gate_full": gate_full,                  # list of (L, d_sub) arrays
        "h_norms": h_norms,                      # (n_steps, L)
        "h_pca": h_pca,                          # (n_steps, L, 3)
        "delta_z_norms": delta_z_norms,          # (n_steps, L)
        "delta_h_norms": delta_h_norms,          # (n_steps, L)
        "ungated_delta_z_norms": ungated_delta_z_norms,  # (n_steps, L)
        "settlement_map": settlement_map,        # (n_steps, L)
        "growth_map": growth_map,                # (n_steps, L)
        "h_trajectories": h_trajectories,        # (n_steps, n_track, d_h)
        "gate_dim_importance": gate_dim_importance,  # (d_h,)
        "W_gate_weight": W_gate_np,              # (d_model, d_h)
        "W_gate_bias": W_gate_bias,              # (d_model,)
        "logits": logits[0].cpu().float().numpy(),  # (L, V)
        "n_steps": n_steps,
        "d_h": cfg.d_h,
        "d_model": cfg.d_model,
        "scaffold": scaffold_info,               # scaffold template data
    }


# ═══════════════════════════════════════════════════════════════════════
# HTML Report Generator
# ═══════════════════════════════════════════════════════════════════════

def _normalize_for_heatmap(arr: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] for colormap rendering."""
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-10:
        return np.full_like(arr, 0.5)
    return (arr - vmin) / (vmax - vmin)


def _to_color(val: float, cmap: str = "viridis") -> str:
    """Map [0,1] value to hex color using simple colormaps."""
    val = max(0.0, min(1.0, val))
    if cmap == "viridis":
        # Simplified viridis: dark purple → teal → yellow
        if val < 0.5:
            t = val * 2
            r = int(68 + t * (32 - 68))
            g = int(1 + t * (145 - 1))
            b = int(84 + t * (140 - 84))
        else:
            t = (val - 0.5) * 2
            r = int(32 + t * (253 - 32))
            g = int(145 + t * (231 - 145))
            b = int(140 + t * (37 - 140))
        return f"#{r:02x}{g:02x}{b:02x}"
    elif cmap == "rdbu":
        # Red-Blue diverging
        if val < 0.5:
            t = val * 2
            r = int(178 + t * (247 - 178))
            g = int(24 + t * (247 - 24))
            b = int(43 + t * (247 - 43))
        else:
            t = (val - 0.5) * 2
            r = int(247 - t * (247 - 33))
            g = int(247 - t * (247 - 102))
            b = int(247 - t * (247 - 172))
        return f"#{r:02x}{g:02x}{b:02x}"
    elif cmap == "hot":
        if val < 0.33:
            t = val * 3
            return f"#{int(t*255):02x}0000"
        elif val < 0.66:
            t = (val - 0.33) * 3
            return f"#ff{int(t*255):02x}00"
        else:
            t = (val - 0.66) * 3
            return f"#ffff{int(t*255):02x}"
    # Default grayscale
    v = int(val * 255)
    return f"#{v:02x}{v:02x}{v:02x}"


def _rgb_from_pca(pca_vals: np.ndarray) -> str:
    """Convert 3D PCA values to hex RGB color."""
    # Normalize each channel to [0, 255]
    mins = pca_vals.min()
    maxs = pca_vals.max()
    if maxs - mins < 1e-10:
        return "#808080"
    normalized = (pca_vals - mins) / (maxs - mins)
    r, g, b = (int(v * 255) for v in normalized[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def generate_html_report(
    data: dict,
    tokens: list[str],
    input_text: str,
    config_info: dict,
) -> str:
    """Generate interactive HTML report of morphogenetic structures."""
    
    n_steps = data["n_steps"]
    L = len(tokens)
    d_h = data["d_h"]
    
    # Prepare data for JS (convert numpy to nested lists)
    gate_heatmap_json = json.dumps(data["gate_heatmap"][:, :L].tolist())
    gate_var_json = json.dumps(data["gate_variance"][:, :L].tolist())
    h_norms_json = json.dumps(data["h_norms"][:, :L].tolist())
    delta_z_json = json.dumps(data["delta_z_norms"][:, :L].tolist())
    delta_h_json = json.dumps(data["delta_h_norms"][:, :L].tolist())
    ungated_dz_json = json.dumps(data["ungated_delta_z_norms"][:, :L].tolist())
    settlement_json = json.dumps(data["settlement_map"][:, :L].tolist())
    growth_json = json.dumps(data["growth_map"][:, :L].tolist())
    h_pca_json = json.dumps(data["h_pca"][:, :L, :].tolist())
    gate_dim_imp_json = json.dumps(data["gate_dim_importance"].tolist())
    tokens_json = json.dumps([html.escape(t) for t in tokens])
    
    # Morphogen trajectories for tracked positions
    n_track = data["h_trajectories"].shape[1]
    h_traj_json = json.dumps(data["h_trajectories"][:, :min(L, n_track), :].tolist())
    
    # Scaffold data
    scaffold_json = json.dumps(data.get("scaffold", {}))
    
    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>V11 Morphogenetic Field Visualizer</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ 
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0d1117; color: #c9d1d9; padding: 20px;
    max-width: 1800px; margin: 0 auto;
  }}
  h1 {{ color: #58a6ff; margin-bottom: 8px; font-size: 24px; }}
  h2 {{ color: #79c0ff; margin: 24px 0 12px; font-size: 18px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }}
  h3 {{ color: #d2a8ff; margin: 16px 0 8px; font-size: 14px; }}
  .subtitle {{ color: #8b949e; font-size: 14px; margin-bottom: 20px; }}
  .config {{ background: #161b22; padding: 12px 16px; border-radius: 6px; margin-bottom: 20px; font-family: monospace; font-size: 13px; border: 1px solid #21262d; }}
  .key-insight {{ 
    background: #1a2332; border-left: 3px solid #58a6ff; 
    padding: 12px 16px; margin: 12px 0; border-radius: 0 6px 6px 0;
    font-size: 13px;
  }}
  .key-insight strong {{ color: #58a6ff; }}
  
  /* Heatmap canvas styling */
  .heatmap-container {{ 
    background: #161b22; border: 1px solid #21262d; border-radius: 6px;
    padding: 16px; margin: 12px 0; overflow-x: auto;
  }}
  canvas {{ image-rendering: pixelated; display: block; }}
  .canvas-label {{ 
    display: flex; justify-content: space-between; 
    font-size: 11px; color: #8b949e; margin-top: 4px; 
  }}
  
  /* Token bar styling */
  .token-bar {{ 
    display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 1px;
    padding: 8px; background: #161b22; border-radius: 6px; margin: 8px 0;
    border: 1px solid #21262d;
  }}
  .token-cell {{
    min-width: 4px; height: 28px; display: flex; align-items: center;
    justify-content: center; font-size: 9px; color: #fff;
    cursor: pointer; position: relative; flex-shrink: 0;
    border-radius: 2px; transition: transform 0.1s;
  }}
  .token-cell:hover {{ transform: scaleY(1.3); z-index: 10; }}
  .token-cell .tooltip {{
    display: none; position: absolute; bottom: 110%; left: 50%;
    transform: translateX(-50%); background: #1f2937; color: #e5e7eb;
    padding: 6px 10px; border-radius: 4px; font-size: 11px;
    white-space: nowrap; z-index: 100; border: 1px solid #374151;
    pointer-events: none;
  }}
  .token-cell:hover .tooltip {{ display: block; }}
  
  /* Trajectory plot */
  .traj-container {{ 
    display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 8px; margin: 8px 0;
  }}
  .traj-item {{
    background: #161b22; border: 1px solid #21262d; border-radius: 6px;
    padding: 8px; text-align: center;
  }}
  .traj-item canvas {{ margin: 0 auto; }}
  .traj-label {{ font-size: 11px; color: #8b949e; margin-top: 4px; }}
  
  /* Colorbar */
  .colorbar {{
    display: flex; align-items: center; gap: 8px; margin: 4px 0;
    font-size: 11px; color: #8b949e;
  }}
  .colorbar-gradient {{
    width: 120px; height: 12px; border-radius: 2px;
  }}
  
  /* Controls */
  .controls {{
    background: #161b22; border: 1px solid #21262d; border-radius: 6px;
    padding: 12px 16px; margin: 12px 0; display: flex; gap: 16px;
    align-items: center; flex-wrap: wrap;
  }}
  .controls label {{ font-size: 12px; color: #8b949e; }}
  .controls select, .controls input {{
    background: #0d1117; color: #c9d1d9; border: 1px solid #30363d;
    border-radius: 4px; padding: 4px 8px; font-size: 12px;
  }}
  
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 1000px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  
  .legend {{ font-size: 11px; color: #8b949e; margin: 4px 0; }}
</style>
</head>
<body>

<h1>🧬 V11 Morphogenetic Field Visualizer</h1>
<div class="subtitle">
  Structural analysis of emergent morphogenetic patterns during a forward pass.
  <br>Morphogenesis runs at <strong>both training and inference</strong> — 
  the CA re-grows all structures from the token embedding (seed) every time.
</div>

<div class="config">
  <strong>Model:</strong> d_model={config_info['d_model']}, d_h={config_info['d_h']}, 
  n_steps={config_info['n_steps']}, n_scales={config_info['n_scales']}
  &nbsp;|&nbsp; <strong>Input:</strong> {html.escape(input_text[:100])}{'...' if len(input_text) > 100 else ''}
  &nbsp;|&nbsp; <strong>Tokens:</strong> {L}
  &nbsp;|&nbsp; <strong>Training step:</strong> {config_info.get('train_step', '?')}
  &nbsp;|&nbsp; <strong>Val PPL:</strong> {config_info.get('val_ppl', '?')}
</div>

<div class="key-insight">
  <strong>What you're seeing:</strong> These visualizations show the structures that the CA 
  <em>grows during a single forward pass</em>. The morphogen field h evolves over 
  {config_info['n_steps']} CA steps, progressively gating how content cells perceive their 
  neighbors. Different colors/intensities = different "cell types" = different routing patterns.
  This is the same process whether training or generating text.
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2>1. Gate Heatmap — Morphogen-Controlled Perception (positions × CA steps)</h2>
<p class="legend">
  Each cell shows the average gate value σ(W_gate · h) for that position at that CA step.
  Bright = open (high perception), Dark = closed (filtered). 
  Spatial patterns here = emergent routing structures.
</p>
<div class="heatmap-container">
  <canvas id="canvas-gate" height="200"></canvas>
  <div class="canvas-label">
    <span>Position 0</span>
    <span>← CA Step (top=0, bottom={n_steps - 1}) →</span>
    <span>Position {L - 1}</span>
  </div>
  <div class="colorbar">
    <span>Closed (0)</span>
    <div class="colorbar-gradient" style="background: linear-gradient(to right, #440154, #31688e, #35b779, #fde725);"></div>
    <span>Open (1)</span>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2>2. Morphogen Field (PCA → RGB) — Emergent Cell Types</h2>
<p class="legend">
  The morphogen h ∈ R^{d_h} projected to 3D via PCA, mapped to RGB color.
  Each unique color = unique morphogen signature = unique "cell type".
  If all cells are the same color → no differentiation (bad).
  Diverse, spatially structured colors → emergent specialization (good!).
</p>
<div class="heatmap-container">
  <canvas id="canvas-hpca" height="200"></canvas>
  <div class="canvas-label">
    <span>Position 0</span>
    <span>← CA Step (top=0, bottom={n_steps - 1}) →</span>
    <span>Position {L - 1}</span>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2>3. Content Update Magnitude ||Δz|| — Where Is the CA Building?</h2>
<p class="legend">
  Bright = large content updates (active growth). Dark = converged/permanent.
  Compare GATED (actual V11) vs UNGATED (hypothetical V10) to see morphogen's filtering effect.
</p>
<div class="grid-2">
  <div>
    <h3>Gated (V11: morphogen-filtered perception)</h3>
    <div class="heatmap-container">
      <canvas id="canvas-dz-gated" height="160"></canvas>
    </div>
  </div>
  <div>
    <h3>Ungated (hypothetical V10: full perception)</h3>
    <div class="heatmap-container">
      <canvas id="canvas-dz-ungated" height="160"></canvas>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2>4. Gate Diversity — Per-Position Channel Variance</h2>
<p class="legend">
  Shows how much the gate varies across content dimensions at each position.
  High variance = the morphogen is selectively filtering specific content channels.
  Low variance = uniform scaling (no real routing, just amplitude modulation).
</p>
<div class="heatmap-container">
  <canvas id="canvas-gatevar" height="200"></canvas>
  <div class="colorbar">
    <span>Low variance (uniform gate)</span>
    <div class="colorbar-gradient" style="background: linear-gradient(to right, #000, #ff4500, #ffff00);"></div>
    <span>High variance (selective filtering)</span>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2>5. Morphogen Norm ||h|| — Field Intensity</h2>
<p class="legend">
  The magnitude of the morphogen field at each position/step.
  Spatial patterns here indicate morphogen "concentration gradients" — 
  like chemical gradients in biological morphogenesis.
</p>
<div class="heatmap-container">
  <canvas id="canvas-hnorm" height="200"></canvas>
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2>6. Token-Level Morphogen Signatures (Final Step)</h2>
<p class="legend">
  Each token's final morphogen state projected to color. Hover to see token text and gate values.
  Tokens with similar colors have similar morphogen signatures → similar "cell fates".
</p>
<div id="token-bar-final" class="token-bar"></div>
<div id="token-bar-step0" class="token-bar"></div>
<div class="legend">Top: Final step morphogen. Bottom: Initial step (seed). 
  Compare to see how much differentiation occurs over the CA steps.</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2>7. Settlement — Permanent Memory Formation</h2>
<p class="legend">
  Shows which cells settle into permanent frozen state. 
  Yellow = settled (permanent memory). Dark = active (still growing).
</p>
<div class="heatmap-container">
  <canvas id="canvas-settlement" height="200"></canvas>
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2>8. Morphogen Dimension Importance</h2>
<p class="legend">
  How much each of the {d_h} morphogen dimensions influences the gate (||W_gate[:, j]||).
  A few dominant dims = simple routing. Many active dims = complex routing.
</p>
<div class="heatmap-container">
  <canvas id="canvas-dim-importance" width="600" height="80"></canvas>
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2>9. Morphogen Evolution Trajectories</h2>
<p class="legend">
  How the morphogen field evolves over CA steps for individual token positions.
  Each line is one morphogen dimension. Convergent lines = stable structure. Divergent = active signaling.
</p>
<div id="trajectories" class="traj-container"></div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2>10. Gating Effect: ||Δz_gated|| / ||Δz_ungated|| Ratio</h2>
<p class="legend">
  Ratio of gated to ungated content updates. Values &lt; 1 mean morphogen is 
  <em>suppressing</em> updates (filtering). Values &gt; 1 shouldn't normally occur.
  Spatial variation in this ratio = morphogen creates different processing for different positions.
</p>
<div class="heatmap-container">
  <canvas id="canvas-gating-ratio" height="200"></canvas>
  <div class="colorbar">
    <span>Strong filtering (0)</span>
    <div class="colorbar-gradient" style="background: linear-gradient(to right, #2166ac, #f7f7f7, #b2182b);"></div>
    <span>Full pass-through (1+)</span>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<div id="scaffold-section"></div>

<script>
// ═══════════════════════════════════════════════════════════════════════
// Data (injected from Python)
// ═══════════════════════════════════════════════════════════════════════
const gateHeatmap = {gate_heatmap_json};
const gateVariance = {gate_var_json};
const hNorms = {h_norms_json};
const deltaZ = {delta_z_json};
const deltaH = {delta_h_json};
const ungatedDZ = {ungated_dz_json};
const settlementMap = {settlement_json};
const growthMap = {growth_json};
const hPCA = {h_pca_json};
const gateDimImportance = {gate_dim_imp_json};
const tokens = {tokens_json};
const hTrajectories = {h_traj_json};
const nSteps = {n_steps};
const L = tokens.length;
const dH = {d_h};
const scaffold = {scaffold_json};

// ═══════════════════════════════════════════════════════════════════════
// Colormaps
// ═══════════════════════════════════════════════════════════════════════
function viridis(t) {{
  t = Math.max(0, Math.min(1, t));
  if (t < 0.5) {{
    const s = t * 2;
    return [68 + s*(32-68), 1 + s*(145-1), 84 + s*(140-84)];
  }} else {{
    const s = (t - 0.5) * 2;
    return [32 + s*(253-32), 145 + s*(231-145), 140 + s*(37-140)];
  }}
}}

function hot(t) {{
  t = Math.max(0, Math.min(1, t));
  if (t < 0.33) return [t*3*255, 0, 0];
  if (t < 0.66) return [255, (t-0.33)*3*255, 0];
  return [255, 255, (t-0.66)*3*255];
}}

function rdbu(t) {{
  t = Math.max(0, Math.min(1, t));
  if (t < 0.5) {{
    const s = t * 2;
    return [33 + s*(247-33), 102 + s*(247-102), 172 + s*(247-172)];
  }} else {{
    const s = (t - 0.5) * 2;
    return [247 - s*(247-178), 247 - s*(247-24), 247 - s*(247-43)];
  }}
}}

function yellow(t) {{
  t = Math.max(0, Math.min(1, t));
  return [t*255, t*220, 0];
}}

// ═══════════════════════════════════════════════════════════════════════
// Drawing Utilities
// ═══════════════════════════════════════════════════════════════════════
function drawHeatmap(canvasId, data, colorFn, fixedRange) {{
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const rows = data.length;
  const cols = data[0].length;
  
  const cellW = Math.max(2, Math.floor(Math.min(1400, window.innerWidth - 80) / cols));
  const cellH = Math.max(8, Math.floor(180 / rows));
  
  canvas.width = cols * cellW;
  canvas.height = rows * cellH;
  
  const ctx = canvas.getContext('2d');
  
  let vmin, vmax;
  if (fixedRange) {{
    [vmin, vmax] = fixedRange;
  }} else {{
    vmin = Infinity; vmax = -Infinity;
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {{
        vmin = Math.min(vmin, data[r][c]);
        vmax = Math.max(vmax, data[r][c]);
      }}
  }}
  const range = vmax - vmin || 1e-10;
  
  for (let r = 0; r < rows; r++) {{
    for (let c = 0; c < cols; c++) {{
      const t = (data[r][c] - vmin) / range;
      const [rv, gv, bv] = colorFn(t);
      ctx.fillStyle = `rgb(${{Math.round(rv)}},${{Math.round(gv)}},${{Math.round(bv)}})`;
      ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
    }}
  }}
}}

function drawPCAHeatmap(canvasId, pcaData) {{
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const rows = pcaData.length;
  const cols = pcaData[0].length;
  
  const cellW = Math.max(2, Math.floor(Math.min(1400, window.innerWidth - 80) / cols));
  const cellH = Math.max(8, Math.floor(180 / rows));
  
  canvas.width = cols * cellW;
  canvas.height = rows * cellH;
  
  const ctx = canvas.getContext('2d');
  
  // Find global min/max for normalization
  let globalMin = [Infinity, Infinity, Infinity];
  let globalMax = [-Infinity, -Infinity, -Infinity];
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      for (let k = 0; k < 3; k++) {{
        globalMin[k] = Math.min(globalMin[k], pcaData[r][c][k]);
        globalMax[k] = Math.max(globalMax[k], pcaData[r][c][k]);
      }}
  
  for (let r = 0; r < rows; r++) {{
    for (let c = 0; c < cols; c++) {{
      const rgb = [0, 0, 0];
      for (let k = 0; k < 3; k++) {{
        const range = globalMax[k] - globalMin[k] || 1;
        rgb[k] = Math.round(((pcaData[r][c][k] - globalMin[k]) / range) * 255);
      }}
      ctx.fillStyle = `rgb(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}})`;
      ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
    }}
  }}
}}

function drawTokenBar(containerId, pcaRow, stepLabel) {{
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';
  
  const cols = pcaRow.length;
  let gMin = [Infinity, Infinity, Infinity], gMax = [-Infinity, -Infinity, -Infinity];
  for (let c = 0; c < cols; c++)
    for (let k = 0; k < 3; k++) {{
      gMin[k] = Math.min(gMin[k], pcaRow[c][k]);
      gMax[k] = Math.max(gMax[k], pcaRow[c][k]);
    }}
  
  for (let c = 0; c < cols; c++) {{
    const cell = document.createElement('div');
    cell.className = 'token-cell';
    cell.style.width = Math.max(4, Math.floor(1200 / cols)) + 'px';
    
    const rgb = [0, 0, 0];
    for (let k = 0; k < 3; k++) {{
      const range = gMax[k] - gMin[k] || 1;
      rgb[k] = Math.round(((pcaRow[c][k] - gMin[k]) / range) * 255);
    }}
    cell.style.backgroundColor = `rgb(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}})`;
    
    const tooltip = document.createElement('span');
    tooltip.className = 'tooltip';
    const gateVal = gateHeatmap[gateHeatmap.length-1][c];
    tooltip.innerHTML = `<strong>"${{tokens[c]}}"</strong> (pos ${{c}})<br>` +
      `Gate: ${{gateVal.toFixed(4)}}<br>` +
      `h norm: ${{hNorms[hNorms.length-1][c].toFixed(3)}}<br>` +
      `${{stepLabel}}`;
    cell.appendChild(tooltip);
    
    if (tokens[c].trim().length > 0 && cols <= 80) {{
      cell.textContent = tokens[c].length > 4 ? tokens[c].slice(0,3) + '…' : tokens[c];
    }}
    container.appendChild(cell);
  }}
}}

function drawBarChart(canvasId, values) {{
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const n = values.length;
  const barW = Math.max(4, Math.floor(canvas.width / n));
  canvas.width = n * barW;
  
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#161b22';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  const vmax = Math.max(...values) || 1;
  
  for (let i = 0; i < n; i++) {{
    const h = (values[i] / vmax) * (canvas.height - 10);
    const [r, g, b] = viridis(values[i] / vmax);
    ctx.fillStyle = `rgb(${{Math.round(r)}},${{Math.round(g)}},${{Math.round(b)}})`;
    ctx.fillRect(i * barW + 1, canvas.height - h, barW - 2, h);
    
    if (barW > 15) {{
      ctx.fillStyle = '#8b949e';
      ctx.font = '9px monospace';
      ctx.fillText(i.toString(), i * barW + 2, 10);
    }}
  }}
}}

function drawTrajectory(container, trajData, posIdx, tokenLabel) {{
  // trajData: (nSteps, dH) for one position
  const item = document.createElement('div');
  item.className = 'traj-item';
  
  const canvas = document.createElement('canvas');
  canvas.width = 180;
  canvas.height = 80;
  const ctx = canvas.getContext('2d');
  
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, 180, 80);
  
  const nDims = trajData[0].length;
  const nS = trajData.length;
  
  // Find range
  let vmin = Infinity, vmax = -Infinity;
  for (let t = 0; t < nS; t++)
    for (let d = 0; d < nDims; d++) {{
      vmin = Math.min(vmin, trajData[t][d]);
      vmax = Math.max(vmax, trajData[t][d]);
    }}
  const range = vmax - vmin || 1;
  
  // Draw each dim as a line (show top 8 most variable dims)
  const dimVars = [];
  for (let d = 0; d < nDims; d++) {{
    let mean = 0;
    for (let t = 0; t < nS; t++) mean += trajData[t][d];
    mean /= nS;
    let variance = 0;
    for (let t = 0; t < nS; t++) variance += (trajData[t][d] - mean) ** 2;
    dimVars.push({{ dim: d, v: variance }});
  }}
  dimVars.sort((a, b) => b.v - a.v);
  const topDims = dimVars.slice(0, Math.min(8, nDims));
  
  const colors = ['#58a6ff', '#f97583', '#56d364', '#d2a8ff', '#79c0ff', '#ffa657', '#ff7b72', '#7ee787'];
  
  topDims.forEach((dd, ci) => {{
    ctx.beginPath();
    ctx.strokeStyle = colors[ci % colors.length];
    ctx.lineWidth = 1.2;
    for (let t = 0; t < nS; t++) {{
      const x = (t / (nS - 1)) * 170 + 5;
      const y = 75 - ((trajData[t][dd.dim] - vmin) / range) * 70;
      if (t === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }}
    ctx.stroke();
  }});
  
  item.appendChild(canvas);
  const label = document.createElement('div');
  label.className = 'traj-label';
  label.textContent = `Pos ${{posIdx}}: "${{tokenLabel}}"`;
  item.appendChild(label);
  container.appendChild(item);
}}

// ═══════════════════════════════════════════════════════════════════════
// Render All
// ═══════════════════════════════════════════════════════════════════════
window.addEventListener('load', () => {{
  // 1. Gate heatmap
  drawHeatmap('canvas-gate', gateHeatmap, viridis, [0, 1]);
  
  // 2. Morphogen PCA → RGB
  drawPCAHeatmap('canvas-hpca', hPCA);
  
  // 3. Content updates (gated vs ungated) — use SHARED range for fair comparison
  let dzMin = Infinity, dzMax = -Infinity;
  let dzuMin = Infinity, dzuMax = -Infinity;
  for (let r = 0; r < nSteps; r++)
    for (let c = 0; c < L; c++) {{
      dzMin = Math.min(dzMin, deltaZ[r][c]);
      dzMax = Math.max(dzMax, deltaZ[r][c]);
      dzuMin = Math.min(dzuMin, ungatedDZ[r][c]);
      dzuMax = Math.max(dzuMax, ungatedDZ[r][c]);
    }}
  const sharedMin = Math.min(dzMin, dzuMin);
  const sharedMax = Math.max(dzMax, dzuMax);
  drawHeatmap('canvas-dz-gated', deltaZ, hot, [sharedMin, sharedMax]);
  drawHeatmap('canvas-dz-ungated', ungatedDZ, hot, [sharedMin, sharedMax]);
  
  // 4. Gate variance
  drawHeatmap('canvas-gatevar', gateVariance, hot);
  
  // 5. Morphogen norm
  drawHeatmap('canvas-hnorm', hNorms, viridis);
  
  // 6. Token bars
  drawTokenBar('token-bar-final', hPCA[nSteps - 1], 'Final step');
  drawTokenBar('token-bar-step0', hPCA[0], 'Initial step (seed)');
  
  // 7. Settlement
  drawHeatmap('canvas-settlement', settlementMap, yellow, [0, 1]);
  
  // 8. Dimension importance
  drawBarChart('canvas-dim-importance', gateDimImportance);
  
  // 9. Trajectories
  const trajContainer = document.getElementById('trajectories');
  const nTrack = Math.min(L, hTrajectories[0].length);
  const trackIndices = [];
  if (nTrack <= 12) {{
    for (let i = 0; i < nTrack; i++) trackIndices.push(i);
  }} else {{
    // Sample evenly + first/last
    trackIndices.push(0);
    for (let i = 1; i < 11; i++) trackIndices.push(Math.floor(i * nTrack / 11));
    trackIndices.push(nTrack - 1);
  }}
  
  trackIndices.forEach(pi => {{
    const traj = [];
    for (let t = 0; t < nSteps; t++) traj.push(hTrajectories[t][pi]);
    drawTrajectory(trajContainer, traj, pi, tokens[pi] || '?');
  }});
  
  // 10. Gating ratio
  const ratio = [];
  for (let r = 0; r < nSteps; r++) {{
    const row = [];
    for (let c = 0; c < L; c++) {{
      const ug = ungatedDZ[r][c] || 1e-10;
      row.push(Math.min(1.5, deltaZ[r][c] / ug));
    }}
    ratio.push(row);
  }}
  drawHeatmap('canvas-gating-ratio', ratio, rdbu, [0, 1.2]);
  
  // 11. Scaffold visualization (only if scaffold data exists)
  const scaffoldDiv = document.getElementById('scaffold-section');
  if (scaffold && scaffold.template_norms && scaffold.template_norms.length > 0) {{
    const K = scaffold.K || scaffold.template_norms.length;
    const alpha = scaffold.alpha || 0;
    const nAlive = scaffold.n_alive || 0;
    const scStep = scaffold.step || 0;
    
    let html = `<h2>11. Persistent Scaffold — Epigenetic Memory (K=${{K}} templates)</h2>`;
    html += `<p class="legend">
      The scaffold stores K=${{K}} persistent morphogen templates that survive across batches.
      They bias the initial morphogen seeding via soft selection (α=${{alpha}}).
      Templates are updated via EMA during training — they crystallize the most common morphogen patterns.
      Alive: ${{nAlive}}/${{K}} | Training step: ${{scStep}}
    </p>`;
    
    // Template norms bar chart
    html += `<h3>Template Norms (alive threshold shown)</h3>`;
    html += `<div class="heatmap-container"><canvas id="canvas-scaffold-norms" width="600" height="100"></canvas></div>`;
    
    // Seeding contribution per position
    if (scaffold.seeding_diff_per_pos) {{
      html += `<h3>Scaffold Seeding Contribution ||h_scaffold - h_fresh|| per Position</h3>`;
      html += `<p class="legend">How much the scaffold changes the initial morphogen seed at each position. High = scaffold strongly biases this position's fate.</p>`;
      html += `<div class="heatmap-container"><canvas id="canvas-scaffold-seeding" width="600" height="80"></canvas></div>`;
    }}
    
    // Template weights heatmap (L x K)
    if (scaffold.template_weights) {{
      html += `<h3>Template Selection Weights (positions × templates)</h3>`;
      html += `<p class="legend">Shows which scaffold template each token position selects via softmax attention. Bright = high weight. Uniform = no differentiation. Block structure = position-specific template selection.</p>`;
      html += `<div class="heatmap-container"><canvas id="canvas-scaffold-weights" height="200"></canvas></div>`;
      html += `<div class="colorbar"><span>Low weight (0)</span>`;
      html += `<div class="colorbar-gradient" style="background: linear-gradient(to right, #440154, #31688e, #35b779, #fde725);"></div>`;
      html += `<span>High weight (1)</span></div>`;
    }}
    
    scaffoldDiv.innerHTML = html;
    
    // Draw template norms bar chart
    const normCanvas = document.getElementById('canvas-scaffold-norms');
    if (normCanvas) {{
      const ctx = normCanvas.getContext('2d');
      const w = normCanvas.width, h = normCanvas.height;
      ctx.fillStyle = '#161b22';
      ctx.fillRect(0, 0, w, h);
      const norms = scaffold.template_norms;
      const maxNorm = Math.max(...norms, 0.01);
      const barW = Math.floor((w - 40) / K);
      const pad = 20;
      for (let i = 0; i < K; i++) {{
        const barH = (norms[i] / maxNorm) * (h - 30);
        const alive = norms[i] >= 0.01;
        ctx.fillStyle = alive ? '#58a6ff' : '#484f58';
        ctx.fillRect(pad + i * barW + 2, h - 15 - barH, barW - 4, barH);
        ctx.fillStyle = '#8b949e';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`T${{i}}`, pad + i * barW + barW/2, h - 3);
      }}
      // Label
      ctx.fillStyle = '#8b949e';
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(maxNorm.toFixed(2), pad - 2, 12);
    }}
    
    // Draw seeding diff bar chart
    const seedCanvas = document.getElementById('canvas-scaffold-seeding');
    if (seedCanvas && scaffold.seeding_diff_per_pos) {{
      const ctx = seedCanvas.getContext('2d');
      const diffs = scaffold.seeding_diff_per_pos;
      const w = seedCanvas.width = Math.max(600, diffs.length * 6);
      const h = seedCanvas.height;
      ctx.fillStyle = '#161b22';
      ctx.fillRect(0, 0, w, h);
      const maxD = Math.max(...diffs, 1e-8);
      const barW = Math.max(2, Math.floor(w / diffs.length));
      for (let i = 0; i < diffs.length; i++) {{
        const t = diffs[i] / maxD;
        const [r, g, b] = hot(t);
        ctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
        const barH = t * (h - 10);
        ctx.fillRect(i * barW, h - 5 - barH, barW - 1, barH);
      }}
    }}
    
    // Draw template weights heatmap (L x K)
    const twCanvas = document.getElementById('canvas-scaffold-weights');
    if (twCanvas && scaffold.template_weights) {{
      const weights = scaffold.template_weights;  // (L, K)
      const Lw = weights.length;
      const Kw = weights[0].length;
      const cellW = Math.max(3, Math.floor(800 / Lw));
      const cellH = Math.max(12, Math.floor(180 / Kw));
      twCanvas.width = cellW * Lw;
      twCanvas.height = cellH * Kw;
      const ctx = twCanvas.getContext('2d');
      // Find max for scaling
      let wMax = 0;
      for (let l = 0; l < Lw; l++)
        for (let k = 0; k < Kw; k++)
          wMax = Math.max(wMax, weights[l][k]);
      wMax = wMax || 1;
      for (let k = 0; k < Kw; k++) {{
        for (let l = 0; l < Lw; l++) {{
          const t = weights[l][k] / wMax;
          const [r, g, b] = viridis(t);
          ctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
          ctx.fillRect(l * cellW, k * cellH, cellW, cellH);
        }}
      }}
      // Template labels on left
      ctx.fillStyle = '#8b949e';
      ctx.font = '10px monospace';
      ctx.textAlign = 'left';
      for (let k = 0; k < Kw; k++) {{
        ctx.fillStyle = 'rgba(200,200,200,0.7)';
        ctx.fillText(`T${{k}}`, 4, k * cellH + cellH/2 + 4);
      }}
    }}
  }} else {{
    scaffoldDiv.innerHTML = '<h2>11. Scaffold</h2><p class="legend">No scaffold data — scaffold may be inactive (K=0 or step &lt; warmup).</p>';
  }}
}});
</script>

</body>
</html>"""

    return report_html


# ═══════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualize morphogenetic field structures in V11"
    )
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--text", type=str, default=None,
                        help="Input text to visualize. If omitted, uses a sample from the dataset.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path (default: alongside checkpoint)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load checkpoint ───────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config_dict = ckpt["config"]

    model_name = config_dict.get("model_name", "v11")
    ModelClass = get_model_class(model_name)
    inf_config = dict(config_dict)
    inf_config["dropout"] = 0.0
    model = ModelClass(inf_config)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()

    cfg = model.config
    print(f"Model: d_model={cfg.d_model}, d_h={cfg.d_h}, n_steps={cfg.n_steps}")

    # ── Create tokenizer ──────────────────────────────────────────────
    tokenizer = create_tokenizer(
        config_dict.get("tokenizer_name", "bpe"),
        vocab_size=config_dict.get("tokenizer_vocab_size", config_dict.get("vocab_size", 10000)),
        cache_dir=config_dict.get("tokenizer_cache_dir"),
        pretrained_name=config_dict.get("tokenizer_pretrained_name"),
        dataset_name=config_dict.get("dataset_hf_name"),
    )

    # ── Prepare input ─────────────────────────────────────────────────
    seq_len = config_dict.get("seq_len", 128)

    if args.text:
        input_text = args.text
        token_ids = tokenizer.encode(input_text)
        if len(token_ids) > seq_len:
            token_ids = token_ids[:seq_len]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    else:
        # Use sample from dataset
        from ncalm.data.dataset import TinyStoriesData, ThePileData, SmolTalk2Data, create_data_loader
        dataset_name = config_dict.get("dataset", "tinystories")
        if dataset_name == "the_pile":
            data = ThePileData(
                seq_len=seq_len, tokenizer=tokenizer,
                cache_name=config_dict.get("dataset_cache_name", "pile_tokens_2b.bin"),
                dataset_name=config_dict.get("dataset_hf_name"),
                max_tokens=config_dict.get("max_dataset_tokens", 2_000_000_000),
                cache_dir=config_dict.get("dataset_cache_dir"),
            )
        elif dataset_name == "smoltalk2":
            data = SmolTalk2Data(
                seq_len=seq_len, tokenizer=tokenizer,
                cache_name=config_dict.get("dataset_cache_name", "smoltalk2_sft_tokens_5b.pt"),
                dataset_config=config_dict.get("dataset_config", "SFT"),
                dataset_splits=config_dict.get("dataset_splits"),
                dataset_sources=config_dict.get("dataset_sources"),
                max_tokens=config_dict.get("max_dataset_tokens", 5_000_000_000),
                cache_dir=config_dict.get("dataset_cache_dir"),
            )
        else:
            data = TinyStoriesData(
                seq_len=seq_len, tokenizer=tokenizer,
                cache_name=config_dict.get("dataset_cache_name", "tinystories_tokens_ncalm.pt"),
            )
        val_loader = create_data_loader(data, 1, device, "val")
        input_ids = next(val_loader)  # (1, seq_len)
        input_text = tokenizer.decode(input_ids[0].tolist())

    # Decode token strings for display
    token_id_list = input_ids[0].tolist()
    token_strings = []
    for tid in token_id_list:
        try:
            tok_str = tokenizer.decode([tid])
        except Exception:
            tok_str = f"[{tid}]"
        token_strings.append(tok_str)

    print(f"Input: {input_text[:120]}...")
    print(f"Tokens: {len(token_strings)}")

    # ── Run instrumented forward pass ─────────────────────────────────
    print("Running instrumented forward pass...")
    data = instrumented_forward(model, input_ids)

    # ── Generate HTML ─────────────────────────────────────────────────
    config_info = {
        "d_model": cfg.d_model,
        "d_h": cfg.d_h,
        "n_steps": cfg.n_steps,
        "n_scales": cfg.n_scales,
        "train_step": ckpt.get("step", "?"),
        "val_ppl": f"{ckpt.get('best_val_ppl', 0):.1f}",
    }

    html_content = generate_html_report(data, token_strings, input_text, config_info)

    # ── Save ──────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.checkpoint).parent / "morphogen_structures.html"
    
    out_path.write_text(html_content, encoding="utf-8")
    print(f"\nVisualization saved to: {out_path}")
    print(f"Open in browser to explore morphogenetic structures.")


if __name__ == "__main__":
    main()
