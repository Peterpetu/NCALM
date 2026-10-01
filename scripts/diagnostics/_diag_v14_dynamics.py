#!/usr/bin/env python3
"""Deep diagnostic: Probe V14 for emergent dynamics after training.

Analyzes:
  1. Weight norms (did zero-init weights actually grow?)
  2. Morphogen evolution dynamics (static vs dynamic per step)
  3. Routing specialization (do different rules activate for different content?)
  4. Content perception coupling (is content actually influencing morphogen?)
  5. Multi-scale readout weights (did scale_logits diverge from uniform?)
  6. Jacobian-based attention pattern (does position i attend to position j?)
  7. Ablation: remove content perception and measure degradation
"""
import sys
sys.path.insert(0, ".")

import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from ncalm.models.v14.model import AENCAv14Model
from ncalm.models.v14.config import V14Config


def load_model(ckpt_path, device="cuda"):
    """Load trained V14 model from checkpoint."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Load config
    config_path = Path(ckpt_path).parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    model = AENCAv14Model(config)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def analyze_weight_growth(model):
    """Check if zero-init weights have grown from zero."""
    print("\n" + "=" * 70)
    print("  1. WEIGHT GROWTH FROM ZERO-INIT CASCADE")
    print("=" * 70)
    
    zero_init_weights = {
        "W_h2 (rule outputs)": [model.morphogen.W_h2[k].weight for k in range(model.config.n_rules)],
        "W_router": [model.morphogen.W_router.weight],
        "W_h_feedback": [model.morphogen.W_h_feedback.weight],
    }
    
    # Add per-theta readout weights
    for i in range(model.config.n_theta):
        zero_init_weights[f"theta{i}_W_readout"] = [model.theta_readouts[i].W_readout.weight]
        zero_init_weights[f"theta{i}_W_down (FFN)"] = [model.theta_ffns[i].W_down.weight]
    
    for name, weight_list in zero_init_weights.items():
        norms = [w.norm().item() for w in weight_list]
        mean_norm = np.mean(norms)
        status = "GROWN" if mean_norm > 0.01 else "STILL ~ZERO"
        print(f"  {name:35s}: norm={mean_norm:.6f}  [{status}]")
    
    # Non-zero-init weights for comparison
    print("\n  --- Non-zero-init (reference) ---")
    ref_weights = {
        "W_h1[0] (rule input)": model.morphogen.W_h1[0].weight.norm().item(),
        "W_gate_a": model.morphogen.W_gate_a.weight.norm().item(),
        "W_content_proj": model.morphogen.W_content_proj.weight.norm().item(),
        "W_h_init (seed)": model.morphogen.W_h_init.weight.norm().item(),
    }
    for name, norm in ref_weights.items():
        print(f"  {name:35s}: norm={norm:.6f}")


def analyze_morphogen_dynamics(model, device="cuda"):
    """Probe morphogen evolution step-by-step to check for dynamic behavior."""
    print("\n" + "=" * 70)
    print("  2. MORPHOGEN EVOLUTION DYNAMICS (per-step analysis)")
    print("=" * 70)
    
    cfg = model.config
    B, L = 2, 64
    x = torch.randint(0, cfg.vocab_size, (B, L), device=device)
    
    with torch.no_grad():
        z = model.token_embedding(x) + model.state_pe(L).unsqueeze(0)
        h = model.morphogen.seed(z)
        
        h_init_norm = h.norm(dim=-1).mean().item()
        print(f"\n  Morphogen seed norm (mean): {h_init_norm:.4f}")
        
        # Project content once
        z_proj = model.morphogen.W_content_proj(model.morphogen.norm_z_proj(z))
        
        # Step-by-step evolution
        print(f"\n  {'Step':>4s} {'Dil':>5s} {'|h|':>8s} {'|dh|':>8s} {'dh/h%':>7s} {'RouteEnt':>9s} {'GateMean':>9s} {'ContentInf':>10s}")
        print("  " + "-" * 65)
        
        h_states = [h.clone()]
        
        for m in range(cfg.n_morph_steps):
            dilation = cfg.morph_dilations[m % len(cfg.morph_dilations)]
            h_before = h.clone()
            
            # Morphogen neighborhood
            h_nbr = model.morphogen.morph_conv(h, dilation=dilation)
            
            # Content perception
            c_m = model.morphogen.content_conv(z_proj, dilation=dilation)
            
            # Concatenate + normalize
            h_cat = model.morphogen.rule_norm(torch.cat([h, h_nbr, c_m], dim=-1))
            
            # Rules
            deltas = []
            for k in range(cfg.n_rules):
                hidden = torch.tanh(model.morphogen.W_h1[k](h_cat))
                delta_k = model.morphogen.W_h2[k](hidden)
                deltas.append(delta_k)
            delta_stack = torch.stack(deltas, dim=2)
            
            # Routing
            route_logits = model.morphogen.W_router(h)
            weights = F.softmax(route_logits, dim=-1)
            route_entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean().item()
            
            # Weighted sum
            delta_h = (weights.unsqueeze(-1) * delta_stack).sum(dim=2)
            
            # Gate
            gate = torch.sigmoid(
                F.silu(model.morphogen.W_gate_a(h)) * model.morphogen.W_gate_b(h)
                + model.morphogen.gate_bias
            )
            gate_mean = gate.mean().item()
            
            # Update
            update = gate * delta_h
            h = h + update
            h_states.append(h.clone())
            
            h_norm = h.norm(dim=-1).mean().item()
            delta_norm = update.norm(dim=-1).mean().item()
            delta_pct = (delta_norm / (h_norm + 1e-8)) * 100
            
            # Content influence: how much does c_m contribute relative to h and h_nbr?
            content_ratio = c_m.norm(dim=-1).mean().item() / (h_cat.norm(dim=-1).mean().item() + 1e-8)
            
            print(f"  {m:4d} {dilation:5d} {h_norm:8.4f} {delta_norm:8.6f} {delta_pct:6.2f}% {route_entropy:9.4f} {gate_mean:9.4f} {content_ratio:10.4f}")
        
        # Overall dynamics assessment
        total_evolution = (h_states[-1] - h_states[0]).norm(dim=-1).mean().item()
        seed_norm = h_states[0].norm(dim=-1).mean().item()
        print(f"\n  Total evolution |h_final - h_seed|: {total_evolution:.6f}")
        print(f"  As fraction of seed norm: {total_evolution/(seed_norm+1e-8)*100:.2f}%")
        
        # Check temporal correlation: are step updates correlated (traveling wave) or independent?
        if len(h_states) > 2:
            step_updates = [h_states[i+1] - h_states[i] for i in range(len(h_states)-1)]
            # Cosine similarity between consecutive updates
            cos_sims = []
            for i in range(len(step_updates)-1):
                u1 = step_updates[i].reshape(-1)
                u2 = step_updates[i+1].reshape(-1)
                cos = F.cosine_similarity(u1.unsqueeze(0), u2.unsqueeze(0)).item()
                cos_sims.append(cos)
            print(f"  Consecutive update cosine similarities: {[f'{c:.4f}' for c in cos_sims]}")
            print(f"  Mean cosine sim: {np.mean(cos_sims):.4f} (>0.9=repetitive, ~0=diverse, <0=oscillatory)")


def analyze_routing_specialization(model, device="cuda"):
    """Check if different rules activate for different content types."""
    print("\n" + "=" * 70)
    print("  3. ROUTING SPECIALIZATION (per-rule analysis)")
    print("=" * 70)
    
    cfg = model.config
    
    # Create varied content: different token patterns
    patterns = {
        "uniform_0": torch.zeros(1, 32, dtype=torch.long, device=device),
        "uniform_500": torch.full((1, 32), 500, dtype=torch.long, device=device),
        "sequential": torch.arange(0, 32, dtype=torch.long, device=device).unsqueeze(0),
        "random_A": torch.randint(0, cfg.vocab_size, (1, 32), device=device),
        "random_B": torch.randint(0, cfg.vocab_size, (1, 32), device=device),
    }
    
    all_routes = {}
    with torch.no_grad():
        for name, tokens in patterns.items():
            z = model.token_embedding(tokens) + model.state_pe(32).unsqueeze(0)
            h = model.morphogen.seed(z)
            scale_w = model.theta_readouts[0].get_scale_weights()
            _, _, routing_weights, _ = model.morphogen.evolve(h, z, cfg.n_morph_steps, scale_w)
            
            # Average routing across all steps and positions
            avg_route = torch.stack(routing_weights).mean(dim=(0, 1, 2))  # (K,)
            all_routes[name] = avg_route.cpu().numpy()
            print(f"  {name:15s}: {[f'{r:.4f}' for r in avg_route.tolist()]}")
    
    # Check if routing is content-dependent
    routes_matrix = np.stack(list(all_routes.values()))
    route_variance = routes_matrix.var(axis=0)
    print(f"\n  Per-rule variance across inputs: {[f'{v:.6f}' for v in route_variance]}")
    print(f"  Total routing variance: {route_variance.sum():.6f}")
    if route_variance.sum() < 0.0001:
        print("  DIAGNOSIS: Routing is NOT content-dependent — all inputs use same rule mix")
    else:
        print("  Routing shows content dependence")


def analyze_scale_weights(model):
    """Check if multi-scale readout learned to differentiate scales."""
    print("\n" + "=" * 70)
    print("  4. MULTI-SCALE READOUT WEIGHTS (Phase C)")
    print("=" * 70)
    
    cfg = model.config
    for i in range(cfg.n_theta):
        logits = model.theta_readouts[i].scale_logits.data
        weights = F.softmax(logits, dim=0)
        entropy = -(weights * (weights + 1e-8).log()).sum().item()
        max_entropy = np.log(cfg.n_morph_steps)
        
        print(f"\n  Theta {i}:")
        print(f"    Raw logits:  {[f'{l:.4f}' for l in logits.tolist()]}")
        print(f"    Weights:     {[f'{w:.4f}' for w in weights.tolist()]}")
        print(f"    Peak step:   {weights.argmax().item()} (dilation={cfg.morph_dilations[weights.argmax().item()]})")
        print(f"    Entropy:     {entropy:.4f} / {max_entropy:.4f} ({entropy/max_entropy*100:.1f}% of max)")
        
        if entropy / max_entropy > 0.95:
            print(f"    DIAGNOSIS: Still near-uniform — scale_logits haven't specialized")
        else:
            print(f"    Scale specialization detected!")


def analyze_jacobian_attention(model, device="cuda"):
    """Compute Jacobian ∂z_out[i]/∂z_in[j] to see emergent attention patterns."""
    print("\n" + "=" * 70)
    print("  5. EMERGENT ATTENTION PATTERNS (Jacobian analysis)")
    print("=" * 70)
    
    cfg = model.config
    L = 16  # small for Jacobian computation
    
    x = torch.randint(0, cfg.vocab_size, (1, L), device=device)
    
    # We need gradients w.r.t. the embedding
    z_embed = model.token_embedding(x) + model.state_pe(L).unsqueeze(0)
    z = z_embed.detach().requires_grad_(True)
    
    # Run one theta cycle
    h = model.morphogen.seed(z)
    scale_w = model.theta_readouts[0].get_scale_weights()
    h_evolved, h_multi, _, _ = model.morphogen.evolve(h, z, cfg.n_morph_steps, scale_w)
    z_update = model.theta_readouts[0](h_multi)
    
    # Compute Jacobian: for each output position i, which input positions j contribute?
    # Use the norm of z_update[i] as the scalar to differentiate
    attention_map = torch.zeros(L, L, device=device)
    
    for i in range(L):
        if z.grad is not None:
            z.grad.zero_()
        scalar = z_update[0, i].norm()
        scalar.backward(retain_graph=True)
        
        # Gradient magnitude per input position
        grad_per_pos = z.grad[0].norm(dim=-1)  # (L,)
        attention_map[i] = grad_per_pos
    
    # Normalize rows
    row_sums = attention_map.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    attention_map = attention_map / row_sums
    
    print(f"\n  Jacobian attention map (normalized, {L}x{L}):")
    print(f"  Rows=output position, Cols=input position influence")
    
    # Print as compact heatmap
    map_np = attention_map.cpu().numpy()
    for i in range(L):
        row = "  "
        for j in range(L):
            v = map_np[i, j]
            if j > i:
                row += "  .  "  # future = should be zero (causal)
            elif v > 0.15:
                row += f" {v:.2f}"
            elif v > 0.05:
                row += f" {v:.2f}"
            else:
                row += "  __ "
        print(f"  pos{i:2d}: {row}")
    
    # Analysis
    causal_violation = attention_map.triu(diagonal=1).sum().item()
    self_attention = attention_map.diag().mean().item()
    
    # How much does each position attend to recent vs distant past?
    for lag in [0, 1, 2, 3, 5, 10]:
        if lag >= L:
            break
        diag_vals = []
        for i in range(lag, L):
            diag_vals.append(map_np[i, i - lag])
        mean_attn = np.mean(diag_vals) if diag_vals else 0
        print(f"  Mean attention at lag {lag}: {mean_attn:.4f}")
    
    print(f"\n  Causal violation mass: {causal_violation:.6f} (should be ~0)")
    print(f"  Self-attention (diagonal): {self_attention:.4f}")
    
    # Entropy of attention distribution (per row)
    entropies = []
    for i in range(L):
        row = attention_map[i, :i+1]  # only causal part
        if row.sum() > 0:
            row = row / row.sum()
            ent = -(row * (row + 1e-8).log()).sum().item()
            entropies.append(ent)
    print(f"  Mean attention entropy: {np.mean(entropies):.4f} (higher = more spread out)")
    print(f"  Max possible entropy (uniform over causal): {np.mean([np.log(i+1) for i in range(L)]):.4f}")


def analyze_content_ablation(model, device="cuda"):
    """Ablate content perception and measure output degradation."""
    print("\n" + "=" * 70)
    print("  6. CONTENT PERCEPTION ABLATION")
    print("=" * 70)
    
    cfg = model.config
    B, L = 2, 32
    x = torch.randint(0, cfg.vocab_size, (B, L), device=device)
    
    with torch.no_grad():
        # Normal forward
        logits_normal = model.get_logits(x)
        
        # Ablate: zero out the content projection weight
        original_weight = model.morphogen.W_content_proj.weight.data.clone()
        model.morphogen.W_content_proj.weight.data.zero_()
        
        logits_ablated = model.get_logits(x)
        
        # Restore
        model.morphogen.W_content_proj.weight.data = original_weight
    
    # Measure difference
    diff = (logits_normal - logits_ablated).norm() / logits_normal.norm()
    kl_div = F.kl_div(
        F.log_softmax(logits_ablated[:, :-1].reshape(-1, cfg.vocab_size), dim=-1),
        F.softmax(logits_normal[:, :-1].reshape(-1, cfg.vocab_size), dim=-1),
        reduction="batchmean",
    )
    
    print(f"  Output difference (normalized L2): {diff:.6f}")
    print(f"  KL divergence (ablated vs normal): {kl_div:.6f}")
    
    if diff < 0.01:
        print("  DIAGNOSIS: Content perception has MINIMAL effect — morphogen ignores content")
    elif diff < 0.1:
        print("  DIAGNOSIS: Content perception has MODERATE effect")
    else:
        print("  DIAGNOSIS: Content perception has STRONG effect — morphogen uses content")


def analyze_dynamics_diversity(model, device="cuda"):
    """Check if the dynamics diversity loss created varied temporal behavior."""
    print("\n" + "=" * 70)
    print("  7. DYNAMICS DIVERSITY (temporal behavior)")
    print("=" * 70)
    
    cfg = model.config
    B, L = 4, 64
    x = torch.randint(0, cfg.vocab_size, (B, L), device=device)
    
    with torch.no_grad():
        z = model.token_embedding(x) + model.state_pe(L).unsqueeze(0)
        h = model.morphogen.seed(z)
        scale_w = model.theta_readouts[0].get_scale_weights()
        _, _, routing_weights, step_deltas = model.morphogen.evolve(h, z, cfg.n_morph_steps, scale_w)
    
    # Step delta analysis
    delta_norms = [sd.mean().item() for sd in step_deltas]
    print(f"\n  Step delta norms (mean across batch/positions):")
    for m, dn in enumerate(delta_norms):
        dil = cfg.morph_dilations[m % len(cfg.morph_dilations)]
        bar = "#" * int(dn * 1000)
        print(f"    Step {m:2d} (d={dil:4d}): {dn:.6f} {bar}")
    
    print(f"\n  Delta norm mean: {np.mean(delta_norms):.6f}")
    print(f"  Delta norm std:  {np.std(delta_norms):.6f}")
    print(f"  Delta norm CV:   {np.std(delta_norms)/(np.mean(delta_norms)+1e-8):.4f}")
    
    if np.mean(delta_norms) < 0.001:
        print("  DIAGNOSIS: Morphogen is nearly STATIC — dynamics diversity loss too weak")
    elif np.std(delta_norms) / (np.mean(delta_norms) + 1e-8) < 0.1:
        print("  DIAGNOSIS: All steps have similar dynamics — no step specialization")
    else:
        print("  DIAGNOSIS: Steps show varied dynamics — potential specialization")
    
    # Routing entropy across steps
    print(f"\n  Routing entropy across steps:")
    for m, rw in enumerate(routing_weights):
        ent = -(rw * (rw + 1e-8).log()).sum(dim=-1).mean().item()
        max_ent = np.log(cfg.n_rules)
        dil = cfg.morph_dilations[m % len(cfg.morph_dilations)]
        print(f"    Step {m:2d} (d={dil:4d}): entropy={ent:.4f} / {max_ent:.4f} ({ent/max_ent*100:.1f}%)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "experiments/AENCA v14 (Emergent Dynamics)/v14_smoke_20min_20260331_124646/best_model.pt"
    
    print("=" * 70)
    print("  V14 EMERGENT DYNAMICS -- POST-TRAINING DEEP DIAGNOSTIC")
    print("=" * 70)
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Device: {device}")
    
    model = load_model(ckpt_path, device)
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    
    analyze_weight_growth(model)
    analyze_morphogen_dynamics(model, device)
    analyze_routing_specialization(model, device)
    analyze_scale_weights(model)
    analyze_dynamics_diversity(model, device)
    analyze_content_ablation(model, device)
    analyze_jacobian_attention(model, device)
    
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
