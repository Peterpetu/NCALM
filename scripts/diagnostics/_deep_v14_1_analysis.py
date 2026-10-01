#!/usr/bin/env python3
"""Deep analysis: V14.1 fixes — did emergent dynamics actually emerge?"""
import sys, glob
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F

from ncalm.models.v14.model import AENCAv14Model

device = "cuda" if torch.cuda.is_available() else "cpu"

# Find latest V14.1 checkpoint
pattern = "experiments/AENCA v14 (Emergent Dynamics)/v14_1_20min_*/best_model.pt"
ckpts = sorted(glob.glob(pattern))
if not ckpts:
    print(f"No checkpoint found matching {pattern}")
    sys.exit(1)
ckpt_path = ckpts[-1]
print(f"Loading: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model = AENCAv14Model(ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device).eval()
cfg = model.config

print("=" * 70)
print("  V14.1 DEEP DYNAMICS ANALYSIS")
print("=" * 70)

# 1. Position variance
print("\n[1] POSITION VARIANCE IN MORPHOGEN")
x = torch.randint(0, cfg.vocab_size, (1, 64), device=device)
with torch.no_grad():
    z = model.token_embedding(x) + model.state_pe(64).unsqueeze(0)
    h = model.morphogen.seed(z)
    
    print(f"  Seed h std across positions: {h[0].std(dim=0).mean():.4f}")
    
    sw = model.theta_readouts[0].get_scale_weights()
    h_final, h_multi, routing, deltas = model.morphogen.evolve(h, z, cfg.n_morph_steps, sw)
    
    print(f"  Final h std across positions: {h_final[0].std(dim=0).mean():.4f}")

# 2. Rule output analysis
print("\n[2] RULE OUTPUT ANALYSIS")
with torch.no_grad():
    z_proj = model.morphogen.W_content_proj(model.morphogen.norm_z_proj(z))
    h_nbr = model.morphogen.morph_conv(h, dilation=1)
    c_m = model.morphogen.content_conv(z_proj, dilation=1)
    h_cat = model.morphogen.rule_norm(torch.cat([h, h_nbr, c_m], dim=-1))
    
    norms = []
    for k in range(cfg.n_rules):
        hidden = torch.tanh(model.morphogen.W_h1[k](h_cat))
        delta_k = model.morphogen.W_h2[k](hidden)
        norms.append(model.morphogen.W_h2[k].weight.norm().item())
        print(f"  Rule {k}: ||delta||={delta_k.norm():.2f}, W_h2 norm={norms[-1]:.4f}")
    
    max_norm = max(norms)
    min_norm = min(norms)
    print(f"  Norm ratio (max/min): {max_norm/min_norm:.2f} (V14.0 was 3.4x)")

# 3. Gate analysis (CRITICAL: was 0.95 → should be ~0.5 now)
print("\n[3] GATE ANALYSIS (bias was 3.0, now 0.0)")
with torch.no_grad():
    gate = torch.sigmoid(
        F.silu(model.morphogen.W_gate_a(h)) * model.morphogen.W_gate_b(h) + model.morphogen.gate_bias
    )
    print(f"  Gate mean: {gate.mean():.4f} (V14.0 was 0.9468)")
    print(f"  Gate min:  {gate.min():.4f}")
    print(f"  Gate max:  {gate.max():.4f}")
    print(f"  Gate std (position): {gate[0].std(dim=0).mean():.6f} (V14.0 was 0.009)")
    print(f"  Gate bias value: {model.morphogen.gate_bias.mean():.4f}")

# 4. Router analysis — CRITICAL: was 99.6% to rule 2
print("\n[4] ROUTING DISTRIBUTION (V14.0 collapsed to 99.6% rule 2)")
with torch.no_grad():
    # Check step embedding + step-dependent routing
    has_step_embed = hasattr(model.morphogen, 'step_embed')
    print(f"  Has step_embed: {has_step_embed}")
    
    # Simulate routing at each step
    for step_idx in range(min(cfg.n_morph_steps, 6)):
        logits = model.morphogen.W_router(h)  # (B, L, K)
        if has_step_embed:
            step_t = torch.tensor([step_idx], device=device)
            step_bias = model.morphogen.W_step_route(model.morphogen.step_embed(step_t))
            logits = logits + step_bias.unsqueeze(0)
        probs = F.softmax(logits, dim=-1)
        mean_probs = probs.mean(dim=(0, 1))
        top1 = probs.argmax(dim=-1)
        
        fracs = []
        for k in range(cfg.n_rules):
            fracs.append((top1 == k).float().mean().item())
        
        prob_str = " ".join(f"{p:.3f}" for p in mean_probs.tolist())
        frac_str = " ".join(f"{f:.3f}" for f in fracs)
        
        entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum()
        max_entropy = torch.tensor(cfg.n_rules, dtype=torch.float).log()
        
        print(f"  Step {step_idx}: probs=[{prob_str}] top1_frac=[{frac_str}] entropy={entropy:.3f}/{max_entropy:.3f}")

# 5. Position-specific information
print("\n[5] POSITION-DEPENDENT INFO (V14.0: cos_sim=0.999990)")
with torch.no_grad():
    h_norm = F.normalize(h_final[0], dim=-1)
    sim_matrix = h_norm @ h_norm.T
    off_diag = sim_matrix[~torch.eye(64, dtype=bool, device=device)]
    print(f"  Mean off-diagonal cosine sim: {off_diag.mean():.6f}")
    print(f"  Min off-diagonal cosine sim:  {off_diag.min():.6f}")
    print(f"  (Target: significantly below 0.999990)")

# 6. Step update diversity — CRITICAL: was cosine sim 0.997 between consecutive steps
print("\n[6] STEP UPDATE DIVERSITY (V14.0: consecutive cos_sim=0.997)")
with torch.no_grad():
    h_curr = model.morphogen.seed(z)
    step_deltas = []
    step_routing = []
    for m in range(cfg.n_morph_steps):
        dil = cfg.morph_dilations[m % len(cfg.morph_dilations)]
        eff_dil = min(dil, max(1, z.shape[1] // 2))
        
        z_proj_m = model.morphogen.W_content_proj(model.morphogen.norm_z_proj(z))
        h_nbr_m = model.morphogen.morph_conv(h_curr, dilation=eff_dil)
        c_m_m = model.morphogen.content_conv(z_proj_m, dilation=eff_dil)
        h_cat_m = model.morphogen.rule_norm(torch.cat([h_curr, h_nbr_m, c_m_m], dim=-1))
        
        logits_m = model.morphogen.W_router(h_curr)
        if has_step_embed:
            step_t = torch.tensor([m], device=device)
            step_bias = model.morphogen.W_step_route(model.morphogen.step_embed(step_t))
            logits_m = logits_m + step_bias.unsqueeze(0)
        w_m = F.softmax(logits_m, dim=-1)
        step_routing.append(w_m[0, 0].clone())  # routing at pos 0
        
        delta_m = torch.zeros_like(h_curr)
        for k in range(cfg.n_rules):
            hidden = torch.tanh(model.morphogen.W_h1[k](h_cat_m))
            delta_k = model.morphogen.W_h2[k](hidden)
            delta_m = delta_m + w_m[:, :, k:k+1] * delta_k
        
        step_deltas.append(delta_m[0].clone())
        
        gate_m = torch.sigmoid(
            F.silu(model.morphogen.W_gate_a(h_curr)) * model.morphogen.W_gate_b(h_curr) + model.morphogen.gate_bias
        )
        h_curr = h_curr + gate_m * delta_m
    
    # Cosine similarity between consecutive step updates
    for i in range(len(step_deltas) - 1):
        d1 = step_deltas[i].flatten()
        d2 = step_deltas[i+1].flatten()
        cos = F.cosine_similarity(d1.unsqueeze(0), d2.unsqueeze(0))
        print(f"  Step {i}→{i+1}: update cos_sim={cos.item():.6f}")
    
    # Routing diversity across steps
    print("\n  Routing at pos 0 across steps:")
    for i, rw in enumerate(step_routing):
        print(f"    Step {i}: [{' '.join(f'{r:.3f}' for r in rw.tolist())}]")

# 7. Jacobian analysis — how much does each position attend to others?
print("\n[7] JACOBIAN / EFFECTIVE ATTENTION (V14.0: 99.7% self-attention)")
with torch.enable_grad():
    seq_len = 16  # small for Jacobian
    x_small = torch.randint(0, cfg.vocab_size, (1, seq_len), device=device)
    # Create leaf tensor we can get gradients from
    z_embed = model.token_embedding(x_small) + model.state_pe(seq_len).unsqueeze(0)
    z_small = z_embed.detach().clone().requires_grad_(True)
    
    h_s = model.morphogen.seed(z_small)
    sw = model.theta_readouts[0].get_scale_weights()
    h_out, _, _, _ = model.morphogen.evolve(h_s, z_small, cfg.n_morph_steps, sw)
    
    # Compute Jacobian: dh_i / dz_j
    J = torch.zeros(seq_len, seq_len, device=device)
    for i in range(seq_len):
        grad_out = torch.zeros_like(h_out)
        grad_out[0, i, :] = 1.0
        z_small.grad = None
        h_out.backward(grad_out, retain_graph=True)
        if z_small.grad is not None:
            J[i] = z_small.grad[0].norm(dim=-1)
    
    # Normalize rows
    J_norm = J / (J.sum(dim=1, keepdim=True) + 1e-8)
    
    self_attn = J_norm.diag().mean().item()
    off_diag_mean = J_norm[~torch.eye(seq_len, dtype=bool, device=device)].mean().item()
    
    # Entropy of attention distribution
    entropy = -(J_norm * (J_norm + 1e-8).log()).sum(dim=1).mean().item()
    max_ent = torch.tensor(seq_len, dtype=torch.float).log().item()
    
    print(f"  Self-attention fraction: {self_attn:.4f} (V14.0 was 0.9967)")
    print(f"  Off-diagonal mean: {off_diag_mean:.6f}")
    print(f"  Entropy: {entropy:.3f}/{max_ent:.3f} ({entropy/max_ent*100:.1f}%)")

# 8. FFN vs readout balance
print("\n[8] FFN vs READOUT BALANCE (V14.0: FFN 2.2x dominant)")
with torch.no_grad():
    readout = model.theta_readouts[0](h_final)  # from multi-scale
    z_after_readout = z + readout
    ffn_out = model.theta_ffns[0](z_after_readout)
    print(f"  Readout norm: {readout.norm():.2f}")
    print(f"  FFN norm:     {ffn_out.norm():.2f}")
    print(f"  Ratio: {ffn_out.norm()/readout.norm():.1f}x (V14.0 was 2.2x)")

print("\n" + "=" * 70)
print("  V14.1 ANALYSIS COMPLETE")
print("=" * 70)
