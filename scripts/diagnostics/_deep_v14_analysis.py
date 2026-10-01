#!/usr/bin/env python3
"""Deep analysis: Is morphogen doing ANYTHING position-dependent?"""
import sys
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F

from ncalm.models.v14.model import AENCAv14Model

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(
    "experiments/AENCA v14 (Emergent Dynamics)/v14_smoke_20min_20260331_124646/best_model.pt",
    map_location=device, weights_only=False
)
model = AENCAv14Model(ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device).eval()
cfg = model.config

print("=" * 70)
print("  DEEP DYNAMICS ANALYSIS")
print("=" * 70)

# 1. Check if morphogen h varies across positions
print("\n[1] POSITION VARIANCE IN MORPHOGEN")
x = torch.randint(0, cfg.vocab_size, (1, 64), device=device)
with torch.no_grad():
    z = model.token_embedding(x) + model.state_pe(64).unsqueeze(0)
    h = model.morphogen.seed(z)
    
    print(f"  Seed h std across positions: {h[0].std(dim=0).mean():.4f}")
    print(f"  Seed h std across channels:  {h[0].std(dim=1).mean():.4f}")
    
    sw = model.theta_readouts[0].get_scale_weights()
    h_final, h_multi, routing, deltas = model.morphogen.evolve(h, z, cfg.n_morph_steps, sw)
    
    print(f"  Final h std across positions: {h_final[0].std(dim=0).mean():.4f}")
    print(f"  Final h std across channels:  {h_final[0].std(dim=1).mean():.4f}")

# 2. Check what the dominant rule is actually computing
print("\n[2] RULE OUTPUT ANALYSIS")
with torch.no_grad():
    z_proj = model.morphogen.W_content_proj(model.morphogen.norm_z_proj(z))
    h_nbr = model.morphogen.morph_conv(h, dilation=1)
    c_m = model.morphogen.content_conv(z_proj, dilation=1)
    h_cat = model.morphogen.rule_norm(torch.cat([h, h_nbr, c_m], dim=-1))
    
    for k in range(cfg.n_rules):
        hidden = torch.tanh(model.morphogen.W_h1[k](h_cat))
        delta_k = model.morphogen.W_h2[k](hidden)
        print(f"  Rule {k}: ||delta||={delta_k.norm():.2f}, mean={delta_k.mean():.4f}, std={delta_k.std():.4f}, W_h2 norm={model.morphogen.W_h2[k].weight.norm():.4f}")

# 3. What does the gate look like?
print("\n[3] GATE ANALYSIS")
with torch.no_grad():
    gate = torch.sigmoid(
        F.silu(model.morphogen.W_gate_a(h)) * model.morphogen.W_gate_b(h) + model.morphogen.gate_bias
    )
    print(f"  Gate mean: {gate.mean():.4f}")
    print(f"  Gate min:  {gate.min():.4f}")
    print(f"  Gate max:  {gate.max():.4f}")
    print(f"  Gate std across positions: {gate[0].std(dim=0).mean():.6f}")
    print(f"  Gate bias value: {model.morphogen.gate_bias.mean():.4f}")

# 4. Router analysis - what determines rule 2 wins?
print("\n[4] ROUTER WEIGHT ANALYSIS")
with torch.no_grad():
    W = model.morphogen.W_router.weight  # (K, d_h)
    print(f"  Router weight shape: {W.shape}")
    for k in range(cfg.n_rules):
        print(f"  Rule {k}: W_router row norm={W[k].norm():.6f}, mean={W[k].mean():.6f}")
    
    # What logits does the router produce?
    logits = model.morphogen.W_router(h)  # (B, L, K)
    probs = F.softmax(logits, dim=-1)
    print(f"\n  Router logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"  Router logits for pos 0: {logits[0, 0].tolist()}")
    print(f"  Router probs  for pos 0: {[f'{p:.6f}' for p in probs[0, 0].tolist()]}")

# 5. Content influence across different dilations
print("\n[5] CONTENT INFLUENCE BY DILATION")
with torch.no_grad():
    for d in [1, 2, 4, 8, 32, 128, 512, 2048]:
        c_d = model.morphogen.content_conv(z_proj, dilation=d)
        # How much does Conv change the content at this dilation?
        identity_diff = (c_d - z_proj).norm() / z_proj.norm()
        print(f"  dil={d:5d}: content_conv output norm={c_d.norm():.2f}, relative change from identity={identity_diff:.4f}")

# 6. Does the readout actually carry morphogen information?
print("\n[6] READOUT INFORMATION CONTENT")
with torch.no_grad():
    readout_0 = model.theta_readouts[0](h_multi)
    readout_final = model.theta_readouts[0](h_final)
    print(f"  h_multi norm:  {h_multi.norm():.2f}")
    print(f"  readout norm:  {readout_0.norm():.4f}")
    print(f"  z norm:        {z.norm():.2f}")
    print(f"  readout/z ratio: {readout_0.norm()/z.norm():.6f}")
    print(f"  W_readout spectral norm approx: {model.theta_readouts[0].W_readout.weight.norm():.4f}")
    
    # How much does the FFN contribute vs readout?
    z_after_readout = z + readout_0
    ffn_out = model.theta_ffns[0](z_after_readout)
    print(f"  FFN output norm: {ffn_out.norm():.4f}")
    print(f"  FFN/readout ratio: {ffn_out.norm()/readout_0.norm():.1f}x")

# 7. Does the morphogen state carry position-specific information?
print("\n[7] POSITION-DEPENDENT INFORMATION IN MORPHOGEN")
with torch.no_grad():
    # Measure cosine sim between morphogen states at different positions
    h_norm = F.normalize(h_final[0], dim=-1)  # (L, d_h)
    sim_matrix = h_norm @ h_norm.T  # (L, L)
    # Diagonal = 1
    off_diag = sim_matrix[~torch.eye(64, dtype=bool, device=device)]
    print(f"  Mean off-diagonal cosine sim: {off_diag.mean():.6f}")
    print(f"  Min off-diagonal cosine sim:  {off_diag.min():.6f}")
    print(f"  Max off-diagonal cosine sim:  {off_diag.max():.6f}")
    print(f"  (1.0 = all positions identical, 0.0 = fully differentiated)")

print("\n" + "=" * 70)
print("  ANALYSIS COMPLETE")
print("=" * 70)
