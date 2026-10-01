#!/usr/bin/env python3
"""Quick evaluation of V13 smoke training results."""
import sys
sys.path.insert(0, ".")

import torch
from ncalm.models.v13.model import AENCAv13Model

ckpt_path = "experiments/AENCA v13 (Morphogenetic Attention)/v13_smoke_audit_20260330_144722/best_model.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = AENCAv13Model(ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("=" * 60)
print("  V13 SMOKE TRAINING — POST-TRAINING ANALYSIS")
print("=" * 60)

# 1. Extra metrics
m = model.extra_metrics()
print("\n[1] Weight Norms (zero-init → should grow during training)")
for k, v in sorted(m.items()):
    print(f"  {k:30s} = {v:.6f}")

# 2. Gate activations on real input
print("\n[2] Gate Activations on Random Input")
x = torch.randint(0, 10000, (1, 64))
z = model.token_embedding(x) + model.state_pe(64).unsqueeze(0)
h = model.morphogen.seed(z)
h_evolved, rw = model.morphogen.evolve(h, model.config.n_morph_steps)
q_gate, k_gate = model.morphogen.generate_gates(h_evolved)

print(f"  q_gate: mean={q_gate.mean():.4f} std={q_gate.std():.4f} "
      f"min={q_gate.min():.4f} max={q_gate.max():.4f}")
print(f"  k_gate: mean={k_gate.mean():.4f} std={k_gate.std():.4f} "
      f"min={k_gate.min():.4f} max={k_gate.max():.4f}")

# 3. Routing weights diversity
print("\n[3] Routing Weight Diversity (K=2 rules)")
for i, w in enumerate(rw):
    rule_std = w.std(dim=-1).mean()
    rule_max = w.max(dim=-1).values.mean()
    print(f"  Step {i}: rule_std={rule_std:.4f} max_weight={rule_max:.4f} "
          f"example={w[0, 0].tolist()}")

# 4. Morphogen evolution delta
h_delta = (h_evolved - h).norm() / h.norm()
print(f"\n[4] Morphogen Evolution")
print(f"  h_seed norm      = {h.norm():.4f}")
print(f"  h_evolved norm   = {h_evolved.norm():.4f}")
print(f"  relative delta   = {h_delta:.4f} ({h_delta*100:.1f}%)")

# 5. Feedback effect
h_fb = model.morphogen.update_from_content(h_evolved, z)
fb_delta = (h_fb - h_evolved).norm() / h_evolved.norm()
print(f"  feedback delta   = {fb_delta:.4f} ({fb_delta*100:.1f}%)")

# 6. Halting analysis
print(f"\n[5] Halting")
print(f"  halt_bias = {model.halting.W_halt.bias.item():.4f}")
hp = model.halting.compute_halt_prob(z)
print(f"  halt_prob: mean={hp.mean():.4f} std={hp.std():.4f}")

print("\n" + "=" * 60)
print("  VERDICT")
print("=" * 60)

# Check if morphogen is active
morph_active = any(v > 0.01 for k, v in m.items() if "morph_rule" in k)
gates_deviated = q_gate.std() > 0.01
routing_diverse = all(w.std(dim=-1).mean() > 0.01 for w in rw)

print(f"  Morphogen rules active:  {'YES' if morph_active else 'NO'}")
print(f"  Gates deviated from 1.0: {'YES' if gates_deviated else 'NO'}")
print(f"  Routing diverse:         {'YES' if routing_diverse else 'NO (collapsed)'}")
print(f"  Training healthy:        {'YES' if morph_active and gates_deviated else 'NEEDS INVESTIGATION'}")
