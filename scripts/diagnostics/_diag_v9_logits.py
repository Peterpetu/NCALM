"""Diagnose tied embedding logit explosion."""
import torch, math

torch.manual_seed(42)
d = 2048
V = 50304

# With 0.02 init (current)
emb = torch.randn(V, d) * 0.02
pe = torch.randn(1, d) * 0.01
z = emb[0:1] + pe
rms = z.pow(2).mean(dim=-1, keepdim=True).sqrt()
z_norm = z / rms
logits = z_norm @ emb.t()
print(f"self-dot product: {logits[0, 0]:.2f}")
print(f"logit mean: {logits.mean():.4f}, std: {logits.std():.4f}")
print(f"logit max: {logits.max():.2f}, min: {logits.min():.2f}")
print(f"Expected CE for uniform: {math.log(V):.2f}")
ce = torch.nn.functional.cross_entropy(logits, torch.tensor([0]))
print(f"Actual CE: {ce:.2f}")
print()

# The problem: after RMSNorm, z has ||z|| ~ sqrt(d) = 45.3
# So logits = z_norm @ emb.T where z_norm elements have std~1
# The self-dot product is z_norm[i] @ emb[i] which is z/rms @ emb[i]
# = (emb[i] + pe) / rms @ emb[i]
# This is biased high because emb[i] @ emb[i] ~ d * 0.02^2 = 0.819
# And rms = ||emb+pe|| / sqrt(d) ~ 0.022
# So self-dot ~ d * 0.02^2 / 0.022 = 0.819 / 0.022 = 37.2
print(f"Predicted self-dot: {d * 0.02**2 / 0.022:.1f}")
print()

# Fix: scale logits by 1/sqrt(d)
logits_scaled = logits / math.sqrt(d)
ce_scaled = torch.nn.functional.cross_entropy(logits_scaled, torch.tensor([0]))
print(f"[/sqrt(d)] CE: {ce_scaled:.2f}, logit max: {logits_scaled.max():.2f}")
