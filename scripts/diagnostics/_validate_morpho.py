"""Quick validation: Morphogenesis smoke profile builds and trains."""
import torch
from ncalm.profiles import PROFILES
from ncalm.models.v8.config import V8Config
from ncalm.models.v8.model import AENCAv8Model

p = PROFILES["smoke_morpho"]
cfg = V8Config(**{k: v for k, v in p.items() if hasattr(V8Config, k)})
print(f"Config: d={cfg.d_model}, dc={cfg.d_concept}, theta={cfg.n_theta}, gamma={cfg.t_gamma}")
print(f"d_ff_theta={cfg.d_ff_theta}, eff_depth={cfg.n_theta*(cfg.t_gamma+1)}")

model = AENCAv8Model(p)
total = sum(p2.numel() for p2 in model.parameters())
print(f"Total params: {total:,} ({total/1e6:.1f}M)")

desc = model.describe()
for k, v in desc.items():
    if "params" in k:
        print(f"  {k}: {v:,}")

# Quick forward pass
batch = torch.randint(0, cfg.vocab_size, (2, 256))
model.train()
model.on_step_start(0, 2000)
loss, info = model.forward_train(batch)
print(f"Forward OK: loss={loss.item():.4f}")
loss.backward()
grad_norm = sum(
    p2.grad.norm().item() ** 2 for p2 in model.parameters() if p2.grad is not None
) ** 0.5
print(f"Backward OK: grad_norm={grad_norm:.4f}")

# Also verify 250M profile builds a config (can't instantiate - too big for memory)
p250m = PROFILES["pile_250m_morpho"]
cfg250 = V8Config(**{k: v for k, v in p250m.items() if hasattr(V8Config, k)})
d_ff = cfg250.compute_d_ff_theta(p250m["target_params"])
print(f"\n250M profile: d={cfg250.d_model}, dc={cfg250.d_concept}, theta={cfg250.n_theta}, gamma={cfg250.t_gamma}")
print(f"  Auto d_ff_theta={d_ff} (ratio: {d_ff/cfg250.d_model:.2f}x)")

# And 250B
p250b = PROFILES["morpho_250b"]
cfg250b = V8Config(**{k: v for k, v in p250b.items() if hasattr(V8Config, k)})
d_ff_250b = cfg250b.compute_d_ff_theta(p250b["target_params"])
print(f"\n250B profile: d={cfg250b.d_model}, dc={cfg250b.d_concept}, theta={cfg250b.n_theta}, gamma={cfg250b.t_gamma}")
print(f"  Auto d_ff_theta={d_ff_250b} (ratio: {d_ff_250b/cfg250b.d_model:.2f}x)")
shared_kernel = 5 * cfg250b.d_concept * cfg250b.d_full
attn_per_block = 4 * cfg250b.d_model ** 2
ffn_per_block = 3 * cfg250b.d_model * d_ff_250b
embed = cfg250b.vocab_size * cfg250b.d_model
conv = cfg250b.d_full ** 2 * (cfg250b.kernel_radius + 1)
total_250b = embed + conv + shared_kernel + cfg250b.n_theta * (attn_per_block + ffn_per_block)
print(f"  Estimated total: {total_250b/1e9:.1f}B params")
print(f"  Kernel reuse: {cfg250b.n_theta * cfg250b.t_gamma * cfg250b.max_seq_len:,}x per sample")

print("\nAll Morphogenesis profiles validated!")
