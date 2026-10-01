"""Quick param count verification for Strategy F configs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ncalm.models.v8.config import V8Config

# ── Strategy F: Production 160M Config ──
cfg_prod = V8Config(d_model=768, d_concept=2560, n_theta=6, t_gamma=3,
                    n_heads=12, d_head=64, vocab_size=50304, max_seq_len=1024,
                    tie_embeddings=True, kernel_radius=2)
ff_prod = cfg_prod.compute_d_ff_theta(162_000_000)
print(f"Production d_ff_theta (auto): {ff_prod}")

# ── Strategy F Smoke: Proportional Mini ──
d, df, dc, V, n_th = 128, 129, 432, 10000, 6
embed = V * d
readout = V * d  # untied for smoke
kernel = dc * 5 * df + 3 * df + df
conv = df * df * 2 + df  # kernel_radius=1
attn_per = 3*d*d + d*d + 3*d
ffn_per = 3 * d * 512
theta_per = attn_per + ffn_per
halt = d + 1 + d
total = embed + readout + kernel + conv + n_th * theta_per + halt + d

print(f"\nSmoke config (Strategy F proportions):")
print(f"  Embed:          {embed:>10,}  ({embed/total*100:.1f}%)")
print(f"  Readout:        {readout:>10,}  ({readout/total*100:.1f}%)")
print(f"  Hebbian kernel: {kernel:>10,}  ({kernel/total*100:.1f}%)")
print(f"  Conv:           {conv:>10,}  ({conv/total*100:.1f}%)")
print(f"  6x Theta:       {n_th*theta_per:>10,}  ({n_th*theta_per/total*100:.1f}%)")
print(f"    attn/block:   {attn_per:>10,}")
print(f"    ffn/block:    {ffn_per:>10,}")
print(f"  Halt:           {halt:>10,}")
print(f"  TOTAL:          {total:>10,}")
print(f"  dc/d ratio:     {dc/d:.2f} (prod: {2560/768:.2f})")
print(f"  d_ff/d ratio:   {512/d:.2f} (prod: {ff_prod/768:.2f})")
print(f"  Eff depth:      {n_th*(3+1)} steps")
print(f"  Total gamma:    {n_th*3} shared kernel apps")
print(f"  Kernel reuse:   {n_th*3*256:,}x per forward")

# FLOP estimate for smoke (L=256)
L = 256
fg = L * (10*dc*df+dc) * n_th * 3
ft = n_th * (2*L*4*d*d + 2*L*L*d + 2*L*3*d*512)
print(f"\nSmoke FLOPs: gamma={fg/1e6:.0f}M + theta={ft/1e6:.0f}M = {(fg+ft)/1e6:.0f}M")
print(f"Token budget: 2000 * 2 * 256 = {2000*2*256:,}")
