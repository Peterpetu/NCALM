"""V8 scaling analysis: 160M parameter budget strategies."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ncalm.models.v8.config import V8Config

print("=" * 78)
print("V8 SCALING ANALYSIS FOR 160M PARAMETER TARGET")
print("=" * 78)

strategies = {}

# A: Spec default
auto_a = V8Config(d_model=768, d_concept=2048, n_theta=4, t_gamma=2,
                  n_heads=12, d_head=64, vocab_size=50304, max_seq_len=1024,
                  tie_embeddings=True).compute_d_ff_theta(162_000_000)
strategies["A: Spec default (d=768, theta=4, gamma=2)"] = V8Config(
    d_model=768, d_concept=2048, n_theta=4, t_gamma=2,
    n_heads=12, d_head=64, vocab_size=50304, max_seq_len=1024,
    tie_embeddings=True, d_ff_theta=auto_a)

# B: Deeper growth (more gamma = more CA dynamics)
auto_b = V8Config(d_model=768, d_concept=2048, n_theta=4, t_gamma=4,
                  n_heads=12, d_head=64, vocab_size=50304, max_seq_len=1024,
                  tie_embeddings=True).compute_d_ff_theta(162_000_000)
strategies["B: Deeper growth (d=768, theta=4, gamma=4)"] = V8Config(
    d_model=768, d_concept=2048, n_theta=4, t_gamma=4,
    n_heads=12, d_head=64, vocab_size=50304, max_seq_len=1024,
    tie_embeddings=True, d_ff_theta=auto_b)

# C: More theta cycles (more global integration points)
auto_c = V8Config(d_model=768, d_concept=2048, n_theta=8, t_gamma=2,
                  n_heads=12, d_head=64, vocab_size=50304, max_seq_len=1024,
                  tie_embeddings=True).compute_d_ff_theta(162_000_000)
strategies["C: More theta (d=768, theta=8, gamma=2)"] = V8Config(
    d_model=768, d_concept=2048, n_theta=8, t_gamma=2,
    n_heads=12, d_head=64, vocab_size=50304, max_seq_len=1024,
    tie_embeddings=True, d_ff_theta=auto_c)

# D: Wider Hebbian kernel (richer concept space)
auto_d = V8Config(d_model=768, d_concept=3072, n_theta=4, t_gamma=2,
                  n_heads=12, d_head=64, vocab_size=50304, max_seq_len=1024,
                  tie_embeddings=True).compute_d_ff_theta(162_000_000)
strategies["D: Wider kernel (d=768, dc=3072, theta=4, gamma=2)"] = V8Config(
    d_model=768, d_concept=3072, n_theta=4, t_gamma=2,
    n_heads=12, d_head=64, vocab_size=50304, max_seq_len=1024,
    tie_embeddings=True, d_ff_theta=auto_d)

# E: Maximum growth (24 gamma steps total)
auto_e = V8Config(d_model=768, d_concept=2048, n_theta=4, t_gamma=6,
                  n_heads=12, d_head=64, vocab_size=50304, max_seq_len=1024,
                  tie_embeddings=True).compute_d_ff_theta(162_000_000)
strategies["E: Max growth (d=768, theta=4, gamma=6)"] = V8Config(
    d_model=768, d_concept=2048, n_theta=4, t_gamma=6,
    n_heads=12, d_head=64, vocab_size=50304, max_seq_len=1024,
    tie_embeddings=True, d_ff_theta=auto_e)

# F: Balanced (wider kernel + more theta + more gamma)
auto_f = V8Config(d_model=768, d_concept=2560, n_theta=6, t_gamma=3,
                  n_heads=12, d_head=64, vocab_size=50304, max_seq_len=1024,
                  tie_embeddings=True).compute_d_ff_theta(162_000_000)
strategies["F: Balanced (d=768, dc=2560, theta=6, gamma=3)"] = V8Config(
    d_model=768, d_concept=2560, n_theta=6, t_gamma=3,
    n_heads=12, d_head=64, vocab_size=50304, max_seq_len=1024,
    tie_embeddings=True, d_ff_theta=auto_f)

print()
for name, cfg in strategies.items():
    d = cfg.d_model
    df = cfg.d_full
    dc = cfg.d_concept
    V = cfg.vocab_size
    n_th = cfg.n_theta
    t_g = cfg.t_gamma

    # Parameter count
    embed = V * d
    kernel = dc * 5 * df + 3 * df + df
    conv = df * df * (cfg.kernel_radius + 1) + df
    attn_per = 3 * d * d + d * d + 3 * d
    ffn_per = 3 * d * cfg.d_ff_theta
    theta_per = attn_per + ffn_per
    halt = d + 1 + d
    total = embed + kernel + conv + n_th * theta_per + halt + d

    # FLOPs (dense, L=1024)
    L = 1024
    flop_gamma_cell = 10 * dc * df + dc
    flop_gamma = L * flop_gamma_cell * n_th * t_g
    flop_theta_per = 2*L*4*d*d + 2*L*L*d + 2*L*3*d*cfg.d_ff_theta
    flop_theta = n_th * flop_theta_per
    flop_total = flop_gamma + flop_theta

    eff_depth = n_th * (t_g + 1)
    gamma_reuse = n_th * t_g * L

    print(f"--- {name} ---")
    print(f"  d_model={d}  d_concept={dc}  n_theta={n_th}  t_gamma={t_g}  d_ff_theta={cfg.d_ff_theta}")
    print(f"  Total params:     {total:>12,}")
    print(f"  Embed:            {embed:>12,}  ({embed/total*100:.1f}%)")
    print(f"  Hebbian kernel:   {kernel:>12,}  ({kernel/total*100:.1f}%)")
    print(f"  Conv:             {conv:>12,}  ({conv/total*100:.1f}%)")
    print(f"  Theta blocks:     {n_th*theta_per:>12,}  ({n_th*theta_per/total*100:.1f}%) [{n_th}x {theta_per:,} each]")
    print(f"    attn/block:     {attn_per:>12,}")
    print(f"    FFN/block:      {ffn_per:>12,}")
    print(f"  Halt:             {halt:>12,}")
    print(f"  Effective depth:  {eff_depth} steps ({n_th}th x ({t_g}g+1th))")
    print(f"  Total gamma:      {n_th*t_g} shared kernel apps")
    print(f"  Kernel reuse:     {gamma_reuse:,}x per forward (L={L})")
    print(f"  Dense FLOPs:      {flop_total/1e9:.1f} GFLOPs")
    print(f"    gamma FLOPs:    {flop_gamma/1e9:.1f}G ({flop_gamma/flop_total*100:.0f}%)")
    print(f"    theta FLOPs:    {flop_theta/1e9:.1f}G ({flop_theta/flop_total*100:.0f}%)")
    print(f"  FLOPs/param:      {flop_total/total:.0f}")

    # Sparsity potential
    # With 50% settlement by mid-training: gamma FLOPs roughly halve
    sparse_gamma = flop_gamma * 0.5
    sparse_total = sparse_gamma + flop_theta
    print(f"  Sparse FLOPs:     {sparse_total/1e9:.1f}G (50% settlement)")
    print()

# GPT-2 reference
print("--- GPT-2 Small (REFERENCE) ---")
gd, gl, gff, gV = 768, 12, 3072, 50257
ge = gV * gd
glayer = 4*gd*gd + 2*gd*gff*2
gtotal = ge + gl * glayer + gd
L = 1024
gflop_layer = 2*L*4*gd*gd + 2*L*L*gd + 2*L*2*gd*gff
gflop = gl * gflop_layer
print(f"  d=768, layers=12, ff=3072, V=50257")
print(f"  Total params:     {gtotal:>12,}")
print(f"  Dense FLOPs:      {gflop/1e9:.1f} GFLOPs")
print(f"  Depth:            {gl} (fixed, independent)")
print(f"  FLOPs/param:      {gflop/gtotal:.0f}")
print(f"  Reuse per layer:  {L:,}x (seq length only)")
print()

# Key comparison table
print("=" * 78)
print("COMPARISON: FLOPs vs GPT-2 at same param count")
print("=" * 78)
print(f"{'Strategy':<45} {'Params':>8} {'Dense GF':>9} {'Sparse GF':>10} {'vs GPT2':>8}")
print("-" * 78)

for name, cfg in strategies.items():
    d = cfg.d_model; df = cfg.d_full; dc = cfg.d_concept
    n_th = cfg.n_theta; t_g = cfg.t_gamma
    embed = cfg.vocab_size * d
    kernel = dc * 5 * df + 3 * df + df
    conv = df * df * (cfg.kernel_radius + 1) + df
    attn_per = 3*d*d + d*d + 3*d
    ffn_per = 3 * d * cfg.d_ff_theta
    theta_per = attn_per + ffn_per
    total = embed + kernel + conv + n_th * theta_per + d + 1 + d + d

    L = 1024
    fg = L * (10*dc*df+dc) * n_th * t_g
    ft_per = 2*L*4*d*d + 2*L*L*d + 2*L*3*d*cfg.d_ff_theta
    ft = n_th * ft_per
    dense = fg + ft
    sparse = fg*0.5 + ft

    short = name.split("(")[0].strip()
    print(f"  {short:<43} {total/1e6:>6.1f}M {dense/1e9:>8.1f}G {sparse/1e9:>9.1f}G {dense/gflop:>7.2f}x")

print(f"  {'GPT-2 Small':<43} {gtotal/1e6:>6.1f}M {gflop/1e9:>8.1f}G {'N/A':>10} {'1.00':>7}x")
