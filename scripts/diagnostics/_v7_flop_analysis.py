#!/usr/bin/env python3
"""V7 FLOP analysis — find optimal config for fair GPT-2 comparison."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ncalm.models.v7.config import V7Config
from ncalm.models.v7.model import AENCAv7Model

# GPT-2 (Pythia-160M) reference
d=768; L=12; dff=3072; V=50304; seq=1024
gpt2_per_layer = 4*d*d + 3*d*dff
gpt2_non_embed = L*gpt2_per_layer + 2*d
gpt2_embed = V*d
gpt2_total = gpt2_embed + gpt2_non_embed
gpt2_matmul = 2*gpt2_non_embed
gpt2_attn = L*2*2*seq*d
gpt2_flops = gpt2_matmul + gpt2_attn
print(f"GPT-2: {gpt2_total:,} params, {gpt2_flops/1e6:.0f}M FLOPs/tok")
print()

def analyze(n_th, t_g, d_fft, tie=False, target=None):
    cfg = {'d_model':768,'d_ff':768,'d_pe':64,'n_streams':8,'d_state':32,'d_css':256,
           'n_theta':n_th,'t_gamma':t_g,'n_heads':12,'d_head':64,'d_ff_theta':d_fft,
           'kernel_radius':2,'enable_halting':True,'halt_eps':0.01,'lambda_halt':0.01,
           'mos_components':1,'tie_embeddings':tie,'vocab_size':50304,'max_seq_len':1024,
           'lambda_reg':0.005,'lambda_aux':0.05,'lambda_agc':0.02,'checkpoint_interval':4,
           'dropout':0.0,'seq_len':1024}
    if target and d_fft==0:
        c = V7Config(**{k:v for k,v in cfg.items() if k in V7Config.__dataclass_fields__})
        cfg['d_ff_theta'] = c.compute_d_ff_theta(target)
    m = AENCAv7Model(cfg)
    params = sum(p.numel() for p in m.parameters())
    D=768; dki=768+256+64; df=768
    tg=n_th*t_g
    conv = 2*D*D*(2+1)
    css = 2*(D*256 + D*8*2 + 256*256) + 8*32*3
    kern = 2*(dki*df + 3*df*df + df*D + df*D)
    gamma_total = tg*(conv+css+kern)
    attn_p = 2*4*D*D; attn_c = 2*2*1024*D
    ffn = 2*3*D*cfg['d_ff_theta']
    theta_total = n_th*(attn_p+attn_c+ffn)
    flops = gamma_total+theta_total
    ratio = flops/gpt2_flops
    del m
    return params, flops, ratio, cfg['d_ff_theta']

hdr = f"{'Config':40} | {'Params':>10} | {'FLOPs/tok':>10} | {'Ratio':>6} | d_ff_theta"
print(hdr)
print("-"*len(hdr))
configs = [
    ("4t*3g, auto-162M (current)",    4,3,0,False,162_000_000),
    ("4t*3g, auto-162M, tied",        4,3,0,True,162_000_000),
    ("3t*3g, auto-162M",              3,3,0,False,162_000_000),
    ("3t*4g, auto-162M",              3,4,0,False,162_000_000),
    ("4t*2g, auto-162M",              4,2,0,False,162_000_000),
    ("3t*3g, d_ff=4096",              3,3,4096,False,None),
    ("3t*3g, d_ff=5120",              3,3,5120,False,None),
    ("3t*3g, d_ff=6144",              3,3,6144,False,None),
    ("4t*2g, d_ff=4096",              4,2,4096,False,None),
    ("4t*2g, d_ff=5120",              4,2,5120,False,None),
    ("3t*2g, auto-162M",              3,2,0,False,162_000_000),
    ("3t*2g, d_ff=8192",              3,2,8192,False,None),
    ("2t*3g, auto-162M",              2,3,0,False,162_000_000),
]
for name,nt,tg,dff,tie,tgt in configs:
    p,f,r,dfft = analyze(nt,tg,dff,tie,tgt)
    print(f"{name:40} | {p/1e6:>8.1f}M | {f/1e6:>8.0f}M | {r:>5.2f}x | {dfft}")
