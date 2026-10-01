#!/usr/bin/env python3
"""Verify V7 250M profile configuration."""
import sys
sys.path.insert(0, "c:/koodi/NCALM")

from ncalm.profiles import PROFILES
from ncalm.models.v7.model import AENCAv7Model
from ncalm.models.v7.config import V7Config

p = dict(PROFILES["pile_250m_v7"])
p["vocab_size"] = 50304  # Set correct vocab (trainer does this from tokenizer)
m = AENCAv7Model(p)
params = sum(x.numel() for x in m.parameters())

c = V7Config(**{k: v for k, v in p.items() if k in V7Config.__dataclass_fields__})
d_ff_theta = c.compute_d_ff_theta(p["target_params"])

print(f"Profile:    pile_250m_v7")
print(f"n_theta:    {p['n_theta']}")
print(f"t_gamma:    {p['t_gamma']}")
print(f"d_ff_theta: {d_ff_theta} (auto-sized)")
print(f"Params:     {params:,} ({params/1e6:.1f}M)")
print(f"Eff depth:  {p['n_theta']} x ({p['t_gamma']} kernel + 1 attn+FFN) = {p['n_theta']*(p['t_gamma']+1)} ops")
print(f"Steps:      {p['max_steps']:,}")
print(f"Budget:     {p['max_steps']*p['batch_size']*p['seq_len']*p['grad_accum_steps']:,} tokens")
print()
desc = m.describe()
for k, v in desc.items():
    print(f"  {k}: {v}")
print()
print("Run: python scripts/train.py --profile pile_250m_v7 --model v7")
