"""Quick preflight check for pile_2b_v7 profile."""
from ncalm.profiles import PROFILES
from ncalm.model import get_model_class
import ncalm.models.v7.model  # noqa: ensure registration

p = dict(PROFILES["pile_2b_v7"])
p["vocab_size"] = p.get("tokenizer_vocab_size", 50304)

cls = get_model_class("v7")
m = cls(p)
total = sum(par.numel() for par in m.parameters())
d = m.describe()

print(f"=== pile_2b_v7 Preflight ===")
print(f"Total params:    {total:,}")
print(f"d_ff_theta auto: {d['d_ff_theta']}")
print(f"Architecture:    {d['architecture']}")
print(f"n_theta:         {d['n_theta']}")
print(f"t_gamma:         {d['t_gamma']}")
print(f"d_concept:       {d['d_concept']}")
print(f"n_heads:         {d['n_heads']}")

tps = p["batch_size"] * p["seq_len"] * p["grad_accum_steps"]
print(f"\nmax_steps:       {p['max_steps']:,}")
print(f"tokens/step:     {tps:,}")
print(f"total tokens:    {p['max_steps'] * tps:,}")
print(f"dataset cache:   {p['dataset_cache_name']}")

import os
cache_path = os.path.join("data_cache", p["dataset_cache_name"])
if os.path.exists(cache_path):
    size_gb = os.path.getsize(cache_path) / 1e9
    print(f"cache file:      EXISTS ({size_gb:.1f} GB)")
else:
    print(f"cache file:      MISSING — run prepare_longrun_assets.py first!")

# GPT-2 baseline for comparison
print(f"\n=== GPT-2 2B baseline ===")
print(f"Best val PPL:    14.45")
print(f"Training time:   ~11 hours")
print(f"\nV7.1 must beat PPL 14.45 to win.")
print(f"\nRun command:")
print(f"  python scripts/train.py --profile pile_2b_v7 --model v7")
