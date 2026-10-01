#!/usr/bin/env python3
"""Verify V14 parameter counting at full 162M scale."""
import sys
sys.path.insert(0, ".")

from ncalm.models.v14.config import V14Config
from ncalm.models.v14.model import AENCAv14Model

# Full default config (162M target)
cfg = V14Config()
print("=== FULL SCALE CONFIG ===")
print(f"d_model={cfg.d_model}, d_h={cfg.d_h}, n_rules={cfg.n_rules}")
print(f"n_morph_steps={cfg.n_morph_steps}, n_theta={cfg.n_theta}")
print(f"d_ff={cfg.d_ff} (auto-computed)")
print(f"effective_rule_hidden={cfg.effective_rule_hidden}")
print(f"target_params={cfg.target_params:,}")

# Instantiate model and verify
model = AENCAv14Model({})
total = sum(p.numel() for p in model.parameters())
print(f"\nActual total params: {total:,}")
print(f"Target: {cfg.target_params:,}")
print(f"Difference: {total - cfg.target_params:,} ({(total-cfg.target_params)/cfg.target_params*100:.2f}%)")

# Breakdown
desc = model.describe()
for k, v in desc.items():
    if "params" in k:
        print(f"  {k}: {v:,}")

# Detailed per-module breakdown
print("\n=== DETAILED MODULE BREAKDOWN ===")
for name, module in model.named_children():
    n = sum(p.numel() for p in module.parameters())
    print(f"  {name:30s}: {n:>12,}")

# Check if d_ff makes the FFN reasonable
print(f"\n=== FFN ANALYSIS ===")
print(f"d_ff = {cfg.d_ff}")
print(f"FFN expansion ratio = {cfg.d_ff / cfg.d_model:.2f}x")
print(f"FFN params per theta = {3 * cfg.d_model * cfg.d_ff + cfg.d_model:,}")
print(f"Total FFN params     = {cfg.n_theta * (3 * cfg.d_model * cfg.d_ff + cfg.d_model):,}")

# Channel/neuron ratio analysis
print(f"\n=== NCA DYNAMICS ANALYSIS ===")
print(f"d_h (state channels) = {cfg.d_h}")
print(f"rule_hidden (neurons) = {cfg.effective_rule_hidden}")
print(f"channel/neuron ratio = {cfg.effective_rule_hidden / cfg.d_h:.2f}")
print(f"Xu et al. recommends ratio > 1.0 for dynamic behavior")
print(f"Current: {'DYNAMIC' if cfg.effective_rule_hidden > cfg.d_h else 'BORDERLINE STATIC'}")

# Receptive field analysis
print(f"\n=== RECEPTIVE FIELD ===")
kernel = cfg.morph_conv_kernel
dilations = cfg.morph_dilations[:cfg.n_morph_steps]
total_rf = sum(d * (kernel - 1) for d in dilations) + 1
print(f"Kernel size = {kernel}")
print(f"Steps used = {cfg.n_morph_steps}")
print(f"Dilations used = {list(dilations)}")
print(f"Total receptive field = {total_rf}")
print(f"Max seq len = {cfg.max_seq_len}")
print(f"Coverage: {'FULL' if total_rf >= cfg.max_seq_len else 'PARTIAL'}")
