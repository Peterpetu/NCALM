#!/usr/bin/env python3
"""Quick smoke test for V14 Emergent Dynamics architecture.

Validates:
  1. Model instantiation and shape correctness
  2. Zero-init cascade (fresh model produces near-zero residuals)
  3. Forward pass + backward pass without NaN/Inf
  4. Causality check (perturbing future tokens doesn't affect current output)
  5. Parameter count report
  6. Extra metrics work correctly
  7. d_ff auto-computation

Usage:
    python scripts/_eval_v14_smoke.py
"""
import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn

# This import triggers the @register_model("v14") decorator
from ncalm.models.v14.model import AENCAv14Model
from ncalm.models.v14.config import V14Config


def make_small_config() -> dict:
    """Create a small V14 config for quick testing (< 5M params)."""
    return {
        "d_model": 128,
        "d_h": 64,
        "n_rules": 4,
        "morph_rule_hidden": 0,  # = d_h
        "n_morph_steps": 6,
        "n_theta": 2,
        "d_ff": 0,               # auto-compute
        "vocab_size": 1024,
        "max_seq_len": 256,
        "enable_halting": True,
        "halt_eps": 0.01,
        "lambda_halt": 0.01,
        "tie_embeddings": False,
        "mos_components": 1,
        "morph_dilations": [1, 2, 4, 8, 16, 32],
        "morph_gate_bias": 3.0,
        "morph_conv_kernel": 3,
        "lambda_reg": 0.005,
        "lambda_aux": 0.05,
        "lambda_div": 0.01,
        "lambda_dyn": 0.001,
        "dropout": 0.0,
        "checkpoint_interval": 0,   # no checkpointing for test
        "target_params": 3_000_000,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 60)
    print("  V14 EMERGENT DYNAMICS — SMOKE TEST")
    print("=" * 60)

    # ── 1. INSTANTIATION ─────────────────────────────────────────────
    print("\n[1] Model Instantiation")
    cfg_dict = make_small_config()
    model = AENCAv14Model(cfg_dict)
    model = model.to(device)
    cfg = model.config

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  d_model={cfg.d_model}, d_h={cfg.d_h}, n_rules={cfg.n_rules}")
    print(f"  n_morph_steps={cfg.n_morph_steps}, n_theta={cfg.n_theta}")
    print(f"  d_ff={cfg.d_ff} (auto-computed)")
    print(f"  morph_dilations={list(cfg.morph_dilations)}")
    print(f"  effective_rule_hidden={cfg.effective_rule_hidden}")
    print(f"  Total params: {total_params:,}")
    print(f"  Target params: {cfg.target_params:,}")
    print(f"  OK: Model instantiated successfully")

    # ── 2. DESCRIBE ──────────────────────────────────────────────────
    print("\n[2] Architecture Description")
    desc = model.describe()
    for k, v in desc.items():
        print(f"  {k:30s} = {v}")

    # ── 3. ZERO-INIT CASCADE CHECK ───────────────────────────────────
    print("\n[3] Zero-Init Cascade Check")
    B, L = 2, 32
    x = torch.randint(0, cfg.vocab_size, (B, L + 1), device=device)

    z = model.token_embedding(x) + model.state_pe(L + 1).unsqueeze(0)
    z0 = z.clone()
    h = model.morphogen.seed(z)

    # Run one theta cycle
    z_after, h_after, routing, deltas = model._run_theta_cycle(z, h, 0)
    residual = (z_after - z0).norm() / z0.norm()
    print(f"  z residual after 1 theta cycle: {residual:.6f}")
    print(f"  Expected: near 0 (zero-init cascade)")
    if residual < 0.01:
        print(f"  OK: Zero-init cascade working (residual < 0.01)")
    else:
        print(f"  WARNING: Residual {residual:.6f} > 0.01 — cascade may be broken")

    # ── 4. FORWARD PASS ──────────────────────────────────────────────
    print("\n[4] Forward Pass (Training)")
    model.train()
    loss, info = model.forward_train(x)
    print(f"  loss = {loss.item():.4f}")
    print(f"  CE = {info['ce'].item():.4f}")
    for k, v in info.items():
        if k != "ce":
            val = v.item() if torch.is_tensor(v) else v
            print(f"  {k} = {val}")
    assert not torch.isnan(loss), "NaN loss!"
    assert not torch.isinf(loss), "Inf loss!"
    print(f"  OK: Forward pass clean (no NaN/Inf)")

    # ── 5. BACKWARD PASS ─────────────────────────────────────────────
    print("\n[5] Backward Pass")
    loss.backward()
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gn = param.grad.norm().item()
            grad_norms[name] = gn
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"  FAIL: NaN/Inf gradient in {name}")
                sys.exit(1)

    # Show largest and smallest grad norms
    sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)
    print(f"  Total params with grad: {len(grad_norms)}")
    print(f"  Top 5 grad norms:")
    for name, gn in sorted_grads[:5]:
        print(f"    {name:50s} = {gn:.6f}")
    print(f"  Bottom 5 grad norms:")
    for name, gn in sorted_grads[-5:]:
        print(f"    {name:50s} = {gn:.6f}")
    print(f"  OK: All gradients clean")

    # ── 6. CAUSALITY CHECK ───────────────────────────────────────────
    print("\n[6] Causality Check")
    model.eval()
    with torch.no_grad():
        # Original logits
        x_test = torch.randint(0, cfg.vocab_size, (1, 16), device=device)
        logits_orig = model.get_logits(x_test)

        # Perturb future token (position 10) and check position 5 unchanged
        x_perturbed = x_test.clone()
        x_perturbed[0, 10] = (x_perturbed[0, 10] + 1) % cfg.vocab_size
        logits_perturbed = model.get_logits(x_perturbed)

        # Positions 0-9 should be identical (causal = no future influence)
        diff_causal = (logits_orig[0, :10] - logits_perturbed[0, :10]).abs().max()
        # Position 10+ may differ
        diff_future = (logits_orig[0, 10:] - logits_perturbed[0, 10:]).abs().max()

    print(f"  Max diff at causal positions [0:10]: {diff_causal:.8f}")
    print(f"  Max diff at future positions [10:]:  {diff_future:.8f}")
    if diff_causal < 1e-4:
        print(f"  OK: Causality preserved")
    else:
        print(f"  WARNING: Causality may be violated (diff={diff_causal:.8f})")

    # ── 7. EXTRA METRICS ─────────────────────────────────────────────
    print("\n[7] Extra Metrics")
    m = model.extra_metrics()
    for k, v in sorted(m.items()):
        print(f"  {k:40s} = {v:.6f}")

    # ── 8. MORPHOGEN DYNAMICS ANALYSIS ───────────────────────────────
    print("\n[8] Morphogen Dynamics Analysis")
    model.eval()
    with torch.no_grad():
        x_dyn = torch.randint(0, cfg.vocab_size, (1, 32), device=device)
        z_dyn = model.token_embedding(x_dyn) + model.state_pe(32).unsqueeze(0)
        h_dyn = model.morphogen.seed(z_dyn)

        # Use multi-scale readout weights from theta 0
        scale_w = model.theta_readouts[0].get_scale_weights()
        h_evolved, h_multi, routing_weights, step_deltas = model.morphogen.evolve(
            h_dyn, z_dyn, cfg.n_morph_steps, scale_w,
        )

        h_delta = (h_evolved - h_dyn).norm() / h_dyn.norm()
        print(f"  Morphogen seed norm:    {h_dyn.norm():.4f}")
        print(f"  Morphogen evolved norm: {h_evolved.norm():.4f}")
        print(f"  h_multi norm:           {h_multi.norm():.4f}")
        print(f"  Relative evolution:     {h_delta:.4f} ({h_delta*100:.1f}%)")

        # Multi-scale readout weights per theta (Phase C)
        print(f"\n  Multi-Scale Readout Weights:")
        for t in range(cfg.n_theta):
            sw = model.theta_readouts[t].get_scale_weights()
            sw_list = [f"{w:.4f}" for w in sw.cpu().tolist()]
            ent = -(sw * (sw + 1e-8).log()).sum().item()
            print(f"    Theta {t}: [{', '.join(sw_list)}] entropy={ent:.4f}")

        for i, (rw, sd) in enumerate(zip(routing_weights, step_deltas)):
            dilation = cfg.morph_dilations[i % len(cfg.morph_dilations)]
            route_entropy = -(rw * (rw + 1e-8).log()).sum(dim=-1).mean()
            delta_norm = sd.mean()
            print(
                f"  Step {i:2d} (d={dilation:4d}): "
                f"route_entropy={route_entropy:.4f} "
                f"delta_norm={delta_norm:.6f} "
                f"route_example={rw[0, 0].cpu().tolist()}"
            )

    # ── 9. INFERENCE SHAPES ──────────────────────────────────────────
    print("\n[9] Inference Shape Check")
    with torch.no_grad():
        x_inf = torch.randint(0, cfg.vocab_size, (2, 64), device=device)
        logits_inf = model.get_logits(x_inf)
        print(f"  Input shape:  {tuple(x_inf.shape)}")
        print(f"  Output shape: {tuple(logits_inf.shape)}")
        expected = (2, 64, cfg.vocab_size)
        assert logits_inf.shape == expected, f"Shape mismatch: {logits_inf.shape} != {expected}"
        print(f"  OK: Shapes correct")

    # ── 10. PARAMETER GROUP CHECK ────────────────────────────────────
    print("\n[10] Parameter Groups")
    groups = model.get_param_groups()
    for g in groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['label']:15s}: {n:>10,} params ({len(g['params'])} tensors)")

    # ── VERDICT ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)
    print(f"  Instantiation:      PASS")
    print(f"  Zero-init cascade:  {'PASS' if residual < 0.01 else 'WARN'}")
    print(f"  Forward pass:       PASS")
    print(f"  Backward pass:      PASS")
    print(f"  Causality:          {'PASS' if diff_causal < 1e-4 else 'WARN'}")
    print(f"  Shapes:             PASS")

    all_pass = residual < 0.01 and diff_causal < 1e-4
    if all_pass:
        print(f"\n  ALL CHECKS PASSED — V14 is ready for training")
    else:
        print(f"\n  WARNINGS present — review before training")


if __name__ == "__main__":
    main()
