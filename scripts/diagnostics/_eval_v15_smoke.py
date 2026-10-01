#!/usr/bin/env python3
"""Smoke evaluation & diagnostic for AENCA V15 (2D RD-NCA).

Verifies Phase 1 critical metrics:
  1. Zero-init cascade: z_out ≈ z₀ at initialization
  2. Shape correctness: all tensors match spec
  3. Fold/unfold roundtrip: fold(unfold(x)) == x
  4. Causal mask correctness: verify mask patterns
  5. Position cosine similarity (target < 0.99, V14: 1.000)
  6. Step update diversity (target < 0.99, V14: 0.999998)
  7. Routing entropy (target > 0.5, V14.0: 0.021)
  8. Gradient stability: no NaN, stable norms

Usage:
    python scripts/_eval_v15_smoke.py                    # Fresh model diagnostics
    python scripts/_eval_v15_smoke.py path/to/ckpt.pt    # Trained model analysis
"""
import sys
sys.path.insert(0, ".")

import math
import torch
import torch.nn.functional as F


def test_causal_masks():
    """Verify causal 2D masks match the plan spec."""
    from ncalm.models.v15.reaction_diffusion import build_causal_2d_mask

    # 3×3 mask: M₃ = [[1,1,1],[1,1,0],[0,0,0]]
    m3 = build_causal_2d_mask(3).squeeze()
    expected_3 = torch.tensor([
        [1, 1, 1],
        [1, 1, 0],
        [0, 0, 0],
    ], dtype=torch.float32)
    assert torch.equal(m3, expected_3), f"3×3 mask mismatch:\n{m3}\nexpected:\n{expected_3}"
    print("  [PASS] 3×3 causal mask (5 active of 9)")

    # 5×5 mask: M₅ = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    m5 = build_causal_2d_mask(5).squeeze()
    expected_5 = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32)
    assert torch.equal(m5, expected_5), f"5×5 mask mismatch:\n{m5}\nexpected:\n{expected_5}"
    print("  [PASS] 5×5 causal mask (13 active of 25)")


def test_fold_unfold_roundtrip():
    """Verify fold(unfold(x)) == x."""
    from ncalm.models.v15.reaction_diffusion import fold_1d_to_2d, unfold_2d_to_1d

    B, N, d = 2, 64, 32
    H = W = int(math.ceil(math.sqrt(N)))

    x = torch.randn(B, N, d)
    x_2d = fold_1d_to_2d(x, H, W)
    x_back = unfold_2d_to_1d(x_2d, N)

    err = (x - x_back).abs().max().item()
    assert err < 1e-6, f"Fold/unfold roundtrip error: {err}"
    print(f"  [PASS] Fold/unfold roundtrip (max error: {err:.2e})")


def test_shapes(cfg_dict: dict):
    """Verify all tensor shapes through a forward pass."""
    from ncalm.models.v15.model import AENCAv15Model

    model = AENCAv15Model(cfg_dict)
    model.eval()

    B = 2
    N = cfg_dict.get("max_seq_len", 64)
    x = torch.randint(0, cfg_dict.get("vocab_size", 50304), (B, N + 1))

    with torch.no_grad():
        loss, info = model.forward_train(x)

    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
    assert "ce" in info, "Missing 'ce' in info dict"
    print(f"  [PASS] Forward pass shapes OK (loss={loss.item():.4f})")

    # Check logits shape
    logits = model.get_logits(x[:, :-1])
    assert logits.shape == (B, N, cfg_dict.get("vocab_size", 50304)), \
        f"Logits shape mismatch: {logits.shape}"
    print(f"  [PASS] Inference logits shape OK: {logits.shape}")


def test_zero_init_cascade(cfg_dict: dict):
    """Verify that a fresh model produces z_out ≈ z₀."""
    from ncalm.models.v15.model import AENCAv15Model

    model = AENCAv15Model(cfg_dict)
    model.eval()

    B = 2
    N = cfg_dict.get("max_seq_len", 64)
    x = torch.randint(0, cfg_dict.get("vocab_size", 50304), (B, N + 1))

    # Compute z₀
    with torch.no_grad():
        z0 = model.token_embedding(x)
        z0 = z0 + model.state_pe(N + 1).unsqueeze(0)

        # Full forward
        loss, info = model.forward_train(x)

        # Get logits to see if they come from z0
        logits_fresh = model.readout(z0)
        logits_trained = model.get_logits(x[:, :-1])

    # At init, readout(z_out) should ≈ readout(z₀)
    # This is approximate since halting weights add slight differences
    diff = (logits_fresh[:, :-1] - logits_trained).abs().mean().item()
    print(f"  Zero-init logit diff: {diff:.6f} (should be small)")
    if diff < 0.1:
        print("  [PASS] Zero-init cascade verified")
    else:
        print("  [WARN] Zero-init cascade: logit diff larger than expected")


def test_turing_symmetry_breaking(cfg_dict: dict):
    """Verify that non-zero W_r2 breaks spatial homogeneity.

    With W_r2 = 0 (init), a uniform morphogen field stays uniform.
    With W_r2 ≠ 0, the differential diffusion (k_b > k_a) should
    break symmetry and produce spatially diverse patterns — the
    Turing instability mechanism described in §16.
    """
    from ncalm.models.v15.model import AENCAv15Model
    from ncalm.models.v15.reaction_diffusion import unfold_2d_to_1d

    model = AENCAv15Model(cfg_dict)
    model.eval()
    cfg = model.config
    N = cfg_dict.get("max_seq_len", 64)

    # ── Baseline: W_r2 = 0 (default init), uniform morphogen should stay uniform ──
    B = 1
    x = torch.randint(0, cfg.vocab_size, (B, N + 1))
    with torch.no_grad():
        z = model.token_embedding(x) + model.state_pe(N + 1).unsqueeze(0)
        a0, b0 = model.rd_field.seed(z)

        # Make morphogen UNIFORM: copy position (0,0) everywhere
        a_uniform = a0[:, :, 0:1, 0:1].expand_as(a0).clone()
        b_uniform = b0[:, :, 0:1, 0:1].expand_as(b0).clone()

        a_out, b_out, _, _, _ = model.rd_field.evolve(
            a_uniform, b_uniform, z, cfg.n_morph_steps,
        )

    a_1d = unfold_2d_to_1d(a_out, N + 1)
    n_pairs = 100
    idx = torch.randint(0, N, (n_pairs, 2))
    mask = idx[:, 0] == idx[:, 1]
    idx[mask, 1] = (idx[mask, 1] + 1) % N
    ai = a_1d[:, idx[:, 0], :]
    aj = a_1d[:, idx[:, 1], :]
    cos_zero = F.cosine_similarity(ai, aj, dim=-1).mean().item()
    print(f"  W_r2=0 uniform→evolved cos_sim = {cos_zero:.6f} (should be ≈1.0)")

    # ── Perturbed: set W_r2 to non-zero to activate Turing instability ──
    with torch.no_grad():
        # Give W_r2 moderate magnitude to trigger instability
        model.rd_field.W_r2.data.normal_(0, 0.3)

        a_out2, b_out2, _, _, _ = model.rd_field.evolve(
            a_uniform.clone(), b_uniform.clone(), z, cfg.n_morph_steps,
        )

    a_1d2 = unfold_2d_to_1d(a_out2, N + 1)
    ai2 = a_1d2[:, idx[:, 0], :]
    aj2 = a_1d2[:, idx[:, 1], :]
    cos_nonzero = F.cosine_similarity(ai2, aj2, dim=-1).mean().item()
    print(f"  W_r2≠0 uniform→evolved cos_sim = {cos_nonzero:.6f} (should be < 0.95)")

    if cos_nonzero < 0.95:
        print("  [PASS] Turing symmetry-breaking: non-zero rules break homogeneity!")
    else:
        print("  [WARN] Turing symmetry-breaking: homogeneity not broken (may need larger W_r2)")

    if cos_zero > cos_nonzero:
        print("  [PASS] Differential diffusion creates spatial diversity (cos_zero > cos_nonzero)")
    else:
        print("  [WARN] Unexpected: zero-W_r2 less uniform than non-zero")


def analyze_model(model, cfg_dict: dict, device: str = "cpu"):
    """Run full diagnostic analysis on a model."""
    from ncalm.models.v15.reaction_diffusion import unfold_2d_to_1d

    model.eval()
    model.to(device)

    B = 2
    N = cfg_dict.get("max_seq_len", 64)
    cfg = model.config

    x = torch.randint(0, cfg.vocab_size, (B, N + 1), device=device)

    print("\n" + "=" * 60)
    print("  V15 RD-NCA DIAGNOSTIC ANALYSIS")
    print("=" * 60)

    # ── 1. Model Description ──
    desc = model.describe()
    print("\n[1] Architecture")
    for k, v in desc.items():
        print(f"  {k:30s} = {v}")

    # ── 2. Extra Metrics (weight norms) ──
    print("\n[2] Weight Norms (zero-init → should grow during training)")
    metrics = model.extra_metrics()
    for k, v in sorted(metrics.items()):
        print(f"  {k:35s} = {v:.6f}")

    # ── 3. Forward pass with diagnostics ──
    with torch.no_grad():
        z = model.token_embedding(x)
        z = z + model.state_pe(N + 1).unsqueeze(0)

        # Seed
        a, b = model.rd_field.seed(z)
        print(f"\n[3] Morphogen Seed")
        print(f"  a: shape={list(a.shape)} mean={a.mean():.4f} std={a.std():.4f}")
        print(f"  b: shape={list(b.shape)} mean={b.mean():.4f} std={b.std():.4f}")

        # Evolve one theta cycle
        a_ev, b_ev, rw_list, ua_list, ub_list = model.rd_field.evolve(
            a, b, z, cfg.n_morph_steps,
        )

        # ── 4. Position Cosine Similarity ──
        print(f"\n[4] CRITICAL: Position Cosine Similarity (target < 0.99, V14: 1.000)")
        a_1d = unfold_2d_to_1d(a_ev, N + 1)  # (B, N+1, d_a)
        # Sample 100 random pairs
        n_pairs = min(100, N * (N - 1) // 2)
        idx = torch.randint(0, N, (n_pairs, 2), device=device)
        mask = idx[:, 0] == idx[:, 1]
        idx[mask, 1] = (idx[mask, 1] + 1) % N

        ai = a_1d[:, idx[:, 0], :]
        aj = a_1d[:, idx[:, 1], :]
        pos_cos = F.cosine_similarity(ai, aj, dim=-1).mean().item()
        print(f"  Position cos_sim = {pos_cos:.6f}")
        if pos_cos < 0.99:
            print(f"  [PASS] Position differentiation achieved!")
        else:
            print(f"  [WARN] Positions still too similar (V14-like)")

        # ── 5. Step Update Diversity ──
        print(f"\n[5] CRITICAL: Step Update Diversity (target < 0.99, V14: 0.999998)")
        if len(ua_list) >= 2:
            step_cos_sims = []
            for m in range(len(ua_list) - 1):
                ua_m = torch.cat([ua_list[m], ub_list[m]], dim=1)  # (B, d_s, H, W)
                ua_m1 = torch.cat([ua_list[m + 1], ub_list[m + 1]], dim=1)
                # Spatial mean
                v1 = ua_m.mean(dim=(2, 3))  # (B, d_s)
                v2 = ua_m1.mean(dim=(2, 3))  # (B, d_s)
                cos = F.cosine_similarity(v1, v2, dim=-1).mean().item()
                step_cos_sims.append(cos)
                print(f"  Step {m}→{m+1} cos_sim = {cos:.6f}")
            mean_step_cos = sum(step_cos_sims) / len(step_cos_sims)
            print(f"  Mean step cos_sim = {mean_step_cos:.6f}")
            if mean_step_cos < 0.99:
                print(f"  [PASS] Step diversity achieved!")
            else:
                print(f"  [WARN] Steps still too similar")

        # ── 6. Routing Entropy ──
        print(f"\n[6] Routing Entropy (target > 0.5, V14.0: 0.021)")
        for m, rw in enumerate(rw_list):
            entropy = -(rw * (rw + 1e-8).log()).sum(dim=-1).mean().item()
            max_entropy = math.log(cfg.n_rules)
            norm_entropy = entropy / max_entropy
            print(f"  Step {m}: H={entropy:.4f} (normalized={norm_entropy:.4f})")
        all_rw = torch.stack(rw_list, dim=0)
        mean_entropy = -(all_rw * (all_rw + 1e-8).log()).sum(dim=-1).mean().item()
        print(f"  Mean routing entropy = {mean_entropy:.4f}")
        if mean_entropy > 0.5:
            print(f"  [PASS] Routing entropy sufficient")
        else:
            print(f"  [INFO] Low routing entropy (expected at init with zero-init router)")

        # ── 7. Turing Ratio ──
        print(f"\n[7] Turing Architecture Verification")
        perc_a_active = model.rd_field.perc_a.causal_mask.sum().item()
        perc_b_active = model.rd_field.perc_b.causal_mask.sum().item()
        turing_ratio = (perc_b_active - 1) / (perc_a_active - 1)  # excl. self
        print(f"  Activator causal neighbors: {perc_a_active - 1:.0f} (+self)")
        print(f"  Inhibitor causal neighbors: {perc_b_active - 1:.0f} (+self)")
        print(f"  D_b/D_a ≈ {turing_ratio:.1f} (>1 required for Turing)")
        print(f"  ρ = d_rule/d_s = {cfg.d_rule}/{cfg.d_s} = {cfg.d_rule/cfg.d_s:.1f}")

    # ── 8. Gradient Test ──
    print(f"\n[8] Gradient Stability (100 steps)")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    nan_count = 0
    max_grad = 0.0
    for step in range(100):
        x = torch.randint(0, cfg.vocab_size, (2, N + 1), device=device)
        loss, _ = model.forward_train(x)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            nan_count += 1
        else:
            max_grad = max(max_grad, grad_norm.item())
        optimizer.step()
        optimizer.zero_grad()
    print(f"  NaN/Inf gradients: {nan_count}/100")
    print(f"  Max grad norm: {max_grad:.4f}")
    if nan_count == 0:
        print(f"  [PASS] Gradient stability OK")
    else:
        print(f"  [FAIL] {nan_count} NaN gradients detected!")


def main():
    # Smoke config
    smoke_cfg = {
        "d_model": 128,
        "d_a": 32,
        "d_b": 16,
        "d_c": 32,
        "d_pos": 16,
        "d_rule": 96,
        "n_rules": 4,
        "n_morph_steps": 4,
        "n_theta": 2,
        "d_ff": 0,
        "step_embed_dim": 8,
        "k_a": 3,
        "k_b": 5,
        "p_async": 0.5,
        "max_seq_len": 64,
        "vocab_size": 50304,
        "enable_halting": True,
        "halt_eps": 0.01,
        "lambda_halt": 0.01,
        "mos_components": 1,
        "tie_embeddings": True,
        "lambda_reg": 0.01,
        "lambda_aux": 0.05,
        "lambda_div": 0.1,
        "lambda_bal": 0.1,
        "lambda_dyn": 0.01,
        "lambda_spatial": 0.1,
        "lambda_temporal": 0.05,
        "target_params": 3_000_000,
        "n_levels": 1,
        "dropout": 0.0,
        "checkpoint_interval": 0,
    }

    print("=" * 60)
    print("  AENCA V15 SMOKE EVALUATION")
    print("=" * 60)

    # Unit Tests
    print("\n── Unit Tests ──")
    test_causal_masks()
    test_fold_unfold_roundtrip()
    test_shapes(smoke_cfg)
    test_zero_init_cascade(smoke_cfg)
    test_turing_symmetry_breaking(smoke_cfg)

    # Full Diagnostic
    if len(sys.argv) > 1:
        # Load trained model
        ckpt_path = sys.argv[1]
        print(f"\n── Loading checkpoint: {ckpt_path} ──")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        from ncalm.models.v15.model import AENCAv15Model
        model = AENCAv15Model(ckpt["config"])
        model.load_state_dict(ckpt["model_state_dict"])
        analyze_model(model, ckpt["config"])
    else:
        # Fresh model diagnostic
        print("\n── Fresh Model Diagnostic ──")
        from ncalm.models.v15.model import AENCAv15Model
        model = AENCAv15Model(smoke_cfg)
        analyze_model(model, smoke_cfg)

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
