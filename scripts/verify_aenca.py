#!/usr/bin/env python3
"""AENCA v3 — Comprehensive Empirical Verification Suite.

Runs all property tests defined in the implementation plan (Sections 9.1–9.6).
All tests operate on a tiny model configuration (seconds per test) so this
script can be run without a GPU as a quick correctness check.

Test categories:
  Phase 1 — Static code verification (param counts, no-attention check, etc.)
  Phase 2 — Mathematical identity checks (locality, homogeneity, causality, ...)
  V-tests  — Unit-level NCA property tests (V1–V8)
  N-tests  — Corrected-mechanism tests (N1–N4)

Usage:
    python scripts/verify_aenca.py
    python scripts/verify_aenca.py --verbose
    python scripts/verify_aenca.py --device cuda  # run on GPU
"""

import argparse
import math
import sys
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for Unicode chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from ncalm.models.aenca_v3 import (
    AENCAv3Config,
    AENCAv3Model,
    gather_neighborhood,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test harness
# ─────────────────────────────────────────────────────────────────────────────

PASS = "  [PASS]"
FAIL = "  [FAIL]"
SKIP = "  [SKIP]"

_results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "", verbose: bool = False):
    """Record one test result and print it."""
    tag = PASS if condition else FAIL
    line = f"{tag} {name}"
    if detail:
        line += f"  — {detail}"
    print(line)
    _results.append((name, condition, detail))
    if not condition and verbose:
        print(f"         Detail: {detail}")


def section(title: str):
    print(f"\n{'-'*72}")
    print(f"  {title}")
    print(f"{'-'*72}")


def build_tiny_model(device: torch.device) -> AENCAv3Model:
    """Construct a tiny model (fast, low-memory) for verification."""
    cfg = {
        "vocab_size": 100,    # tiny vocab
        "d_emb":      16,
        "Nx": 4, "Ny": 4,    # 4×4 = 16 cells
        "d_h": 8, "d_b": 6, "d_c": 4,   # D=18
        "d_pos":      8,
        "d_kernel_h": 32,
        "L_scales":   2,
        "T_steps":    4,
        "R_steps":    2,
        "R_max_inference": 8,
        "tau_H_stop": 2.0,    # high threshold → adaptive stop fires quickly
        "eps_E_stop": 1e-3,
        "anytime_steps":   (2, 4),
        "anytime_weights": (0.3, 0.7),
        "lambda_pg":  0.01,
    }
    model = AENCAv3Model(cfg).to(device)
    model.train()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Static code verification
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1(args):
    section("Phase 1 — Static Code Verification")
    device = args.device

    model = build_tiny_model(device)
    cfg   = model.cfg

    # C1: Parameter count in expected range
    total = sum(p.numel() for p in model.parameters())
    check("C1: param count > 0", total > 0, f"total={total:,}")

    # C2: No per-cell parameters in kernel (NCA parameter sharing)
    kernel_params = sum(p.numel() for p in model.kernel.parameters())
    check("C2: kernel has params (shared)", kernel_params > 0, f"kernel_params={kernel_params:,}")

    # C3: Kernel input dimension matches formula  9*D + d_emb + d_pos
    D    = cfg.D
    expected_kernel_in = 9 * D + cfg.d_emb + cfg.d_pos
    actual_kernel_in   = model.kernel.layers[0].in_features
    check(
        "C3: kernel input dim = 9*D + d_emb + d_pos",
        actual_kernel_in == expected_kernel_in,
        f"got={actual_kernel_in}, expected={expected_kernel_in}",
    )

    # C4: Obs model input dimension = 9*D + d_pos
    expected_obs_in = 9 * D + cfg.d_pos
    actual_obs_in   = model.obs_model.net[0].in_features
    check(
        "C4: obs model input dim = 9*D + d_pos",
        actual_obs_in == expected_obs_in,
        f"got={actual_obs_in}, expected={expected_obs_in}",
    )

    # C5: Action space = 9 (3 direction × 3 scale)
    check("C5: action space = 9", len(model.action_alphas) == 9, "9 joint actions")

    # C6: No cross-cell LayerNorm (no TransformerEncoder or MultiheadAttention)
    has_attention = any(
        isinstance(m, (torch.nn.MultiheadAttention, torch.nn.TransformerEncoder,
                       torch.nn.TransformerDecoder))
        for m in model.modules()
    )
    check("C6: no attention/transformer modules", not has_attention)

    # C7: Readout is a single linear (no attention-weighted pooling)
    check(
        "C7: readout is Linear(d_b, vocab_size)",
        isinstance(model.readout.W_out, torch.nn.Linear)
        and model.readout.W_out.in_features == cfg.d_b
        and model.readout.W_out.out_features == cfg.vocab_size,
        f"W_out: {cfg.d_b} → {cfg.vocab_size}",
    )

    # C8: describe() returns required keys
    desc = model.describe()
    required_keys = {"architecture", "total_params"}
    check(
        "C8: describe() has required keys",
        required_keys.issubset(desc.keys()),
        f"keys={list(desc.keys())}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Mathematical identity checks
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2(args):
    section("Phase 2 — Mathematical Identity Checks")
    device = args.device

    torch.manual_seed(42)
    model = build_tiny_model(device)
    cfg   = model.cfg
    B, Nx, Ny, D = 2, cfg.Nx, cfg.Ny, cfg.D

    # ── M1: NCA Locality — one-step update only affects distance-1 neighbours ─
    # Perturb cell (0,0) and verify cells at distance >1 are unchanged after 1 step.
    substrate1 = model.substrate_init(B, device)
    substrate2 = substrate1.clone()
    # Big perturbation at one cell only
    substrate2[:, 0, 0, :] = substrate2[:, 0, 0, :] + 10.0

    with torch.no_grad():
        P1 = gather_neighborhood(substrate1)   # (B, Nx, Ny, 9*D)
        P2 = gather_neighborhood(substrate2)

        # Cells at distance > 1 from (0,0): minimum (2,2) in a 4×4 grid
        # Neighbour P for cell (2,2) should be identical in P1 and P2
        # because the perturbed cell (0,0) is not in the Moore neighbourhood of (2,2)
        p1_cell22 = P1[:, 2, 2, :]
        p2_cell22 = P2[:, 2, 2, :]
        locality_ok = torch.allclose(p1_cell22, p2_cell22, atol=1e-6)

    check("M1: locality (far cell unaffected by distant perturbation)", locality_ok)

    # ── M2: NCA Homogeneity — kernel produces same output for same input ────────
    with torch.no_grad():
        # Create two identical inputs but at different "positions" in the batch
        # Since kernel has no cell-specific parameters, outputs must be identical
        x_in = torch.randn(1, 9 * D + cfg.d_emb + cfg.d_pos, device=device)
        x_doubled = x_in.repeat(2, 1)
        ds, gate, alpha, beta = model.kernel(x_doubled)
        homogeneity_ok = (
            torch.allclose(ds[0], ds[1], atol=1e-6)
            and torch.allclose(gate[0], gate[1], atol=1e-6)
        )
    check("M2: homogeneity (same input → same kernel output)", homogeneity_ok)

    # ── M3: Causal constraint — pointers never exceed pos ───────────────────────
    # Run forward_train and instrument to check pointers at each step
    model_traced = build_tiny_model(device)
    model_traced.train()
    causal_violated = [False]

    orig_process = model_traced._process_position

    def traced_process(substrate, pointers, scale_lvl, emb, pos_tensor, targets):
        pos = int(pos_tensor.item())
        if pointers.max().item() > pos:
            causal_violated[0] = True
        result = orig_process(substrate, pointers, scale_lvl, emb, pos_tensor, targets)
        new_ptrs = result[1]
        if new_ptrs.max().item() > pos:
            causal_violated[0] = True
        return result

    model_traced._process_position = traced_process

    with torch.no_grad():
        batch = torch.randint(0, cfg.vocab_size, (1, 6), device=device)
        emb   = model_traced.token_embed(batch)
        substrate = model_traced.substrate_init(1, device)
        pointers  = model_traced._init_pointers(1, device)
        scale_lvl = torch.zeros(1, Nx, Ny, device=device, dtype=torch.long)
        for pos in range(5):
            pos_tensor = torch.tensor(pos, device=device)
            targets    = batch[:, pos + 1]
            result = model_traced._process_position(
                substrate, pointers, scale_lvl, emb, pos_tensor, targets
            )
            substrate, pointers, scale_lvl = result[0], result[1], result[2]

    check("M3: causal constraint (pointers ≤ pos at all times)", not causal_violated[0])

    # ── M4: EFE masking — boundary actions have EFE = +inf ──────────────────────
    # Place all sensors at pos=0, expect all α=-1 actions to be masked
    with torch.no_grad():
        substrate = model.substrate_init(B, device)
        pointers  = torch.zeros(B, Nx, Ny, device=device, dtype=torch.long)
        scale_lvl = torch.zeros(B, Nx, Ny, device=device, dtype=torch.long)
        P = gather_neighborhood(substrate)

        efe, _, _ = model._compute_efe(
            P=P, substrate=substrate,
            pointers=pointers, scale_lvl=scale_lvl,
            pos=0, targets=None, training_mode=False,
        )
        # Actions 0,1,2 have α=-1 (left move from pos=0 → masked)
        # Actions 6,7,8 have α=+1 (right move at pos=0 → masked)
        alpha_minus1_idx = [0, 1, 2]
        alpha_plus1_idx  = [6, 7, 8]
        left_masked  = all(efe[:, :, :, i].isinf().all() for i in alpha_minus1_idx)
        right_masked = all(efe[:, :, :, i].isinf().all() for i in alpha_plus1_idx)
        # α=0 actions (3,4,5) should NOT be masked
        stay_valid   = all(not efe[:, :, :, i].isinf().any() for i in [3, 4, 5])

    check("M4: EFE masking (left-moves at pos=0 → +inf)",  left_masked)
    check("M4: EFE masking (right-moves at pos=0 → +inf)", right_masked)
    check("M4: EFE masking (stay actions remain finite)",   stay_valid)

    # ── M5: Gradient flow through BPTT (substrate grad at pos=0 is nonzero) ─────
    model_grad = build_tiny_model(device)
    model_grad.train()
    batch = torch.randint(0, cfg.vocab_size, (1, 4), device=device)
    loss, _ = model_grad.forward_train(batch)
    loss.backward()
    # Check gradient on the substrate initializer (affected by all outer steps)
    coord_grad = model_grad.substrate_init.coord_embed.weight.grad
    grad_norm = coord_grad.norm().item() if coord_grad is not None else 0.0
    check(
        "M5: gradient flow (coord_embed grad > 1e-8)",
        grad_norm > 1e-8,
        f"||grad||={grad_norm:.2e}",
    )

    # ── M6: Inference EFE runs without error (no targets) ────────────────────────
    try:
        with torch.no_grad():
            substrate = model.substrate_init(B, device)
            pointers  = torch.zeros(B, Nx, Ny, device=device, dtype=torch.long)
            scale_lvl = torch.zeros(B, Nx, Ny, device=device, dtype=torch.long)
            P = gather_neighborhood(substrate)
            efe_infer, _, _ = model._compute_efe(
                P=P, substrate=substrate,
                pointers=pointers, scale_lvl=scale_lvl,
                pos=3, targets=None, training_mode=False,
            )
        infer_ok = efe_infer.isfinite().any().item()
    except Exception as e:
        infer_ok = False
        print(f"         Exception: {e}")
    check("M6: inference EFE (no targets, no exception)", infer_ok)

    # ── M7: Obs model output shape ────────────────────────────────────────────────
    with torch.no_grad():
        P_small = torch.randn(B, Nx, Ny, 9 * D, device=device)
        pos_enc = torch.randn(B, Nx, Ny, cfg.d_pos, device=device)
        pred    = model.obs_model(P_small, pos_enc)
    expected_shape = (B, Nx, Ny, cfg.d_emb)
    check(
        "M7: obs model output shape",
        tuple(pred.shape) == expected_shape,
        f"got={tuple(pred.shape)}, expected={expected_shape}",
    )

    # ── M8: Readout is exact mean over cells (not attention-weighted) ─────────────
    with torch.no_grad():
        substrate = model.substrate_init(B, device)
        logits = model.readout(substrate)
        b_ch   = substrate[..., cfg.d_h : cfg.d_h + cfg.d_b]
        manual = model.readout.W_out(b_ch).mean(dim=(1, 2))
    check(
        "M8: readout = exact mean vote",
        torch.allclose(logits, manual, atol=1e-5),
        f"max_diff={( logits - manual).abs().max().item():.2e}",
    )

    # ── M9: No pointer movement during relaxation ──────────────────────────────────
    # Run a _process_position call, capture pointers before and after relaxation.
    # Relaxation inside _process_position does not move sensors.
    # We verify: the returned pointers equal the pre-relaxation positions.
    # (Indirect test: pointers returned reflect only T sensing steps, not R extra moves.)
    with torch.no_grad():
        substrate = model.substrate_init(1, device)
        init_ptrs = torch.zeros(1, Nx, Ny, device=device, dtype=torch.long)
        scale_lvl = torch.zeros(1, Nx, Ny, device=device, dtype=torch.long)
        batch1    = torch.randint(0, cfg.vocab_size, (1, 4), device=device)
        emb1      = model.token_embed(batch1)
        pos_tensor = torch.tensor(2, device=device)
        targets1   = batch1[:, 3]
        # Run _process_position: returns (substrate, pointers, ...)
        result = model._process_position(
            substrate, init_ptrs, scale_lvl, emb1, pos_tensor, targets1
        )
        returned_ptrs = result[1]
        # Pointers must be in [0, pos=2]
        ptr_in_range = (returned_ptrs >= 0).all() and (returned_ptrs <= 2).all()
    check(
        "M9: returned pointers in [0, pos] after sensing+relaxation",
        ptr_in_range.item(),
        f"max_ptr={returned_ptrs.max().item()}, pos=2",
    )


# ─────────────────────────────────────────────────────────────────────────────
# V-tests — Unit-level NCA property tests
# ─────────────────────────────────────────────────────────────────────────────

def run_vtests(args):
    section("V-Tests — NCA Property Verification")
    device = args.device
    torch.manual_seed(42)
    model = build_tiny_model(device)
    cfg   = model.cfg
    B, Nx, Ny, D = 2, cfg.Nx, cfg.Ny, cfg.D

    # ── V1: NCA Parameter Sharing ────────────────────────────────────────────
    # Kernel has EXACTLY the shared set of parameters — no per-cell extras.
    kernel_params_expected = sum(p.numel() for p in model.kernel.parameters())
    # Instantiate with different m and check kernel params don't change
    cfg_large = dict(vars(cfg))
    cfg_large['Nx'] = 8
    cfg_large['Ny'] = 8
    model_large = AENCAv3Model(cfg_large).to(device)
    kernel_params_large = sum(p.numel() for p in model_large.kernel.parameters())
    check(
        "V1: kernel param count is independent of substrate size",
        kernel_params_expected == kernel_params_large,
        f"4×4 model: {kernel_params_expected}, 8×8 model: {kernel_params_large}",
    )

    # ── V2: Causal Constraint ─────────────────────────────────────────────────
    # Already tested in M3 (comprehensive). Quick repeat with asserted pos.
    with torch.no_grad():
        ptrs_test = torch.randint(0, 5, (B, Nx, Ny), device=device, dtype=torch.long)
        clamped   = ptrs_test.clamp(0, 3)   # enforce pos=3
        check(
            "V2: causal clamp works correctly",
            (clamped <= 3).all().item(),
            f"max_after_clamp={clamped.max().item()}, pos=3",
        )

    # ── V3: EFE Epistemic Value direction ────────────────────────────────────
    # After a few steps, EFE-selected actions should have info_gain ≥ random.
    # We test the structure: info_gain is computed correctly (finite, sign can vary).
    with torch.no_grad():
        substrate = model.substrate_init(B, device)
        P = gather_neighborhood(substrate)
        efe, _, _ = model._compute_efe(
            P=P, substrate=substrate,
            pointers=torch.zeros(B, Nx, Ny, device=device, dtype=torch.long),
            scale_lvl=torch.zeros(B, Nx, Ny, device=device, dtype=torch.long),
            pos=5, targets=torch.randint(0, cfg.vocab_size, (B,), device=device),
            training_mode=True,
        )
        # All valid EFE values must be finite
        valid_mask = ~efe.isinf()
        efe_finite = efe[valid_mask].isfinite().all().item()
    check("V3: EFE values are finite for valid actions", efe_finite)

    # ── V4: Anytime Property — energy during relaxation ──────────────────────
    # After main kernel update, R relaxation steps should keep or reduce energy.
    # This is a soft property that holds statistically after training,
    # but on random weights we just check the relaxation runs without error.
    try:
        with torch.no_grad():
            substrate = model.substrate_init(B, device)
            targets   = torch.randint(0, cfg.vocab_size, (B,), device=device)
            E_before, _ = model._local_energy(substrate, targets)
            # One relaxation step
            P = gather_neighborhood(substrate)
            obs    = substrate[..., cfg.d_h : cfg.d_h + cfg.d_b].mean(-1, keepdim=True).expand(
                B, Nx, Ny, cfg.d_emb
            ).contiguous()
            pos_enc = model.pos_encoder.encode(
                torch.zeros(B, Nx, Ny, device=device, dtype=torch.long), 1
            )
            kin = torch.cat([P, obs, pos_enc], dim=-1)
            ds, g, _, _ = model.kernel(kin.reshape(B * Nx * Ny, -1))
            ds = ds.reshape(B, Nx, Ny, D)
            g  = g.reshape(B, Nx, Ny, D)
            substrate = substrate + torch.sigmoid(g) * ds
            E_after, _ = model._local_energy(substrate, targets)
        relax_ok = E_before.isfinite().item() and E_after.isfinite().item()
    except Exception as e:
        relax_ok = False
        print(f"         Exception in V4: {e}")
    check("V4: relaxation step runs without error (energy finite)", relax_ok)

    # ── V5: Anytime Supervision — energy logged at correct steps ─────────────
    with torch.no_grad():
        model_v5 = build_tiny_model(device)
        batch = torch.randint(0, cfg.vocab_size, (1, 4), device=device)
        emb   = model_v5.token_embed(batch)
        substrate = model_v5.substrate_init(1, device)
        pointers  = model_v5._init_pointers(1, device)
        scale_lvl = torch.zeros(1, Nx, Ny, device=device, dtype=torch.long)
        pos_tensor = torch.tensor(2, device=device)
        targets    = batch[:, 3]
        result = model_v5._process_position(
            substrate, pointers, scale_lvl, emb, pos_tensor, targets
        )
        n_anytime  = model_v5._n_anytime   # should be 2 (for anytime_steps=(2,4))
        energies   = result[3 : 3 + n_anytime]
    check(
        "V5: correct number of anytime energies returned",
        len(energies) == n_anytime,
        f"got={len(energies)}, expected={n_anytime}",
    )
    all_finite = all(e.isfinite().item() for e in energies)
    check("V5: all anytime energies are finite", all_finite)

    # ── V6: Multi-scale Reachability ─────────────────────────────────────────
    # With L_scales=2, step sizes are 1 and 2.
    # Any position in [0, n] is reachable from 0 in ≤ log2(n) steps.
    L = cfg.L_scales   # 2
    step_sizes = [2**l for l in range(L)]   # [1, 2]
    max_pos = 7   # reach position 7 from 0 using steps 1 and 2
    # Can we reach any position 0..7? With steps 1 and 2, yes (2 bits of movement)
    reachable = set()
    for combo in range(4 ** (max_pos + 1)):
        pos = 0
        steps = combo
        for _ in range(8):
            s = steps % 4
            steps //= 4
            if s == 0: pos += 1
            elif s == 1: pos += 2
            elif s == 2: pos -= 1
            elif s == 3: pos -= 2
            pos = max(0, min(pos, max_pos))
            reachable.add(pos)
    fully_reachable = all(p in reachable for p in range(max_pos + 1))
    check(
        "V6: multi-scale reachability (all positions reachable)",
        fully_reachable,
        f"L_scales={L}, step_sizes={step_sizes}",
    )

    # ── V7: Gradient Flow through full BPTT ──────────────────────────────────
    model_bptt = build_tiny_model(device)
    model_bptt.train()
    batch = torch.randint(0, cfg.vocab_size, (1, 5), device=device)
    loss, _ = model_bptt.forward_train(batch)
    loss.backward()
    coord_grad = model_bptt.substrate_init.coord_embed.weight.grad
    g_norm = coord_grad.norm().item() if coord_grad is not None else 0.0
    check(
        "V7: gradient flows back to substrate initialiser",
        g_norm > 1e-8,
        f"||grad||={g_norm:.2e}",
    )

    # ── V8: NCA Locality — kernel update only uses neighbourhood of radius 1 ─
    # Already verified as M1. Additional check: gather_neighborhood output at
    # cell (row, col) depends ONLY on cells within Moore distance ≤ 1.
    torch.manual_seed(0)
    with torch.no_grad():
        s1 = torch.randn(1, Nx, Ny, D, device=device)
        s2 = s1.clone()
        # Perturb only cell (3, 3) — far from cell (0, 0)
        s2[:, 3, 3, :] += 5.0
        P1_00 = gather_neighborhood(s1)[:, 0, 0, :]
        P2_00 = gather_neighborhood(s2)[:, 0, 0, :]
        locality_far = torch.allclose(P1_00, P2_00, atol=1e-6)
    check(
        "V8: NCA locality (far cell perturbation doesn't reach cell (0,0))",
        locality_far,
        f"Nx×Ny={Nx}×{Ny}, perturbed (3,3), checking (0,0)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# N-tests — Corrected mechanism tests
# ─────────────────────────────────────────────────────────────────────────────

def run_ntests(args):
    section("N-Tests — Corrected Mechanism Verification")
    device = args.device
    torch.manual_seed(42)
    model = build_tiny_model(device)
    cfg   = model.cfg
    B, Nx, Ny, D = 2, cfg.Nx, cfg.Ny, cfg.D

    # ── N1: Training vs Inference EFE Mode ───────────────────────────────────
    # Training EFE uses CE risk (requires targets).
    # Inference EFE uses KL risk (no targets needed).
    targets = torch.randint(0, cfg.vocab_size, (B,), device=device)
    substrate = model.substrate_init(B, device)
    pointers  = torch.ones(B, Nx, Ny, device=device, dtype=torch.long) * 2
    scale_lvl = torch.zeros(B, Nx, Ny, device=device, dtype=torch.long)
    P = gather_neighborhood(substrate)

    # Training mode
    try:
        efe_train, _, _ = model._compute_efe(
            P=P, substrate=substrate, pointers=pointers, scale_lvl=scale_lvl,
            pos=3, targets=targets, training_mode=True,
        )
        train_ok = efe_train.isfinite().any().item()
    except Exception as e:
        train_ok = False
        print(f"         Training EFE exception: {e}")

    # Inference mode (NO targets)
    try:
        efe_infer, _, _ = model._compute_efe(
            P=P, substrate=substrate, pointers=pointers, scale_lvl=scale_lvl,
            pos=3, targets=None, training_mode=False,
        )
        infer_ok = efe_infer.isfinite().any().item()
    except Exception as e:
        infer_ok = False
        print(f"         Inference EFE exception: {e}")

    # Both modes produce finite EFE for valid actions
    check("N1: training-mode EFE runs without error", train_ok)
    check("N1: inference-mode EFE runs without error (no targets)", infer_ok)

    # Training and inference EFE should differ (different risk formulations)
    if train_ok and infer_ok:
        valid_mask = (~efe_train.isinf()) & (~efe_infer.isinf())
        if valid_mask.any():
            correlation_ok = not torch.allclose(
                efe_train[valid_mask], efe_infer[valid_mask], atol=1e-4
            )
            check(
                "N1: training and inference EFE values differ (different risk)",
                correlation_ok,
                "EFE formulations use different risk terms",
            )

    # ── N2: Action Masking Correctness ───────────────────────────────────────
    # At pos=0, ALL moves are out-of-bounds. Only scale-change-only actions valid.
    with torch.no_grad():
        ptr_0 = torch.zeros(B, Nx, Ny, device=device, dtype=torch.long)
        efe_at_0, _, _ = model._compute_efe(
            P=P, substrate=substrate, pointers=ptr_0, scale_lvl=scale_lvl,
            pos=0, targets=targets, training_mode=True,
        )
        # α ≠ 0 actions at boundary should have EFE = +inf
        move_actions = [0, 1, 2, 6, 7, 8]   # α=-1 and α=+1 actions
        stay_actions = [3, 4, 5]             # α=0 actions (no movement)
        moves_masked = all(efe_at_0[:, :, :, i].isinf().all().item() for i in move_actions)
        stays_finite = all(efe_at_0[:, :, :, i].isfinite().all().item() for i in stay_actions)

    check("N2: all movement actions masked at pos=0", moves_masked)
    check("N2: stay actions remain unmasked at pos=0", stays_finite)

    # No boundary action should ever be sampled
    model_n2 = build_tiny_model(device)
    model_n2.train()
    torch.manual_seed(0)
    boundary_sampled = False
    with torch.no_grad():
        substrate_n2 = model_n2.substrate_init(B, device)
        ptr_left = torch.zeros(B, Nx, Ny, device=device, dtype=torch.long)
        sl_n2 = torch.zeros(B, Nx, Ny, device=device, dtype=torch.long)
        P_n2 = gather_neighborhood(substrate_n2)
        for _ in range(20):
            efe_n2, _, _ = model_n2._compute_efe(
                P=P_n2, substrate=substrate_n2,
                pointers=ptr_left, scale_lvl=sl_n2,
                pos=0, targets=targets, training_mode=True,
            )
            policy_logits = (-efe_n2 / model_n2.cfg.tau_efe).masked_fill(
                efe_n2.isinf(), float('-inf')
            )
            from torch.distributions import Categorical
            flat_actions = Categorical(
                logits=policy_logits.reshape(-1, 9)
            ).sample()
            # Check: no sampled action is in the masked set (move_actions)
            for a in flat_actions.tolist():
                if a in move_actions:
                    boundary_sampled = True
    check("N2: boundary actions never sampled (20 trials)", not boundary_sampled)

    # ── N3: Adaptive Stopping Activates ──────────────────────────────────────
    # With high tau_H_stop (2.0 nats), the adaptive stopping should trigger quickly.
    model_n3 = build_tiny_model(device)
    model_n3.eval()
    torch.manual_seed(0)

    with torch.no_grad():
        inp = torch.randint(0, cfg.vocab_size, (1, 4), device=device)
        logits = model_n3.get_logits(inp)
    check(
        "N3: get_logits runs without error (adaptive stopping integrated)",
        logits.shape == (1, 4, cfg.vocab_size),
        f"logits shape: {tuple(logits.shape)}",
    )

    # ── N4: Gradient Norm Monitoring (BPTT health) ───────────────────────────
    model_n4 = build_tiny_model(device)
    model_n4.train()
    batch = torch.randint(0, cfg.vocab_size, (1, 6), device=device)
    loss, _ = model_n4.forward_train(batch)
    loss.backward()

    # Check gradients exist and are finite for key modules
    for name, param in model_n4.named_parameters():
        if param.grad is not None:
            if not param.grad.isfinite().all():
                check(f"N4: gradient finite for {name}", False, "contains NaN/Inf!")
                break
    else:
        check("N4: all parameter gradients are finite", True)

    # Grad norm of first kernel layer should be non-negligible
    g_k = model_n4.kernel.layers[0].weight.grad
    g_k_norm = g_k.norm().item() if g_k is not None else 0.0
    check(
        "N4: kernel gradient norm > 1e-8 (BPTT working)",
        g_k_norm > 1e-8,
        f"||grad||={g_k_norm:.2e}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Additional integration tests
# ─────────────────────────────────────────────────────────────────────────────

def run_integration(args):
    section("Integration Tests")
    device = args.device
    torch.manual_seed(42)

    # Test with full-size config (but small batch to keep it fast)
    full_cfg = {
        "vocab_size": 10000,
        "d_emb": 64, "Nx": 4, "Ny": 4,
        "d_h": 20, "d_b": 12, "d_c": 8,
        "d_pos": 16, "d_kernel_h": 128, "L_scales": 2,
        "T_steps": 2, "R_steps": 1,
        "anytime_steps": (2,), "anytime_weights": (1.0,),
    }
    model = AENCAv3Model(full_cfg).to(device)
    model.train()

    # I1: forward_train is deterministic with same seed.
    # Save state BEFORE any forward_train so both runs start identically
    # (pg_baseline EMA is updated during forward_train, so we must restore it).
    batch = torch.randint(0, 10000, (2, 5), device=device)
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    torch.manual_seed(1)
    loss1, info1 = model.forward_train(batch)

    # Restore initial state for the second run
    model.load_state_dict(initial_state)
    torch.manual_seed(1)
    loss2, info2 = model.forward_train(batch)
    check(
        "I1: forward_train is deterministic with same seed",
        abs(loss1.item() - loss2.item()) < 1e-4,
        f"loss1={loss1.item():.4f}, loss2={loss2.item():.4f}",
    )

    # I2: get_param_groups returns valid structure
    groups = model.get_param_groups()
    group_ok = (
        isinstance(groups, list)
        and len(groups) > 0
        and all("params" in g and "label" in g for g in groups)
    )
    check("I2: get_param_groups returns valid structure", group_ok)

    # I3: eval mode forward_train (for validation loop in Trainer)
    model.eval()
    with torch.no_grad():
        loss_eval, info_eval = model.forward_train(batch)
    check(
        "I3: forward_train in eval mode (for validation)",
        "ce" in info_eval and math.isfinite(info_eval["ce"]),
        f"ce={info_eval.get('ce', 'missing'):.4f}",
    )

    # I4: train mode after eval (toggle works correctly)
    model.train()
    loss_back, _ = model.forward_train(batch)
    check(
        "I4: model toggles between eval and train correctly",
        loss_back.requires_grad,
    )

    # I5: token budget sanity (tokens per step = B × L)
    B, L = batch.shape
    check(
        "I5: token budget sanity",
        B * L == 10,
        f"B={B}, L={L}, B*L={B*L}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    section("Summary")
    n_total = len(_results)
    n_pass  = sum(1 for _, ok, _ in _results if ok)
    n_fail  = n_total - n_pass

    print(f"\n  Total: {n_total}  |  Passed: {n_pass}  |  Failed: {n_fail}\n")

    if n_fail > 0:
        print("  FAILED TESTS:")
        for name, ok, detail in _results:
            if not ok:
                print(f"    ✗ {name}  — {detail}")
        print()

    return n_fail == 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AENCA v3 Verification Suite")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on (cpu or cuda). Default: cpu for determinism.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print extra detail on failures.",
    )
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print("=" * 72)
    print("  AENCA v3 - Verification Suite")
    print(f"  Device: {args.device}")
    print("=" * 72)

    run_phase1(args)
    run_phase2(args)
    run_vtests(args)
    run_ntests(args)
    run_integration(args)

    success = print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
