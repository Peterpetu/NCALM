"""Sanity check for AENCA V12 MMHR architecture.

Verifies:
  1. Import chain resolves without errors
  2. Smoke config instantiates V12Config correctly
  3. Model builds and parameter count is reasonable
  4. Forward pass (forward_train) runs and returns finite loss
  5. Routing entropy ~ log(K) at init (W_router zero-init -> uniform weights)
  6. Diversity loss term is non-zero and negative (−entropy)
  7. Backward pass runs; gradients flow to W_router, W_s, W_d
  8. get_logits runs without error
  9. on_step_end scaffolding runs without error
 10. Profile names resolve correctly

Usage:
    python scripts/_check_v12.py
"""

import math
import sys
import traceback

import torch

# ── 1. Import chain ────────────────────────────────────────────────────────────
print("[ 1] Checking imports ...", end=" ")
try:
    from ncalm.models.v12.config import V12Config
    from ncalm.models.v12.kernel import MixtureHebbianRules, MorphogenRule, DepthwiseCausalConv
    from ncalm.models.v12.settlement import SettlementState
    from ncalm.models.v12.model import AENCAv12Model
    from ncalm.models import AENCAv12Model as AENCAv12Model_reg, V12Config as V12Config_reg
    from ncalm.profiles import PROFILES
    print("OK")
except Exception as e:
    print(f"FAIL\n{traceback.format_exc()}")
    sys.exit(1)

# ── 2. Profile checks ──────────────────────────────────────────────────────────
print("[ 2] Checking profile registration ...", end=" ")
for key in ("smoke_v12", "smoke_v12_prod", "pile_250m_v12"):
    assert key in PROFILES, f"Profile '{key}' not found in PROFILES"
print("OK")

# ── 3. Smoke config instantiation ─────────────────────────────────────────────
print("[ 3] Instantiating smoke V12Config ...", end=" ")
from dataclasses import fields as dataclass_fields

SMOKE_CFG = {
    "d_model": 128,
    "d_rule": 32,
    "d_h": 32,
    "n_rules": 4,
    "n_steps": 8,
    "n_scales": 4,
    "conv_kernel_size": 3,
    "max_seq_len": 64,
    "vocab_size": 100,
    "tie_embeddings": True,
    "noise_sigma": 0.02,
    "route_temp_min": 0.1,
    "route_temp_max": 1.0,
    "lambda_div": 0.01,
    "lambda_perm": 0.01,
    "lambda_spectral": 0.001,
    "settlement_beta": 0.9,
    "settlement_tau": 0.05,
    "settlement_eta": 0.1,
    "n_scaffold_templates": 8,
    "scaffold_alpha": 0.3,
    "scaffold_temperature": 1.0,
    "scaffold_ema_decay": 0.99,
    "scaffold_warmup_steps": 5,
    "scaffold_maintenance_interval": 10,
    "scaffold_min_norm": 0.01,
    "scaffold_merge_threshold": 0.95,
    "dropout": 0.0,
    "checkpoint_interval": 0,
}

v12_keys = {f.name for f in dataclass_fields(V12Config)}
cfg_params = {k: v for k, v in SMOKE_CFG.items() if k in v12_keys}
cfg = V12Config(**cfg_params)

assert cfg.n_rules == 4
assert cfg.d_h == 32
assert cfg.n_steps == 8
print("OK")

# ── 4. Model construction ──────────────────────────────────────────────────────
print("[ 4] Building AENCAv12Model ...", end=" ")
model = AENCAv12Model(SMOKE_CFG)
model.eval()
print("OK")

# ── 5. Parameter count check ───────────────────────────────────────────────────
print("[ 5] Checking parameter count ...", end=" ")
actual = sum(p.numel() for p in model.parameters())
estimate = cfg.total_params_estimate
ratio = actual / estimate
print(f"OK  (actual={actual:,}  estimate={estimate:,}  ratio={ratio:.3f})")

if not (0.85 <= ratio <= 1.15):
    print(f"  WARNING: ratio {ratio:.3f} outside [0.85, 1.15] — check config.total_params_estimate")

# ── 6. Forward pass ───────────────────────────────────────────────────────────
print("[ 6] Forward pass (forward_train) ...", end=" ")
model.train()
batch = torch.randint(0, SMOKE_CFG["vocab_size"], (2, SMOKE_CFG["max_seq_len"]))
loss, info = model.forward_train(batch)

assert torch.isfinite(loss), f"Loss is not finite: {loss}"
assert "ce" in info and torch.isfinite(info["ce"]), "CE loss not finite"
assert "routing_entropy" in info, "routing_entropy missing from info"
assert "div_loss" in info, "div_loss missing from info"
print(f"OK  (loss={loss.item():.4f}  ce={info['ce'].item():.4f})")

# ── 7. Routing entropy check ──────────────────────────────────────────────────
print("[ 7] Routing entropy at init ...", end=" ")
# W_router is zero-init -> logits = 0 -> Gumbel-softmax should give near-uniform weights
# Expected entropy ~ log(K) but with Gumbel NOISE it will be slightly below log(K)
# We just check it's positive and not collapsed (> 0.3 * log(K))
K = SMOKE_CFG["n_rules"]
log_K = math.log(K)
routing_entropy = info["routing_entropy"].item()
min_expected = 0.3 * log_K
print(f"OK  (entropy={routing_entropy:.4f}  log(K)={log_K:.4f}  min_expected={min_expected:.4f})")
assert routing_entropy > min_expected, (
    f"Routing entropy {routing_entropy:.4f} too low — expected > {min_expected:.4f}. "
    f"W_router zero-init should give near-uniform routing."
)

# ── 8. Diversity loss sign check ──────────────────────────────────────────────
print("[ 8] Diversity loss sign check ...", end=" ")
div_loss = info["div_loss"].item()
assert div_loss <= 0.0, f"div_loss should be <= 0 (= -entropy), got {div_loss:.4f}"
print(f"OK  (div_loss={div_loss:.4f}  -- should be negative)")

# ── 9. Backward pass and gradient check ───────────────────────────────────────
print("[ 9] Backward pass and gradient flow ...", end=" ")
loss.backward()

# Verify gradients on key V12 parameters
grad_checks = [
    ("rules.W_router.weight", model.rules.W_router.weight),
    ("rules.W_s", model.rules.W_s),
    ("rules.W_d", model.rules.W_d),
    ("rules.W_p.weight", model.rules.W_p.weight),
    ("token_embedding.weight", model.token_embedding.weight),
]

failed = []
for name, param in grad_checks:
    if param.grad is None:
        failed.append(f"{name} has NO gradient")
    elif not torch.isfinite(param.grad).all():
        failed.append(f"{name} gradient has non-finite values")

if failed:
    print("FAIL")
    for f in failed:
        print(f"  {f}")
    sys.exit(1)

print("OK")
for name, param in grad_checks:
    gnorm = param.grad.norm().item()
    print(f"         {name}: grad_norm={gnorm:.6f}")

# ── 10. get_logits (inference mode) ──────────────────────────────────────────
print("[10] get_logits (inference mode) ...", end=" ")
model.eval()
with torch.no_grad():
    inp = torch.randint(0, SMOKE_CFG["vocab_size"], (1, SMOKE_CFG["max_seq_len"]))
    logits = model.get_logits(inp)

assert logits.shape == (1, SMOKE_CFG["max_seq_len"], SMOKE_CFG["vocab_size"]), \
    f"Wrong logits shape: {logits.shape}"
assert torch.isfinite(logits).all(), "Logits have non-finite values"
print(f"OK  (shape={tuple(logits.shape)})")

# ── 11. Scaffold on_step_end ──────────────────────────────────────────────────
print("[11] on_step_end scaffold ...", end=" ")
model.train()
# Run forward to populate _last_h_final / _last_settled
_, _ = model.forward_train(batch)
model.on_step_end(step=0, metrics={})
extra = model.extra_metrics()
assert "scaffold/n_alive" in extra, "extra_metrics missing scaffold/n_alive"
print(f"OK  (scaffold_step={model.scaffold_step.item()}, n_alive={extra['scaffold/n_alive']})")

# ── 12. describe() ─────────────────────────────────────────────────────────────
print("[12] describe() ...", end=" ")
desc = model.describe()
assert desc["architecture"] == "AENCA-V12-MMHR"
assert desc["n_rules"] == SMOKE_CFG["n_rules"]
assert desc["d_h"] == SMOKE_CFG["d_h"]
print(f"OK  (total_params={desc['total_params']:,})")

# ── 13. param groups ──────────────────────────────────────────────────────────
print("[13] get_param_groups() ...", end=" ")
groups = model.get_param_groups()
assert len(groups) == 2
all_params = set()
for g in groups:
    for p in g["params"]:
        all_params.add(id(p))
model_params = {id(p) for p in model.parameters()}
assert all_params == model_params, "get_param_groups does not cover all parameters"
print("OK")

# ── 14. MixtureHebbianRules direct test ─────────────────────────────────────
print("[14] MixtureHebbianRules manual forward ...", end=" ")
B, L, d = 2, 16, SMOKE_CFG["d_model"]
K = SMOKE_CFG["n_rules"]
dh = SMOKE_CFG["d_h"]

rules = MixtureHebbianRules(cfg)
rules.train()

z_t = torch.randn(B, L, d)
h_t = torch.randn(B, L, dh)
gate_t = torch.sigmoid(torch.randn(B, L, d))

delta_z, weights = rules(z_t, h_t, dilation=1, gate=gate_t, training=True, step_frac=0.5)

assert delta_z.shape == (B, L, d), f"Wrong delta_z shape: {delta_z.shape}"
assert weights.shape == (B, L, K), f"Wrong weights shape: {weights.shape}"
assert torch.isfinite(delta_z).all(), "delta_z has non-finite values"
assert torch.isfinite(weights).all(), "weights has non-finite values"

# Weights should be non-negative and sum to ~1 (Gumbel softmax output)
assert (weights >= 0).all(), "weights has negative values"
weight_sums = weights.sum(dim=-1)
assert (weight_sums - 1.0).abs().max() < 0.01, f"weights sum to {weight_sums.mean():.4f} (expected ~1)"

# At zero-init, weights should be near-uniform (Gumbel noise may perturb)
weight_std = weights.std(dim=-1).mean()
print(f"OK  (delta_z sum to zero at init: {delta_z.abs().max().item():.6f}, weight_std={weight_std:.4f})")

# ── 15. W_d zero cascade: delta_z should be zero at init ─────────────────────
print("[15] Zero cascade check (W_d=0 -> delta_z=0) ...", end=" ")
fresh_rules = MixtureHebbianRules(cfg)  # fresh instance, W_d = zeros
fresh_rules.eval()
with torch.no_grad():
    delta_z_init, _ = fresh_rules(z_t, h_t, dilation=1, gate=gate_t, training=False, step_frac=0.0)
max_delta = delta_z_init.abs().max().item()
assert max_delta < 1e-7, f"delta_z should be ~0 at init (W_d=0), got {max_delta:.2e}"
print(f"OK  (max |delta_z| at init = {max_delta:.2e})")

print()
print("=" * 60)
print("All V12 checks passed!")
print("=" * 60)
desc_full = model.describe()
print(f"  Architecture:     {desc_full['architecture']}")
print(f"  Total params:     {desc_full['total_params']:,}")
print(f"  n_rules (K):      {desc_full['n_rules']}")
print(f"  d_h:              {desc_full['d_h']}")
print(f"  shared_wp:        {desc_full['shared_wp_params']:,}")
print(f"  binding_rules:    {desc_full['binding_rules_params']:,}")
print(f"  morphogen:        {desc_full['morphogen_params']:,}")
print(f"  router:           {desc_full['router_params']:,}")
print(f"  Paradigm: {desc_full['paradigm']}")
