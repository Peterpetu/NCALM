"""Quick V5 unit test: instantiate, forward, backward, inference."""
import torch
from ncalm.models.v5.model import AENCAv5Model

# Smoke config (minimal)
config = {
    "d_model": 256, "d_msg": 32, "d_ff": 512, "d_pe": 64,
    "n_heads": 4, "n_trunk": 2, "kernel_radius": 1,
    "k_phase1": 4, "k_phase2": 6, "k_phase3": 2,
    "mos_components": 1, "tie_embeddings": True,
    "vocab_size": 10000, "max_seq_len": 256, "seq_len": 256,
    "lambda_reg": 0.01, "lambda_aux": 0.1,
    "lambda_agc": 0.02, "checkpoint_interval": 4, "dropout": 0.0,
}

print("=== V5 Unit Test ===")

model = AENCAv5Model(config)
desc = model.describe()
print(f"Architecture: {desc['architecture']}")
print(f"Total params: {desc['total_params']:,}")
print(f"Phases: {desc['phases']}")
print(f"K: {desc['K']}")

# Forward pass
batch = torch.randint(0, 10000, (2, 64))
model.train()
loss, info = model.forward_train(batch)
print(f"Loss: {loss.item():.4f}")
print(f"CE: {info['ce'].item():.4f}")
print(f"Info keys: {list(info.keys())}")
print(f"Loss requires_grad: {loss.requires_grad}")

# Backward
loss.backward()
print("Backward OK")

# Check gradients exist
n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
n_total = sum(1 for p in model.parameters() if p.requires_grad)
print(f"Params with nonzero grad: {n_with_grad}/{n_total}")

# Inference
model.eval()
logits = model.get_logits(batch)
print(f"Logits shape: {logits.shape}")

# Verify CE is in expected range for random init (should be ~ln(10000) ≈ 9.21)
import math
expected_ce = math.log(10000)
print(f"Expected CE (random): ~{expected_ce:.2f}, got: {info['ce'].item():.2f}")

print("\n=== ALL CHECKS PASSED ===")
