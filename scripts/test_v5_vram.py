"""VRAM stress test for V5 configs. Find the largest safe config for 16GB RTX 4080 Super."""
import torch
import gc
from ncalm.models.v5.model import AENCAv5Model


def test_config(name, config):
    """Test if a config fits in VRAM with forward+backward."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        model = AENCAv5Model(config).cuda()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*60}")
        print(f"Config: {name}")
        print(f"Params: {n_params:,}")

        batch = torch.randint(0, 10000, (config["batch_size"], config["seq_len"]), device="cuda")

        # Forward
        model.train()
        loss, info = model.forward_train(batch)
        fwd_mem = torch.cuda.max_memory_allocated() / 1024**3

        # Backward
        loss.backward()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3

        print(f"CE: {info['ce'].item():.2f}")
        print(f"After forward: {fwd_mem:.2f} GB")
        print(f"Peak (fwd+bwd): {peak_mem:.2f} GB")
        print(f"Headroom: {16.0 - peak_mem:.2f} GB")
        print(f"STATUS: OK")

        del model, batch, loss
        gc.collect()
        torch.cuda.empty_cache()
        return True, peak_mem

    except torch.cuda.OutOfMemoryError as e:
        print(f"STATUS: OOM - {e}")
        gc.collect()
        torch.cuda.empty_cache()
        return False, 16.0


# Base config keys shared across all tests
base = {
    "mos_components": 1, "tie_embeddings": True,
    "vocab_size": 10000, "max_seq_len": 256, "seq_len": 256,
    "lambda_reg": 0.01, "lambda_aux": 0.1,
    "lambda_agc": 0.02, "checkpoint_interval": 4, "dropout": 0.0,
}

# Config A: Medium spec (d_model=512, K=16, n_trunk=3), batch=32
configs = [
    ("Medium_b32", {**base,
        "d_model": 512, "d_msg": 64, "d_ff": 1024, "d_pe": 64,
        "n_heads": 4, "n_trunk": 3, "kernel_radius": 1,
        "k_phase1": 4, "k_phase2": 8, "k_phase3": 4,
        "batch_size": 32,
    }),
    ("Medium_b48", {**base,
        "d_model": 512, "d_msg": 64, "d_ff": 1024, "d_pe": 64,
        "n_heads": 4, "n_trunk": 3, "kernel_radius": 1,
        "k_phase1": 4, "k_phase2": 8, "k_phase3": 4,
        "batch_size": 48,
    }),
    ("Medium_b64", {**base,
        "d_model": 512, "d_msg": 64, "d_ff": 1024, "d_pe": 64,
        "n_heads": 4, "n_trunk": 3, "kernel_radius": 1,
        "k_phase1": 4, "k_phase2": 8, "k_phase3": 4,
        "batch_size": 64,
    }),
    ("Large_b32", {**base,
        "d_model": 640, "d_msg": 80, "d_ff": 1280, "d_pe": 64,
        "n_heads": 4, "n_trunk": 3, "kernel_radius": 1,
        "k_phase1": 4, "k_phase2": 8, "k_phase3": 4,
        "batch_size": 32,
    }),
    ("Large_b48", {**base,
        "d_model": 640, "d_msg": 80, "d_ff": 1280, "d_pe": 64,
        "n_heads": 4, "n_trunk": 3, "kernel_radius": 1,
        "k_phase1": 4, "k_phase2": 8, "k_phase3": 4,
        "batch_size": 48,
    }),
]

print("RTX 4080 Super — 16 GB VRAM")
print("Testing V5 configs for maximum safe utilization...")

for name, cfg in configs:
    ok, mem = test_config(name, cfg)
    if not ok:
        print(f"\n>>> Stopping at {name} (OOM). Previous config is the max.\n")
        break

print("\nDone.")
