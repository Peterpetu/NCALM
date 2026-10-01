"""VRAM stress test round 2 — push to the limit on 16GB."""
import torch
import gc
from ncalm.models.v5.model import AENCAv5Model


def test_config(name, config):
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
        model.train()
        loss, info = model.forward_train(batch)
        loss.backward()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak (fwd+bwd): {peak_mem:.2f} GB")
        print(f"Headroom: {16.0 - peak_mem:.2f} GB")
        print(f"STATUS: OK")
        del model, batch, loss
        gc.collect()
        torch.cuda.empty_cache()
        return True, peak_mem
    except torch.cuda.OutOfMemoryError as e:
        print(f"STATUS: OOM")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return False, 16.0


base = {
    "mos_components": 1, "tie_embeddings": True,
    "vocab_size": 10000, "max_seq_len": 256, "seq_len": 256,
    "lambda_reg": 0.01, "lambda_aux": 0.1,
    "lambda_agc": 0.02, "checkpoint_interval": 4, "dropout": 0.0,
}

configs = [
    # Large with big batch
    ("Large_b64", {**base,
        "d_model": 640, "d_msg": 80, "d_ff": 1280, "d_pe": 64,
        "n_heads": 4, "n_trunk": 3, "kernel_radius": 1,
        "k_phase1": 4, "k_phase2": 8, "k_phase3": 4,
        "batch_size": 64,
    }),
    ("Large_b80", {**base,
        "d_model": 640, "d_msg": 80, "d_ff": 1280, "d_pe": 64,
        "n_heads": 4, "n_trunk": 3, "kernel_radius": 1,
        "k_phase1": 4, "k_phase2": 8, "k_phase3": 4,
        "batch_size": 80,
    }),
    # XLarge: d_model=768
    ("XLarge_b32", {**base,
        "d_model": 768, "d_msg": 96, "d_ff": 1536, "d_pe": 64,
        "n_heads": 4, "n_trunk": 3, "kernel_radius": 1,
        "k_phase1": 4, "k_phase2": 8, "k_phase3": 4,
        "batch_size": 32,
    }),
    ("XLarge_b48", {**base,
        "d_model": 768, "d_msg": 96, "d_ff": 1536, "d_pe": 64,
        "n_heads": 4, "n_trunk": 3, "kernel_radius": 1,
        "k_phase1": 4, "k_phase2": 8, "k_phase3": 4,
        "batch_size": 48,
    }),
    ("XLarge_b64", {**base,
        "d_model": 768, "d_msg": 96, "d_ff": 1536, "d_pe": 64,
        "n_heads": 4, "n_trunk": 3, "kernel_radius": 1,
        "k_phase1": 4, "k_phase2": 8, "k_phase3": 4,
        "batch_size": 64,
    }),
    # XXLarge: d_model=896
    ("XXLarge_b32", {**base,
        "d_model": 896, "d_msg": 112, "d_ff": 1792, "d_pe": 64,
        "n_heads": 4, "n_trunk": 3, "kernel_radius": 1,
        "k_phase1": 4, "k_phase2": 8, "k_phase3": 4,
        "batch_size": 32,
    }),
    ("XXLarge_b48", {**base,
        "d_model": 896, "d_msg": 112, "d_ff": 1792, "d_pe": 64,
        "n_heads": 4, "n_trunk": 3, "kernel_radius": 1,
        "k_phase1": 4, "k_phase2": 8, "k_phase3": 4,
        "batch_size": 48,
    }),
]

print("RTX 4080 Super — 16 GB VRAM — Round 2")
for name, cfg in configs:
    ok, mem = test_config(name, cfg)
    if not ok:
        print(f"\n>>> OOM at {name}. Previous config is the max.\n")
        break

print("\nDone.")
