"""Find d_ff / d_slow_ff for V6 to match GPT-2's ~162M params."""
import io, contextlib, sys
sys.path.insert(0, ".")

from ncalm.model import get_model_class

target = 162_300_000

M6 = get_model_class("v6")

results = []
for d_ff in range(2048, 4096+1, 128):
    for d_slow_ff in [1536, 2048, 2560, 3072]:
        cfg = {
            "vocab_size": 50304, "d_model": 768, "d_msg": 96,
            "d_ff": d_ff, "d_pe": 64, "d_slow_ff": d_slow_ff,
            "max_seq_len": 1024, "n_heads": 4, "n_trunk": 3,
            "kernel_radius": 2, "n_theta_cycles": 4, "t_max_gamma": 6,
            "t_min_gamma": 1, "halt_prior_p": 0.2, "lambda_halt": 0.01,
            "mos_components": 1, "tie_embeddings": False,
            "lambda_reg": 0.005, "lambda_aux": 0.05, "lambda_agc": 0.02,
            "checkpoint_interval": 4, "dropout": 0.0,
        }
        with contextlib.redirect_stdout(io.StringIO()):
            m = M6(cfg)
        total = sum(p.numel() for p in m.parameters())
        diff = total - target
        if abs(diff) < 3_000_000:
            fast_n = sum(p.numel() for p in m.fast_kernel.parameters())
            slow_n = sum(p.numel() for p in m.slow_updater.parameters())
            results.append((abs(diff), d_ff, d_slow_ff, total, diff, fast_n, slow_n))

results.sort()
print("Closest V6 configs to 162.3M:")
print(f"{'d_ff':>6s}  {'d_slow_ff':>10s}  {'total':>8s}  {'diff':>8s}  {'fast':>8s}  {'slow':>8s}")
for _, d_ff, d_slow_ff, total, diff, fast_n, slow_n in results[:10]:
    print(f"{d_ff:6d}  {d_slow_ff:10d}  {total/1e6:7.1f}M  {diff/1e6:+7.1f}M  {fast_n/1e6:7.1f}M  {slow_n/1e6:7.1f}M")
