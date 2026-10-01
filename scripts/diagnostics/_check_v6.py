import io, contextlib, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from ncalm.model import get_model_class

M6 = get_model_class("v6")

for d_ff, d_slow_ff in [(2688, 2048), (2720, 2048), (2752, 2048), (2752, 1536)]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m = M6({
            "vocab_size": 50304, "d_model": 768, "d_msg": 96,
            "d_ff": d_ff, "d_pe": 64, "d_slow_ff": d_slow_ff,
            "max_seq_len": 1024, "n_heads": 4, "n_trunk": 3,
            "kernel_radius": 2, "n_theta_cycles": 4, "t_max_gamma": 6,
            "t_min_gamma": 1, "halt_prior_p": 0.2, "lambda_halt": 0.01,
            "mos_components": 1, "tie_embeddings": False,
            "lambda_reg": 0.005, "lambda_aux": 0.05, "lambda_agc": 0.02,
            "checkpoint_interval": 4, "dropout": 0.0,
        })
    t = sum(p.numel() for p in m.parameters())
    diff = t - 162_300_000
    sys.stderr.write(f"d_ff={d_ff} d_slow_ff={d_slow_ff}: {t:,} ({t/1e6:.1f}M) diff={diff/1e6:+.2f}M\n")
