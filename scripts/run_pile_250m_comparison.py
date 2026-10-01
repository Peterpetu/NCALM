#!/usr/bin/env python3
"""Run a quick 250M-token comparison: GPT-2 vs AENCA v5.1 vs AENCA v6.

Trains all three models sequentially on The Pile with matched hyperparameters.
250M tokens ≈ 3815 steps each.  Estimated wall times (RTX 4080 Super):
  GPT-2       ~1.75h
  AENCA v5.1  ~8h
  AENCA v6    ~5h   (fewer attention calls than v5.1)
  Total       ~15h

Usage:
    python scripts/run_pile_250m_comparison.py
    python scripts/run_pile_250m_comparison.py --seed 42
    python scripts/run_pile_250m_comparison.py --only gpt2
    python scripts/run_pile_250m_comparison.py --only v5
    python scripts/run_pile_250m_comparison.py --only v6
    python scripts/run_pile_250m_comparison.py --skip gpt2,v5
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────

RUNS = [
    {
        "key": "gpt2",
        "name": "GPT-2 Baseline (~162M params)",
        "profile": "pile_250m_gpt2",
        "model": "gpt2",
        "est_hours": 1.75,
    },
    {
        "key": "v6",
        "name": "AENCA v6 param-matched (~162M params, K=16, t_max=16)",
        "profile": "pile_250m_v6",
        "model": "v6",
        "est_hours": 6.0,
    },
    {
        "key": "v6_flops",
        "name": "AENCA v6 FLOP-matched (~59M params, same FLOPs as GPT-2)",
        "profile": "pile_250m_v6_flops",
        "model": "v6",
        "est_hours": 1.5,
    },
]

# ── Helpers ──────────────────────────────────────────────────────────────

def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{timestamp()}] {msg}", flush=True)


def fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def get_python() -> str:
    return sys.executable


def check_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            log("ERROR: CUDA not available. These runs require a GPU.")
            sys.exit(1)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        return gpu_name
    except ImportError:
        log("ERROR: PyTorch not installed.")
        sys.exit(1)


def check_disk_space(min_gb: float = 20.0):
    import shutil
    usage = shutil.disk_usage(Path.cwd())
    free_gb = usage.free / 1024**3
    if free_gb < min_gb:
        log(f"WARNING: Only {free_gb:.1f} GB free. Recommend ≥{min_gb} GB.")
    else:
        log(f"Disk: {free_gb:.1f} GB free")


def run_training(profile: str, model: str, seed: int,
                 backup_dir: str | None = None,
                 device: str = "cuda") -> tuple[int, float]:
    cmd = [
        get_python(), "scripts/train.py",
        "--profile", profile,
        "--model", model,
        "--device", device,
        "--seed", str(seed),
    ]
    if backup_dir:
        cmd.extend(["--backup-dir", backup_dir])

    log(f"Command: {' '.join(cmd)}")
    start = time.time()

    result = subprocess.run(
        cmd,
        cwd=str(Path(__file__).parent.parent),
    )

    duration = time.time() - start
    return result.returncode, duration


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run GPT-2 vs AENCA v5.1 vs AENCA v6 comparison (250M tokens each)"
    )
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
    p.add_argument("--device", default="cuda",
                    help="Device (default: cuda)")
    p.add_argument("--backup-dir", default=None,
                    help="Mirror checkpoints to this directory")
    p.add_argument("--only", default=None,
                    help="Run only this model (gpt2, v5, v6)")
    p.add_argument("--skip", default=None,
                    help="Comma-separated list of models to skip (gpt2, v5, v6)")
    return p.parse_args()


def main():
    args = parse_args()

    log("=" * 70)
    log("  NCALM 250M Quick Comparison")
    log("  GPT-2/NeoX vs AENCA v5.1 vs AENCA v6")
    log("  The Pile, 250M tokens, 3815 steps, seed=" + str(args.seed))
    log("=" * 70)
    log("")

    # Pre-flight checks
    gpu_name = check_gpu()
    check_disk_space()
    log(f"Python: {get_python()}")
    log(f"Working dir: {Path.cwd()}")
    log("")

    # Filter runs
    skip_keys = set()
    if args.skip:
        skip_keys = {k.strip().lower() for k in args.skip.split(",")}
    if args.only:
        only_key = args.only.strip().lower()
        skip_keys = {r["key"] for r in RUNS if r["key"] != only_key}

    active_runs = [r for r in RUNS if r["key"] not in skip_keys]

    if not active_runs:
        log("Nothing to run (all models skipped).")
        return

    total_est = sum(r["est_hours"] for r in active_runs)
    log(f"Planned runs: {len(active_runs)}, estimated total: ~{total_est:.1f}h")
    for r in active_runs:
        log(f"  • {r['name']} ({r['profile']}) — ~{r['est_hours']:.1f}h")
    log("")

    # ── Run each model ───────────────────────────────────────────────
    results = []
    campaign_start = time.time()

    for i, run in enumerate(active_runs, 1):
        log("─" * 70)
        log(f"  [{i}/{len(active_runs)}] Starting: {run['name']}")
        log(f"  Profile: {run['profile']}, Model: {run['model']}")
        log(f"  Estimated: ~{run['est_hours']:.1f}h")
        log("─" * 70)
        log("")

        exit_code, duration = run_training(
            profile=run["profile"],
            model=run["model"],
            seed=args.seed,
            backup_dir=args.backup_dir,
            device=args.device,
        )

        status = "SUCCESS" if exit_code == 0 else f"FAILED (exit code {exit_code})"
        log("")
        log(f"  {run['name']}: {status} in {fmt_duration(duration)}")
        log("")

        results.append({
            "name": run["name"],
            "key": run["key"],
            "profile": run["profile"],
            "exit_code": exit_code,
            "duration_sec": duration,
            "duration_human": fmt_duration(duration),
        })

        if exit_code != 0:
            log(f"ERROR: {run['name']} failed. Stopping campaign.")
            log(f"TIP: Fix the issue, then resume with: --skip {','.join(r['key'] for r in results if r['exit_code'] == 0)}")
            break

    # ── Final Summary ────────────────────────────────────────────────
    campaign_duration = time.time() - campaign_start

    log("")
    log("=" * 70)
    log("  CAMPAIGN COMPLETE")
    log("=" * 70)
    log(f"  Total wall time: {fmt_duration(campaign_duration)}")
    log(f"  Seed: {args.seed}")
    log(f"  GPU: {gpu_name}")
    log("")
    for r in results:
        icon = "✓" if r["exit_code"] == 0 else "✗"
        log(f"  {icon} {r['name']:30s}  {r['duration_human']:>12s}  "
            f"({'OK' if r['exit_code'] == 0 else 'FAILED'})")
    log("")

    # Save campaign summary
    summary_path = Path("experiments") / f"pile_250m_campaign_{args.seed}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "campaign": "GPT-2 vs AENCA v5.1 vs AENCA v6 — The Pile 250M",
        "seed": args.seed,
        "gpu": gpu_name,
        "tokens_per_model": 250_000_000,
        "steps_per_model": 3815,
        "started": datetime.datetime.now().isoformat(),
        "total_wall_time_sec": campaign_duration,
        "total_wall_time_human": fmt_duration(campaign_duration),
        "runs": results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"  Campaign summary saved to: {summary_path}")

    if any(r["exit_code"] != 0 for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
