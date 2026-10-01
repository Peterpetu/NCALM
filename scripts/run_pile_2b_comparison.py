#!/usr/bin/env python3
"""Run the full AENCA v5.1 vs GPT-2 academic comparison (2B tokens each).

Trains both models sequentially on The Pile with matched hyperparameters.
GPT-2 runs first (~14h), then AENCA (~67h). Total: ~3.5 days.

Usage:
    python scripts/run_pile_2b_comparison.py
    python scripts/run_pile_2b_comparison.py --seed 42
    python scripts/run_pile_2b_comparison.py --skip-gpt2      # resume AENCA only
    python scripts/run_pile_2b_comparison.py --skip-aenca      # GPT-2 only
    python scripts/run_pile_2b_comparison.py --backup-dir /path/to/backups
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
        "name": "GPT-2 Baseline",
        "profile": "pile_2b_gpt2",
        "model": "gpt2",
        "est_hours": 14,
    },
    {
        "name": "AENCA v5.1",
        "profile": "pile_2b_aenca",
        "model": "v5",
        "est_hours": 67,
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
    """Return the current Python executable (should be the venv one)."""
    return sys.executable


def check_gpu():
    """Verify CUDA is available before starting."""
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


def check_disk_space(min_gb: float = 40.0):
    """Ensure enough disk for checkpoints (~1.8GB each × several saves)."""
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
    """Launch a training run as a subprocess. Returns (exit_code, duration_sec)."""
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

    # Stream output directly to console (no buffering)
    result = subprocess.run(
        cmd,
        cwd=str(Path(__file__).parent.parent),
        # Inherit stdout/stderr so user sees live output
    )

    duration = time.time() - start
    return result.returncode, duration


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run AENCA v5.1 vs GPT-2 academic comparison (2B tokens each)"
    )
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
    p.add_argument("--device", default="cuda",
                    help="Device (default: cuda)")
    p.add_argument("--backup-dir", default=None,
                    help="Mirror checkpoints to this directory")
    p.add_argument("--skip-gpt2", action="store_true",
                    help="Skip GPT-2 training (e.g. if already completed)")
    p.add_argument("--skip-aenca", action="store_true",
                    help="Skip AENCA training (e.g. if already completed)")
    return p.parse_args()


def main():
    args = parse_args()

    log("=" * 70)
    log("  NCALM Academic Comparison: AENCA v5.1 vs GPT-2/NeoX")
    log("  The Pile, 2B tokens, ~160M params each, seed=" + str(args.seed))
    log("=" * 70)
    log("")

    # Pre-flight checks
    gpu_name = check_gpu()
    check_disk_space()
    log(f"Python: {get_python()}")
    log(f"Working dir: {Path.cwd()}")
    log("")

    # Filter runs based on skip flags
    skip_map = {"GPT-2 Baseline": args.skip_gpt2, "AENCA v5.1": args.skip_aenca}
    active_runs = [r for r in RUNS if not skip_map.get(r["name"], False)]

    if not active_runs:
        log("Nothing to run (both models skipped).")
        return

    total_est = sum(r["est_hours"] for r in active_runs)
    log(f"Planned runs: {len(active_runs)}, estimated total: ~{total_est:.0f}h")
    for r in active_runs:
        log(f"  • {r['name']} ({r['profile']}) — ~{r['est_hours']}h")
    log("")

    # ── Run each model ───────────────────────────────────────────────
    results = []
    campaign_start = time.time()

    for i, run in enumerate(active_runs, 1):
        log("─" * 70)
        log(f"  [{i}/{len(active_runs)}] Starting: {run['name']}")
        log(f"  Profile: {run['profile']}, Model: {run['model']}")
        log(f"  Estimated: ~{run['est_hours']}h")
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
            "profile": run["profile"],
            "exit_code": exit_code,
            "duration_sec": duration,
            "duration_human": fmt_duration(duration),
        })

        if exit_code != 0:
            log(f"ERROR: {run['name']} failed. Stopping campaign.")
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
        log(f"  {icon} {r['name']:20s}  {r['duration_human']:>12s}  "
            f"({'OK' if r['exit_code'] == 0 else 'FAILED'})")
    log("")

    # Save campaign summary
    summary_path = Path("experiments") / f"pile_2b_campaign_{args.seed}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "campaign": "AENCA v5.1 vs GPT-2/NeoX — The Pile 2B",
        "seed": args.seed,
        "gpu": gpu_name,
        "started": datetime.datetime.now().isoformat(),
        "total_wall_time_sec": campaign_duration,
        "total_wall_time_human": fmt_duration(campaign_duration),
        "runs": results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"  Campaign summary saved to: {summary_path}")

    # Exit with error if any run failed
    if any(r["exit_code"] != 0 for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
