#!/usr/bin/env python3
"""Resume the latest checkpoint from an experiment directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume NCALM training from latest.pt")
    parser.add_argument("--exp-dir", required=True, help="Experiment directory")
    parser.add_argument("--profile", required=True, help="Profile name to use")
    parser.add_argument("--model", required=True, help="Registered model name")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--model-module", default=None, help="Optional model module import")
    args, extra = parser.parse_known_args()

    exp_dir = Path(args.exp_dir)
    latest = exp_dir / "latest.pt"
    if not latest.exists():
        raise FileNotFoundError(f"Could not find latest checkpoint: {latest}")

    cmd = [
        sys.executable,
        "scripts/train.py",
        "--profile",
        args.profile,
        "--model",
        args.model,
        "--device",
        args.device,
        "--resume",
        str(latest),
    ]
    if args.model_module:
        cmd.extend(["--model-module", args.model_module])
    cmd.extend(extra)

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
