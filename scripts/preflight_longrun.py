#!/usr/bin/env python3
"""Preflight checks for long-running NCALM training jobs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ncalm.profiles import PROFILES
from ncalm.env_utils import load_local_env
def bytes_to_gb(n: int) -> float:
    return n / 1024**3


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description="Preflight long-run training configuration")
    parser.add_argument("--profile", required=True, choices=list(PROFILES.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backup-dir", default=None)
    parser.add_argument("--data-cache-dir", default=None)
    parser.add_argument("--tokenizer-cache-dir", default=None)
    parser.add_argument("--hf-cache-dir", default=None)
    args = parser.parse_args()

    config = dict(PROFILES[args.profile])
    if args.data_cache_dir:
        config["dataset_cache_dir"] = args.data_cache_dir
    if args.tokenizer_cache_dir:
        config["tokenizer_cache_dir"] = args.tokenizer_cache_dir
    if args.hf_cache_dir:
        config["hf_cache_dir"] = args.hf_cache_dir
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if config.get("hf_cache_dir"):
        hf_cache = str(config["hf_cache_dir"])
        os.environ.setdefault("HF_HOME", hf_cache)
        os.environ.setdefault("HF_DATASETS_CACHE", hf_cache)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_cache)
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    report: dict[str, object] = {
        "profile": args.profile,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available() and device.type == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["gpu_total_gb"] = round(bytes_to_gb(torch.cuda.get_device_properties(0).total_memory), 2)

    disk = shutil.disk_usage(Path.cwd())
    report["cwd_free_gb"] = round(bytes_to_gb(disk.free), 2)

    report["tokenizer_name"] = config.get("tokenizer_name", "bpe")
    report["tokenizer_vocab_size"] = config.get(
        "tokenizer_vocab_size", config.get("vocab_size", 10000)
    )
    report["dataset"] = config.get("dataset", "tinystories")
    report["dataset_config"] = config.get("dataset_config")
    report["dataset_splits"] = config.get("dataset_splits")
    report["dataset_cache_dir"] = config.get("dataset_cache_dir")
    report["tokenizer_cache_dir"] = config.get("tokenizer_cache_dir")
    report["hf_cache_dir"] = config.get("hf_cache_dir")

    batch_size = int(config["batch_size"])
    seq_len = int(config["seq_len"])
    max_steps = int(config["max_steps"])
    grad_accum_steps = int(config.get("grad_accum_steps", 1))
    report["micro_batch_size"] = batch_size
    report["grad_accum_steps"] = grad_accum_steps
    report["effective_batch_tokens"] = batch_size * seq_len * grad_accum_steps
    report["total_tokens"] = batch_size * seq_len * max_steps * grad_accum_steps

    if args.backup_dir:
        backup_path = Path(args.backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        backup_usage = shutil.disk_usage(backup_path)
        report["backup_dir"] = str(backup_path)
        report["backup_free_gb"] = round(bytes_to_gb(backup_usage.free), 2)

    if report["cwd_free_gb"] < 150:
        report["warning_low_working_disk"] = (
            "Working drive free space may be tight for large token caches and checkpoints"
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
