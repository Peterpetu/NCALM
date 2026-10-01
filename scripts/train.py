#!/usr/bin/env python3
"""NCALM Training Script.

Usage:
    python scripts/train.py --profile smoke --model <model_name>
    python scripts/train.py --profile overnight --model <model_name> --name my_run_v1
    python scripts/train.py --profile smoke --model <model_name> --device cpu
"""

import argparse
import importlib
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ncalm.model import get_model_class, list_models
from ncalm.profiles import PROFILES
from ncalm.trainer import Trainer
from ncalm.logging_utils import format_console_line, format_val_line
from ncalm.data.dataset import SmolTalk2Data, ThePileData, TinyStoriesData, create_data_loader
from ncalm.data.tokenizer import create_tokenizer
from ncalm.env_utils import load_local_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="NCALM Training Harness")
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        required=True,
        help="Training profile",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device (cuda/cpu). Default: cuda",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Custom experiment name",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Registered model name",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Path to model config JSON (architecture params)",
    )
    parser.add_argument(
        "--model-module",
        default=None,
        help="Python module path to import (triggers model registration)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Override tokenizer name from profile/config",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override dataset name from profile/config",
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help="Optional directory for mirrored checkpoints",
    )
    parser.add_argument(
        "--data-cache-dir",
        default=None,
        help="Override tokenized dataset cache directory",
    )
    parser.add_argument(
        "--tokenizer-cache-dir",
        default=None,
        help="Override tokenizer cache directory",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Override Hugging Face datasets/hub cache directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def print_final_summary(trainer, val_metrics, total_time, config):
    """Print the final training summary."""
    avg_tps = trainer.tokens_processed / total_time if total_time > 0 else 0

    trainer.log("")
    trainer.log("=" * 78)
    trainer.log("  TRAINING COMPLETE")
    trainer.log("=" * 78)
    trainer.log("")
    trainer.log("  RESULTS")
    trainer.log(f"    Final Val PPL:    {val_metrics['val_ppl']:.2f}")
    trainer.log(f"    Best Val PPL:     {trainer.best_val_ppl:.2f}")
    trainer.log(f"    Final Val CE:     {val_metrics['val_ce']:.4f}")
    trainer.log("")
    trainer.log("  TRAINING STATISTICS")
    trainer.log(
        f"    Total time:       {total_time / 3600:.2f} hours "
        f"({total_time / 60:.1f} minutes)"
    )
    trainer.log(f"    Steps completed:  {trainer.step:,}")
    trainer.log(
        f"    Tokens processed: {trainer.tokens_processed:,} "
        f"({trainer.tokens_processed / 1e6:.1f}M)"
    )
    trainer.log(f"    Avg throughput:   {avg_tps:.0f} tok/s")
    trainer.log("")
    trainer.log("  GPU MEMORY")
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        trainer.log(f"    Peak allocated:   {peak_mem:.2f} GB")
    trainer.log("")
    trainer.log("=" * 78)


def write_analysis_json(
    trainer, val_metrics, total_time, config, exp_dir, model_desc, device, args
):
    """Write analysis.json for post-hoc comparison."""
    avg_tps = trainer.tokens_processed / total_time if total_time > 0 else 0

    analysis = {
        "experiment": {
            "name": args.name or args.profile,
            "profile": args.profile,
            "seed": args.seed,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "total_time_hours": total_time / 3600,
            "total_time_minutes": total_time / 60,
        },
        "results": {
            "final_val_ppl": val_metrics["val_ppl"],
            "final_val_ce": val_metrics["val_ce"],
            "best_val_ppl": trainer.best_val_ppl,
        },
        "model": model_desc,
        "training": {
            "max_steps": config["max_steps"],
            "steps_completed": trainer.step,
            "batch_size": config["batch_size"],
            "seq_len": config["seq_len"],
            "grad_accum_steps": int(config.get("grad_accum_steps", 1)),
            "effective_batch_tokens": (
                config["batch_size"]
                * config["seq_len"]
                * int(config.get("grad_accum_steps", 1))
            ),
            "total_tokens": trainer.tokens_processed,
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "grad_clip": config["grad_clip"],
            "warmup_steps": config["warmup_steps"],
            "beta2": config.get("beta2", 0.95),
            "lambda_agc": config.get("lambda_agc", 0.0),
            "lambda_reg": config.get("lambda_reg", 0.0),
            "lambda_aux": config.get("lambda_aux", 0.0),
            "dropout": config.get("dropout", 0.0),
            "optimizer": (
                f"AdamW (beta1=0.9, beta2={config.get('beta2', 0.95)})"
            ),
            "lr_schedule": "cosine with linear warmup (min_lr_ratio=0.1)",
            "amp": device.type == "cuda",
        },
        "performance": {
            "avg_tokens_per_sec": avg_tps,
            "gpu_peak_memory_gb": (
                torch.cuda.max_memory_allocated() / 1024**3
                if device.type == "cuda"
                else 0
            ),
        },
        "hardware": {
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A"
            ),
            "gpu_memory_total_gb": (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
                if device.type == "cuda"
                else 0
            ),
        },
    }
    with open(exp_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)


def main():
    load_local_env()
    args = parse_args()

    # ─── Load profile ───
    config = dict(PROFILES[args.profile])  # Copy so we don't mutate

    # ─── Load model config (architecture params) ───
    model_config = {}
    if args.model_config:
        with open(args.model_config) as f:
            model_config = json.load(f)
        config.update(model_config)

    if args.dataset is not None:
        config["dataset"] = args.dataset
    if args.tokenizer is not None:
        config["tokenizer_name"] = args.tokenizer
    if args.backup_dir is not None:
        config["backup_checkpoint_dir"] = args.backup_dir
    if args.data_cache_dir is not None:
        config["dataset_cache_dir"] = args.data_cache_dir
    if args.tokenizer_cache_dir is not None:
        config["tokenizer_cache_dir"] = args.tokenizer_cache_dir
    if args.hf_cache_dir is not None:
        config["hf_cache_dir"] = args.hf_cache_dir

    # ─── Device ───
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ─── Reproducibility seed ───
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    # Note: torch.backends.cudnn.deterministic is NOT set because it significantly
    # degrades performance.  The seed controls init + data order, which is sufficient
    # for reproducibility of aggregate metrics across runs.
    logger.info(f"Random seed: {seed}")

    # Record seed in config for traceability
    config["seed"] = seed

    if config.get("hf_cache_dir"):
        hf_cache = str(config["hf_cache_dir"])
        os.environ.setdefault("HF_HOME", hf_cache)
        os.environ.setdefault("HF_DATASETS_CACHE", hf_cache)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_cache)
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    # ─── Import model module if specified ───
    if args.model_module:
        importlib.import_module(args.model_module)

    # ─── Get model class ───
    if args.model is None:
        available = list_models()
        if len(available) == 0:
            print("ERROR: No models registered. Use --model-module to import one,")
            print("       or register a model with @register_model('name').")
            sys.exit(1)
        elif len(available) == 1:
            args.model = available[0]
            logger.info(f"Auto-selected only registered model: '{args.model}'")
        else:
            print(f"ERROR: Multiple models registered: {available}")
            print("       Use --model <name> to select one.")
            sys.exit(1)

    ModelClass = get_model_class(args.model)

    resume_state = None
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        resume_state = torch.load(args.resume, map_location=device, weights_only=False)
        ckpt_config = resume_state.get("config")
        if ckpt_config:
            merged = dict(ckpt_config)
            merged.update(config)
            config = merged

    # ─── Create tokenizer ───
    tokenizer_name = config.get("tokenizer_name", "bpe")
    tokenizer = create_tokenizer(
        tokenizer_name,
        vocab_size=config.get("tokenizer_vocab_size", config.get("vocab_size", 10000)),
        cache_dir=config.get("tokenizer_cache_dir"),
        cache_stem=config.get("tokenizer_cache_stem"),
        dataset_name=config.get("tokenizer_dataset_name", config.get("dataset_hf_name")),
        dataset_config=config.get("tokenizer_dataset_config", config.get("dataset_config")),
        dataset_splits=config.get("tokenizer_dataset_splits"),
        train_max_examples=config.get("tokenizer_train_max_examples"),
        pretrained_name=config.get("tokenizer_pretrained_name"),
    )
    # Take the larger of tokenizer's actual size and profile's requested size.
    # This ensures: (1) embedding table fits all token IDs, (2) profile can
    # request padding for GPU alignment (e.g. Pythia pads to 50304 for ×128).
    config["vocab_size"] = max(
        tokenizer.vocab_size,
        config.get("tokenizer_vocab_size", 0),
    )

    # ─── Create model ───
    model = ModelClass(config)
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"], strict=False)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    model_desc = model.describe()

    # ─── Create experiment directory ───
    model_name = model_desc.get("architecture", args.model)
    if args.resume is not None:
        exp_dir = Path(args.resume).resolve().parent
        exp_name = exp_dir.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = args.name or args.profile
        exp_dir = Path(f"experiments/{model_name}/{exp_name}_{timestamp}")
        exp_dir.mkdir(parents=True, exist_ok=True)

    # ─── Create dataset ───
    dataset_name = config.get("dataset", "tinystories")
    if dataset_name == "smoltalk2":
        data = SmolTalk2Data(
            seq_len=config["seq_len"],
            tokenizer=tokenizer,
            cache_name=config.get("dataset_cache_name", "smoltalk2_sft_tokens_5b.pt"),
            dataset_config=config.get("dataset_config", "SFT"),
            dataset_splits=config.get("dataset_splits"),
            dataset_sources=config.get("dataset_sources"),
            max_tokens=config.get("max_dataset_tokens", 5_000_000_000),
            cache_dir=config.get("dataset_cache_dir"),
        )
    elif dataset_name == "the_pile":
        data = ThePileData(
            seq_len=config["seq_len"],
            tokenizer=tokenizer,
            cache_name=config.get("dataset_cache_name", "pile_tokens_2b.bin"),
            dataset_name=config.get("dataset_hf_name"),
            max_tokens=config.get("max_dataset_tokens", 2_000_000_000),
            cache_dir=config.get("dataset_cache_dir"),
        )
    else:
        cache_name = config.get("dataset_cache_name", "tinystories_tokens_ncalm.pt")
        data = TinyStoriesData(
            seq_len=config["seq_len"],
            tokenizer=tokenizer,
            cache_name=cache_name,
        )
    train_loader = create_data_loader(data, config["batch_size"], device, "train")
    val_loader = create_data_loader(data, config["batch_size"], device, "val")

    # ─── Create trainer ───
    trainer = Trainer(model, config, device, exp_dir, resume_state=resume_state)

    # ─── Print banner ───
    token_budget = config["max_steps"] * config["batch_size"] * config["seq_len"]
    trainer.log("=" * 78)
    trainer.log(f"  {model_name}")
    trainer.log("=" * 78)
    trainer.log(f"  Params: {n_params:,} total")
    trainer.log(
        f"  Config: {config['max_steps']} steps, batch={config['batch_size']}, "
        f"seq_len={config['seq_len']}"
    )
    trainer.log(f"  Token budget: {token_budget:,}")
    trainer.log(f"  Device: {device}")
    trainer.log("=" * 78)

    # ─── Training loop ───
    log_interval = config["log_interval"]
    eval_interval = config["eval_interval"]
    eval_batches = config["eval_batches"]
    generate_interval = config["generate_interval"]
    save_interval = config["save_interval"]
    start_time = time.time()

    for step_idx in range(trainer.step, config["max_steps"]):
        grad_accum_steps = int(config.get("grad_accum_steps", 1))
        if grad_accum_steps > 1:
            batch = [next(train_loader) for _ in range(grad_accum_steps)]
        else:
            batch = next(train_loader)
        metrics = trainer.train_step(batch)
        trainer.log_metrics(metrics)

        step = metrics["step"]

        # ─── Console display ───
        if step % log_interval == 0 or step <= 5:
            trainer.log(format_console_line(metrics, config, start_time))

        # ─── Evaluation ───
        if step % eval_interval == 0:
            val_metrics = trainer.evaluate(val_loader, n_batches=eval_batches)
            trainer.log(format_val_line(val_metrics))
            trainer.log_metrics(
                {
                    "step": step,
                    "type": "eval",
                    **val_metrics,
                }
            )

            if val_metrics["val_ppl"] < trainer.best_val_ppl:
                trainer.best_val_ppl = val_metrics["val_ppl"]
                trainer.save_checkpoint("best_model.pt")
                trainer.log(f"  >>> New best Val PPL: {trainer.best_val_ppl:.1f}")

        # ─── Generation sample ───
        if step % generate_interval == 0:
            sample = trainer.generate_sample(tokenizer)
            trainer.log(f"  >>> Sample: {sample[:200]}...")

        # ─── Checkpoint ───
        if step % save_interval == 0:
            trainer.save_checkpoint(f"checkpoint_step{step}.pt")
            trainer.save_checkpoint("latest.pt")

    # ─── Final evaluation ───
    total_time = time.time() - start_time
    val_metrics = trainer.evaluate(val_loader, n_batches=50)
    trainer.save_checkpoint("final_model.pt")
    trainer.save_checkpoint("latest.pt")

    # Print final summary
    print_final_summary(trainer, val_metrics, total_time, config)

    # Write analysis.json
    write_analysis_json(
        trainer, val_metrics, total_time, config, exp_dir, model_desc, device, args
    )

    # Final generation samples
    trainer.log("\nFinal generation samples:")
    for prompt in ["Once upon a time", "The little cat", "One day, a"]:
        sample = trainer.generate_sample(tokenizer, prompt=prompt)
        trainer.log(f"  [{prompt}]: {sample[:300]}")

    trainer.close()
    logger.info(f"Experiment saved to: {exp_dir}")


if __name__ == "__main__":
    main()
