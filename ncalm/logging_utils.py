"""Logging utilities: MetricsWriter, console formatting, log helpers.

Three log outputs per training run:
  1. metrics.jsonl — one JSON object per line (machine-readable)
  2. training.log  — human-readable mirror of console
  3. Console        — formatted step lines with ETA
"""

import json
import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class MetricsWriter:
    """Writes one JSON line per step to metrics.jsonl.

    Converts tensors to Python scalars for JSON serialization.
    Flushes after every write for crash safety.
    """

    def __init__(self, filepath: Path):
        self._file = open(filepath, "w", encoding="utf-8")

    def write(self, metrics: dict):
        serializable = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                serializable[k] = v.cpu().tolist() if v.dim() > 0 else v.item()
            else:
                serializable[k] = v
        self._file.write(json.dumps(serializable) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()


class TrainingLogger:
    """Writes to both Python logger (stderr) and training.log file."""

    def __init__(self, log_path: Path):
        self._file = open(log_path, "w", encoding="utf-8")

    def log(self, msg: str):
        logger.info(msg)
        self._file.write(msg + "\n")
        self._file.flush()

    def close(self):
        self._file.close()


def format_console_line(metrics: dict, config: dict, start_time: float) -> str:
    """Format a training step for console display.

    Matches the exact format from the blueprint:
      Step   50/5000 [A] | CE=7.1234 PPL=1238.5 | 670 tok/s ∇=0.41 GPU=4.3G ETA=9.2h
    """
    step = metrics["step"]
    max_steps = config["max_steps"]

    # ETA calculation
    elapsed = time.time() - start_time
    eta_str = ""
    if step >= 5:
        sec_per_step = elapsed / step
        remaining = sec_per_step * (max_steps - step)
        if remaining >= 3600:
            eta_str = f" ETA={remaining / 3600:.1f}h"
        else:
            eta_str = f" ETA={remaining / 60:.0f}m"

    # Core line
    line = (
        f"Step {step:5d}/{max_steps} [{metrics['phases']}] | "
        f"CE={metrics['ce']:.4f} PPL={metrics['ppl']:.1f}"
    )

    # Model-specific losses — any extra keys from the model's info dict
    # are included in metrics; the model can add them via forward_train's
    # info dict or via extra_metrics(). We don't hardcode them here —
    # the model controls what gets logged via its own metrics.

    # Infrastructure (always present)
    line += (
        f" | {metrics['tokens_per_sec']:.0f} tok/s "
        f"\u2207={metrics['grad_norm']:.2f}"
    )
    if metrics["gpu_mem_gb"] > 0:
        line += f" GPU={metrics['gpu_mem_gb']:.1f}G"
    line += eta_str

    return line


def format_val_line(val_metrics: dict) -> str:
    """Format validation results for console display."""
    return (
        f"  >>> VAL: CE={val_metrics['val_ce']:.4f} "
        f"PPL={val_metrics['val_ppl']:.1f}"
    )
