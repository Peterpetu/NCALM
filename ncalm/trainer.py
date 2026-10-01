"""Model-agnostic Trainer class.

Handles optimizer construction, LR scheduling, mixed precision (AMP),
training steps, validation, generation samples, and checkpointing.
The model is treated as a black box via the NCALMModel interface.
"""

import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

from ncalm.model import NCALMModel
from ncalm.scheduler import get_lr_scheduler
from ncalm.logging_utils import MetricsWriter, TrainingLogger


class Trainer:
    """Model-agnostic training harness.

    Handles:
    - Optimizer construction (from model.get_param_groups())
    - LR scheduling (cosine with linear warmup)
    - Mixed precision (AMP + GradScaler on CUDA)
    - Training step (forward, backward, clip, step)
    - Validation loop
    - Generation samples
    - Checkpointing (periodic, best, final)
    - Logging (metrics.jsonl, training.log, console)
    """

    def __init__(
        self,
        model: NCALMModel,
        config: dict,
        device: torch.device,
        exp_dir: Path,
        *,
        resume_state: dict | None = None,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.exp_dir = exp_dir

        # State
        self.step = 0
        self.tokens_processed = 0
        self.best_val_ppl = float("inf")
        self.phase_a_time = 0.0

        # Logging
        self.training_logger = TrainingLogger(exp_dir / "training.log")
        self.metrics_writer = MetricsWriter(exp_dir / "metrics.jsonl")

        # Save config
        with open(exp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # ─── Optimizer construction ───
        param_groups = model.get_param_groups()
        self.param_list = []
        optimizer_groups = []
        for group in param_groups:
            params = list(group["params"])
            self.param_list.extend(params)
            opt_group = {
                "params": params,
                "lr": group.get("lr", config["learning_rate"]),
                "weight_decay": group.get("weight_decay", config["weight_decay"]),
                "betas": group.get("betas", (0.9, config.get("beta2", 0.95))),
                "eps": 1e-8,
            }
            optimizer_groups.append(opt_group)
            label = group.get("label", "unnamed")
            n_params = sum(p.numel() for p in params)
            self.log(f"  Param group '{label}': {n_params:,} params, "
                     f"wd={opt_group['weight_decay']}")

        self.optimizer = AdamW(optimizer_groups)

        # ─── LR scheduler ───
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            warmup_steps=config["warmup_steps"],
            max_steps=config["max_steps"],
        )

        # ─── Mixed precision ───
        self.use_amp = device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp) if self.use_amp else None

        if resume_state is not None:
            self._load_resume_state(resume_state)

    def log(self, msg: str):
        """Write to both logger and training.log."""
        self.training_logger.log(msg)

    def _load_resume_state(self, resume_state: dict) -> None:
        """Restore trainer/optimizer/scheduler/scaler state from checkpoint."""
        self.step = int(resume_state.get("step", 0))
        self.tokens_processed = int(resume_state.get("tokens_processed", 0))
        self.best_val_ppl = float(resume_state.get("best_val_ppl", float("inf")))
        self.phase_a_time = float(resume_state.get("phase_a_time", 0.0))

        optimizer_state = resume_state.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        scheduler_state = resume_state.get("scheduler_state_dict")
        if scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        scaler_state = resume_state.get("scaler_state_dict")
        if self.scaler is not None and scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

        self.log(
            f"  Resumed from checkpoint: step={self.step:,}, "
            f"tokens={self.tokens_processed:,}, best_val_ppl={self.best_val_ppl:.2f}"
        )

    def log_metrics(self, metrics: dict):
        """Write one JSON line to metrics.jsonl."""
        self.metrics_writer.write(metrics)

    def _maybe_update_gradnorm(self, info: dict) -> None:
        """Run GradNorm every 100 steps if model provides individual losses.

        GradNorm (V4.1 §10.4, Eq 10.6-10.8) dynamically adjusts task loss
        weights based on relative training rates. Called BEFORE backward()
        so the computation graph is still alive for per-task gradient computation.

        Args:
            info: Dict returned by forward_train, may contain '_individual_losses'.
        """
        individual_losses = info.get("_individual_losses")
        if individual_losses is None:
            return
        if not hasattr(self.model, "loss_computer"):
            return
        # GradNorm update interval (Eq 10.9: "every 100 steps")
        # Configurable via profile; default 100. Higher values reduce VRAM spikes
        # from retain_graph=True (5 task gradients computed on the live graph).
        gradnorm_interval = self.config.get("gradnorm_interval", 100)
        if gradnorm_interval <= 0:
            return  # Disabled
        if self.step > 0 and self.step % gradnorm_interval == 0:
            loss_computer = getattr(self.model, "loss_computer", None)
            if loss_computer is None:
                return
            loss_computer.update_gradnorm(
                self.model, individual_losses, alpha=1.5
            )
            # GradNorm's retain_graph=True can spike VRAM; reclaim immediately
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _apply_agc(self, lambda_agc: float, eps: float = 1e-3) -> int:
        """Adaptive Gradient Clipping (V4.1 §4.1, Eq 4.2-4.3).

        Per-tensor gradient clipping based on gradient-to-weight norm ratio.
        Clips gradients where ||grad|| / max(||weight||, eps) > lambda_agc.

        Args:
            lambda_agc: Maximum allowed gradient-to-weight ratio (default 0.02).
            eps:        Minimum weight norm to prevent division issues.

        Returns:
            clip_count: Number of tensors that were clipped.
        """
        clip_count = 0
        for param in self.param_list:
            if param.grad is None:
                continue
            w_norm = param.data.norm(2).item()
            g_norm = param.grad.data.norm(2).item()
            ratio = g_norm / max(w_norm, eps)
            if ratio > lambda_agc:
                param.grad.data.mul_(lambda_agc / ratio)
                clip_count += 1
        return clip_count

    def _atomic_torch_save(self, obj: dict, path: Path) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)

    def _min_free_space_bytes(self) -> int:
        min_gb = float(self.config.get("min_free_disk_gb", 10.0))
        return int(min_gb * 1024**3)

    def _disk_free_gb(self, path: Path) -> float:
        return shutil.disk_usage(path).free / 1024**3

    def _ensure_free_space_for_checkpoint(self, path: Path) -> None:
        usage = shutil.disk_usage(path.parent)
        if usage.free < self._min_free_space_bytes():
            raise RuntimeError(
                f"Refusing to save checkpoint to {path.parent}: free space "
                f"{usage.free / 1024**3:.2f} GB is below configured minimum "
                f"{self.config.get('min_free_disk_gb', 10.0)} GB"
            )

    def _should_mirror_checkpoint(self, filename: str) -> bool:
        policy = str(self.config.get("backup_policy", "critical_only")).lower()
        if not self.config.get("backup_checkpoint_dir"):
            return False
        if policy == "all":
            return True
        if policy == "none":
            return False
        # default: mirror only critical recovery artifacts
        return filename in {"latest.pt", "best_model.pt", "final_model.pt"}

    def _mirror_checkpoint(self, path: Path) -> None:
        backup_root = self.config.get("backup_checkpoint_dir")
        if not backup_root:
            return

        backup_dir = Path(backup_root) / self.exp_dir.name
        backup_dir.mkdir(parents=True, exist_ok=True)
        dst = backup_dir / path.name
        tmp_dst = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(path, tmp_dst)
        os.replace(tmp_dst, dst)

    def _cleanup_old_step_checkpoints(self) -> None:
        keep_last = int(self.config.get("keep_last_n_checkpoints", 0))
        if keep_last <= 0:
            return

        checkpoints = sorted(
            self.exp_dir.glob("checkpoint_step*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        old = checkpoints[:-keep_last]
        for path in old:
            path.unlink(missing_ok=True)

            backup_root = self.config.get("backup_checkpoint_dir")
            if backup_root:
                backup_path = Path(backup_root) / self.exp_dir.name / path.name
                backup_path.unlink(missing_ok=True)

    def train_step(self, batch: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> dict:
        """Single training step. Returns metrics dict."""
        self.model.train()
        step_start = time.time()
        micro_batches = list(batch) if isinstance(batch, (list, tuple)) else [batch]
        accum_steps = len(micro_batches)
        if accum_steps <= 0:
            raise ValueError("train_step requires at least one batch")

        B, T = micro_batches[0].shape
        total_tokens_in_step = 0
        ce_weighted_sum = 0.0
        loss_value_sum = 0.0
        info_sums: dict[str, float] = {}
        info_counts: dict[str, int] = {}

        # Notify model
        self.model.on_step_start(self.step + 1, self.config["max_steps"])

        # ─── Phase A: Main loss ───
        phase_a_start = time.time()

        self.optimizer.zero_grad()

        if self.use_amp:
            assert self.scaler is not None
            for micro_batch in micro_batches:
                b_micro, t_micro = micro_batch.shape
                total_tokens_in_step += b_micro * t_micro
                with autocast("cuda", enabled=self.use_amp):
                    loss, info = self.model.forward_train(micro_batch)
                self._maybe_update_gradnorm(info)
                self.scaler.scale(loss / accum_steps).backward()
                loss_value_sum += float(loss.item())
                ce_val = info["ce"]
                if isinstance(ce_val, torch.Tensor):
                    ce_val = ce_val.item()
                ce_weighted_sum += float(ce_val) * b_micro * (t_micro - 1)
                for k, v in info.items():
                    if k.startswith("_"):
                        continue
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    info_sums[k] = info_sums.get(k, 0.0) + float(v)
                    info_counts[k] = info_counts.get(k, 0) + 1
            self.scaler.unscale_(self.optimizer)
            agc_clip_count = 0
            lambda_agc = self.config.get("lambda_agc", 0.0)
            if lambda_agc > 0:
                agc_clip_count = self._apply_agc(lambda_agc)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.param_list, self.config["grad_clip"]
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            for micro_batch in micro_batches:
                b_micro, t_micro = micro_batch.shape
                total_tokens_in_step += b_micro * t_micro
                loss, info = self.model.forward_train(micro_batch)
                self._maybe_update_gradnorm(info)
                (loss / accum_steps).backward()
                loss_value_sum += float(loss.item())
                ce_val = info["ce"]
                if isinstance(ce_val, torch.Tensor):
                    ce_val = ce_val.item()
                ce_weighted_sum += float(ce_val) * b_micro * (t_micro - 1)
                for k, v in info.items():
                    if k.startswith("_"):
                        continue
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    info_sums[k] = info_sums.get(k, 0.0) + float(v)
                    info_counts[k] = info_counts.get(k, 0) + 1
            agc_clip_count = 0
            lambda_agc = self.config.get("lambda_agc", 0.0)
            if lambda_agc > 0:
                agc_clip_count = self._apply_agc(lambda_agc)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.param_list, self.config["grad_clip"]
            )
            self.optimizer.step()

        self.scheduler.step()
        self.phase_a_time += time.time() - phase_a_start

        # ─── Tracking ───
        self.step += 1
        self.tokens_processed += total_tokens_in_step
        step_time = time.time() - step_start

        # ─── Build metrics dict ───
        predicted_tokens = sum(b.shape[0] * (b.shape[1] - 1) for b in micro_batches)
        ce_val = ce_weighted_sum / max(1, predicted_tokens)

        gpu_mem_gb = 0.0
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.memory_allocated() / 1024**3

        phases = self.model.get_phases(self.step, self.config["max_steps"])

        gpu_peak_gb = 0.0
        if torch.cuda.is_available():
            gpu_peak_gb = torch.cuda.max_memory_allocated() / 1024**3

        metrics = {
            "step": self.step,
            "timestamp": time.time(),
            # Core metrics (required for comparison)
            "loss": loss_value_sum / accum_steps,
            "ce": ce_val,
            "ppl": math.exp(min(ce_val, 20)),
            # Infrastructure metrics (required)
            "lr": self.scheduler.get_last_lr()[0],
            "grad_norm": (
                grad_norm.item()
                if isinstance(grad_norm, torch.Tensor)
                else grad_norm
            ),
            "tokens_per_sec": total_tokens_in_step / step_time,
            "step_time": step_time,
            "phase_a_time": self.phase_a_time,
            "gpu_mem_gb": gpu_mem_gb,
            "gpu_peak_gb": gpu_peak_gb,
            "phases": phases,
            "tokens_processed": self.tokens_processed,
            "agc_clip_count": agc_clip_count,
            "grad_accum_steps": accum_steps,
        }

        # Merge averaged model-specific metrics
        for k, total_v in info_sums.items():
            if k == "ce" or k.startswith("_"):
                continue
            metrics[k] = total_v / max(1, info_counts.get(k, 1))

        # Merge extra_metrics from model
        metrics.update(self.model.extra_metrics())

        # Notify model
        self.model.on_step_end(self.step, metrics)

        return metrics

    @torch.no_grad()
    def evaluate(self, val_loader, n_batches: int = 20) -> dict:
        """Compute validation CE and perplexity.

        Args:
            val_loader: Iterator yielding (B, T) batches from val split.
            n_batches: Number of batches to evaluate.

        Returns:
            {'val_ce': float, 'val_ppl': float}
        """
        self.model.eval()
        total_ce = 0.0
        total_tokens = 0

        for _ in range(n_batches):
            batch = next(val_loader)
            B, T = batch.shape

            if self.use_amp:
                with autocast("cuda", enabled=self.use_amp):
                    _, info = self.model.forward_train(batch)
            else:
                _, info = self.model.forward_train(batch)

            ce_val = info["ce"]
            if isinstance(ce_val, torch.Tensor):
                ce_val = ce_val.item()

            # Accumulate: predict T-1 positions (shifted by 1)
            total_ce += ce_val * B * (T - 1)
            total_tokens += B * (T - 1)

        avg_ce = total_ce / max(1, total_tokens)
        val_ppl = math.exp(min(avg_ce, 20))  # Cap exponent to prevent overflow

        self.model.train()
        return {"val_ce": avg_ce, "val_ppl": val_ppl}

    @torch.no_grad()
    def generate_sample(
        self,
        tokenizer,
        prompt: str = "Once upon a time",
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        """Generate a text sample for monitoring training progress."""
        self.model.eval()
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long, device=self.device
        )

        generated = input_ids.clone()
        for _ in range(max_tokens):
            # Context window
            ctx = generated[:, -512:] if generated.shape[1] > 512 else generated

            logits = self.model.get_logits(ctx)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.squeeze().item() == tokenizer.eos_token_id:
                break

        self.model.train()
        return tokenizer.decode(generated.squeeze().tolist())

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = self.exp_dir / filename
        local_free_before = self._disk_free_gb(path.parent)
        self.log(
            f"  Disk free before save ({path.parent.drive or path.parent}): {local_free_before:.2f} GB"
        )

        backup_root = self.config.get("backup_checkpoint_dir")
        if backup_root:
            backup_dir = Path(backup_root)
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_free_before = self._disk_free_gb(backup_dir)
            self.log(f"  Backup free before save ({backup_dir}): {backup_free_before:.2f} GB")

        self._ensure_free_space_for_checkpoint(path)
        checkpoint = {
            "step": self.step,
            "tokens_processed": self.tokens_processed,
            "phase_a_time": self.phase_a_time,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_ppl": self.best_val_ppl,
            "config": self.config,
        }
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        self._atomic_torch_save(checkpoint, path)
        if self._should_mirror_checkpoint(filename):
            self._mirror_checkpoint(path)
        if filename.startswith("checkpoint_step"):
            self._cleanup_old_step_checkpoints()
        self.log(f"  Saved checkpoint: {path.name}")
        local_free_after = self._disk_free_gb(path.parent)
        self.log(
            f"  Disk free after save ({path.parent.drive or path.parent}): {local_free_after:.2f} GB"
        )
        if backup_root:
            backup_dir = Path(backup_root)
            backup_free_after = self._disk_free_gb(backup_dir)
            self.log(f"  Backup free after save ({backup_dir}): {backup_free_after:.2f} GB")

    def close(self):
        """Close file handles."""
        self.training_logger.close()
        self.metrics_writer.close()
