#!/usr/bin/env python3
"""Interactive Chat Script for AENCA v5.1 models.

Features:
  - Proper chat-template formatting (<|user|> / <|assistant|> role tokens)
  - NCA introspection: per-step gate magnitudes, state convergence, PRMP activity
  - Auto-discovers latest checkpoint from experiment directory
  - Streaming token output with per-token entropy display
  - Multi-turn conversation with context management

Usage:
    python scripts/chat.py --checkpoint experiments/AENCA\ v5.1/smoltalk2_5b_mixed_run1_20260312_170948/best_model.pt
    python scripts/chat.py --experiment "AENCA v5.1/smoltalk2_5b_mixed_run1_20260312_170948"
    python scripts/chat.py --checkpoint path/to/best_model.pt --nca-diagnostics
"""

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncalm.model import get_model_class, list_models
from ncalm.data.tokenizer import create_tokenizer


# ─── Terminal Formatting ─────────────────────────────────────────────────────


class C:
    """ANSI color codes."""
    BOLD      = "\033[1m"
    DIM       = "\033[2m"
    ITALIC    = "\033[3m"
    UNDERLINE = "\033[4m"
    CYAN      = "\033[36m"
    GREEN     = "\033[32m"
    YELLOW    = "\033[33m"
    MAGENTA   = "\033[35m"
    RED       = "\033[31m"
    BLUE      = "\033[34m"
    WHITE     = "\033[37m"
    GRAY      = "\033[90m"
    RESET     = "\033[0m"


def styled(text: str, *styles: str) -> str:
    prefix = "".join(styles)
    return f"{prefix}{text}{C.RESET}" if prefix else text


def bar_chart(value: float, max_val: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    """Render a mini bar chart."""
    ratio = min(max(value / max_val, 0.0), 1.0) if max_val > 0 else 0.0
    filled = int(ratio * width)
    return fill * filled + empty * (width - filled)


# ─── NCA Diagnostics Engine ─────────────────────────────────────────────────


class NCADiagnostics:
    """Captures per-step NCA dynamics during inference."""

    def __init__(self):
        self.steps: list[dict] = []
        self.initial_state_norm: float = 0.0
        self.final_state_norm: float = 0.0

    def clear(self):
        self.steps = []
        self.initial_state_norm = 0.0
        self.final_state_norm = 0.0

    def record_step(self, k: int, phase: int, gate: torch.Tensor,
                    delta_s: torch.Tensor, prmp_out: torch.Tensor,
                    state: torch.Tensor):
        """Record diagnostics for one NCA step."""
        with torch.no_grad():
            # Gate statistics (how open are the gates?)
            gate_mean = gate.mean().item()
            gate_std = gate.std().item()
            gate_max = gate.max().item()
            gate_min = gate.min().item()

            # State update magnitude
            delta_norm = delta_s.norm(dim=-1).mean().item()
            effective_update = (gate * delta_s).norm(dim=-1).mean().item()

            # PRMP activity (how much global communication?)
            prmp_norm = prmp_out.norm(dim=-1).mean().item()

            # State norm after update
            state_norm = state.norm(dim=-1).mean().item()

            # Per-position gate variance (are different tokens getting different compute?)
            gate_pos_var = gate.mean(dim=-1).var(dim=-1).mean().item()

            self.steps.append({
                "k": k,
                "phase": phase,
                "gate_mean": gate_mean,
                "gate_std": gate_std,
                "gate_max": gate_max,
                "gate_min": gate_min,
                "delta_norm": delta_norm,
                "effective_update": effective_update,
                "prmp_norm": prmp_norm,
                "state_norm": state_norm,
                "gate_pos_var": gate_pos_var,
            })

    def format_report(self) -> str:
        """Format a pretty diagnostics report."""
        if not self.steps:
            return "  No diagnostics captured."

        lines = []
        lines.append("")
        lines.append(styled("  ╔══════════════════════════════════════════════════════════════════════╗", C.CYAN))
        lines.append(styled("  ║               NCA STEP-BY-STEP DIAGNOSTICS                         ║", C.CYAN, C.BOLD))
        lines.append(styled("  ╚══════════════════════════════════════════════════════════════════════╝", C.CYAN))
        lines.append("")

        # Header
        lines.append(styled(
            "  Step  Phase  Gate \u03bc   Gate \u03c3   \u0394S Norm  Eff.Updt  PRMP \u2016M\u2016  Gate Visual",
            C.BOLD
        ))
        lines.append(styled("  " + "\u2500" * 72, C.DIM))

        phase_names = {0: "Prop", 1: "Intg", 2: "Refn"}
        phase_colors = {0: C.GREEN, 1: C.YELLOW, 2: C.MAGENTA}
        max_gate = max(s["gate_mean"] for s in self.steps) if self.steps else 1.0

        for s in self.steps:
            phase_name = styled(f"{phase_names[s['phase']]:>4}", phase_colors[s['phase']])
            gate_bar = bar_chart(s["gate_mean"], max(max_gate, 0.01), width=16)

            # Color gate mean by magnitude
            gm = s["gate_mean"]
            if gm > 0.3:
                gate_color = C.RED
            elif gm > 0.15:
                gate_color = C.YELLOW
            else:
                gate_color = C.GREEN

            line = (
                f"  {s['k']:>3}   {phase_name}   "
                f"{styled(f'{gm:.4f}', gate_color)}   "
                f"{s['gate_std']:.4f}   "
                f"{s['delta_norm']:.4f}   "
                f"{s['effective_update']:.5f}   "
                f"{s['prmp_norm']:>7.3f}   "
                f"{styled(gate_bar, gate_color)}"
            )
            lines.append(line)

        # Phase boundaries
        lines.append(styled("  " + "\u2500" * 72, C.DIM))

        # Phase summary
        lines.append(styled("\n  Phase Summary:", C.BOLD))
        for p_idx in range(3):
            phase_steps = [s for s in self.steps if s["phase"] == p_idx]
            if not phase_steps:
                continue
            avg_gate = sum(s["gate_mean"] for s in phase_steps) / len(phase_steps)
            avg_update = sum(s["effective_update"] for s in phase_steps) / len(phase_steps)
            avg_prmp = sum(s["prmp_norm"] for s in phase_steps) / len(phase_steps)
            total_update = sum(s["effective_update"] for s in phase_steps)

            lines.append(
                f"    {styled(phase_names[p_idx], phase_colors[p_idx], C.BOLD):>20}  "
                f"({len(phase_steps)} steps)  "
                f"avg gate={avg_gate:.4f}  "
                f"avg update={avg_update:.5f}  "
                f"total update={total_update:.4f}  "
                f"avg PRMP={avg_prmp:.3f}"
            )

        # Convergence analysis
        lines.append(styled("\n  Convergence Analysis:", C.BOLD))
        if len(self.steps) >= 2:
            last_updates = [s["effective_update"] for s in self.steps[-4:]]
            first_updates = [s["effective_update"] for s in self.steps[:4]]
            avg_last = sum(last_updates) / len(last_updates)
            avg_first = sum(first_updates) / len(first_updates)
            if avg_first > 0:
                convergence_ratio = avg_last / avg_first
                if convergence_ratio < 0.3:
                    conv_status = styled("CONVERGED", C.GREEN, C.BOLD)
                elif convergence_ratio < 0.7:
                    conv_status = styled("CONVERGING", C.YELLOW)
                else:
                    conv_status = styled("STILL ACTIVE", C.RED)
                lines.append(f"    Last/First update ratio: {convergence_ratio:.3f} \u2014 {conv_status}")
            else:
                lines.append("    First updates near zero \u2014 model may be under-activated")

        # State drift
        if self.initial_state_norm > 0:
            drift = abs(self.final_state_norm - self.initial_state_norm) / self.initial_state_norm
            lines.append(f"    State norm: {self.initial_state_norm:.2f} \u2192 {self.final_state_norm:.2f} (drift: {drift:.1%})")

        lines.append("")
        return "\n".join(lines)


# ─── Model Loading ───────────────────────────────────────────────────────────


def find_checkpoint(experiment_name: str) -> Path:
    """Find the best checkpoint in an experiment directory."""
    exp_dir = PROJECT_ROOT / "experiments" / experiment_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    for name in ["best_model.pt", "final_model.pt", "latest.pt"]:
        path = exp_dir / name
        if path.exists():
            return path

    raise FileNotFoundError(f"No checkpoint found in {exp_dir}")


def load_model(
    checkpoint_path: str, device: torch.device, model_name: str | None = None
) -> tuple:
    """Load trained model from checkpoint.

    Returns: (model, config_dict, info_dict)
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    print(styled(f"  Loading checkpoint: {path}", C.DIM))
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Get config
    config_dict = checkpoint.get("config", None)
    if config_dict is None:
        for parent in [path.parent, path.parent.parent]:
            config_json = parent / "config.json"
            if config_json.exists():
                with open(config_json) as f:
                    config_dict = json.load(f)
                break
        if config_dict is None:
            raise RuntimeError("No config found in checkpoint or experiment directory")

    # Determine model class
    if model_name is None:
        model_name = config_dict.get("model_name", "v5")
    ModelClass = get_model_class(model_name)

    # Build model with dropout=0.0 for inference
    inf_config = dict(config_dict)
    inf_config["dropout"] = 0.0
    model = ModelClass(inf_config)

    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if missing:
        print(styled(f"  New parameters (initialized fresh): {len(missing)}", C.DIM))
    if unexpected:
        print(styled(f"  Deprecated parameters (ignored): {len(unexpected)}", C.DIM))

    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    info = {
        "step": checkpoint.get("step", "?"),
        "best_val_ppl": checkpoint.get("best_val_ppl", "?"),
        "parameters": n_params,
        "checkpoint_name": path.name,
        "experiment_dir": str(path.parent.name),
    }
    return model, config_dict, info


def create_tokenizer_from_config(config_dict: dict):
    """Create the correct tokenizer based on training config."""
    tok_name = config_dict.get("tokenizer_name", "bpe")
    tok_vocab = config_dict.get("tokenizer_vocab_size", config_dict.get("vocab_size", 32000))

    return create_tokenizer(
        tok_name,
        vocab_size=tok_vocab,
        cache_dir=config_dict.get("tokenizer_cache_dir", config_dict.get("dataset_cache_dir")),
        pretrained_name=config_dict.get("tokenizer_pretrained_name"),
        dataset_name=config_dict.get("dataset_hf_name"),
    )


# ─── Instrumented Forward Pass ───────────────────────────────────────────────


@torch.no_grad()
def forward_with_diagnostics(
    model, input_ids: torch.Tensor, diagnostics: NCADiagnostics
) -> torch.Tensor:
    """Run model forward pass while capturing NCA step diagnostics.

    Returns logits (B, L, V) and populates the diagnostics object.
    """
    diagnostics.clear()

    B, L = input_ids.shape
    cfg = model.config

    # Embed tokens
    s = model.token_embedding(input_ids)
    s = s + model.state_pe(L).unsqueeze(0)

    # Position encoding for kernel
    pe = model.pos_encoder(L).unsqueeze(0).expand(B, -1, -1)

    # Record initial state
    diagnostics.initial_state_norm = s.norm(dim=-1).mean().item()

    # Build phase map
    phase_map = []
    for p_idx, p_steps in enumerate(cfg.phase_steps):
        phase_map.extend([p_idx] * p_steps)

    # Run K NCA steps with instrumentation
    K = cfg.total_steps
    for k in range(K):
        p_idx = phase_map[k]
        P = model.convs[p_idx](s)
        M = model.prmp(s)
        delta_s, gate = model.kernels[p_idx](P, M, pe)

        # Record diagnostics BEFORE update
        diagnostics.record_step(k, p_idx, gate, delta_s, M, s)

        # Apply update
        s = s + gate * delta_s

    # Record final state
    diagnostics.final_state_norm = s.norm(dim=-1).mean().item()

    # Readout
    logits = model.readout(s)
    return logits


# ─── Token Analysis ──────────────────────────────────────────────────────────


def compute_token_stats(logits_last: torch.Tensor, next_token_id: int) -> dict:
    """Compute statistics about the model's prediction."""
    probs = F.softmax(logits_last, dim=-1)  # (V,)
    log_probs = F.log_softmax(logits_last, dim=-1)

    # Entropy of prediction distribution (bits)
    entropy = -(probs * log_probs).sum().item() / math.log(2)

    # Rank of chosen token
    sorted_indices = probs.argsort(descending=True)
    rank = (sorted_indices == next_token_id).nonzero(as_tuple=True)[0].item() + 1

    # Probability of chosen token
    chosen_prob = probs[next_token_id].item()

    # Top-5 predictions
    top5_probs, top5_indices = probs.topk(5)
    top5 = [(idx.item(), p.item()) for idx, p in zip(top5_indices, top5_probs)]

    return {
        "entropy_bits": entropy,
        "rank": rank,
        "chosen_prob": chosen_prob,
        "top5": top5,
    }


# ─── Chat Formatting ─────────────────────────────────────────────────────────


def format_chat_prompt(conversation: list[dict], system_prompt: str | None = None) -> str:
    """Format conversation history into the training chat template.

    Training data format (from formatting.py):
        <|system|>
        {system text}
        <|user|>
        {user message}
        <|assistant|>
        {assistant response}
    """
    parts = []

    if system_prompt:
        parts.append("<|system|>")
        parts.append(system_prompt)

    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            parts.append("<|user|>")
            parts.append(content)
        elif role == "assistant":
            parts.append("<|assistant|>")
            parts.append(content)

    # Add assistant prefix to prompt the model to respond
    parts.append("<|assistant|>")

    return "\n".join(parts)


# ─── Streaming Generation ────────────────────────────────────────────────────


@torch.no_grad()
def generate_streaming(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 768,
    temperature: float = 0.7,
    top_k: int | None = 50,
    top_p: float | None = 0.9,
    repetition_penalty: float = 1.1,
    diagnostics: NCADiagnostics | None = None,
    show_token_info: bool = False,
    suppress_think_ids: list[int] | None = None,
    hide_think: bool = False,
) -> tuple[list[int], dict]:
    """Generate tokens with streaming output.

    Args:
        suppress_think_ids: Token IDs to suppress at early positions,
            used to prevent the model from starting with <think>.
        hide_think: If True, buffer output during <think>...</think> blocks
            and only display the text after </think>. If </think> never
            appears, print the full output at the end.

    Returns: (generated_token_ids, stats_dict)
    """
    model.eval()
    generated = input_ids.clone()
    seq_len = model.config.seq_len

    generated_tokens = []
    token_times = []
    token_stats_list = []
    start_time = time.time()
    prev_decoded_len = 0  # Track length of previously printed text
    in_think_block = False  # For dimming <think> content
    think_ended = False      # True once </think> has been seen
    thinking_shown = False   # True once (thinking...) indicator was printed
    started_printing = False # True once we've started streaming visible answer

    # Detect special token IDs for stopping
    eos_id = tokenizer.eos_token_id

    for i in range(max_new_tokens):
        step_start = time.time()

        # Context window management
        ctx = generated[:, -seq_len:] if generated.shape[1] > seq_len else generated

        # Use instrumented forward on first token, standard after
        if diagnostics is not None and i == 0:
            logits = forward_with_diagnostics(model, ctx, diagnostics)
        else:
            logits = model.get_logits(ctx)

        # Get logits for last position
        next_logits = logits[:, -1, :].clone()  # (B, V)

        # Suppress <think> at the first few token positions.
        # SentencePiece may produce a whitespace/newline token before '<',
        # so we suppress '<' variants for positions 0-2 to catch sequences
        # like: [newline] [<] [think] ...
        if suppress_think_ids and i < 3:
            for tid in suppress_think_ids:
                next_logits[0, tid] = float("-inf")

        # Repetition penalty
        if repetition_penalty != 1.0 and len(generated_tokens) > 0:
            for prev_id in set(generated_tokens[-64:]):
                if next_logits[0, prev_id] > 0:
                    next_logits[0, prev_id] /= repetition_penalty
                else:
                    next_logits[0, prev_id] *= repetition_penalty

        # Temperature scaling
        next_logits = next_logits / max(temperature, 1e-8)

        # Top-k filtering
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p is not None and 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_logits[indices_to_remove] = float("-inf")

        # Sample
        if temperature <= 0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        next_token_id = next_token.squeeze().item()
        elapsed = time.time() - step_start
        token_times.append(elapsed)
        generated_tokens.append(next_token_id)

        # Compute token stats if requested
        if show_token_info:
            raw_logits = logits[:, -1, :].squeeze(0)
            stats = compute_token_stats(raw_logits, next_token_id)
            token_stats_list.append(stats)

        # Stop conditions
        if next_token_id == eos_id:
            break

        # Decode the FULL generated sequence so far to get correct spacing.
        # SentencePiece encodes spaces as part of tokens (▁ prefix), so
        # single-token decode loses spaces. Full-sequence decode preserves them.
        full_decoded = tokenizer.decode(generated_tokens)
        new_text = full_decoded[prev_decoded_len:]
        prev_decoded_len = len(full_decoded)

        # Stop if model generates a role token (starting new turn)
        if new_text.strip() in ("<|user|>", "<|system|>", "<|endoftext|>"):
            break

        # Track <think> block boundaries
        if "<think>" in full_decoded and not in_think_block and not think_ended:
            in_think_block = True
            if hide_think and not thinking_shown:
                print(styled("(thinking...) ", C.DIM), end="", flush=True)
                thinking_shown = True
        if "</think>" in full_decoded and in_think_block:
            in_think_block = False
            think_ended = True
            if hide_think:
                # The model finished thinking — now stream the answer.
                # Extract only the text after </think>
                post_think = full_decoded.split("</think>", 1)[1].lstrip()
                if post_think:
                    print(post_think, end="", flush=True)
                    started_printing = True
                # Reset prev_decoded_len so subsequent tokens print correctly
                prev_decoded_len = len(full_decoded)
                generated = torch.cat([generated, next_token], dim=1)
                continue

        # Decide whether to print this token
        should_hide = hide_think and (in_think_block or (not think_ended and not started_printing))

        if should_hide:
            # Still in think block — silently buffer
            pass
        else:
            # Stream the NEW characters to terminal
            if show_token_info:
                stats = token_stats_list[-1]
                entropy = stats["entropy_bits"]
                if entropy < 2:
                    color = C.GREEN
                elif entropy < 5:
                    color = C.YELLOW
                else:
                    color = C.RED
                if in_think_block:
                    color = C.DIM
                print(styled(new_text, color), end="", flush=True)
            else:
                if in_think_block:
                    print(styled(new_text, C.DIM), end="", flush=True)
                else:
                    print(new_text, end="", flush=True)
            started_printing = True

        # Extend generated sequence
        generated = torch.cat([generated, next_token], dim=1)

    # If hide_think was active and </think> never appeared
    if hide_think and not think_ended and generated_tokens:
        if thinking_shown and not started_printing:
            # Model spent all tokens thinking and never produced </think>.
            # Don't dump the think block — just show a notice.
            print(styled("(ran out of tokens before finishing thought)", C.DIM), end="", flush=True)
        elif not started_printing:
            # No <think> was ever produced either — show raw output
            full_text = tokenizer.decode(generated_tokens)
            print(full_text, end="", flush=True)

    total_time = time.time() - start_time
    n_tokens = len(generated_tokens)

    print()  # newline after generation

    stats = {
        "tokens_generated": n_tokens,
        "total_time_s": total_time,
        "tokens_per_sec": n_tokens / total_time if total_time > 0 else 0,
        "avg_token_ms": (sum(token_times) / len(token_times) * 1000) if token_times else 0,
        "first_token_ms": token_times[0] * 1000 if token_times else 0,
        "token_stats": token_stats_list,
    }
    return generated_tokens, stats


# ─── Chat Session ─────────────────────────────────────────────────────────────


class ChatSession:
    """Interactive chat session with NCA diagnostics."""

    def __init__(self, model, tokenizer, device, config_dict, model_info):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config_dict = config_dict
        self.model_info = model_info

        # Generation settings
        self.temperature = 0.6
        self.top_k = 50
        self.top_p = 0.9
        self.max_new_tokens = 768
        self.repetition_penalty = 1.1

        # Think mode: when OFF (default), suppress <think> at first token
        self.think_mode = False
        self._think_suppress_ids = self._compute_think_suppress_ids()

        # Display modes
        self.show_nca = False          # NCA step-by-step diagnostics
        self.show_token_info = False   # Per-token entropy / rank coloring
        self.show_stats = True         # Generation speed stats

        # System prompt
        self.system_prompt = None

        # Conversation history
        self.conversation: list[dict] = []
        self.max_context_tokens = config_dict.get("seq_len", 1024) - 64  # leave room

        # Diagnostics engine
        self.diagnostics = NCADiagnostics()

    def _compute_think_suppress_ids(self) -> list[int]:
        """Find token IDs that could start a <think> block.

        SentencePiece encodes '<' differently based on context (e.g., token 523
        at string start vs 28789 after newline). We collect ALL '<' variants
        from multiple contexts so suppression works regardless of what precedes
        the generation point.
        """
        suppress = set()
        # Encode '<think>' in different contexts to find all '<' token variants
        for text in ["<think>", "<think", " <think>", " <think",
                     "\n<think>", "\n<think", "a<think>", ".<think>"]:
            try:
                enc = self.tokenizer.encode(text)
                if enc:
                    # Find the token that represents '<' in this encoding
                    for tid in enc:
                        decoded = self.tokenizer.decode([tid])
                        if '<' in decoded and 'think' not in decoded.lower():
                            suppress.add(tid)
            except Exception:
                pass
        # Also directly encode '<' alone for good measure
        try:
            enc = self.tokenizer.encode("<")
            suppress.update(enc)
        except Exception:
            pass
        return list(suppress)

    def print_banner(self):
        cfg = self.model.config
        K = cfg.total_steps
        n_params = self.model_info["parameters"]

        print()
        print(styled("\u2554" + "\u2550" * 70 + "\u2557", C.CYAN))
        print(styled("\u2551           ", C.CYAN) + styled("AENCA v5.1 \u2014 Neural Cellular Automaton Chat", C.BOLD + C.WHITE) + styled("           \u2551", C.CYAN))
        print(styled("\u255a" + "\u2550" * 70 + "\u255d", C.CYAN))

        print()
        print(styled("  Architecture", C.BOLD + C.UNDERLINE))
        print(f"    Model:         {styled('AENCA V5.1 (Precision-Routed NCA)', C.GREEN)}")
        print(f"    Parameters:    {styled(f'{n_params:,}', C.GREEN)} ({n_params/1e6:.1f}M)")
        print(f"    NCA Steps:     {styled(str(K), C.YELLOW)} total  "
              f"({cfg.k_phase1} propagation + {cfg.k_phase2} integration + {cfg.k_phase3} refinement)")
        print(f"    State dim:     d_model={styled(str(cfg.d_model), C.YELLOW)}  "
              f"d_msg={styled(str(cfg.d_msg), C.YELLOW)} ({cfg.d_model//cfg.d_msg}x compression)  "
              f"d_ff={styled(str(cfg.d_ff), C.YELLOW)}")
        print(f"    PRMP:          {cfg.n_heads} heads x d_head={cfg.d_head}  "
              f"(shared across all {K} steps)")
        print(f"    Kernels:       3 phase kernels x {cfg.n_trunk} SwiGLU blocks")
        print(f"    Context:       {styled(str(cfg.seq_len), C.YELLOW)} tokens")

        print()
        print(styled("  Training", C.BOLD + C.UNDERLINE))
        step = self.model_info["step"]
        print(f"    Checkpoint:    {styled(self.model_info.get('checkpoint_name', '?'), C.MAGENTA)}")
        print(f"    Steps:         {styled(f'{step:,}' if isinstance(step, int) else str(step), C.GREEN)}")
        best_ppl = self.model_info.get("best_val_ppl", "?")
        if isinstance(best_ppl, (int, float)):
            print(f"    Best Val PPL:  {styled(f'{best_ppl:.2f}', C.YELLOW)}")
        print(f"    Tokenizer:     {styled(self.config_dict.get('tokenizer_pretrained_name', 'unknown'), C.GREEN)}")

        print()
        print(styled("  Device", C.BOLD + C.UNDERLINE))
        print(f"    Compute:       {styled(str(self.device), C.MAGENTA)}", end="")
        if self.device.type == "cuda":
            name = torch.cuda.get_device_name(self.device)
            mem_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            print(f"  ({styled(name, C.DIM)}, {mem_total:.1f} GB)")
        else:
            print()

        print()
        print(styled("  Settings", C.BOLD + C.UNDERLINE))
        print(f"    Temp={self.temperature}  Top-k={self.top_k}  Top-p={self.top_p}  "
              f"Rep.Pen={self.repetition_penalty}  Max={self.max_new_tokens} tokens")
        print(f"    Think mode: {styled('ON (shows reasoning)', C.YELLOW) if self.think_mode else styled('OFF (answer only)', C.GREEN)}")
        print(f"    NCA diagnostics: {styled('ON', C.GREEN) if self.show_nca else styled('OFF', C.DIM)}  "
              f"Token info: {styled('ON', C.GREEN) if self.show_token_info else styled('OFF', C.DIM)}")

        print()
        print(styled("  Commands", C.BOLD + C.UNDERLINE))
        print(styled("    /think          Toggle think mode (allow/suppress <think> blocks)", C.DIM))
        print(styled("    /nca            Toggle NCA step-by-step diagnostics", C.DIM))
        print(styled("    /tokens         Toggle per-token entropy coloring (green=sure, red=unsure)", C.DIM))
        print(styled("    /inspect        Run NCA diagnostics on last input WITHOUT generating", C.DIM))
        print(styled("    /architecture   Show full model architecture details", C.DIM))
        print(styled("    /system <text>  Set system prompt", C.DIM))
        print(styled("    /temp <float>   Set temperature      /topk <int>     Set top-k", C.DIM))
        print(styled("    /topp <float>   Set top-p            /len <int>      Set max tokens", C.DIM))
        print(styled("    /rep <float>    Set repetition penalty", C.DIM))
        print(styled("    /reset          Clear conversation    /settings       Show settings", C.DIM))
        print(styled("    /help           Show all commands     /quit           Exit", C.DIM))
        print()
        print(styled("\u2500" * 72, C.CYAN))
        print()

    def handle_command(self, cmd: str) -> bool:
        """Handle a /command. Returns True to continue, False to quit."""
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if command in ("/quit", "/exit", "/q"):
            return False

        elif command in ("/help", "/h"):
            self._show_help()

        elif command == "/nca":
            self.show_nca = not self.show_nca
            status = styled("ON",  C.GREEN, C.BOLD) if self.show_nca else styled("OFF", C.DIM)
            print(f"  NCA diagnostics: {status}")
            if self.show_nca:
                print(styled("  Will show per-step gate magnitudes, convergence, and PRMP activity", C.DIM))

        elif command == "/tokens":
            self.show_token_info = not self.show_token_info
            status = styled("ON", C.GREEN, C.BOLD) if self.show_token_info else styled("OFF", C.DIM)
            print(f"  Token info: {status}")
            if self.show_token_info:
                print(styled("  Tokens colored by entropy: green=confident, yellow=moderate, red=uncertain", C.DIM))

        elif command == "/think":
            self.think_mode = not self.think_mode
            status = styled("ON", C.GREEN, C.BOLD) if self.think_mode else styled("OFF", C.DIM)
            print(f"  Think mode: {status}")
            if self.think_mode:
                print(styled("  Model will show <think> blocks (dimmed). Full reasoning visible.", C.DIM))
            else:
                print(styled("  Reasoning hidden. Shows (thinking...) then only the final answer.", C.DIM))
                print(styled("  Use /think to see the full reasoning trace.", C.DIM))

        elif command == "/architecture":
            self._show_architecture()

        elif command == "/inspect":
            self._inspect_last()

        elif command == "/system":
            if arg:
                self.system_prompt = arg
                print(f"  System prompt set: {styled(arg[:80] + ('...' if len(arg) > 80 else ''), C.DIM)}")
            else:
                if self.system_prompt:
                    print(f"  Current system prompt: {self.system_prompt}")
                else:
                    print("  No system prompt set. Usage: /system You are a helpful assistant.")

        elif command == "/temp":
            if arg:
                try:
                    self.temperature = float(arg)
                    print(f"  Temperature: {self.temperature}")
                except ValueError:
                    print("  Invalid. Usage: /temp 0.7")
            else:
                print(f"  Temperature: {self.temperature}")

        elif command == "/topk":
            if arg:
                try:
                    val = int(arg)
                    self.top_k = val if val > 0 else None
                    print(f"  Top-k: {self.top_k or 'disabled'}")
                except ValueError:
                    print("  Invalid. Usage: /topk 50")
            else:
                print(f"  Top-k: {self.top_k or 'disabled'}")

        elif command == "/topp":
            if arg:
                try:
                    val = float(arg)
                    self.top_p = val if 0 < val < 1 else None
                    print(f"  Top-p: {self.top_p or 'disabled'}")
                except ValueError:
                    print("  Invalid. Usage: /topp 0.9")
            else:
                print(f"  Top-p: {self.top_p or 'disabled'}")

        elif command == "/len":
            if arg:
                try:
                    self.max_new_tokens = max(1, int(arg))
                    print(f"  Max tokens: {self.max_new_tokens}")
                except ValueError:
                    print("  Invalid. Usage: /len 256")
            else:
                print(f"  Max tokens: {self.max_new_tokens}")

        elif command == "/rep":
            if arg:
                try:
                    self.repetition_penalty = max(1.0, float(arg))
                    print(f"  Repetition penalty: {self.repetition_penalty}")
                except ValueError:
                    print("  Invalid. Usage: /rep 1.1")
            else:
                print(f"  Repetition penalty: {self.repetition_penalty}")

        elif command == "/reset":
            self.conversation = []
            print(styled("  Conversation cleared.", C.DIM))

        elif command == "/settings":
            self._show_settings()

        elif command == "/stats":
            self._show_model_stats()

        else:
            print(f"  Unknown command: {command}. Type /help for commands.")

        return True

    def _show_help(self):
        print(styled("\n  Commands:", C.BOLD))
        print("    /think           Toggle think mode (allow/suppress <think> blocks)")
        print("    /nca             Toggle NCA step-by-step diagnostics")
        print("    /tokens          Toggle per-token entropy coloring")
        print("    /inspect         Run NCA diagnostics on current conversation")
        print("    /architecture    Show full architecture details")
        print("    /system <text>   Set system prompt")
        print("    /temp <float>    Set temperature (e.g. /temp 0.6)")
        print("    /topk <int>      Set top-k (e.g. /topk 50, 0=disable)")
        print("    /topp <float>    Set top-p (e.g. /topp 0.9, 0=disable)")
        print("    /len <int>       Max tokens to generate (e.g. /len 256)")
        print("    /rep <float>     Repetition penalty (e.g. /rep 1.1)")
        print("    /reset           Clear conversation context")
        print("    /settings        Show current settings")
        print("    /stats           GPU memory and model stats")
        print("    /quit            Exit")
        print()

    def _show_settings(self):
        print(styled("\n  Current Settings:", C.BOLD))
        print(f"    Temperature:       {self.temperature}")
        print(f"    Top-k:             {self.top_k or 'disabled'}")
        print(f"    Top-p:             {self.top_p or 'disabled'}")
        print(f"    Repetition Pen.:   {self.repetition_penalty}")
        print(f"    Max tokens:        {self.max_new_tokens}")
        think_status = 'ON (allows <think>)' if self.think_mode else f'OFF (suppressing {len(self._think_suppress_ids)} token IDs)'
        print(f"    Think mode:        {think_status}")
        print(f"    NCA diagnostics:   {'ON' if self.show_nca else 'OFF'}")
        print(f"    Token info:        {'ON' if self.show_token_info else 'OFF'}")
        print(f"    System prompt:     {self.system_prompt or '(none)'}")
        print(f"    Conversation:      {len(self.conversation)} messages")
        print()

    def _show_model_stats(self):
        print(styled("\n  Model Statistics:", C.BOLD))
        print(f"    Total params:  {self.model_info['parameters']:,}")
        print(f"    Step:          {self.model_info['step']}")
        print(f"    Best Val PPL:  {self.model_info.get('best_val_ppl', '?')}")
        if torch.cuda.is_available() and self.device.type == "cuda":
            alloc = torch.cuda.memory_allocated(self.device) / 1024**2
            reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            print(f"    GPU allocated:   {alloc:.0f} MB")
            print(f"    GPU reserved:    {reserved:.0f} MB")
        print()

    def _show_architecture(self):
        cfg = self.model.config
        K = cfg.total_steps
        n_params = self.model_info["parameters"]

        # Compute parameter breakdown
        embed_params = cfg.vocab_size * cfg.d_model
        prmp_params = sum(p.numel() for p in self.model.prmp.parameters())
        conv_params = sum(p.numel() for c in self.model.convs for p in c.parameters())
        kernel_params = sum(p.numel() for k in self.model.kernels for p in k.parameters())

        print(styled("\n  == AENCA V5.1 Architecture Detail ==", C.BOLD + C.CYAN))
        print()

        print(styled("  Forward Pass (for each token position):", C.BOLD))
        print(f"    1. Token Embed:    x_i -> W_E[x_i]  (d_model={cfg.d_model})     ({embed_params:,} params)")
        print(f"    2. + Sinusoidal PE: s_i^(0) = W_E[x_i] + SinPE(i)")
        print(f"    3. NCA Loop (K={K} steps, 3 phases):")
        print(f"       +----------------------------------------------------+")
        print(f"       |  for k = 0..{K-1}:                                     |")
        print(f"       |    P = CausalConv1D(s, radius={cfg.kernel_radius})        ({conv_params:,} params)")
        print(f"       |    M = PRMP(s)  [{cfg.d_model}->{cfg.d_msg}->attn->{cfg.d_model}]      ({prmp_params:,} params, shared)")
        print(f"       |    ds, g = Kernel([P;M;pe])               ({kernel_params:,} params, 3 kernels)")
        print(f"       |        [{cfg.d_kernel_in}->{cfg.d_ff} -> {cfg.n_trunk}xSwiGLU -> {cfg.d_model}+gate]")
        print(f"       |    s = s + sigmoid(g) * ds                 (gated update)")
        print(f"       +----------------------------------------------------+")
        print(f"    4. Readout:        logits = RMSNorm(s^(K)) @ W_E^T  (V={cfg.vocab_size})")

        print()
        print(styled("  Parameter Budget:", C.BOLD))
        print(f"    Embeddings:    {embed_params:>12,}   ({embed_params/n_params*100:4.1f}%)")
        print(f"    PRMP:          {prmp_params:>12,}   ({prmp_params/n_params*100:4.1f}%)")
        print(f"    Convolutions:  {conv_params:>12,}   ({conv_params/n_params*100:4.1f}%)")
        print(f"    Kernels (x3):  {kernel_params:>12,}   ({kernel_params/n_params*100:4.1f}%)")
        print(f"    {'_'*42}")
        print(f"    Total:         {n_params:>12,}   (100.0%)")

        print()
        print(styled("  Key Properties:", C.BOLD))
        print(f"    * Weight sharing: PRMP shared across all {K} steps -> {K}x reuse")
        print(f"    * Effective depth: {K} steps x (conv + attn + SwiGLU) = ~{K*3} 'layer equivalents'")
        print(f"    * Message compression: {cfg.d_model}d -> {cfg.d_msg}d ({cfg.d_model//cfg.d_msg}x compression)")
        print(f"    * Zero-init cascade: W_O=0, W_down=0, W_ds=0 (starts as identity)")
        print(f"    * Gated updates: sigmoid(b_g) ~ 1/K_phase initial opening")
        print()

    def _inspect_last(self):
        """Run NCA diagnostics on the current conversation state."""
        if not self.conversation:
            print(styled("  No conversation to inspect. Send a message first.", C.DIM))
            return

        prompt_text = format_chat_prompt(self.conversation, self.system_prompt)
        tokens = self.tokenizer.encode(prompt_text)
        tokens = tokens[-self.max_context_tokens:]

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        print(styled(f"  Inspecting {len(tokens)} tokens through {self.model.config.total_steps} NCA steps...", C.DIM))

        diag = NCADiagnostics()
        logits = forward_with_diagnostics(self.model, input_ids, diag)
        print(diag.format_report())

        # Also show prediction entropy for last few positions
        print(styled("  Token Predictions (last 10 positions):", C.BOLD))
        n_show = min(10, logits.shape[1])
        for pos in range(logits.shape[1] - n_show, logits.shape[1]):
            pos_logits = logits[0, pos, :]
            probs = F.softmax(pos_logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum().item() / math.log(2)
            top_prob, top_idx = probs.topk(3)
            top_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_idx]
            top_strs = "  ".join(
                f"{styled(repr(t), C.GREEN)}:{p:.3f}" for t, p in zip(top_tokens, top_prob.tolist())
            )
            actual = self.tokenizer.decode([tokens[pos]]) if pos < len(tokens) else "?"
            print(
                f"    pos {pos:>4}: H={entropy:5.2f}b  "
                f"actual={styled(repr(actual), C.YELLOW):>12}  "
                f"top3: {top_strs}"
            )
        print()

    def generate_response(self, user_message: str):
        """Generate a chat response."""
        # Add user message to conversation
        self.conversation.append({"role": "user", "content": user_message})

        # Format full prompt
        prompt_text = format_chat_prompt(self.conversation, self.system_prompt)
        tokens = self.tokenizer.encode(prompt_text)

        # Trim if too long (keep most recent context)
        if len(tokens) > self.max_context_tokens:
            tokens = tokens[-self.max_context_tokens:]

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # Print response header
        print(f"\n{styled('AENCA', C.CYAN + C.BOLD)} ", end="", flush=True)

        # Generate with optional diagnostics
        diag = self.diagnostics if self.show_nca else None
        # When think mode is OFF:
        #   - Let the model generate <think>...</think> naturally (don't
        #     suppress tokens — that breaks the format and removes </think>)
        #   - hide_think=True: silently buffer <think> block, only display
        #     the post-</think> answer
        #   - Use a larger token budget: the model typically spends 500-700
        #     tokens thinking, then needs room for the actual answer
        # When think mode is ON:
        #   - Show everything including <think> blocks (dimmed)
        hide = not self.think_mode
        effective_max = self.max_new_tokens
        if hide:
            # Ensure enough room for think block + answer
            effective_max = max(self.max_new_tokens, 1500)
        generated_tokens, stats = generate_streaming(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            max_new_tokens=effective_max,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            diagnostics=diag,
            show_token_info=self.show_token_info,
            suppress_think_ids=None,
            hide_think=hide,
        )

        # Decode response and add to conversation
        response_text = self.tokenizer.decode(generated_tokens).strip()
        self.conversation.append({"role": "assistant", "content": response_text})

        # Show generation stats
        if self.show_stats:
            first_ms = stats["first_token_ms"]
            avg_ms = stats["avg_token_ms"]
            print(styled(
                f"  [{stats['tokens_generated']} tokens  "
                f"{stats['tokens_per_sec']:.1f} tok/s  "
                f"first={first_ms:.0f}ms  avg={avg_ms:.0f}ms/tok  "
                f"ctx={len(tokens)} tokens]",
                C.DIM
            ))

        # Show token-level stats summary if enabled
        if self.show_token_info and stats["token_stats"]:
            ts = stats["token_stats"]
            avg_entropy = sum(s["entropy_bits"] for s in ts) / len(ts)
            avg_rank = sum(s["rank"] for s in ts) / len(ts)
            rank1_pct = sum(1 for s in ts if s["rank"] == 1) / len(ts) * 100
            avg_prob = sum(s["chosen_prob"] for s in ts) / len(ts)
            print(styled(
                f"  [entropy: {avg_entropy:.1f} bits  "
                f"avg rank: {avg_rank:.1f}  "
                f"rank-1: {rank1_pct:.0f}%  "
                f"avg P(chosen): {avg_prob:.3f}]",
                C.DIM
            ))

        # Show NCA diagnostics if enabled
        if self.show_nca and diag:
            print(diag.format_report())

    def run(self):
        """Main chat loop."""
        self.print_banner()

        while True:
            try:
                prompt = input(styled("\nYou > ", C.BOLD + C.BLUE))
            except (EOFError, KeyboardInterrupt):
                print(styled("\n\nGoodbye!", C.CYAN))
                break

            prompt = prompt.strip()
            if not prompt:
                continue

            if prompt.startswith("/"):
                if not self.handle_command(prompt):
                    print(styled("\nGoodbye!", C.CYAN))
                    break
                continue

            try:
                self.generate_response(prompt)
            except Exception as e:
                print(styled(f"\n  Error during generation: {e}", C.RED))
                import traceback
                traceback.print_exc()


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Chat with AENCA v5.1 -- Neural Cellular Automaton Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/chat.py --checkpoint experiments/AENCA\\ v5.1/smoltalk2_5b_mixed_run1_20260312_170948/best_model.pt
  python scripts/chat.py --experiment "AENCA v5.1/smoltalk2_5b_mixed_run1_20260312_170948"
  python scripts/chat.py --checkpoint best_model.pt --nca-diagnostics --token-info
        """,
    )

    # Checkpoint specification (mutually exclusive)
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument(
        "--checkpoint", type=str,
        help="Path to model checkpoint (.pt file)",
    )
    ckpt_group.add_argument(
        "--experiment", type=str,
        help="Experiment name (e.g. 'AENCA v5.1/smoltalk2_5b_mixed_run1_20260312_170948')",
    )

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu). Default: cuda")

    # Generation defaults
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=768)
    parser.add_argument("--rep_penalty", type=float, default=1.1,
                        help="Repetition penalty (1.0 = disabled)")

    # Display modes
    parser.add_argument("--nca-diagnostics", action="store_true",
                        help="Enable NCA step-by-step diagnostics from start")
    parser.add_argument("--token-info", action="store_true",
                        help="Enable per-token entropy coloring from start")
    parser.add_argument("--think", action="store_true",
                        help="Allow <think> blocks (default: suppressed for direct answers)")
    parser.add_argument("--system", type=str, default=None,
                        help="System prompt")

    # Model selection
    parser.add_argument("--model", type=str, default=None,
                        help="Registered model name (default: v5)")
    parser.add_argument("--model-module", type=str, default=None,
                        help="Python module path to import for model registration")

    args = parser.parse_args()

    # Device setup
    if args.device == "cuda" and not torch.cuda.is_available():
        print(styled("  CUDA not available, falling back to CPU", C.YELLOW))
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Import model module if specified
    if args.model_module:
        importlib.import_module(args.model_module)

    # Resolve checkpoint
    if args.experiment:
        checkpoint_path = str(find_checkpoint(args.experiment))
    else:
        checkpoint_path = args.checkpoint

    # Load model
    print(styled("\n  Loading AENCA v5.1...", C.CYAN + C.BOLD))
    model, config_dict, model_info = load_model(
        checkpoint_path, device, model_name=args.model
    )

    # Create tokenizer from training config
    print(styled("  Loading tokenizer...", C.DIM))
    tokenizer = create_tokenizer_from_config(config_dict)
    print(styled(f"  Tokenizer ready: vocab_size={tokenizer.vocab_size}", C.DIM))

    # Run chat
    session = ChatSession(model, tokenizer, device, config_dict, model_info)
    session.temperature = args.temperature
    session.top_k = args.top_k if args.top_k > 0 else None
    session.top_p = args.top_p if 0 < args.top_p < 1 else None
    session.max_new_tokens = args.max_tokens
    session.repetition_penalty = args.rep_penalty
    session.show_nca = args.nca_diagnostics
    session.show_token_info = args.token_info
    session.think_mode = args.think
    session.system_prompt = args.system
    session.run()


if __name__ == "__main__":
    main()
