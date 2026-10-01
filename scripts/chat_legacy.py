#!/usr/bin/env python3
"""Interactive Chat Script for NCALM models.

Usage:
    python scripts/chat.py --checkpoint path/to/best_model.pt
    python scripts/chat.py --checkpoint path/to/best_model.pt --temperature 0.7 --top_k 40
    python scripts/chat.py --checkpoint path/to/best_model.pt --model my_model
"""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ncalm.model import get_model_class, list_models
from ncalm.data.tokenizer import BPETokenizer


# ─── Terminal Colors ─────────────────────────────────────────────────────────


class Colors:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


def colored(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"


# ─── Model Loading ───────────────────────────────────────────────────────────


def load_model(
    checkpoint_path: str, device: torch.device, model_name: str | None = None
) -> tuple:
    """Load trained model from checkpoint.

    Returns: (model, config_dict, info_dict)
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    print(colored(f"Loading checkpoint: {path}", Colors.DIM))
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
            raise RuntimeError(
                "No config found in checkpoint or experiment directory"
            )

    # Determine model class
    if model_name is None:
        model_name = config_dict.get("model_name")
    if model_name is None:
        available = list_models()
        if len(available) == 1:
            model_name = available[0]
        elif len(available) == 0:
            raise RuntimeError(
                "No models registered. Use --model or --model-module."
            )
        else:
            raise RuntimeError(
                f"Multiple models registered: {available}. Use --model."
            )

    ModelClass = get_model_class(model_name)

    # Build model with dropout=0.0 for inference
    if "dropout" in config_dict:
        config_dict["dropout"] = 0.0
    model = ModelClass(config_dict)

    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if missing:
        print(
            colored(
                f"  New parameters (initialized fresh): {len(missing)}", Colors.DIM
            )
        )
    if unexpected:
        print(
            colored(
                f"  Deprecated parameters (ignored): {len(unexpected)}", Colors.DIM
            )
        )

    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    info = {
        "step": checkpoint.get("step", "?"),
        "best_val_ppl": checkpoint.get("best_val_ppl", "?"),
        "parameters": n_params,
    }
    return model, config_dict, info


# ─── Streaming Generation ────────────────────────────────────────────────────


@torch.no_grad()
def generate_streaming(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
) -> tuple:
    """Generate tokens with streaming output to terminal.

    Returns: (generated_token_ids, stats_dict)
    """
    model.eval()
    generated = input_ids.clone()

    generated_tokens = []
    token_times = []
    start_time = time.time()

    for i in range(max_new_tokens):
        step_start = time.time()

        # Context window (512 token limit for memory efficiency)
        ctx = generated[:, -512:] if generated.shape[1] > 512 else generated

        logits = model.get_logits(ctx)

        # Temperature scaling
        logits = logits[:, -1, :] / max(temperature, 1e-8)  # (B, V)

        # Top-k filtering
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p is not None and 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # Sample
        if temperature <= 0:
            next_token = logits.argmax(dim=-1, keepdim=True)  # Greedy
        else:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        next_token_id = next_token.squeeze().item()
        token_times.append(time.time() - step_start)
        generated_tokens.append(next_token_id)

        # Stop on EOS
        if next_token_id == tokenizer.eos_token_id:
            break

        # Stream token to terminal
        token_text = tokenizer.decode([next_token_id])
        print(token_text, end="", flush=True)

        # Extend generated sequence
        generated = torch.cat([generated, next_token], dim=1)

    total_time = time.time() - start_time
    n_tokens = len(generated_tokens)

    print()  # newline after generation

    stats = {
        "tokens_generated": n_tokens,
        "total_time_s": total_time,
        "tokens_per_sec": n_tokens / total_time if total_time > 0 else 0,
        "avg_token_ms": (
            (sum(token_times) / len(token_times) * 1000) if token_times else 0
        ),
    }
    return generated_tokens, stats


# ─── Chat Session ─────────────────────────────────────────────────────────────


class ChatSession:
    """Interactive chat session with an NCALM model."""

    def __init__(self, model, tokenizer, device, config_dict, model_info):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config_dict = config_dict
        self.model_info = model_info

        # Generation settings (defaults)
        self.temperature = 0.8
        self.top_k = 50
        self.top_p = 0.9
        self.max_new_tokens = 200

        # Conversation context
        self.context_tokens = []
        self.max_context = config_dict.get("seq_len", 256) - 20

    def print_banner(self):
        print()
        print(colored("=" * 70, Colors.CYAN))
        arch = self.model_info.get("architecture", "NCALM Model")
        print(colored(f"  {arch} — Interactive Chat", Colors.BOLD + Colors.CYAN))
        print(colored("=" * 70, Colors.CYAN))
        print()

        param_str = f'{self.model_info["parameters"]:,} params'
        print(f"  Model:        {colored(param_str, Colors.GREEN)}")

        step = self.model_info["step"]
        train_str = f"{step} steps" if isinstance(step, int) else str(step)
        print(f"  Trained:      {colored(train_str, Colors.GREEN)}")

        best_ppl = self.model_info.get("best_val_ppl", "?")
        if isinstance(best_ppl, (int, float)):
            print(f"  Best Val PPL: {colored(f'{best_ppl:.1f}', Colors.YELLOW)}")

        print(f"  Device:       {colored(str(self.device), Colors.MAGENTA)}")
        print()

        print(colored("  Settings:", Colors.DIM))
        print(
            f"    Temp: {self.temperature}  |  Top-k: {self.top_k}  |  "
            f"Top-p: {self.top_p}  |  Max: {self.max_new_tokens}"
        )
        print()
        print(
            colored(
                "  Note: Small model trained on TinyStories — use simple story prompts",
                Colors.DIM,
            )
        )
        print(colored("  Type /help for commands, /quit to exit", Colors.DIM))
        print(colored("-" * 70, Colors.CYAN))
        print()

    def handle_command(self, cmd: str) -> bool:
        """Handle a /command. Returns True to continue, False to quit."""
        parts = cmd.strip().split()
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        if command in ("/quit", "/exit", "/q"):
            return False

        elif command in ("/help", "/h"):
            print(colored("\n  Commands:", Colors.BOLD))
            print("    /temp <float>    — Set temperature (e.g. /temp 0.7)")
            print("    /topk <int>      — Set top-k, 0 to disable (e.g. /topk 40)")
            print(
                "    /topp <float>    — Set top-p, 0 to disable (e.g. /topp 0.95)"
            )
            print("    /len <int>       — Max tokens to generate (e.g. /len 300)")
            print("    /stats           — Show model stats")
            print("    /reset           — Clear conversation context")
            print("    /settings        — Show current settings")
            print("    /quit            — Exit")
            print()

        elif command == "/temp":
            if args:
                try:
                    self.temperature = float(args[0])
                    print(f"  Temperature: {self.temperature}")
                except ValueError:
                    print("  Invalid temperature. Usage: /temp 0.8")
            else:
                print(f"  Temperature: {self.temperature}")

        elif command == "/topk":
            if args:
                try:
                    val = int(args[0])
                    self.top_k = val if val > 0 else None
                    print(f"  Top-k: {self.top_k or 'disabled'}")
                except ValueError:
                    print("  Invalid top-k. Usage: /topk 50")
            else:
                print(f"  Top-k: {self.top_k or 'disabled'}")

        elif command == "/topp":
            if args:
                try:
                    val = float(args[0])
                    self.top_p = val if 0 < val < 1 else None
                    print(f"  Top-p: {self.top_p or 'disabled'}")
                except ValueError:
                    print("  Invalid top-p. Usage: /topp 0.9")
            else:
                print(f"  Top-p: {self.top_p or 'disabled'}")

        elif command == "/len":
            if args:
                try:
                    self.max_new_tokens = max(1, int(args[0]))
                    print(f"  Max tokens: {self.max_new_tokens}")
                except ValueError:
                    print("  Invalid length. Usage: /len 200")
            else:
                print(f"  Max tokens: {self.max_new_tokens}")

        elif command == "/stats":
            print(colored("\n  Model Statistics:", Colors.BOLD))
            print(f"    Total params:  {self.model_info['parameters']:,}")
            print(f"    Step:          {self.model_info['step']}")
            print(
                f"    Best Val PPL:  {self.model_info.get('best_val_ppl', '?')}"
            )
            if torch.cuda.is_available() and self.device.type == "cuda":
                mem = torch.cuda.memory_allocated() / 1024**2
                print(f"    GPU memory:    {mem:.0f} MB")
            print()

        elif command == "/reset":
            self.context_tokens = []
            print(colored("  Context cleared.", Colors.DIM))

        elif command == "/settings":
            print(colored("\n  Current Settings:", Colors.BOLD))
            print(f"    Temperature: {self.temperature}")
            print(f"    Top-k:       {self.top_k or 'disabled'}")
            print(f"    Top-p:       {self.top_p or 'disabled'}")
            print(f"    Max tokens:  {self.max_new_tokens}")
            print(f"    Context:     {len(self.context_tokens)} tokens")
            print()

        else:
            print(f"  Unknown command: {command}. Type /help for commands.")

        return True

    def generate_response(self, prompt: str):
        """Generate a response to the user's prompt."""
        # Tokenize
        prompt_tokens = self.tokenizer.encode(prompt)

        # Build context: previous + new
        all_tokens = self.context_tokens + prompt_tokens

        # Trim if too long
        if len(all_tokens) > self.max_context:
            all_tokens = all_tokens[-self.max_context :]

        # Prepare input
        input_ids = torch.tensor(
            [all_tokens], dtype=torch.long, device=self.device
        )

        # Print generation mode indicator
        print(f"\n{colored('[GEN]', Colors.MAGENTA)} ", end="", flush=True)

        # Generate
        generated_tokens, stats = generate_streaming(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )

        # Update context
        self.context_tokens = all_tokens + generated_tokens
        if len(self.context_tokens) > self.max_context:
            self.context_tokens = self.context_tokens[-self.max_context :]

        # Print stats
        print(
            colored(
                f"  [{stats['tokens_generated']} tokens, "
                f"{stats['tokens_per_sec']:.1f} tok/s, "
                f"{stats['avg_token_ms']:.0f}ms/tok]",
                Colors.DIM,
            )
        )

    def run(self):
        """Main chat loop."""
        self.print_banner()

        while True:
            try:
                prompt = input(colored("\nYou> ", Colors.BOLD + Colors.BLUE))
            except (EOFError, KeyboardInterrupt):
                print(colored("\n\nGoodbye!", Colors.CYAN))
                break

            prompt = prompt.strip()
            if not prompt:
                continue

            if prompt.startswith("/"):
                if not self.handle_command(prompt):
                    print(colored("\nGoodbye!", Colors.CYAN))
                    break
                continue

            try:
                self.generate_response(prompt)
            except Exception as e:
                print(colored(f"\n  Error during generation: {e}", Colors.RED))
                import traceback

                traceback.print_exc()


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Chat with a trained NCALM model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu). Default: cuda",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Registered model name (if multiple registered)",
    )
    parser.add_argument(
        "--model-module",
        type=str,
        default=None,
        help="Python module path to import (triggers model registration)",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Import model module if specified
    if args.model_module:
        importlib.import_module(args.model_module)

    # Load model
    model, config_dict, model_info = load_model(
        args.checkpoint, device, model_name=args.model
    )

    # Create tokenizer
    tokenizer = BPETokenizer()

    # Run chat
    session = ChatSession(model, tokenizer, device, config_dict, model_info)
    session.temperature = args.temperature
    session.top_k = args.top_k if args.top_k > 0 else None
    session.top_p = args.top_p if 0 < args.top_p < 1 else None
    session.max_new_tokens = args.max_tokens
    session.run()


if __name__ == "__main__":
    main()
