"""Abstract base class for all models in the NCALM training harness.

Any model architecture that wants to use the NCALM training infrastructure
must subclass NCALMModel and implement all abstract methods. The harness
handles everything else: data loading, optimizer construction, LR scheduling,
AMP, logging, checkpointing, evaluation, and chat.

Use the @register_model decorator to make a model discoverable by name
from the training script's --model flag.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class NCALMModel(ABC, nn.Module):
    """Abstract base class for all models in the NCALM training harness.

    The harness calls:
      - forward_train(batch)  during training steps
      - get_logits(input_ids) during generation / evaluation
      - get_param_groups()    during optimizer construction
      - describe()            for logging / analysis.json

    The model owns its own loss computation. The harness only needs the
    total loss (for backward), the CE component (for PPL tracking), and
    any extra metrics the model wants logged.
    """

    @abstractmethod
    def forward_train(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass for training. Computes loss internally.

        Args:
            batch: Token IDs of shape (B, T), dtype torch.long.

        Returns:
            loss: Scalar tensor — the total training loss to .backward().
                  May combine CE + auxiliary losses.
            info: Dict with at minimum:
                  - 'ce': torch.Tensor or float — the cross-entropy component
                         (mean per predicted token, NOT summed).
                  Optionally any model-specific metrics (e.g. 'hier',
                  'efe_loss', etc.) — these are logged to metrics.jsonl.
        """
        ...

    @abstractmethod
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference/generation. Returns raw logits.

        Args:
            input_ids: Token IDs of shape (B, T), dtype torch.long.

        Returns:
            logits: Shape (B, T, vocab_size), dtype float.
        """
        ...

    @abstractmethod
    def get_param_groups(self) -> list[dict]:
        """Return parameter groups for the optimizer.

        Each group is a dict with at minimum:
          - 'params': list of nn.Parameter
          - 'label': str (e.g. 'backbone', 'policy') — for logging only

        Optional per-group overrides (take precedence over profile defaults):
          - 'lr': float
          - 'weight_decay': float
          - 'betas': tuple

        If the model has no special grouping, return a single group:
          [{'params': list(self.parameters()), 'label': 'all'}]
        """
        ...

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return a description dict for analysis.json and logging.

        Must include at minimum:
          - 'architecture': str (human-readable name)
          - 'total_params': int

        May include any additional architecture-specific keys.
        """
        ...

    def get_phases(self, step: int, max_steps: int) -> str:
        """Return active training phases string for this step.

        Override for multi-phase training (e.g. "A", "A+B", "A+B+C").
        Default: always "A".
        """
        return "A"

    def on_step_start(self, step: int, max_steps: int) -> None:
        """Hook called before each training step.

        Override to implement phase transitions, lambda ramps, etc.
        """
        pass

    def on_step_end(self, step: int, metrics: dict) -> None:
        """Hook called after each training step with the metrics dict.

        Override to implement custom tracking or adaptive behavior.
        """
        pass

    def extra_metrics(self) -> dict[str, float]:
        """Return model-specific metrics to merge into step logging.

        Called after forward_train. Use for metrics computed outside
        the forward pass (running averages, internal state monitors, etc.).
        """
        return {}


# ─── Model Registry ─────────────────────────────────────────────────────────

_MODEL_REGISTRY: dict[str, type[NCALMModel]] = {}


def register_model(name: str):
    """Decorator to register a model class by name.

    Usage:
        @register_model("my_transformer")
        class MyTransformer(NCALMModel):
            ...
    """

    def decorator(cls):
        if not issubclass(cls, NCALMModel):
            raise TypeError(f"{cls.__name__} must subclass NCALMModel")
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(name: str) -> type[NCALMModel]:
    """Look up a registered model by name."""
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model '{name}'. Registered: {available}")
    return _MODEL_REGISTRY[name]


def list_models() -> list[str]:
    """List all registered model names."""
    return list(_MODEL_REGISTRY.keys())
