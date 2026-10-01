"""Adaptive halting for AENCA V13 — PonderNet at theta boundaries.

Same mechanism as V7.1: per-position halt probability, remaining mass
tracking, weighted z accumulation. KL regularization against a Geometric
prior encourages moderate compute usage.

Reference: V7.1 halting.py — AdaptiveHalting (identical mechanism).
Reference: Banino et al. (2021), "PonderNet: Learning to Ponder."
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ncalm.models.v13.config import V13Config


class AdaptiveHalting(nn.Module):
    """Per-position readout accumulation at theta cycle boundaries.

    Each position has a "remaining mass" starting at 1.0. At each theta
    boundary, the model outputs halt_prob fraction weighted by current z.
    The last theta cycle absorbs all remaining mass.

    Parameters: d_model + 1 (W_halt weight + bias) + d_model (norm).
    """

    def __init__(self, config: V13Config):
        super().__init__()
        self.n_theta = config.n_theta
        self.halt_eps = config.halt_eps
        self.lambda_halt = config.lambda_halt

        self.norm = nn.RMSNorm(config.d_model)
        self.W_halt = nn.Linear(config.d_model, 1, bias=True)

        self._init_weights(config)

    def _init_weights(self, config: V13Config) -> None:
        """Initialize halt head.

        Weight: small xavier → halt probability starts near-constant.
        Bias: σ(bias) ≈ 2/n_theta for n_theta > 2, else 0.0.
        """
        nn.init.xavier_normal_(self.W_halt.weight, gain=0.1)
        if config.n_theta > 2:
            bias_val = math.log(2.0 / max(config.n_theta - 2, 0.5))
        else:
            bias_val = 0.0
        nn.init.constant_(self.W_halt.bias, bias_val)

    def compute_halt_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Compute per-position halt probability.

        Args:
            z: (B, L, d_model) cell states

        Returns:
            halt_prob: (B, L) ∈ (0, 1) halt probability per position
        """
        return torch.sigmoid(self.W_halt(self.norm(z)).squeeze(-1))

    def accumulate(
        self,
        z: torch.Tensor,
        z_out: torch.Tensor,
        remaining: torch.Tensor,
        is_last: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Accumulate readout-weighted state at a theta boundary.

        Args:
            z:         (B, L, d_model) current cell states
            z_out:     (B, L, d_model) accumulated readout state
            remaining: (B, L) remaining mass per position
            is_last:   whether this is the final theta cycle

        Returns:
            z_out:     updated readout accumulator
            remaining: updated remaining mass
            halt_prob: (B, L) halt probability (for KL regularization)
        """
        halt_prob = self.compute_halt_prob(z)

        if is_last:
            effective = remaining
        else:
            effective = torch.min(halt_prob * remaining, remaining)

        z_out = z_out + effective.unsqueeze(-1) * z
        remaining = remaining - effective

        return z_out, remaining, halt_prob

    def compute_kl_loss(
        self, halt_probs: list[torch.Tensor]
    ) -> torch.Tensor:
        """Compute PonderNet KL regularization loss.

        KL(empirical halt distribution || Geometric(p=1/n_theta)).

        Args:
            halt_probs: list of (B, L) halt probabilities at each theta boundary

        Returns:
            kl_loss: scalar KL divergence
        """
        n = len(halt_probs)
        if n == 0:
            return torch.tensor(0.0)

        device = halt_probs[0].device
        p_prior = 1.0 / self.n_theta

        kl_total = torch.tensor(0.0, device=device)
        remaining = torch.ones_like(halt_probs[0])

        for i, hp in enumerate(halt_probs):
            p_actual = hp * remaining
            p_geo = p_prior * ((1.0 - p_prior) ** i)

            p_actual_clamped = p_actual.clamp(min=1e-8)
            p_geo_clamped = max(p_geo, 1e-8)

            kl_total = kl_total + (
                p_actual_clamped
                * (p_actual_clamped.log() - math.log(p_geo_clamped))
            ).mean()

            remaining = remaining * (1.0 - hp).clamp(min=0.0)

        return kl_total
