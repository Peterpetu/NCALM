"""Morphogen-informed adaptive halting for AENCA V15 — PonderNet at theta boundaries.

Enhanced from V14: the halt signal uses BOTH content convergence AND
activator morphogen convergence. The activator's spatial pattern stability
provides a natural stopping criterion for the NCA dynamics.

halt_input = [RMSNorm(z); ā_pos] ∈ R^{d_model + d_a}
p_halt = σ(W_halt · halt_input) ∈ (0, 1)

where ā_pos is the per-position activator (unfolded from 2D grid).

Reference: §13 of plan.md — Morphogen-informed halting.
Reference: Banino et al. (2021), "PonderNet: Learning to Ponder."
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ncalm.models.v15.config import V15Config
from ncalm.models.v15.reaction_diffusion import unfold_2d_to_1d


class AdaptiveHalting(nn.Module):
    """Per-position readout accumulation at theta cycle boundaries.

    Uses both content state z and activator morphogen a for halt signal.
    Each position has "remaining mass" starting at 1.0. At each theta
    boundary, halt_prob fraction is weighted by current z.

    Parameters: (d_model + d_a) + 1 (W_halt) + d_model (norm).
    """

    def __init__(self, config: V15Config):
        super().__init__()
        self.n_theta = config.n_theta
        self.halt_eps = config.halt_eps
        self.d_a = config.d_a

        self.norm = nn.RMSNorm(config.d_model)
        # Halt input: [Norm(z); a_unfold] → (d_model + d_a)
        self.W_halt = nn.Linear(config.d_model + config.d_a, 1, bias=True)

        self._init_weights(config)

    def _init_weights(self, config: V15Config) -> None:
        """Initialize halt head."""
        nn.init.xavier_normal_(self.W_halt.weight, gain=0.1)
        if config.n_theta > 2:
            bias_val = math.log(2.0 / max(config.n_theta - 2, 0.5))
        else:
            bias_val = 0.0
        nn.init.constant_(self.W_halt.bias, bias_val)

    def compute_halt_prob(
        self, z: torch.Tensor, a: torch.Tensor, N: int
    ) -> torch.Tensor:
        """Compute per-position halt probability.

        Args:
            z: (B, N, d_model) content state
            a: (B, d_a, H, W) activator on 2D grid
            N: sequence length

        Returns:
            halt_prob: (B, N) ∈ (0, 1)
        """
        # Unfold activator to 1D
        a_1d = unfold_2d_to_1d(a, N)  # (B, N, d_a)

        # Concatenate [Norm(z); a_1d]
        halt_input = torch.cat([self.norm(z), a_1d], dim=-1)  # (B, N, d_model+d_a)

        return torch.sigmoid(self.W_halt(halt_input).squeeze(-1))  # (B, N)

    def accumulate(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        N: int,
        z_out: torch.Tensor,
        remaining: torch.Tensor,
        is_last: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Accumulate readout-weighted state at a theta boundary.

        Args:
            z:         (B, N, d_model) current content state
            a:         (B, d_a, H, W) current activator
            N:         sequence length
            z_out:     (B, N, d_model) accumulated readout state
            remaining: (B, N) remaining mass per position
            is_last:   whether this is the final theta cycle

        Returns:
            z_out:     updated readout accumulator
            remaining: updated remaining mass
            halt_prob: (B, N) halt probability (for KL loss)
        """
        halt_prob = self.compute_halt_prob(z, a, N)

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

        KL(empirical halt || Geometric(1/n_theta)).

        Args:
            halt_probs: list of (B, N) halt probs at each theta boundary

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
