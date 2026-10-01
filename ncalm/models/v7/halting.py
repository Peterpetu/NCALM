"""Adaptive halting for AENCA V7 — per-position readout at theta boundaries.

Implements PonderNet-style adaptive computation: each token position
independently decides how much of its representation to "output" at each
theta boundary. Tokens that have converged early (common words, easy syntax)
use fewer theta cycles; complex tokens use all n_theta cycles.

Mechanism:
    After each theta cycle θ ∈ {1, ..., n_theta}:
        halt_prob = σ(W_halt(RMSNorm(z)))         — per-position halt probability
        effective = min(halt_prob, remaining)       — can't output more than remaining
        z_out += effective × z                     — accumulate weighted output
        remaining -= effective                      — track remaining mass
    At the last theta cycle:
        z_out += remaining × z                     — absorb all remaining mass

PonderNet regularization:
    λ_halt × KL(halt_distribution || Geometric(p=1/n_theta))
    Encourages moderate compute usage — neither always minimal nor always maximal.

Reference: Banino et al. (2021), "PonderNet: Learning to Ponder."
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v7.config import V7Config


class AdaptiveHalting(nn.Module):
    """Per-position readout accumulation at theta cycle boundaries.

    Each position has a "remaining mass" that starts at 1.0. At each theta
    boundary, the model outputs halt_prob fraction of the remaining mass,
    weighted by the current state z. The accumulated z_out is the final
    representation used for readout.

    For inference: positions with remaining < halt_eps are skipped,
    providing genuine computational savings for easy tokens.

    Parameters: d_model + 1 (weight + bias) + d_model (norm) ≈ 2*d_model
    """

    def __init__(self, config: V7Config):
        super().__init__()
        self.n_theta = config.n_theta
        self.halt_eps = config.halt_eps
        self.lambda_halt = config.lambda_halt

        # Halt probability head
        self.norm = nn.RMSNorm(config.d_model)
        self.W_halt = nn.Linear(config.d_model, 1, bias=True)

        self._init_weights(config)

    def _init_weights(self, config: V7Config) -> None:
        """Initialize halt head.

        Bias: σ(bias) ≈ 2/n_theta → initial readout approximately uniform
        across all theta cycles (slight front-loading).
        Weight: small Xavier → halt probability starts near-constant.
        """
        nn.init.xavier_normal_(self.W_halt.weight, gain=0.1)
        # σ(b) ≈ 2/n_theta → b = ln(2/(n_theta - 2)) for n_theta > 2
        # For n_theta=4: b = ln(2/2) = 0.0 → σ(0) = 0.5 → reasonable
        # For n_theta=2: b = ln(2/0) → use fallback
        if config.n_theta > 2:
            bias_val = math.log(2.0 / max(config.n_theta - 2, 0.5))
        else:
            bias_val = 0.0  # σ(0) = 0.5
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
        halt_prob = self.compute_halt_prob(z)           # (B, L)

        if is_last:
            # Last theta cycle: absorb all remaining mass
            effective = remaining
        else:
            # Normal: consume halt_prob fraction, clamped to remaining
            effective = torch.min(halt_prob * remaining, remaining)

        # Accumulate weighted state
        z_out = z_out + effective.unsqueeze(-1) * z     # (B, L, d_model)
        remaining = remaining - effective                # (B, L)

        return z_out, remaining, halt_prob

    def compute_kl_loss(
        self, halt_probs: list[torch.Tensor]
    ) -> torch.Tensor:
        """Compute PonderNet KL regularization loss.

        Encourages the halt distribution to be close to a Geometric(p_prior)
        distribution, preventing degenerate solutions (always halt immediately
        or never halt).

        The prior is Geometric(p=1/n_theta) — uniform expected compute.

        Args:
            halt_probs: list of (B, L) halt probabilities at each theta boundary

        Returns:
            kl_loss: scalar KL divergence
        """
        n = len(halt_probs)
        if n == 0:
            return torch.tensor(0.0, device=halt_probs[0].device if halt_probs else "cpu")

        device = halt_probs[0].device
        p_prior = 1.0 / self.n_theta  # Geometric prior parameter

        # Compute empirical halt distribution (how much mass is consumed at each step)
        # p_halt[θ] = halt_prob[θ] × Π_{j<θ}(1 - halt_prob[j])
        # Trick: we track remaining mass implicitly
        kl_total = torch.tensor(0.0, device=device)
        remaining = torch.ones_like(halt_probs[0])

        for i, hp in enumerate(halt_probs):
            # Actual halting mass at this step
            p_actual = hp * remaining  # (B, L)

            # Prior halting mass at this step: Geometric(p_prior)
            # p_geo[i] = p_prior × (1 - p_prior)^i
            p_geo = p_prior * ((1.0 - p_prior) ** i)

            # KL contribution: p_actual × log(p_actual / p_geo)
            # Clamp for numerical stability
            p_actual_clamped = p_actual.clamp(min=1e-8)
            p_geo_clamped = max(p_geo, 1e-8)

            kl_total = kl_total + (
                p_actual_clamped * (p_actual_clamped.log() - math.log(p_geo_clamped))
            ).mean()

            # Update remaining mass
            remaining = remaining * (1.0 - hp).clamp(min=0.0)

        return kl_total
