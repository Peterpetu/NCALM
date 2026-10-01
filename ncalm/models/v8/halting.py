"""Adaptive halting and edge-of-chaos regularization for AENCA V8.

Three components:

1. **AdaptiveHalting**: PonderNet-style per-position readout accumulation
   at theta boundaries. Preserved from V7 with one change: operates on
   d_model hidden dimension (sliced from d+1 full state). Active cells
   and settled cells both contribute to halting decisions.

2. **EdgeOfChaosRegularizer**: Implements the Lyapunov exponent estimation
   and band loss from Spec Section 11.1-11.2.
     Def 11.1: λ̂_max = (1/K) · ln(‖Φ^K(z₀+ε) − Φ^K(z₀)‖ / ‖ε‖)
     Def 11.2: L_edge = α_edge · [ReLU(λ̂_max − λ₊) + ReLU(λ₋ − λ̂_max)]
   This keeps the dynamics in the Class IV edge-of-chaos band [-0.05, +0.05].

3. **SpectralEntropyRegularizer**: Implements the SVD-based spectral entropy
   from Spec Section 11.3 and the complexity band from Spec Section 19.
     Def 11.3: H_sv = −Σ σ̂_j log₂ σ̂_j (normalized singular values)
     Def 19.1: Target band [η_min, η_max] = [0.3, 0.7] × log₂(min(L,d))
   This keeps the state complexity in the intermediate Class IV regime.

Reference: model_documents/AENCA_V8_GROWING_CELLULAR_AUTOMATA.md
  Section 11 (Edge-of-Chaos Regulation),
  Section 19 (Complexity Regularization),
  Section 16.3 (Halting Loss),
  Section 16.6 (Edge-of-Chaos Loss).

Reference: Banino et al. (2021), "PonderNet: Learning to Ponder."
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ncalm.models.v8.config import V8Config


class AdaptiveHalting(nn.Module):
    """Per-position readout accumulation at theta cycle boundaries.

    Preserved from V7 with one V8 adaptation: operates on d_model
    (the hidden state dimension), NOT d_full. The caller slices
    z[..., :d_model] before passing to compute_halt_prob.

    Mechanism (Spec Section 16.3):
        After each theta cycle θ ∈ {1, ..., n_theta}:
            halt_prob = σ(W_halt(RMSNorm(z_hidden)))     — per-position
            effective = min(halt_prob, remaining)          — can't exceed remaining
            z_out += effective × z_hidden                 — accumulate
            remaining -= effective                         — track mass
        Last cycle: z_out += remaining × z_hidden         — absorb all

    PonderNet regularization (Spec Section 16.3):
        L_halt = KL(halt_distribution ∥ Geometric(p = 1/n_theta))

    Halting bias initialization (Spec Section 16.3):
        b_h = ln(2 / (n_theta − 2))
        → σ(b_h) ≈ 2/n_theta initially (moderate compute usage)

    Parameters: d_model + 1 (weight + bias) + d_model (norm)
    """

    def __init__(self, config: V8Config):
        super().__init__()
        self.n_theta = config.n_theta
        self.halt_eps = config.halt_eps
        self.lambda_halt = config.lambda_halt

        self.norm = nn.RMSNorm(config.d_model)
        self.W_halt = nn.Linear(config.d_model, 1, bias=True)

        self._init_weights(config)

    def _init_weights(self, config: V8Config) -> None:
        """Initialize halt head.

        Bias: σ(bias) ≈ 2/n_theta → initial readout approximately uniform.
        Weight: small Xavier → halt probability starts near-constant.
        """
        nn.init.xavier_normal_(self.W_halt.weight, gain=0.1)
        if config.n_theta > 2:
            bias_val = math.log(2.0 / max(config.n_theta - 2, 0.5))
        else:
            bias_val = 0.0
        nn.init.constant_(self.W_halt.bias, bias_val)

    def compute_halt_prob(self, z_hidden: torch.Tensor) -> torch.Tensor:
        """Compute per-position halt probability.

        Args:
            z_hidden: (B, L, d_model) hidden cell states (alpha sliced off)

        Returns:
            halt_prob: (B, L) ∈ (0, 1)
        """
        return torch.sigmoid(self.W_halt(self.norm(z_hidden)).squeeze(-1))

    def accumulate(
        self,
        z_hidden: torch.Tensor,
        z_out: torch.Tensor,
        remaining: torch.Tensor,
        is_last: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Accumulate readout-weighted state at a theta boundary.

        Args:
            z_hidden:  (B, L, d_model) current hidden cell states
            z_out:     (B, L, d_model) accumulated readout state
            remaining: (B, L) remaining mass per position
            is_last:   whether this is the final theta cycle

        Returns:
            z_out:     updated readout accumulator
            remaining: updated remaining mass
            halt_prob: (B, L) halt probability (for KL computation)
        """
        halt_prob = self.compute_halt_prob(z_hidden)

        if is_last:
            effective = remaining
        else:
            effective = torch.min(halt_prob * remaining, remaining)

        z_out = z_out + effective.unsqueeze(-1) * z_hidden
        remaining = remaining - effective

        return z_out, remaining, halt_prob

    def compute_kl_loss(
        self, halt_probs: list[torch.Tensor]
    ) -> torch.Tensor:
        """Compute PonderNet KL regularization loss.

        Spec Section 16.3: D_KL(p_halt ∥ Geometric(1/n_theta))

        The prior is Geometric(p = 1/n_theta) — uniform expected compute.

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


class EdgeOfChaosRegularizer(nn.Module):
    """Lyapunov exponent estimation and edge-of-chaos band loss.

    Spec Section 11.1-11.2: Estimates the maximal Lyapunov exponent of the
    growth dynamics and penalizes deviations from the Class IV band.

    Lyapunov exponent estimation (Spec Definition 11.1):
        λ̂_max(K) = (1/K) · ln(‖Φ^K(z₀+ε) − Φ^K(z₀)‖ / ‖ε‖)
        where ε ~ N(0, σ_ε² I) with σ_ε = 10⁻⁴

    Edge-of-chaos loss (Spec Definition 11.2):
        L_edge = α_edge · [ReLU(λ̂_max − λ₊) + ReLU(λ₋ − λ̂_max)]
        where λ₊ = +0.05 (upper chaos threshold)
              λ₋ = −0.05 (lower order threshold)

    Interpretation:
        λ̂_max < 0  → Class I/II (contractive, permanent)
        λ̂_max ≈ 0  → Class IV  (edge of chaos — target)
        λ̂_max > 0  → Class III (chaotic, divergent)

    The loss is zero when λ̂_max ∈ [λ₋, λ₊] = [-0.05, +0.05].

    Note: Lyapunov estimation requires a perturbed forward pass,
    effectively doubling compute for the batch fraction used.
    Default: computed on 25% of batch (config.edge_batch_fraction).
    """

    def __init__(self, config: V8Config):
        super().__init__()
        self.eps_std = config.lyapunov_eps
        self.lambda_plus = config.edge_lambda_plus
        self.lambda_minus = config.edge_lambda_minus

    def estimate_lyapunov(
        self,
        z_clean: torch.Tensor,
        z_perturbed: torch.Tensor,
        epsilon_norm: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        """Estimate maximal Lyapunov exponent from clean and perturbed trajectories.

        This is called AFTER running both clean and perturbed forward passes
        through the gamma steps. The caller is responsible for:
            1. Creating ε ~ N(0, σ_ε² I) perturbation
            2. Running Φ^K(z₀) and Φ^K(z₀ + ε) through the gamma loop
            3. Passing the final states and original perturbation norm here

        Spec Definition 11.1:
            λ̂_max = (1/K) · ln(‖Φ^K(z₀+ε) − Φ^K(z₀)‖ / ‖ε‖)

        Args:
            z_clean:      (B_sub, L, d+1) clean trajectory final state
            z_perturbed:  (B_sub, L, d+1) perturbed trajectory final state
            epsilon_norm: (B_sub,) ‖ε‖ per batch element
            n_steps:      K, number of growth steps applied

        Returns:
            lambda_max: scalar, mean estimated Lyapunov exponent
        """
        # ‖Φ^K(z+ε) − Φ^K(z)‖ per batch element
        divergence = (z_perturbed - z_clean).flatten(1).norm(dim=1)  # (B_sub,)

        # Avoid log(0) — clamp both terms
        ratio = divergence / epsilon_norm.clamp(min=1e-10)
        ratio = ratio.clamp(min=1e-10)

        # λ̂_max = (1/K) · ln(ratio)
        lambda_max = torch.log(ratio) / max(n_steps, 1)

        return lambda_max.mean()

    def compute_loss(self, lambda_max: torch.Tensor) -> torch.Tensor:
        """Compute edge-of-chaos band loss.

        Spec Definition 11.2:
            L_edge = ReLU(λ̂_max − λ₊) + ReLU(λ₋ − λ̂_max)

        Zero when λ̂_max ∈ [λ₋, λ₊], grows linearly outside.

        Args:
            lambda_max: scalar estimated Lyapunov exponent

        Returns:
            loss: scalar edge-of-chaos loss (before weighting by α_edge)
        """
        loss_chaos = torch.relu(lambda_max - self.lambda_plus)
        loss_order = torch.relu(self.lambda_minus - lambda_max)
        return loss_chaos + loss_order


class SpectralEntropyRegularizer(nn.Module):
    """SVD-based spectral entropy and complexity band regularization.

    Spec Section 11.3 (Spectral Entropy):
        Given Z ∈ R^{L×d}, compute SVD singular values {σ₁, ..., σ_min(L,d)}.
        Normalize: σ̂_j = σ_j / Σ σ_j'
        H_sv(Z) = −Σ σ̂_j log₂ σ̂_j

    Properties (Spec Section 11.3):
        H_sv = 0                   when rank(Z) = 1 (Class I: all rows identical)
        H_sv = log₂(min(L,d))     when all σ equal (Class III: max entropy)
        Intermediate H_sv          when structured but non-degenerate (Class IV)

    Complexity band (Spec Definition 19.1):
        η_min = 0.3 · log₂(min(L, d))    (above Class I/II)
        η_max = 0.7 · log₂(min(L, d))    (below Class III)

    Loss (Spec Section 19.3):
        L_complexity = ReLU(η_min − H_sv) + ReLU(H_sv − η_max)
        Zero when H_sv ∈ [η_min, η_max].

    Correlation with BDM (Spec Proposition 11.1):
        Corr(H_sv, BDM) > 0.8 for Class II–IV dynamics.
    """

    def __init__(self, config: V8Config):
        super().__init__()
        self.eta_min_frac = config.spectral_eta_min_frac
        self.eta_max_frac = config.spectral_eta_max_frac

    def compute_spectral_entropy(
        self, Z: torch.Tensor
    ) -> torch.Tensor:
        """Compute spectral entropy of state matrix.

        Spec Definition 11.3:
            σ̂_j = σ_j / Σ σ_j'
            H_sv = −Σ σ̂_j log₂ σ̂_j

        Args:
            Z: (B, L, d) hidden state matrix (alpha channel excluded)

        Returns:
            H_sv: (B,) spectral entropy per batch element
        """
        # Compute singular values (no need for full U, V)
        # torch.linalg.svdvals is differentiable
        sigma = torch.linalg.svdvals(Z)  # (B, min(L, d))

        # Normalize to probability distribution
        sigma_sum = sigma.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        sigma_hat = sigma / sigma_sum  # (B, min(L, d))

        # Entropy: −Σ σ̂_j log₂ σ̂_j
        # Clamp to avoid log(0)
        log2_sigma = torch.log2(sigma_hat.clamp(min=1e-10))
        H_sv = -(sigma_hat * log2_sigma).sum(dim=-1)  # (B,)

        return H_sv

    def compute_loss(
        self, H_sv: torch.Tensor, L: int, d: int
    ) -> torch.Tensor:
        """Compute spectral entropy band loss.

        Spec Definition 19.1 and Section 19.3:
            η_min = 0.3 · log₂(min(L, d))
            η_max = 0.7 · log₂(min(L, d))
            L_complexity = ReLU(η_min − H_sv) + ReLU(H_sv − η_max)

        Args:
            H_sv: (B,) spectral entropy per batch element
            L: sequence length
            d: hidden dimension (d_model)

        Returns:
            loss: scalar complexity band loss
        """
        max_entropy = math.log2(max(min(L, d), 2))
        eta_min = self.eta_min_frac * max_entropy
        eta_max = self.eta_max_frac * max_entropy

        H_mean = H_sv.mean()
        loss_low = torch.relu(eta_min - H_mean)
        loss_high = torch.relu(H_mean - eta_max)
        return loss_low + loss_high
