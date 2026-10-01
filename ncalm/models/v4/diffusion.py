"""Masked diffusion process for AENCA V4.

Implements the discrete masked diffusion language model (MDLM) framework:
cosine masking schedule, mixed masking strategies, importance weighting.

Reference: model_documents/aenca_v4_architecture.md §3.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


# ── Masking Schedule (§3.5, Eq 3.11) ────────────────────────────────────────


def cosine_masking_schedule(t: torch.Tensor) -> torch.Tensor:
    """Fraction of tokens masked at diffusion time t.

    gamma(t) = 1 - cos^2(pi*t/2) = sin^2(pi*t/2)

    Properties:
        gamma(0) = 0  (clean data)
        gamma(0.5) = 0.5
        gamma(1) = 1  (fully masked)
    """
    return torch.sin(math.pi * t / 2.0) ** 2


def survival_probability(t: torch.Tensor) -> torch.Tensor:
    """Fraction of tokens surviving (unmasked) at time t.

    alpha(t) = 1 - gamma(t) = cos^2(pi*t/2)
    """
    return torch.cos(math.pi * t / 2.0) ** 2


def importance_weight(t: torch.Tensor) -> torch.Tensor:
    """MDLM importance weight (Eq 3.9, 8.8).

    w(t) = gamma_dot(t) / gamma(t) = pi / tan(pi*t/2)

    Numerical reference:
        w(0.001) ~ 2000
        w(0.01)  ~ 200
        w(0.1)   ~ 18.9
        w(0.5)   ~ pi ~ 3.14
        w(0.9)   ~ 0.53
        w(0.99)  ~ 0.05

    Clamped to avoid division by zero at endpoints.
    """
    half_pi_t = math.pi * t / 2.0
    sin_val = torch.sin(half_pi_t).clamp(min=1e-8)
    cos_val = torch.cos(half_pi_t)
    return math.pi * cos_val / sin_val


# ── Diffusion Time Sampling (§8.1) ──────────────────────────────────────────


def sample_diffusion_times(
    B: int, eps: float = 1e-4, device: torch.device | None = None
) -> torch.Tensor:
    """Sample diffusion times t ~ U[eps, 1 - eps] per example.

    Args:
        B: Batch size.
        eps: Endpoint clamping (Eq 3.12).
        device: Target device.

    Returns:
        t: Shape (B,), float32 in [eps, 1-eps].
    """
    return torch.rand(B, device=device) * (1.0 - 2.0 * eps) + eps


def sample_diffusion_times_curriculum(
    B: int,
    step: int,
    easy_steps: int = 5000,
    mid_steps: int = 20000,
    eps: float = 1e-4,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Curriculum diffusion time sampling (V4.1 §6.3, Eq 6.11-6.13).

    Phase 1 (step < easy_steps):  t ~ Beta(2, 5),   E[t] ≈ 0.286 (easy: more tokens visible)
    Phase 2 (easy ≤ step < mid):  t ~ U[eps, 1-eps], E[t] = 0.5   (standard)
    Phase 3 (step ≥ mid_steps):   t ~ Beta(5, 2),   E[t] ≈ 0.714 (hard: fewer tokens visible)

    Args:
        B:           Batch size.
        step:        Current training step.
        easy_steps:  Boundary for Phase 1 (default 5000).
        mid_steps:   Boundary for Phase 2 (default 20000).
        eps:         Endpoint clamping.
        device:      Target device.

    Returns:
        t: Shape (B,), float32 in [eps, 1-eps].
    """
    if step < easy_steps:
        # Phase 1: Easy — more tokens visible (Beta(2,5), mode ≈ 0.25)
        t = torch.distributions.Beta(2.0, 5.0).sample((B,))
    elif step < mid_steps:
        # Phase 2: Standard — uniform
        t = torch.rand(B)
    else:
        # Phase 3: Hard — fewer tokens visible (Beta(5,2), mode ≈ 0.75)
        t = torch.distributions.Beta(5.0, 2.0).sample((B,))

    return t.clamp(eps, 1.0 - eps).to(device=device)


# ── Masking Strategy Sampling (Definition 3.5) ──────────────────────────────


def sample_masking_strategy(
    B: int,
    p_random: float = 0.5,
    p_causal: float = 0.4,
    p_block: float = 0.1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample masking strategy per example.

    Returns:
        strategy: Shape (B,), int64. 0=random, 1=causal, 2=block.
    """
    probs = torch.tensor(
        [p_random, p_causal, p_block], dtype=torch.float32, device=device
    )
    return torch.multinomial(probs.unsqueeze(0).expand(B, -1), 1).squeeze(1)


# ── Masked Sequence Construction (§3.6, Eq 3.13–3.15) ───────────────────────


def _random_mask(
    gamma_t: torch.Tensor, L: int, device: torch.device
) -> torch.Tensor:
    """Random (i.i.d.) masking (Eq 3.13).

    mu_i ~ Bernoulli(gamma(t)) independently per position.

    Args:
        gamma_t: Shape (B,) masking rates.
        L: Sequence length.
        device: Target device.

    Returns:
        mask: Shape (B, L), bool (True = masked).
    """
    # (B, 1) broadcast → (B, L)
    return torch.rand(gamma_t.shape[0], L, device=device) < gamma_t.unsqueeze(1)


def _causal_mask(B: int, L: int, device: torch.device) -> torch.Tensor:
    """Causal masking (Eq 3.14).

    Sample k ~ U{1, ..., L-1}, mask all positions i > k.
    This creates a random-length prefix (always at least 1 unmasked token).

    Returns:
        mask: Shape (B, L), bool (True = masked).
    """
    # k in {1, ..., L-1}: positions 0..k-1 unmasked, k..L-1 masked
    k = torch.randint(1, L, (B,), device=device)  # (B,)
    positions = torch.arange(L, device=device).unsqueeze(0)  # (1, L)
    return positions >= k.unsqueeze(1)  # (B, L)


def _block_mask(B: int, L: int, device: torch.device) -> torch.Tensor:
    """Block (contiguous) masking (Eq 3.15).

    Sample block length B_len ~ U{floor(0.1*L), ..., floor(0.5*L)},
    then start position B_start ~ U{0, ..., L - B_len}.
    Mask the contiguous block [B_start, B_start + B_len).

    Returns:
        mask: Shape (B, L), bool (True = masked).
    """
    min_len = max(1, int(0.1 * L))
    max_len = max(min_len, int(0.5 * L))
    # Block length per example
    block_len = torch.randint(min_len, max_len + 1, (B,), device=device)  # (B,)
    # Start position per example: 0-indexed, so start in [0, L - block_len]
    max_start = (L - block_len).clamp(min=0)
    # randint needs high > low; when max_start==0, start must be 0
    block_start = torch.zeros(B, dtype=torch.long, device=device)
    nonzero_mask = max_start > 0
    if nonzero_mask.any():
        # For each example with max_start > 0, sample uniformly
        uniform_samples = torch.rand(B, device=device)
        block_start = (uniform_samples * (max_start.float() + 1.0)).long().clamp(
            max=max_start
        )

    positions = torch.arange(L, device=device).unsqueeze(0)  # (1, L)
    start = block_start.unsqueeze(1)  # (B, 1)
    end = (block_start + block_len).unsqueeze(1)  # (B, 1)
    return (positions >= start) & (positions < end)  # (B, L)


def create_masked_sequence(
    x: torch.Tensor,
    t: torch.Tensor,
    strategy: torch.Tensor,
    mask_token_id: int,
    masking_eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create masked sequence z_t from clean tokens x and diffusion time t.

    Applies the appropriate masking strategy per example in the batch.

    Args:
        x: Clean token IDs, shape (B, L), dtype long.
        t: Diffusion times, shape (B,), float in [eps, 1-eps].
        strategy: Masking strategy per example, shape (B,), int64.
            0=random, 1=causal, 2=block.
        mask_token_id: The [MASK] token ID (= vocab_size).
        masking_eps: Endpoint clamping epsilon.

    Returns:
        z_t: Masked token IDs, shape (B, L), with mask_token_id at masked positions.
        mask: Boolean mask, shape (B, L), True at masked positions.
    """
    B, L = x.shape
    device = x.device

    # Compute per-example masking rate
    gamma_t = cosine_masking_schedule(t)  # (B,)

    # Initialize full mask as False
    mask = torch.zeros(B, L, dtype=torch.bool, device=device)

    # Apply each strategy to the corresponding examples
    is_random = strategy == 0
    is_causal = strategy == 1
    is_block = strategy == 2

    # Random masking (Eq 3.13)
    if is_random.any():
        n_random = is_random.sum().item()
        random_idx = is_random.nonzero(as_tuple=True)[0]
        random_mask = _random_mask(gamma_t[random_idx], L, device)
        mask[random_idx] = random_mask

    # Causal masking (Eq 3.14)
    if is_causal.any():
        n_causal = is_causal.sum().item()
        causal_idx = is_causal.nonzero(as_tuple=True)[0]
        causal_mask = _causal_mask(n_causal, L, device)
        mask[causal_idx] = causal_mask

    # Block masking (Eq 3.15)
    if is_block.any():
        n_block = is_block.sum().item()
        block_idx = is_block.nonzero(as_tuple=True)[0]
        block_mask = _block_mask(n_block, L, device)
        mask[block_idx] = block_mask

    # Ensure at least one token is masked per example (for valid loss)
    # If all unmasked, mask the last position
    all_unmasked = ~mask.any(dim=1)  # (B,)
    if all_unmasked.any():
        mask[all_unmasked, -1] = True

    # Ensure at least one token is unmasked per example (for valid context)
    all_masked = mask.all(dim=1)  # (B,)
    if all_masked.any():
        mask[all_masked, 0] = False

    # Apply mask: replace masked positions with mask_token_id
    z_t = x.clone()
    z_t[mask] = mask_token_id

    return z_t, mask
