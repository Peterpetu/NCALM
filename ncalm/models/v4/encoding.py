"""Sinusoidal position and timestep encodings for AENCA V4.

Zero-parameter encodings that provide spatial and temporal information
to the NCA substrate.

Reference: model_documents/aenca_v4_architecture.md §4.5, §5.4.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionEncoding(nn.Module):
    """Sinusoidal position encoding (Eq 5.16).

    rho(i) = [sin(w_1 * i/L_max), cos(w_1 * i/L_max), ...,
              sin(w_{d/2} * i/L_max), cos(w_{d/2} * i/L_max)]

    where w_j = 10000^{-2j/d} for j = 0, ..., d/2 - 1.

    No learnable parameters. Registered as buffer for device/dtype tracking.
    """

    def __init__(self, d: int, L_max: int = 262_144) -> None:
        super().__init__()
        assert d % 2 == 0, f"Encoding dimension must be even, got {d}"
        self.d = d
        self.L_max = L_max

        # Frequency bands: w_j = 10000^{-2j/d}
        j = torch.arange(d // 2, dtype=torch.float32)
        freq = 10000.0 ** (-2.0 * j / d)  # (d/2,)
        self.register_buffer("freq", freq)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Encode integer positions into sinusoidal vectors.

        Args:
            positions: Integer positions of any shape (...).

        Returns:
            Sinusoidal encodings of shape (..., d).
        """
        # Normalize positions by L_max (Eq 5.16)
        x = positions.float().unsqueeze(-1) / self.L_max  # (..., 1)
        # Phase: x * freq → (..., d/2)
        phase = x * self.freq  # broadcast: (..., 1) * (d/2,) → (..., d/2)
        # Interleave sin and cos
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)  # (..., d)


class SinusoidalTimestepEncoding(nn.Module):
    """Sinusoidal timestep encoding for diffusion time t in [0, 1] (Eq 4.10).

    tau(t) = [sin(w_1 * t), cos(w_1 * t), ...,
              sin(w_{d/2} * t), cos(w_{d/2} * t)]

    where w_j = 10000^{-2j/d} for j = 0, ..., d/2 - 1.

    Same frequency formula as position encoding but takes continuous t
    directly (not divided by L_max).
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        assert d % 2 == 0, f"Encoding dimension must be even, got {d}"
        self.d = d

        # Frequency bands: w_j = 10000^{-2j/d}
        j = torch.arange(d // 2, dtype=torch.float32)
        freq = 10000.0 ** (-2.0 * j / d)  # (d/2,)
        self.register_buffer("freq", freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode diffusion timesteps into sinusoidal vectors.

        Args:
            t: Diffusion times in [0, 1], shape (B,).

        Returns:
            Timestep encodings, shape (B, d).
        """
        phase = t.unsqueeze(-1) * self.freq  # (B, 1) * (d/2,) → (B, d/2)
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)  # (B, d)
