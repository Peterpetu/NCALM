"""Settlement state tracking for AENCA V10 — Permanence as Primary Citizen.

Settlement is the structural mechanism that makes V10 a TRUE cellular automaton.
In Game of Life, still lifes are permanent. In V10, settled cells are permanent.

How it works:
    1. Each cell tracks a_bar: EMA of ‖delta‖₂ (how much it's changing)
    2. When a_bar drops below tau, the cell settles: settled → 1
    3. The growth gate (1 - settled) zeroes out updates for settled cells
    4. Settled cells become frozen read-only memory for neighbors

Why this matters:
    - Creates PERMANENT structures (CA axiom)
    - Natural gradient truncation (settled cells don't backprop further)
    - Sparse computation (later steps have fewer active cells)
    - Emergent content-based routing (active cells read from settled neighbors)

All settlement state is DETACHED from the gradient graph — it's an observed
running statistic, not a learned parameter. The permanence loss (L_perm)
separately drives deltas toward zero, which CAUSES settlement naturally.

Critical fix from V8: a_bar initialized to tau + 3*eta ≈ 0.35 (not V8's 1e6),
so settlement is REACHABLE within the 64-step forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ncalm.models.v10.config import V10Config


@dataclass
class SettlementState:
    """Per-cell settlement tracking state.

    Not a nn.Module — just tensors. Created fresh each forward pass.
    All updates are detached from the computation graph.

    Attributes:
        a_bar: (B, L) — EMA of ‖delta‖₂. High = actively changing. Low = converged.
        settled: (B, L) — σ((tau - a_bar) / eta). 0 = active, 1 = permanently settled.
    """
    a_bar: torch.Tensor     # (B, L) — convergence EMA
    settled: torch.Tensor   # (B, L) — settlement score ∈ [0, 1]

    @staticmethod
    def create(
        batch_size: int,
        seq_len: int,
        device: torch.device,
        config: V10Config,
    ) -> "SettlementState":
        """Create initial settlement state — all cells active.

        a_bar starts at tau + 3*eta ≈ 0.35, giving:
            settled = σ((0.05 - 0.35) / 0.1) = σ(-3) ≈ 0.047

        This is clearly unsettled but REACHABLE — unlike V8's 1e6 which
        made settlement impossible within the forward pass.
        """
        a_bar = torch.full(
            (batch_size, seq_len),
            config.a_bar_init,
            device=device,
        )
        settled = torch.sigmoid(
            (config.settlement_tau - a_bar) / config.settlement_eta
        )
        return SettlementState(a_bar=a_bar, settled=settled)

    @torch.no_grad()
    def update(self, delta: torch.Tensor, config: V10Config) -> None:
        """Update settlement state from kernel output.

        All operations are detached — settlement is OBSERVED, not learned.
        The permanence loss L_perm separately drives deltas toward zero,
        which causes settlement to increase naturally.

        Args:
            delta: (B, L, d_model) — raw kernel output (before growth gate)
            config: V10Config with settlement parameters
        """
        # ‖delta‖₂ per cell, detached from gradient graph
        delta_norm = delta.detach().norm(dim=-1)  # (B, L)

        # EMA update: track average update magnitude
        self.a_bar = config.settlement_beta * self.a_bar + (1.0 - config.settlement_beta) * delta_norm

        # Settlement score: sigmoid transition around tau
        # When a_bar >> tau: settled ≈ 0 (actively changing)
        # When a_bar << tau: settled ≈ 1 (converged, permanent)
        self.settled = torch.sigmoid(
            (config.settlement_tau - self.a_bar) / config.settlement_eta
        )

    def growth_gate(self) -> torch.Tensor:
        """Growth gate: (1 - settled). Settled cells get zero update.

        Returns:
            gate: (B, L) — 1.0 for active cells, ~0.0 for settled cells
        """
        return 1.0 - self.settled

    @property
    def mean_settled(self) -> float:
        """Mean settlement score across all cells (for logging)."""
        return self.settled.mean().item()

    @property
    def active_fraction(self) -> float:
        """Fraction of cells that are still active (settled < 0.5)."""
        return (self.settled < 0.5).float().mean().item()
