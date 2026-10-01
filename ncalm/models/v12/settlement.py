"""Settlement state tracking for AENCA V12 — Permanence as Primary Citizen.

Identical to V11's settlement mechanism. V12 uses the same content-driven
settlement: only delta_z drives the convergence EMA; the morphogen field h
is gated by the same growth_gate but does not drive the threshold itself.

Biologically: signaling (morphogen) continues even in differentiated cells,
but what matters for "settling" is whether the cell's function (content z)
has stabilised.

How it works:
    1. Each cell tracks a_bar: EMA of ‖delta_z‖₂
    2. When a_bar < tau, the cell settles (settled → 1)
    3. growth_gate = (1 - settled) zeroes updates for BOTH z and h
    4. Settled cells become frozen read-only memory for neighbours

All settlement state is DETACHED from the gradient graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ncalm.models.v12.config import V12Config


@dataclass
class SettlementState:
    """Per-cell settlement tracking state.

    Not a nn.Module — just tensors. Created fresh each forward pass.
    All updates are detached from the computation graph.

    Attributes:
        a_bar:   (B, L) — EMA of ‖delta_z‖₂. High = actively changing. Low = converged.
        settled: (B, L) — σ((tau - a_bar) / eta). 0 = active, 1 = permanently settled.
    """
    a_bar: torch.Tensor     # (B, L)
    settled: torch.Tensor   # (B, L) ∈ [0, 1]

    @staticmethod
    def create(
        batch_size: int,
        seq_len: int,
        device: torch.device,
        config: V12Config,
    ) -> "SettlementState":
        """Create initial settlement state — all cells active.

        a_bar starts at tau + 3*eta ≈ 0.35, giving:
            settled = σ((0.05 - 0.35) / 0.1) = σ(-3) ≈ 0.047

        Clearly unsettled but REACHABLE within the 64-step pass.
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
    def update(self, delta: torch.Tensor, config: V12Config) -> None:
        """Update settlement state from CONTENT kernel output.

        Settlement tracks delta_z only (not delta_h). The morphogen
        keeps evolving until the cell settles, then both fields freeze.

        All operations are detached — settlement is OBSERVED, not learned.

        Args:
            delta:  (B, L, d_model) — raw content update (before growth gate)
            config: V12Config with settlement parameters
        """
        delta_norm = delta.detach().norm(dim=-1)   # (B, L)
        self.a_bar = (
            config.settlement_beta * self.a_bar
            + (1.0 - config.settlement_beta) * delta_norm
        )
        self.settled = torch.sigmoid(
            (config.settlement_tau - self.a_bar) / config.settlement_eta
        )

    def growth_gate(self) -> torch.Tensor:
        """Growth gate: (1 - settled). Settled cells receive zero update.

        Applied to BOTH content z and morphogen h.

        Returns:
            gate: (B, L) ∈ [0, 1]
        """
        return 1.0 - self.settled

    @property
    def mean_settled(self) -> float:
        return self.settled.mean().item()

    @property
    def active_fraction(self) -> float:
        return (self.settled < 0.5).float().mean().item()
