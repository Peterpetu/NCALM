"""Mixture-of-Softmax Readout for AENCA V5.

Implements the MoS readout head (§7) with optional tied embeddings.
For smoke/small models: single softmax with tied embeddings (no MoS).
For medium+ models: M mixture components for higher effective rank.

Reference: model_documents/aenca_v5_architecture.md §7.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v5.config import V5Config


class MixtureOfSoftmaxReadout(nn.Module):
    """MoS readout head (§7.2-7.6).

    When mos_components=1 and tie_embeddings=True: standard tied linear readout.
    When mos_components>1 and tie_embeddings=True: tied MoS (§7.5, Eq 7.11).
    When mos_components>1 and tie_embeddings=False: full MoS (§7.2, Eq 7.2-7.4).

    IMPORTANT:
        When mos_components == 1: forward() returns raw logits.
        When mos_components > 1: forward() returns log-probabilities.
        Use the `returns_log_probs` property to determine which loss function to use.
    """

    def __init__(self, config: V5Config, embedding_weight: torch.Tensor | None = None):
        """Initialize MoS readout.

        Args:
            config: V5 configuration
            embedding_weight: W_E weight tensor for tied embeddings (V, d_model)
        """
        super().__init__()
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.mos_components = config.mos_components
        self.tie_embeddings = config.tie_embeddings

        # Pre-norm before readout (§6.3)
        self.norm = nn.RMSNorm(config.d_model)

        if config.mos_components == 1:
            # Standard single readout (§7.6, Eq 7.12)
            if config.tie_embeddings and embedding_weight is not None:
                self.embedding_weight = embedding_weight  # Shared reference
                self.readout_proj = None
            else:
                self.embedding_weight = None
                self.readout_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
                nn.init.xavier_normal_(self.readout_proj.weight, gain=1.0)
        else:
            # MoS with M components
            if config.tie_embeddings and embedding_weight is not None:
                # Tied MoS (§7.5, Eq 7.11): per-component projection before shared W_E^T
                self.embedding_weight = embedding_weight
                self.readout_proj = None
                self.mix_projs = nn.ModuleList([
                    nn.Linear(config.d_model, config.d_model, bias=False)
                    for _ in range(config.mos_components)
                ])
                for proj in self.mix_projs:
                    nn.init.eye_(proj.weight)  # Start as identity
            else:
                # Full MoS (§7.2, Eq 7.2): independent W_D^m
                self.embedding_weight = None
                self.readout_proj = None
                self.mix_projs = nn.ModuleList([
                    nn.Linear(config.d_model, config.vocab_size, bias=False)
                    for _ in range(config.mos_components)
                ])
                for proj in self.mix_projs:
                    nn.init.xavier_normal_(proj.weight, gain=1.0)

            # Mixture weight projection (§7.2, Eq 7.3)
            self.W_mix = nn.Linear(config.d_model, config.mos_components, bias=False)
            nn.init.zeros_(self.W_mix.weight)  # Uniform mixture at start (§10.5)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Compute logits (or log-probs for MoS) from final cell states.

        Args:
            s: (B, L, d_model) final cell states s^(K)

        Returns:
            logits: (B, L, V) output logits (mos_components==1)
                    OR log-probabilities (mos_components>1)
        """
        s_norm = self.norm(s)  # (B, L, d_model)

        if self.mos_components == 1:
            return self._single_readout(s_norm)
        else:
            return self._mos_readout(s_norm)

    @property
    def returns_log_probs(self) -> bool:
        """Whether forward() returns log-probabilities instead of raw logits.

        MoS readout (mos_components > 1) returns log p = log Σ_m π_m softmax(l_m),
        which is in log-probability space. Must use F.nll_loss, NOT F.cross_entropy.
        """
        return self.mos_components > 1

    def _single_readout(self, s: torch.Tensor) -> torch.Tensor:
        """Standard single softmax readout."""
        if self.embedding_weight is not None:
            # Tied: logits = s @ W_E^T (§7.6, Eq 7.12)
            return F.linear(s, self.embedding_weight)  # (B, L, V)
        else:
            return self.readout_proj(s)

    def _mos_readout(self, s: torch.Tensor) -> torch.Tensor:
        """Mixture-of-Softmax readout (§7.2).

        Computes log p = log Σ_m π_m · softmax(logits_m).
        Returns log-space for numerical stability, then converts back to logits.
        """
        B, L, D = s.shape
        M = self.mos_components

        # Compute mixture weights (Eq 7.3)
        mix_weights = F.softmax(self.W_mix(s), dim=-1)  # (B, L, M)

        # Compute per-component logits
        if self.embedding_weight is not None:
            # Tied MoS (Eq 7.11)
            all_logits = torch.stack([
                F.linear(proj(s), self.embedding_weight)  # (B, L, V)
                for proj in self.mix_projs
            ], dim=-2)  # (B, L, M, V)
        else:
            # Full MoS (Eq 7.2)
            all_logits = torch.stack([
                proj(s) for proj in self.mix_projs
            ], dim=-2)  # (B, L, M, V)

        # Compute per-component log softmax
        log_probs = F.log_softmax(all_logits, dim=-1)  # (B, L, M, V)

        # Mix: log Σ_m π_m · exp(log_probs_m)
        # = log Σ_m exp(log(π_m) + log_probs_m)
        log_mix_weights = torch.log(mix_weights + 1e-8).unsqueeze(-1)  # (B, L, M, 1)
        log_mixture = torch.logsumexp(log_mix_weights + log_probs, dim=-2)  # (B, L, V)

        return log_mixture  # These are log-probabilities, usable as logits for CE
