"""1D bidirectional dilated NCA substrate for AENCA V4/V4.1.

Contains:
- DilatedConv1DPerception: shared-weight conv with dynamic dilation (§5.6)
- ContentGate: channel-wise selection gate (V4.1 §2, Eq 2.2-2.3)
- TemporalPredictor: self-prediction for Active Inference (V4.1 §5, Eq 5.1-5.2)
- SubstrateInitializer: creates s^(0) from masked sequence z_t (§4.4)
- NCASubstrate: orchestrates K NCA steps with gated updates (§5.3)

The substrate is a 1D chain where each cell IS a sequence position.
All cells update simultaneously using the same kernel — this is the
fundamental NCA property that replaces attention.

V4 reference: model_documents/aenca_v4_architecture.md §4.4, §5.
V4.1 reference: model_documents/aenca_v4_1_architecture.md §1-§5, §8.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v4.config import AENCAv4Config
from ncalm.models.v4.encoding import SinusoidalPositionEncoding, SinusoidalTimestepEncoding
from ncalm.models.v4.kernel import NCAKernel
from ncalm.models.v4.sensing import ActiveSensing


class DilatedConv1DPerception(nn.Module):
    """Dilated 1D convolution for NCA perception (§5.6, Definition 5.7, Eq 5.24).

    A single weight tensor W_conv shared across ALL NCA steps.
    Only the dilation parameter changes between steps: d_k = 2^{k mod M}.

    W_conv in R^{D_conv x D x (2r+1)}
    No bias (§5.3).

    Initialization (Eq 5.32): Xavier uniform, gain = 1/sqrt(2r+1).
    """

    def __init__(self, config: AENCAv4Config) -> None:
        super().__init__()
        self.config = config
        kernel_size = 2 * config.kernel_radius + 1

        # Weight tensor for conv1d: (out_channels, in_channels, kernel_size)
        self.weight = nn.Parameter(
            torch.empty(config.D_conv, config.D, kernel_size)
        )
        # No bias (§5.3)

        # Xavier uniform init with gain = 1/sqrt(2r+1) (Eq 5.32)
        nn.init.xavier_uniform_(self.weight, gain=config.conv_xavier_gain)

    def forward(self, substrate: torch.Tensor, dilation: int) -> torch.Tensor:
        """Apply dilated convolution with dynamic dilation.

        Args:
            substrate: (B, L, D) — current substrate state or prediction error.
            dilation:  int — dilation value d_k = 2^{k mod M}.

        Returns:
            perception: (B, L, D_conv) — local context vector.
        """
        # Conv1d expects (B, C, L) format
        x = substrate.transpose(1, 2)  # (B, D, L)

        # Dynamic padding to preserve spatial dimension: padding = r * dilation
        padding = self.config.kernel_radius * dilation

        # Apply conv with shared weight and dynamic dilation
        P = F.conv1d(x, self.weight, bias=None, dilation=dilation, padding=padding)
        # (B, D_conv, L)

        return P.transpose(1, 2)  # (B, L, D_conv)


class ContentGate(nn.Module):
    """Content-selective gating for perception (V4.1 §2, Eq 2.2-2.3).

    g_sel = sigma(W_sel @ s + b_sel)    — channel-wise gate from substrate state
    P_gated = g_sel * P_raw             — precision-weighted perception

    This acts as a learned precision matrix Π = diag(g_sel), implementing
    the Active Inference principle: weight each channel by its estimated
    reliability.

    Init: Xavier gain=0.1, bias=zeros → gates start at sigmoid(0) = 0.5
    """

    def __init__(self, config: AENCAv4Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.D, config.D)
        # Init: small weights so gates start near 0.5 (uninformative)
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self, substrate: torch.Tensor, perception: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply content gating to perception.

        Args:
            substrate:  (B, L, D) — current substrate state.
            perception: (B, L, D) — raw perception from convolution.

        Returns:
            gated_perception: (B, L, D) — g_sel * perception.
            gate_values:      (B, L, D) — sigmoid gate values (for loss).
        """
        gate_values = torch.sigmoid(self.gate_proj(substrate))  # (B, L, D)
        return gate_values * perception, gate_values


class TemporalPredictor(nn.Module):
    """Self-prediction for Active Inference dynamics (V4.1 §5, Eq 5.1-5.2).

    μ_i^(k) = W_pred @ s_i^(k) + b_pred   — predict next state
    ε_i^(k) = s_i^(k+1) - μ_i^(k)          — prediction error

    At k=0: no prediction available, use raw substrate for conv.
    At k>=1: conv operates on prediction error ε instead of raw state.

    Init: weight = I + N(0, 0.01²), bias = 0 (near-identity, Eq 5.16)
    """

    def __init__(self, config: AENCAv4Config) -> None:
        super().__init__()
        self.predictor = nn.Linear(config.D, config.D)
        # Near-identity initialization (Eq 5.16)
        nn.init.eye_(self.predictor.weight)
        self.predictor.weight.data.add_(torch.randn_like(self.predictor.weight) * 0.01)
        nn.init.zeros_(self.predictor.bias)

    def predict(self, substrate: torch.Tensor) -> torch.Tensor:
        """Predict next substrate state.

        Args:
            substrate: (B, L, D) — current state.

        Returns:
            prediction: (B, L, D) — predicted next state μ.
        """
        return self.predictor(substrate)

    def compute_error(
        self, actual: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        """Compute prediction error.

        Args:
            actual:     (B, L, D) — actual next state s^(k+1).
            prediction: (B, L, D) — predicted state μ^(k).

        Returns:
            error: (B, L, D) — prediction error ε = actual - prediction.
        """
        return actual - prediction


class SubstrateInitializer(nn.Module):
    """Initialize substrate s^(0) from masked sequence z_t (§4.4, Eq 4.6-4.9).

    Cell state: s_i = [z_i, h_i, c_i] in R^D

    Latent channels (Eq 4.6-4.7):
        z_i = E_proj(e_{x_i})  if unmasked
        z_i = z_mask            if masked (learnable)

    Hidden channels (Eq 4.8):
        h_i = MLP_init(rho(i))  — position-dependent initialization

    Control channels (Eq 4.9):
        c_i = 0
    """

    def __init__(self, config: AENCAv4Config) -> None:
        super().__init__()
        self.config = config

        # Token embeddings: |V|+1 entries (includes [MASK] token)
        self.token_embedding = nn.Embedding(config.vocab_size + 1, config.d_emb)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        # Encoder projection: d_emb → d_z (Eq 4.1)
        self.encoder_proj = nn.Linear(config.d_emb, config.d_z)

        # Learnable [MASK] latent embedding (Eq 4.7)
        # Separate from the embedding table — directly in latent space
        self.z_mask = nn.Parameter(torch.randn(config.d_z) * 0.02)

        # Hidden init MLP: d_pos → d_h (Eq 4.8)
        self.hidden_init_mlp = nn.Sequential(
            nn.Linear(config.d_pos, config.d_h),
            nn.GELU(),
            nn.Linear(config.d_h, config.d_h),
        )

        # Position encoder for initialization
        self.pos_encoder = SinusoidalPositionEncoding(config.d_pos, config.L_max)

    def forward(
        self, z_t: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize substrate and return token embeddings.

        Args:
            z_t:  (B, L) masked token IDs (mask_token_id at masked positions).
            mask: (B, L) bool, True at masked positions.

        Returns:
            s0:     (B, L, D) initial substrate state.
            z_t_emb: (B, L, d_emb) token embeddings of z_t (for sensing).
        """
        B, L = z_t.shape
        device = z_t.device

        # Token embeddings of masked sequence (for sensor observations)
        z_t_emb = self.token_embedding(z_t)  # (B, L, d_emb)

        # ── Latent channels z^(0) (Eq 4.6-4.7) ─────────────────────────
        # Project ALL embeddings through encoder
        z_all = self.encoder_proj(z_t_emb)  # (B, L, d_z)

        # Override masked positions with learnable z_mask
        z_latent = z_all.clone()
        z_latent[mask] = self.z_mask.to(z_latent.dtype).unsqueeze(0).expand(mask.sum(), -1)

        # ── Hidden channels h^(0) (Eq 4.8) ──────────────────────────────
        positions = torch.arange(L, device=device)  # (L,)
        pos_enc = self.pos_encoder(positions)  # (L, d_pos)
        h_init = self.hidden_init_mlp(pos_enc)  # (L, d_h)
        # Broadcast to batch: (1, L, d_h) → (B, L, d_h)
        h_init = h_init.unsqueeze(0).expand(B, -1, -1)

        # ── Control channels c^(0) = 0 (Eq 4.9) ────────────────────────
        c_init = torch.zeros(B, L, self.config.d_c, device=device, dtype=z_latent.dtype)

        # ── Assemble cell state (Eq 4.5) ────────────────────────────────
        s0 = torch.cat([z_latent, h_init, c_init], dim=-1)  # (B, L, D)

        return s0, z_t_emb

    def get_token_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings (utility for observation gathering).

        Args:
            token_ids: (...) int64 token IDs.

        Returns:
            embeddings: (..., d_emb).
        """
        return self.token_embedding(token_ids)


class NCASubstrate(nn.Module):
    """Orchestrates the 1D NCA substrate dynamics for K steps (§5.3, Algorithm 1).

    V4.1 additions:
        - ContentGate: channel-wise precision gating of perception
        - TemporalPredictor: prediction-error perception (true Active Inference)
        - Contextual readout: 3-position window [z_{i-1}, z_i, z_{i+1}]
        - Stochastic dynamics: learned noise σ_noise per step
        - Modified nca_step with prediction error flow

    Contains all sub-modules:
        - DilatedConv1DPerception: shared-weight local perception
        - NCAKernel f_theta: shared kernel (THE NCA rule)
        - SubstrateInitializer: s^(0) construction
        - ActiveSensing: pointer management, observation, EFE
        - Readout head: contextual D_proj (V4.1 Eq 3.3-3.5)
        - Position/timestep encoders
    """

    def __init__(self, config: AENCAv4Config) -> None:
        super().__init__()
        self.config = config

        # Core components
        self.perception = DilatedConv1DPerception(config)
        self.kernel = NCAKernel(config)
        self.initializer = SubstrateInitializer(config)
        self.sensing = ActiveSensing(config)

        # V4.1: Content gate (§2, Eq 2.2-2.3)
        self.content_gate = ContentGate(config)

        # V4.1: Temporal predictor (§5, Eq 5.1-5.2)
        self.temporal_predictor = TemporalPredictor(config)

        # V4.1: Contextual readout (§3, Eq 3.3-3.5)
        # W_ctx: 3*d_z → d_z (no bias, Eq 3.4)
        self.context_proj = nn.Linear(3 * config.d_z, config.d_z, bias=False)
        # Xavier gain = 1/sqrt(3) (Eq 3.14)
        nn.init.xavier_uniform_(self.context_proj.weight, gain=1.0 / math.sqrt(3))
        # W_D: d_z → |V| (shared between final and anytime readouts)
        self.readout = nn.Linear(config.d_z, config.vocab_size)

        # V4.1: Stochastic dynamics (§8, Eq 8.2)
        # Learned scalar log(σ_noise), initialized to log(0.01)
        self.log_sigma_noise = nn.Parameter(
            torch.tensor(math.log(config.sigma_noise_init))
        )

        # Encodings
        self.pos_encoder = SinusoidalPositionEncoding(config.d_pos, config.L_max)
        self.timestep_encoder = SinusoidalTimestepEncoding(config.d_t)

    def get_dilation(self, k: int) -> int:
        """Dilation schedule: d_k = 2^{k mod M} (Eq 5.6)."""
        return 2 ** (k % self.config.M)

    def nca_step(
        self,
        substrate: torch.Tensor,
        k: int,
        t_enc: torch.Tensor,
        z_t_emb: torch.Tensor,
        pointers: torch.Tensor,
        scale_levels: torch.Tensor,
        pos_encodings: torch.Tensor,
        tau_pi: float,
        prev_prediction: torch.Tensor | None = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        """Execute one V4.1 NCA step (Definition 5.4, V4.1 Eq 5.2-5.6, 2.2-2.3, 8.2).

        Args:
            substrate:       (B, L, D) current state.
            k:               NCA step index (for dilation only).
            t_enc:           (B, d_t) timestep encoding.
            z_t_emb:         (B, L, d_emb) masked token embeddings.
            pointers:        (B, L, h) sensor pointer positions.
            scale_levels:    (B, L, h) sensor scale levels.
            pos_encodings:   (B, L, d_pos) pre-computed position encodings.
            tau_pi:          Policy temperature for Tier-1 action selection.
            prev_prediction: (B, L, D) or None — prediction from previous step.

        Returns:
            substrate_new:      (B, L, D) updated substrate.
            pointers_new:       (B, L, h) updated pointers.
            scales_new:         (B, L, h) updated scale levels.
            action_log_probs:   (B, L, h) log pi for selected actions.
            action_logits:      (B, L, h*|A|) raw action logits.
            current_prediction: (B, L, D) prediction for next step's error.
            gate_values:        (B, L, D) content gate values (for loss).
        """
        B, L, D = substrate.shape

        # [1. Perception input selection] (V4.1 Eq 5.3 vs 5.6)
        if k == 0 or prev_prediction is None:
            # Bootstrap: use raw substrate (no prediction available yet)
            conv_input = substrate
        else:
            # Prediction error: ε = s - μ (V4.1 Eq 5.2)
            conv_input = substrate - prev_prediction

        # [2. Dilated convolution] (Eq 5.9)
        d_k = self.get_dilation(k)
        P_raw = self.perception(conv_input, dilation=d_k)  # (B, L, D_conv)

        # [3. Content gating] (V4.1 Eq 2.3): precision-weighted perception
        P, gate_values = self.content_gate(substrate, P_raw)  # (B, L, D), (B, L, D)

        # [4. Self-prediction] (V4.1 Eq 5.1): for next step's error
        current_prediction = self.temporal_predictor.predict(substrate)  # (B, L, D)

        # [5. Sensor observation] (Eq 5.10)
        obs = self.sensing.gather_observations(z_t_emb, pointers)  # (B, L, h*d_emb)

        # [6. Positional & timestep encoding] (Eq 5.11)
        psi = torch.cat(
            [pos_encodings, t_enc.unsqueeze(1).expand(-1, L, -1)],
            dim=-1,
        )  # (B, L, d_pos + d_t)

        # [7. Kernel application] (Eq 5.12)
        x_in = torch.cat([P, obs, psi], dim=-1)  # (B, L, d_in)
        delta_s, gate_logits, action_logits = self.kernel(x_in)

        # [8. Gated update with stochastic noise] (V4.1 Eq 8.2)
        gate = torch.sigmoid(gate_logits)  # (B, L, D)
        # FP32 accumulation for stability (§8.6)
        substrate_new = substrate.float() + (gate * delta_s).float()
        # Stochastic noise (training only)
        if self.training:
            sigma = torch.exp(self.log_sigma_noise)
            noise = torch.randn_like(substrate_new) * sigma
            substrate_new = substrate_new + noise
        substrate_new = substrate_new.to(substrate.dtype)

        # [9. Stability clamping] (Eq 5.15)
        substrate_new = substrate_new.clamp(-self.config.C_clamp, self.config.C_clamp)

        # [10. Sensor action selection — Tier 1] (Eq 6.16-6.17)
        actions, log_probs = self.sensing.select_actions_tier1(
            action_logits, tau_pi
        )

        # [11. Sensor movement] (Eq 6.7-6.8)
        pointers_new, scales_new = self.sensing.move_sensors(
            pointers, scale_levels, actions, L
        )

        return (
            substrate_new, pointers_new, scales_new,
            log_probs, action_logits,
            current_prediction, gate_values,
        )

    def get_readout_logits(self, substrate: torch.Tensor) -> torch.Tensor:
        """Contextual readout: 3-position window (V4.1 §3, Eq 3.3-3.5).

        z_i context = [z_{i-1}, z_i, z_{i+1}] → W_ctx → W_D → logits

        This replaces V4's simple W_D · z_i readout, providing direct
        access to neighbor latent channels for bigram information.

        Args:
            substrate: (B, L, D) — substrate state.

        Returns:
            logits: (B, L, vocab_size) — token predictions.
        """
        d_z = self.config.d_z
        z = substrate[:, :, :d_z]  # (B, L, d_z)

        # Pad with zeros at boundaries (Eq 3.5 boundary condition)
        z_pad = F.pad(z, (0, 0, 1, 1), value=0.0)  # (B, L+2, d_z)

        # 3-position window: [z_{i-1}, z_i, z_{i+1}]
        context = torch.cat(
            [z_pad[:, :-2, :], z_pad[:, 1:-1, :], z_pad[:, 2:, :]],
            dim=-1,
        )  # (B, L, 3*d_z)

        # Project context to d_z (Eq 3.4)
        h_ctx = self.context_proj(context)  # (B, L, d_z)

        # Readout to vocabulary (Eq 3.3)
        return self.readout(h_ctx)  # (B, L, vocab_size)
