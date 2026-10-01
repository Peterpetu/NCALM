"""Active sensing module for AENCA V4.

Implements the complete active sensing pipeline:
- Sensor pointer initialization and dynamics (§6.1-6.4)
- Observation gathering (§6.5, Eq 6.9-6.12)
- Observation model g_theta for prediction (§6.5, Eq 6.13-6.14)
- Tier-1 amortised policy (§6.6, Eq 6.16-6.17)
- Tier-2 explicit EFE computation (§6.7, Eq 6.18-6.27)
- REINFORCE advantage computation (§6.8, Eq 6.28-6.32)

Reference: model_documents/aenca_v4_architecture.md §6.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v4.config import AENCAv4Config
from ncalm.models.v4.encoding import SinusoidalPositionEncoding


class ObservationModel(nn.Module):
    """Observation model g_theta (Definition 6.5, Eq 6.13-6.14).

    Predicts what a sensor will observe at a candidate position, given
    the cell's hidden state and relative spatial information.

    g_theta([h_i, RelPosEnc(i - p_cand), ScaleEnc(2^l_cand)]) -> e_hat

    Architecture: 2-layer MLP (Eq 6.14)
        Input:  d_h + 2*d_pos = 256 + 64 = 320
        Hidden: d_g_hidden = 256
        Output: d_emb = 384
    """

    def __init__(self, config: AENCAv4Config) -> None:
        super().__init__()
        input_dim = config.d_h + 2 * config.d_pos  # 256 + 64 = 320
        self.net = nn.Sequential(
            nn.Linear(input_dim, config.d_g_hidden),
            nn.GELU(),
            nn.Linear(config.d_g_hidden, config.d_emb),
        )

    def forward(
        self,
        h: torch.Tensor,
        rel_pos_enc: torch.Tensor,
        scale_enc: torch.Tensor,
    ) -> torch.Tensor:
        """Predict observation embedding.

        Args:
            h:            (..., d_h)   — hidden channels of cell state.
            rel_pos_enc:  (..., d_pos) — sinusoidal encoding of (i - p_cand).
            scale_enc:    (..., d_pos) — sinusoidal encoding of 2^l_cand.

        Returns:
            predicted_emb: (..., d_emb) — predicted token embedding.
        """
        x = torch.cat([h, rel_pos_enc, scale_enc], dim=-1)
        return self.net(x)


class ActiveSensing(nn.Module):
    """Complete active sensing pipeline (§6).

    Manages sensor pointers, observations, action selection (Tier-1 & Tier-2),
    and the observation model for EFE computation.
    """

    def __init__(self, config: AENCAv4Config) -> None:
        super().__init__()
        self.config = config

        self.obs_model = ObservationModel(config)
        # Value baseline V_theta(h) for REINFORCE (Eq 6.30)
        self.value_head = nn.Linear(config.d_h, 1)
        # Position encoder for relative position / scale encoding
        self.pos_encoder = SinusoidalPositionEncoding(config.d_pos, config.L_max)

    # ── Sensor Initialization (§6.1, Eq 6.3-6.5) ────────────────────────

    def init_sensors(
        self, B: int, L: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize sensor pointers and scale levels.

        Pointer offsets (Eq 6.4): Delta_init(j) = (-1)^j * ceil(j/2)
        For h=2 (1-indexed j): offsets = [-1, +1]
        p_{i,j}^(0) = clip(i + offset_j, 0, L-1)   [0-indexed]
        l_{i,j}^(0) = 0                              [finest scale]

        Args:
            B: Batch size.
            L: Sequence length.
            device: Target device.

        Returns:
            pointers:     (B, L, h) int64, initial pointer positions in [0, L-1].
            scale_levels: (B, L, h) int64, all zeros (finest scale).
        """
        h = self.config.n_sensor_heads

        # Compute offsets: for 1-indexed j, Delta_init(j) = (-1)^j * ceil(j/2)
        # j=1: (-1)^1 * 1 = -1,  j=2: (-1)^2 * 1 = +1
        offsets = []
        for j in range(1, h + 1):
            offsets.append(((-1) ** j) * math.ceil(j / 2))
        offsets_t = torch.tensor(offsets, dtype=torch.long, device=device)  # (h,)

        # Position indices: (L,)
        positions = torch.arange(L, dtype=torch.long, device=device)

        # Broadcast: (L, 1) + (h,) → (L, h)
        pointers = (positions.unsqueeze(1) + offsets_t.unsqueeze(0)).clamp(0, L - 1)
        # Expand to batch: (1, L, h) → (B, L, h)
        pointers = pointers.unsqueeze(0).expand(B, -1, -1).clone()

        # Scale levels: heads 1-2 at finest scale (0), heads 3+ at scale 1 (V4.1 Eq 7.8)
        scale_levels = torch.zeros(B, L, h, dtype=torch.long, device=device)
        if h > 2:
            scale_levels[:, :, 2:] = 1

        return pointers, scale_levels

    # ── Observation Gathering (§6.5, Eq 6.9-6.12) ───────────────────────

    def gather_observations(
        self, z_t_emb: torch.Tensor, pointers: torch.Tensor
    ) -> torch.Tensor:
        """Gather sensor observations from token embeddings.

        For each sensor head j: o_{i,j} = e_{z_t^{p_{i,j}}}
        Concatenated: o_i = [o_{i,1}, ..., o_{i,h}]

        Args:
            z_t_emb:  (B, L, d_emb) — token embeddings of masked sequence z_t.
            pointers: (B, L, h)     — current pointer positions (0-indexed).

        Returns:
            obs: (B, L, h*d_emb) — concatenated observations for all heads.
        """
        B, L, d_emb = z_t_emb.shape
        h = pointers.shape[2]

        # Gather embeddings at pointer positions
        # pointers: (B, L, h) → flatten to (B, L*h) for gather
        ptr_flat = pointers.reshape(B, L * h)  # (B, L*h)
        # Expand for embedding dim: (B, L*h, d_emb)
        ptr_expanded = ptr_flat.unsqueeze(-1).expand(-1, -1, d_emb)
        # Gather: z_t_emb is (B, L, d_emb), gather along dim=1
        obs_flat = torch.gather(z_t_emb, dim=1, index=ptr_expanded)  # (B, L*h, d_emb)
        # Reshape to (B, L, h*d_emb)
        obs = obs_flat.reshape(B, L, h * d_emb)

        return obs

    # ── Tier-1 Amortised Policy (§6.6, Eq 6.16-6.17) ───────────────────

    def select_actions_tier1(
        self, action_logits: torch.Tensor, tau_pi: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tier-1 amortised action selection.

        pi_1(a_j | s_i) = softmax(l_a[j*|A|:(j+1)*|A|] / tau_pi)
        a_{i,j} ~ Cat(pi_1)

        Args:
            action_logits: (B, L, h*|A|) — raw action logits from kernel.
            tau_pi: Policy temperature.

        Returns:
            actions:   (B, L, h) — selected action indices in [0, |A|-1].
            log_probs: (B, L, h) — log pi_1(a | s) for selected actions.
        """
        B, L, _ = action_logits.shape
        h = self.config.n_sensor_heads
        n_A = self.config.n_actions

        # Reshape to (B, L, h, |A|)
        logits = action_logits.reshape(B, L, h, n_A)

        # Temperature-scaled softmax
        probs = F.softmax(logits / tau_pi, dim=-1)  # (B, L, h, |A|)

        # Sample actions
        probs_flat = probs.reshape(-1, n_A)  # (B*L*h, |A|)
        actions_flat = torch.multinomial(probs_flat, 1).squeeze(-1)  # (B*L*h,)
        actions = actions_flat.reshape(B, L, h)  # (B, L, h)

        # Log probabilities for selected actions
        log_probs_all = F.log_softmax(logits / tau_pi, dim=-1)  # (B, L, h, |A|)
        log_probs = log_probs_all.gather(
            dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)  # (B, L, h)

        return actions, log_probs

    # ── Sensor Dynamics (§6.2, Eq 6.7-6.8) ──────────────────────────────

    def move_sensors(
        self,
        pointers: torch.Tensor,
        scale_levels: torch.Tensor,
        actions: torch.Tensor,
        L: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply sensor movement dynamics.

        Action encoding (Definition 6.3):
            action a encodes (alpha, beta) in {-1,0,+1} x {-1,0,+1}
            alpha = a // 3 - 1  (pointer direction)
            beta  = a % 3 - 1   (scale change)

        Scale update (Eq 6.7):
            l^(k+1) = clip(l^(k) + beta, 0, L_scales - 1)

        Pointer update (Eq 6.8):
            p^(k+1) = clip(p^(k) + alpha * 2^{l^(k)}, 0, L - 1)

        Args:
            pointers:     (B, L, h) int64, current positions.
            scale_levels: (B, L, h) int64, current scales.
            actions:      (B, L, h) int64, selected actions in [0, |A|-1].
            L: Sequence length.

        Returns:
            new_pointers:     (B, L, h) int64.
            new_scale_levels: (B, L, h) int64.
        """
        # Decode actions → (alpha, beta)
        alpha = actions // 3 - 1  # {-1, 0, +1}: pointer direction
        beta = actions % 3 - 1  # {-1, 0, +1}: scale change

        # Scale update (Eq 6.7)
        new_scale_levels = (scale_levels + beta).clamp(0, self.config.L_scales - 1)

        # Pointer update (Eq 6.8): step size = 2^{l^(k)}
        step_size = (2 ** scale_levels.float()).long()  # Use CURRENT scale for step
        new_pointers = (pointers + alpha * step_size).clamp(0, L - 1)

        return new_pointers, new_scale_levels

    # ── Tier-2 Explicit EFE (§6.7, Eq 6.18-6.27) ───────────────────────

    def compute_efe_tier2(
        self,
        substrate: torch.Tensor,
        pointers: torch.Tensor,
        scale_levels: torch.Tensor,
        z_t_emb: torch.Tensor,
        pos_encodings: torch.Tensor,
        t_enc: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        kernel: nn.Module,
        readout_head: nn.Module,
        perception_fn,
        dilation: int,
        config: AENCAv4Config,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Tier-2 explicit EFE for a random subset of masked positions.

        For each position i in subset S, for each action a in A:
            1. Compute candidate (p_cand, l_cand) after action a
            2. Predict observation via g_theta
            3. Simulate one-step state update with predicted observation
            4. Risk(a) = CE(x_i*, softmax(W_out * z_i^sim))     (Eq 6.24)
            5. InfoGain(a) = H[current] - H[simulated]          (Eq 6.26)
            6. Cost(a) = |alpha| * l / (L_scales - 1)           (Eq 6.27)
            7. G(a) = Risk - beta_efe * InfoGain + kappa * Cost  (Eq 6.21)

        Args:
            substrate:     (B, L, D) current substrate state.
            pointers:      (B, L, h) current pointer positions.
            scale_levels:  (B, L, h) current scale levels.
            z_t_emb:       (B, L, d_emb) masked token embeddings.
            pos_encodings: (B, L, d_pos) position encodings.
            t_enc:         (B, d_t) timestep encoding.
            targets:       (B, L) true token IDs.
            mask:          (B, L) bool, True at masked positions.
            kernel:        NCAKernel module.
            readout_head:  nn.Linear(d_z, vocab_size) readout.
            perception_fn: Callable(substrate, dilation) → perception.
            dilation:      Current dilation value.
            config:        Model configuration.

        Returns:
            advantages:  (n_subset,) per-position advantages for REINFORCE.
            obs_loss:    scalar, observation model MSE loss.
            value_loss:  scalar, value baseline MSE loss.
        """
        B, L, D = substrate.shape
        h = config.n_sensor_heads
        n_A = config.n_actions
        d_z = config.d_z

        # Select random subset of masked positions (p_efe fraction)
        # Flatten mask to find all masked indices
        masked_indices = mask.nonzero(as_tuple=False)  # (N_masked, 2) [batch_idx, pos_idx]
        n_masked = masked_indices.shape[0]
        n_subset = max(1, int(config.p_efe * n_masked))
        if n_masked == 0:
            # No masked positions — return zero losses
            zero = torch.tensor(0.0, device=substrate.device, requires_grad=True)
            return (
                torch.zeros(0, device=substrate.device),
                zero,
                zero,
            )

        # Random subset selection
        perm = torch.randperm(n_masked, device=substrate.device)[:n_subset]
        subset_indices = masked_indices[perm]  # (n_subset, 2)
        batch_idx = subset_indices[:, 0]  # (n_subset,)
        pos_idx = subset_indices[:, 1]  # (n_subset,)

        # Extract state at subset positions
        h_states = substrate[batch_idx, pos_idx, d_z : d_z + config.d_h]  # (n_subset, d_h)
        z_states = substrate[batch_idx, pos_idx, :d_z]  # (n_subset, d_z)

        # Current logits and entropy at subset positions
        current_logits = readout_head(z_states)  # (n_subset, vocab_size)
        current_probs = F.softmax(current_logits, dim=-1)
        current_entropy = -(current_probs * (current_probs + 1e-10).log()).sum(dim=-1)  # (n_subset,)

        # Target tokens at subset positions
        target_tokens = targets[batch_idx, pos_idx]  # (n_subset,)

        # Current positions and scales for first sensor head (use head 0 for EFE)
        cur_ptr = pointers[batch_idx, pos_idx, :]  # (n_subset, h)
        cur_scale = scale_levels[batch_idx, pos_idx, :]  # (n_subset, h)

        # Compute EFE for each action (9 actions)
        G_values = torch.zeros(n_subset, n_A, device=substrate.device)

        for a_idx in range(n_A):
            alpha = a_idx // 3 - 1  # {-1, 0, +1}
            beta = a_idx % 3 - 1  # {-1, 0, +1}

            # Candidate scale and position (use head 0)
            cand_scale = (cur_scale[:, 0] + beta).clamp(0, config.L_scales - 1)
            step_size = (2 ** cur_scale[:, 0].float()).long()
            cand_ptr = (cur_ptr[:, 0] + alpha * step_size).clamp(0, L - 1)

            # Relative position encoding: rho(i - p_cand)
            rel_pos = (pos_idx - cand_ptr).float()  # (n_subset,)
            rel_pos_enc = self.pos_encoder(rel_pos.long())  # (n_subset, d_pos)

            # Scale encoding: rho(2^l_cand)
            scale_val = (2 ** cand_scale.float()).long()
            scale_enc = self.pos_encoder(scale_val)  # (n_subset, d_pos)

            # Predict observation (Eq 6.14)
            predicted_emb = self.obs_model(h_states.detach(), rel_pos_enc, scale_enc)  # (n_subset, d_emb)

            # --- Simulate one-step update with predicted observation ---
            # Build simulated observation: replace head-0 obs with prediction
            # Actual observation from other heads stays the same
            # For simplicity, construct full obs for one position

            # Get current perception at these positions
            # (This is approximate: we use the full conv output, already computed)
            P_i = perception_fn(substrate, dilation)  # (B, L, D_conv)
            P_subset = P_i[batch_idx, pos_idx]  # (n_subset, D_conv)

            # Build simulated observation (predicted for head 0, actual for others)
            sim_obs_parts = []
            for head_j in range(h):
                if head_j == 0:
                    sim_obs_parts.append(predicted_emb)  # (n_subset, d_emb)
                else:
                    # Actual observation from current pointer
                    actual_ptr = cur_ptr[:, head_j]  # (n_subset,)
                    actual_emb = z_t_emb[batch_idx, actual_ptr]  # (n_subset, d_emb)
                    sim_obs_parts.append(actual_emb)
            sim_obs = torch.cat(sim_obs_parts, dim=-1)  # (n_subset, h*d_emb)

            # Position + timestep encoding
            psi = torch.cat([
                pos_encodings[batch_idx, pos_idx],  # (n_subset, d_pos)
                t_enc[batch_idx],  # (n_subset, d_t)
            ], dim=-1)

            # Kernel input
            x_in_sim = torch.cat([P_subset, sim_obs, psi], dim=-1)  # (n_subset, d_in)
            x_in_sim = x_in_sim.unsqueeze(1)  # (n_subset, 1, d_in) for kernel

            with torch.no_grad():
                delta_s_sim, gate_logits_sim, _ = kernel(x_in_sim)
                gate_sim = torch.sigmoid(gate_logits_sim)
                # Simulated state update
                s_sim = substrate[batch_idx, pos_idx].unsqueeze(1)  # (n_subset, 1, D)
                s_sim = s_sim.float() + (gate_sim * delta_s_sim).float()
                s_sim = s_sim.squeeze(1)  # (n_subset, D)

                # Simulated logits
                z_sim = s_sim[:, :d_z]  # (n_subset, d_z)
                sim_logits = readout_head(z_sim)  # (n_subset, vocab_size)
                sim_probs = F.softmax(sim_logits, dim=-1)
                sim_entropy = -(sim_probs * (sim_probs + 1e-10).log()).sum(dim=-1)

            # Risk (Eq 6.24): CE of simulated prediction vs target
            risk = F.cross_entropy(
                sim_logits, target_tokens, reduction="none"
            )  # (n_subset,)

            # Info gain (Eq 6.26): H[current] - H[simulated]
            info_gain = current_entropy.detach() - sim_entropy  # (n_subset,)

            # Cost (Eq 6.27): |alpha| * l / (L_scales - 1)
            cost = abs(alpha) * cur_scale[:, 0].float() / max(1, config.L_scales - 1)

            # G(a) = Risk - beta * InfoGain + kappa * Cost (Eq 6.21)
            G_values[:, a_idx] = risk - config.beta_efe * info_gain + config.kappa_cost * cost

        # Optimal action under Tier-2: argmin G
        # But we use this for REINFORCE training of Tier-1, not for action selection

        # Per-position denoising loss at final step (used as advantage signal)
        final_ce = F.cross_entropy(
            current_logits.detach(), target_tokens, reduction="none"
        )  # (n_subset,)

        # Value baseline (Eq 6.30)
        value = self.value_head(h_states.detach()).squeeze(-1)  # (n_subset,)

        # Advantage (Eq 6.28): l_i - b_i
        advantages = final_ce - value.detach()  # (n_subset,)

        # Value loss (for training the baseline)
        value_loss = F.mse_loss(value, final_ce.detach())

        # Observation model loss (Eq 6.15): MSE(predicted, actual observation)
        # Compare predicted embedding with actual embedding at candidate position
        # Use the first action's prediction as representative
        # Actually: compute obs_loss for actual next-step pointer positions
        actual_obs_emb = z_t_emb[batch_idx, pointers[batch_idx, pos_idx, 0]]  # (n_subset, d_emb)
        rel_pos_actual = (pos_idx - pointers[batch_idx, pos_idx, 0]).float()
        rel_pos_enc_actual = self.pos_encoder(rel_pos_actual.long())
        scale_enc_actual = self.pos_encoder((2 ** scale_levels[batch_idx, pos_idx, 0].float()).long())
        predicted_actual = self.obs_model(
            h_states.detach(), rel_pos_enc_actual, scale_enc_actual
        )
        obs_loss = F.mse_loss(predicted_actual, actual_obs_emb.detach())

        return advantages, obs_loss, value_loss
