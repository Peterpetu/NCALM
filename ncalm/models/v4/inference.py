"""Inference procedures for AENCA V4.

Three generation modes:
1. Autoregressive (Algorithm 2, §9.1): one token at a time, left-to-right.
   Slowest but highest quality — uses the diffusion model as an AR generator.

2. Semi-autoregressive / ReMDM (Algorithm 3, §9.2): parallel decoding with
   iterative remasking. Multiple tokens decoded per diffusion pass.
   Good balance of quality and speed.

3. Anytime (Algorithm 4, §9.3): adaptive compute — runs NCA steps until
   substrate converges or predictions are confident. Can stop early for
   easy tokens, spend more compute on hard ones.

Reference: model_documents/aenca_v4_architecture.md §9.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ncalm.models.v4.config import AENCAv4Config
from ncalm.models.v4.diffusion import cosine_masking_schedule


@torch.no_grad()
def autoregressive_generate(
    model,
    prefix: torch.Tensor,
    max_tokens: int,
    temperature: float = 0.8,
    top_k: int = 50,
) -> torch.Tensor:
    """Generate tokens one at a time, left to right (Algorithm 2, §9.1).

    For each new token position n+1:
    1. z_t = [x_0, ..., x_n, MASK]  (prefix unmasked, target masked)
    2. Run K NCA steps with t near 0 (almost fully revealed context)
    3. logits = W_out * z_{n+1}^(K)
    4. Sample x_{n+1} ~ top-k softmax(logits / temperature)

    Args:
        model: AENCAv4Model instance (eval mode).
        prefix: (1, P) token IDs — the prompt/prefix.
        max_tokens: Number of new tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k filtering (0 = no filtering).

    Returns:
        generated: (1, P + max_tokens) complete sequence.
    """
    model.eval()
    config = model.config
    device = prefix.device
    B = 1  # Generation is batch-1

    seq = prefix.clone()  # (1, P)

    for _ in range(max_tokens):
        cur_len = seq.shape[1]
        if cur_len >= config.seq_len:
            # Truncate to last seq_len-1 tokens + 1 masked position
            seq = seq[:, -(config.seq_len - 1) :]
            cur_len = seq.shape[1]

        # Construct input: prefix tokens + one MASK token
        z_t = torch.cat([
            seq,
            torch.full((1, 1), config.mask_token_id, dtype=torch.long, device=device),
        ], dim=1)  # (1, cur_len+1)

        L = z_t.shape[1]
        mask = torch.zeros(1, L, dtype=torch.bool, device=device)
        mask[0, -1] = True  # Only last position is masked

        # Set t near 0 (almost no masking — strong context)
        t = torch.full((1,), config.masking_eps, device=device)

        # Initialize substrate
        s0, z_t_emb = model.substrate.initializer(z_t, mask)

        # Initialize sensors
        pointers, scale_levels = model.substrate.sensing.init_sensors(B, L, device)

        # Pre-compute encodings
        positions = torch.arange(L, device=device)
        pos_enc = model.substrate.pos_encoder(positions).unsqueeze(0)  # (1, L, d_pos)
        t_enc = model.substrate.timestep_encoder(t)  # (1, d_t)

        # Run K NCA steps
        substrate = s0
        for k in range(config.K):
            substrate, pointers, scale_levels, _, _, _, _ = model.substrate.nca_step(
                substrate, k, t_enc, z_t_emb, pointers, scale_levels,
                pos_enc, config.tau_pi,
            )

        # Readout at masked position (last)
        logits = model.substrate.get_readout_logits(substrate)  # (1, L, |V|)
        next_logits = logits[0, -1]  # (|V|,)

        # Temperature scaling
        next_logits = next_logits / temperature

        # Top-k filtering
        if top_k > 0:
            topk_vals, topk_idx = torch.topk(next_logits, top_k)
            next_logits = torch.full_like(next_logits, float("-inf"))
            next_logits.scatter_(0, topk_idx, topk_vals)

        # Sample
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)  # (1,)

        seq = torch.cat([seq, next_token.unsqueeze(0)], dim=1)

    return seq


@torch.no_grad()
def semi_autoregressive_generate(
    model,
    prefix: torch.Tensor,
    n_tokens: int,
    T_diff: int = 10,
    tau_conf: float = 0.5,
    temperature: float = 0.8,
) -> torch.Tensor:
    """Parallel decoding with ReMDM remasking (Algorithm 3, §9.2).

    1. Initialize: prefix unmasked, n_tokens target positions masked
    2. For each diffusion step s = T_diff, T_diff-1, ..., 1:
        a. Denoise: K NCA steps → logits at all masked positions
        b. Decode: argmax at all masked
        c. Compute confidence: c_i = max_v p_theta(x_i = v | z_t)
        d. Determine target masking rate gamma(s/T_diff)
        e. Remask lowest-confidence predictions to match target rate
    3. Final pass: one more K-step decode, argmax

    ReMDM remasking (Eq 3.16-3.18):
        sigma_{t->s} = min(1, gamma(s) / (1 - gamma(t) + gamma(s)))

    Args:
        model: AENCAv4Model instance (eval mode).
        prefix: (1, P) token IDs — the prompt.
        n_tokens: Number of new tokens to generate.
        T_diff: Number of diffusion steps (more = higher quality).
        tau_conf: Confidence threshold for early stopping.
        temperature: Sampling temperature for final decode.

    Returns:
        generated: (1, P + n_tokens) complete sequence.
    """
    model.eval()
    config = model.config
    device = prefix.device
    B = 1

    P = prefix.shape[1]
    L = min(P + n_tokens, config.seq_len)
    n_gen = L - P

    if n_gen <= 0:
        return prefix

    # Construct initial sequence: prefix + MASK tokens
    z_t = torch.cat([
        prefix,
        torch.full((1, n_gen), config.mask_token_id, dtype=torch.long, device=device),
    ], dim=1)  # (1, L)

    # Mask tracks which positions are still masked (only generation positions)
    is_masked = torch.zeros(1, L, dtype=torch.bool, device=device)
    is_masked[0, P:] = True
    n_to_unmask_total = n_gen

    # Pre-compute encodings (fixed across diffusion steps)
    positions = torch.arange(L, device=device)
    pos_enc = model.substrate.pos_encoder(positions).unsqueeze(0)

    for s in range(T_diff, 0, -1):
        # Current and target masking rates
        t_cur = s / T_diff
        t_next = (s - 1) / T_diff

        gamma_cur = cosine_masking_schedule(torch.tensor(t_cur)).item()
        gamma_next = cosine_masking_schedule(torch.tensor(t_next)).item()

        # Target number of masked positions at next step
        n_target_masked = max(0, int(gamma_next * n_gen))
        if s == 1:
            n_target_masked = 0  # Final step: unmask everything

        # Diffusion time for this step
        t = torch.full((1,), t_cur, device=device)
        t_enc = model.substrate.timestep_encoder(t)

        # Initialize substrate from current z_t
        mask_bool = (z_t == config.mask_token_id)
        s0, z_t_emb = model.substrate.initializer(z_t, mask_bool)
        pointers, scale_levels = model.substrate.sensing.init_sensors(B, L, device)

        # Run K NCA steps
        substrate = s0
        for k in range(config.K):
            substrate, pointers, scale_levels, _, _, _, _ = model.substrate.nca_step(
                substrate, k, t_enc, z_t_emb, pointers, scale_levels,
                pos_enc, config.tau_pi,
            )

        # Readout logits
        logits = model.substrate.get_readout_logits(substrate)  # (1, L, |V|)
        probs = F.softmax(logits, dim=-1)  # (1, L, |V|)

        # Decode at all currently masked positions
        predicted_tokens = probs[0].argmax(dim=-1)  # (L,)
        confidence = probs[0].max(dim=-1).values  # (L,)

        # Fill in predictions at masked positions
        z_t_new = z_t.clone()
        z_t_new[0, is_masked[0]] = predicted_tokens[is_masked[0]]

        if n_target_masked > 0 and s > 1:
            # ReMDM: remask the least confident predictions
            # Among generation positions, find which to remask
            gen_mask = is_masked[0].clone()  # Currently masked positions
            gen_confidence = confidence.clone()
            gen_confidence[~gen_mask] = float("inf")  # Don't remask prefix

            # Sort by confidence (ascending) — remask lowest confidence
            _, sorted_idx = gen_confidence.sort()
            remask_idx = sorted_idx[:n_target_masked]

            # Update z_t: unmask confident, remask uncertain
            z_t = z_t_new
            is_masked[0] = False
            is_masked[0, P:] = False  # Start fresh for gen positions
            z_t[0, remask_idx] = config.mask_token_id
            is_masked[0, remask_idx] = True
        else:
            # Final step: accept all predictions
            z_t = z_t_new
            is_masked[:] = False

        # Early stop if all confident
        if not is_masked.any():
            break

    return z_t


@torch.no_grad()
def anytime_inference(
    model,
    z_t: torch.Tensor,
    mask: torch.Tensor,
    eps_conv: float = 1e-3,
    tau_H: float = 1.0,
    K_max: int = 48,
    t_value: float = 0.5,
) -> torch.Tensor:
    """Adaptive compute: stop when converged or confident (Algorithm 4, §9.3).

    Run NCA steps until one of:
    - Substrate state change < eps_conv (converged)
    - Max prediction entropy < tau_H (all predictions confident)
    - k >= K_max (compute budget exhausted)

    This demonstrates the NCA's natural anytime property: unlike transformers,
    we can stop at any step and get a valid (if less refined) prediction.

    Args:
        model: AENCAv4Model instance (eval mode).
        z_t: (1, L) masked token IDs.
        mask: (1, L) bool, True at masked positions.
        eps_conv: Convergence threshold on state delta norm.
        tau_H: Entropy threshold for confidence stopping.
        K_max: Maximum NCA steps.
        t_value: Diffusion time for the denoising pass.

    Returns:
        logits: (1, L, vocab_size) final predictions.
    """
    model.eval()
    config = model.config
    device = z_t.device
    B, L = z_t.shape

    t = torch.full((B,), t_value, device=device)

    # Initialize
    s0, z_t_emb = model.substrate.initializer(z_t, mask)
    pointers, scale_levels = model.substrate.sensing.init_sensors(B, L, device)

    positions = torch.arange(L, device=device)
    pos_enc = model.substrate.pos_encoder(positions).unsqueeze(0).expand(B, -1, -1)
    t_enc = model.substrate.timestep_encoder(t)

    substrate = s0
    steps_taken = 0

    for k in range(K_max):
        substrate_prev = substrate.clone()

        substrate, pointers, scale_levels, _, _, _, _ = model.substrate.nca_step(
            substrate, k % config.M,  # Cycle dilation pattern
            t_enc, z_t_emb, pointers, scale_levels,
            pos_enc, config.tau_pi,
        )
        steps_taken += 1

        # Check convergence: ||s^(k+1) - s^(k)||
        delta = (substrate - substrate_prev).norm() / (B * L * config.D) ** 0.5
        if delta < eps_conv:
            break

        # Check confidence: max entropy at masked positions
        if mask.any():
            logits = model.substrate.get_readout_logits(substrate)
            masked_logits = logits[mask]  # (n_masked, |V|)
            probs = F.softmax(masked_logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
            if entropy.max() < tau_H:
                break

    # Final readout
    logits = model.substrate.get_readout_logits(substrate)
    return logits
