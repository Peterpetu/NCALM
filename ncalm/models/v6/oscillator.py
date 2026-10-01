"""Coupled Kuramoto oscillator network for AENCA V6.

Per-cell oscillator bank controlling all temporal dynamics of the NCA.
Each token position has K oscillator channels (K=64 default) with
state-dependent frequencies, STATE-DEPENDENT inter-channel coupling,
local spatial coupling (depthwise causal Conv1d), and global causal
mean-field coupling.

All oscillator heads use 2-layer MLPs (d_model → d_osc_hidden → output)
providing sufficient representational capacity for the system to
discover critical dynamics (edge of chaos / Kuramoto synchronization
transition) without explicit design.

The coupling matrix C(z) = C_base + C_dynamic(z) has two components:
  - C_base: learned constant K×K matrix (initialization anchor)
  - C_dynamic(z): state-dependent K×K matrix from MLP, enabling the
    system to self-tune coupling strength per-input toward K_c

The oscillators serve three roles:
  1. Temporal gating — phase-derived gates control WHEN each cell updates
  2. Global coordination — mean-field coupling replaces attention as the
     long-range communication mechanism
  3. Slow context extraction — frequency-ranked oscillator filtering
     provides a "slow state" view for the kernel, replacing z_H

All tensors are per-cell: (B, L, K) for phases, (B, L, d_model) for
projected outputs. The model self-regulates the effective number of
timescales via Kuramoto synchronization (oscillators that couple strongly
merge into one effective timescale), learned relevance gates (inactive
oscillators are suppressed), and frequency diversity regularization
(active oscillators are pushed to distinct frequencies).

Reference: Kuramoto Y. (1984) "Chemical Oscillations, Waves, and Turbulence"
Reference: Acebrón et al. (2005) "The Kuramoto model: A simple paradigm
           for synchronization phenomena" Rev. Mod. Phys. 77, 137.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncalm.models.v6.config import V6Config


class CoupledOscillatorNetwork(nn.Module):
    """Per-cell coupled Kuramoto oscillator network.

    Each cell (token position) has K oscillator channels. The oscillators
    evolve according to the Kuramoto model with three coupling terms:

        φ_k^(i)(t+1) = φ_k^(i)(t) + ω_k(z_i)
            + Σ_j [C_base + C_dyn(z_i)]_kj sin(φ_j^(i) - φ_k^(i))
            + κ_local · DepthwiseConv1d(φ_k)_i            [local spatial]
            + κ_global · sin(mean_field_k^[0:i] - φ_k^(i))  [global]

    The coupling matrix combines a learned base C_base (constant) with
    a state-dependent modulation C_dyn(z) from a 2-layer MLP:
        C_total(z) = antisym(C_raw) + antisym(MLP(z).reshape(K,K))

    This enables per-input coupling adaptation — the system can learn to
    sit near the Kuramoto critical coupling K_c = 2/(πg(0)) where g(ω)
    is the frequency distribution, without hand-designing criticality.

    All heads use 2-layer MLPs: input → Linear(in, d_osc_hidden) → SiLU
    → Linear(d_osc_hidden, out), providing ~15% of total model parameters.

    From the oscillator state, the module produces:
        - osc_gate:     (B, L, d_model) temporal update gate for the kernel
        - readout_prob: (B, L) per-position probability of contributing to output
        - slow_ctx:     (B, L, d_model) slow-frequency-filtered context for kernel

    Args:
        config: V6Config with oscillator parameters
    """

    def __init__(self, config: V6Config) -> None:
        super().__init__()
        K = config.n_osc_channels
        D = config.d_model
        H = config.d_osc_hidden
        self.K = K
        self.d_model = D
        self.d_osc_hidden = H
        self.enable_spatial = config.enable_spatial_coupling
        self.enable_mean_field = config.enable_mean_field

        # ── Frequency Head (2-layer MLP) ───────────────────────────────
        # z → RMSNorm → Linear(D,H) → SiLU → Linear(H,K) → softplus
        # State-dependent natural frequencies per cell per channel
        self.freq_norm = nn.RMSNorm(D)
        self.freq_up = nn.Linear(D, H, bias=True)
        self.freq_down = nn.Linear(H, K, bias=True)

        # ── State-Dependent Coupling Head (2-layer MLP) ────────────────
        # z → RMSNorm → Linear(D,H) → SiLU → Linear(H,K²) → reshape
        # → antisymmetrize → C_dynamic(z) ∈ ℝ^{K×K}
        # Enables per-input coupling modulation toward criticality
        self.coupling_norm = nn.RMSNorm(D)
        self.coupling_up = nn.Linear(D, H, bias=True)
        self.coupling_down = nn.Linear(H, K * K, bias=True)

        # ── Base Coupling Matrix (learned constant) ────────────────────
        # C_raw ∈ ℝ^{K×K}, antisymmetrized to (C - C^T)/2 at forward time
        # Provides a stable initialization anchor; C_dynamic adds adaptation
        self.C_raw = nn.Parameter(torch.empty(K, K))

        # ── Local Spatial Coupling ─────────────────────────────────────
        # Depthwise causal Conv1d: channel k at position i couples with
        # channel k at position i−1. kernel_size=2, left-padded.
        if self.enable_spatial:
            self.spatial_conv = nn.Conv1d(
                in_channels=K, out_channels=K, kernel_size=2,
                padding=0, groups=K, bias=False,
            )
            self.kappa_local = nn.Parameter(torch.tensor(config.spatial_coupling_init))

        # ── Global Mean-Field Coupling ─────────────────────────────────
        # κ_global · sin(mean_field - φ), where mean_field is causal cumsum
        # Initialize at 0.5 (not 0.1) so global communication is active
        # from the start. This is the ONLY non-local mechanism in V6 —
        # it must be strong enough to provide useful cross-position signal
        # before the model can learn to strengthen it.
        if self.enable_mean_field:
            self.kappa_global = nn.Parameter(torch.tensor(0.5))

        # ── Phase-to-Gate Parameters ───────────────────────────────────
        # Per-channel: gate_k = σ(α_k · cos(φ_k) + β_k), learned per channel
        self.alpha = nn.Parameter(torch.empty(K))  # amplitude
        self.beta = nn.Parameter(torch.empty(K))   # bias

        # ── Relevance Head (2-layer MLP) ───────────────────────────────
        # z → RMSNorm → Linear(D,H) → SiLU → Linear(H,K) → sigmoid
        # Per-position per-channel: self-regulates effective K
        self.relevance_norm = nn.RMSNorm(D)
        self.relevance_up = nn.Linear(D, H, bias=True)
        self.relevance_down = nn.Linear(H, K, bias=True)

        # ── Phase Gate Projection (2-layer MLP: K → D) ────────────────
        # Maps K oscillator activations → d_model temporal gate
        # (relevance ⊙ channel_gates) → Linear(K,H) → SiLU → Linear(H,D) → σ
        self.gate_up = nn.Linear(K, H, bias=True)
        self.gate_down = nn.Linear(H, D, bias=True)

        # ── Readout Head (2-layer MLP) ─────────────────────────────────
        # Per-position readout probability: when to contribute to output
        # z → RMSNorm → Linear(D,H) → SiLU → Linear(H,1) → σ
        self.readout_norm = nn.RMSNorm(D)
        self.readout_up = nn.Linear(D, H, bias=True)
        self.readout_down = nn.Linear(H, 1, bias=True)

        # ── Slow Context Extraction (2-layer MLP) ──────────────────────
        # Frequency-filtered view of z, replacing z_H as kernel context
        # Uses the K oscillator frequencies to weight slow/fast contributions
        # z → RMSNorm → Linear(D,H) → SiLU → Linear(H,D), gated by slowness
        self.slow_ctx_norm = nn.RMSNorm(D)
        self.slow_ctx_up = nn.Linear(D, H, bias=True)
        self.slow_ctx_down = nn.Linear(H, D, bias=False)
        # Learned threshold for slow/fast frequency boundary
        self.slow_threshold = nn.Parameter(torch.tensor(0.0))

        # Initialize all parameters
        self._init_weights(config)

    def _init_weights(self, config: V6Config) -> None:
        """Careful initialization for stable oscillator dynamics.

        Key principle: zero-init on "output" projections (coupling_down,
        gate_down, slow_ctx_down) so the oscillator starts near-identity.
        C_dynamic starts at zero → C_raw dominates initially, then the
        system gradually learns input-dependent coupling modulation.
        """
        K = self.K

        # ── Frequency MLP ───────────────────────────────────────────
        # Small weights so initial frequencies are driven by bias
        nn.init.xavier_normal_(self.freq_up.weight, gain=0.1)
        nn.init.zeros_(self.freq_up.bias)
        nn.init.xavier_normal_(self.freq_down.weight, gain=0.1)
        # Bias: logarithmically spaced from base_freq to base_freq * 2^spread
        # This gives each oscillator a distinct initial frequency.
        # With freq clamping at π and spread=2.5, max initial ω ≈ 2.83 < π.
        with torch.no_grad():
            base = config.osc_base_freq
            spread = config.osc_freq_spread
            if K > 1:
                log_freqs = torch.linspace(0, spread, K)
            else:
                log_freqs = torch.zeros(1)
            target_freqs = base * (2.0 ** log_freqs)
            # softplus inverse: if y = ln(1 + e^x), then x = ln(e^y - 1)
            bias_init = torch.log(torch.expm1(target_freqs.clamp(min=1e-4)))
            self.freq_down.bias.copy_(bias_init)

        # ── State-dependent coupling MLP ────────────────────────────
        # Small up-weights, ZERO down-weights → C_dynamic starts at zero
        # C_raw dominates initially; model learns coupling adaptation
        nn.init.xavier_normal_(self.coupling_up.weight, gain=0.1)
        nn.init.zeros_(self.coupling_up.bias)
        nn.init.zeros_(self.coupling_down.weight)
        nn.init.zeros_(self.coupling_down.bias)

        # ── Base coupling C_raw ─────────────────────────────────────
        # Uniform random ± coupling_init
        nn.init.uniform_(
            self.C_raw,
            -config.osc_coupling_init,
            config.osc_coupling_init,
        )

        # ── Spatial coupling ────────────────────────────────────────
        if self.enable_spatial:
            nn.init.uniform_(
                self.spatial_conv.weight,
                -config.spatial_coupling_init,
                config.spatial_coupling_init,
            )

        # ── Phase-to-gate ───────────────────────────────────────────
        # α ≈ 1.0 (cos amplitude), β ≈ 0 (unbiased initially)
        nn.init.ones_(self.alpha)
        nn.init.zeros_(self.beta)

        # ── Relevance MLP ───────────────────────────────────────────
        # Small weights, positive output bias so all channels start active
        # σ(1.5) ≈ 0.82 — most channels active initially
        nn.init.xavier_normal_(self.relevance_up.weight, gain=0.1)
        nn.init.zeros_(self.relevance_up.bias)
        nn.init.xavier_normal_(self.relevance_down.weight, gain=0.1)
        nn.init.constant_(self.relevance_down.bias, 1.5)

        # ── Gate MLP: zero-init weights, positive bias → gates start ≈ σ(2) ≈ 0.88
        # The oscillator gate multiplies the kernel output. Starting near 1.0
        # means the oscillator initially "gets out of the way" and passes kernel
        # updates through almost fully.  The old σ(0)=0.5 init halved all
        # kernel updates from step 1, severely limiting early learning.
        # As training progresses, the oscillator learns per-step modulation.
        nn.init.xavier_normal_(self.gate_up.weight, gain=0.1)
        nn.init.zeros_(self.gate_up.bias)
        nn.init.zeros_(self.gate_down.weight)
        nn.init.constant_(self.gate_down.bias, 2.0)

        # ── Readout MLP ────────────────────────────────────────────
        # Bias so initial readout ≈ 2/T_max (aggressive accumulation)
        # With the old 1/T_max init, readout_prob ≈ 1/16 = 0.0625, and after
        # 16 steps only 64% of mass was distributed during the loop (36%
        # absorbed at the final step). This created near-uniform weighting
        # across steps → no gradient signal to learn WHEN to readout.
        # With 2/T_max ≈ 0.125, after 16 steps 87% of mass is distributed
        # during the loop, creating differentiated per-step gradients.
        nn.init.xavier_normal_(self.readout_up.weight, gain=0.1)
        nn.init.zeros_(self.readout_up.bias)
        nn.init.xavier_normal_(self.readout_down.weight, gain=0.1)
        t_max = config.t_max
        # σ(b) ≈ 2/T_max → b ≈ ln(2/(T_max - 2))
        readout_bias = -math.log(max(t_max / 2 - 1, 1))
        nn.init.constant_(self.readout_down.bias, readout_bias)

        # ── Slow context MLP ───────────────────────────────────────
        # Zero-init output → slow_ctx starts as zero (kernel starts without it)
        nn.init.xavier_normal_(self.slow_ctx_up.weight, gain=0.1)
        nn.init.zeros_(self.slow_ctx_up.bias)
        nn.init.zeros_(self.slow_ctx_down.weight)

    def init_phase(
        self, B: int, L: int, device: torch.device, deterministic: bool = False
    ) -> torch.Tensor:
        """Initialize oscillator phases.

        Args:
            B: batch size
            L: sequence length
            device: torch device
            deterministic: if True, use evenly-spaced phases (inference)

        Returns:
            phi: (B, L, K) initial phases in [0, 2π)
        """
        if deterministic:
            # Evenly spaced across channels, same per position/batch
            phi = torch.linspace(0, 2 * math.pi * (1 - 1 / self.K), self.K,
                                 device=device)
            phi = phi.unsqueeze(0).unsqueeze(0).expand(B, L, -1)
        else:
            # Uniform random initialization per cell
            phi = torch.rand(B, L, self.K, device=device) * (2 * math.pi)
        return phi

    def _base_coupling(self) -> torch.Tensor:
        """Compute antisymmetric base coupling matrix: C = (C_raw - C_raw^T) / 2.

        This is the learned constant component of coupling. It provides
        stable dynamics and serves as an initialization anchor while
        C_dynamic(z) learns input-dependent modulation.

        Returns:
            C: (K, K) antisymmetric base coupling matrix
        """
        return (self.C_raw - self.C_raw.T) / 2

    def _dynamic_coupling(self, z: torch.Tensor) -> torch.Tensor:
        """Compute state-dependent coupling modulation from cell state.

        The MLP produces a per-cell K×K matrix from z, then antisymmetrizes.
        This enables the system to modulate coupling strength per-input,
        potentially self-tuning toward the Kuramoto critical coupling K_c.

        Args:
            z: (B, L, d_model) current cell states

        Returns:
            C_dyn: (B, L, K, K) antisymmetric coupling modulation
        """
        h = self.coupling_norm(z)                              # (B, L, D)
        h = F.silu(self.coupling_up(h))                        # (B, L, H)
        C_flat = self.coupling_down(h)                         # (B, L, K²)
        # Bound dynamic coupling with tanh to prevent unbounded growth.
        # After a gradient perturbation, unbounded C_dyn can spiral,
        # causing phase chaos → gate collapse → readout collapse.
        # tanh keeps each element in [-1, 1], combined with 1/K
        # normalization this ensures well-behaved Kuramoto dynamics.
        C_flat = torch.tanh(C_flat)
        C_dyn = C_flat.view(*z.shape[:2], self.K, self.K)      # (B, L, K, K)
        # Antisymmetrize: C_kj = -C_jk
        C_dyn = (C_dyn - C_dyn.transpose(-1, -2)) / 2
        return C_dyn

    def _compute_frequencies(self, z: torch.Tensor) -> torch.Tensor:
        """Compute state-dependent natural frequencies per cell per channel.

        2-layer MLP: z → RMSNorm → Linear → SiLU → Linear → softplus

        Args:
            z: (B, L, d_model) current cell states

        Returns:
            omega: (B, L, K) natural frequencies (positive, via softplus)
        """
        h = self.freq_norm(z)                        # (B, L, D)
        h = F.silu(self.freq_up(h))                  # (B, L, H)
        omega = F.softplus(self.freq_down(h))         # (B, L, K), positive
        # Clamp frequencies below Nyquist limit (π rad/step).
        # In discrete-time phase updates φ_{t+1} = (φ_t + ω) mod 2π,
        # frequencies above π cause aliasing: the effective phase
        # velocity wraps, making dynamics unpredictable.
        # This also prevents the freq_diversity_loss from pushing
        # frequencies to arbitrarily large values (reward hacking).
        omega = omega.clamp(max=math.pi)
        return omega

    def _compute_relevance(self, z: torch.Tensor) -> torch.Tensor:
        """Compute per-channel relevance gates from cell state.

        2-layer MLP: z → RMSNorm → Linear → SiLU → Linear → sigmoid

        Args:
            z: (B, L, d_model) current cell states

        Returns:
            relevance: (B, L, K) per-channel relevance ∈ (0, 1)
        """
        h = self.relevance_norm(z)                   # (B, L, D)
        h = F.silu(self.relevance_up(h))             # (B, L, H)
        return torch.sigmoid(self.relevance_down(h))  # (B, L, K)

    def _inter_channel_coupling(
        self, phi: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Kuramoto inter-channel coupling with state-dependent modulation.

        For oscillator k at position i:
            C_total = C_base + C_dynamic(z_i)
            Δφ_k = Σ_j C_total_kj sin(φ_j - φ_k)

        The combined coupling enables per-input adaptation of coupling
        strength, allowing the system to approach criticality dynamically.

        Args:
            phi: (B, L, K) current phases
            z:   (B, L, d_model) current cell states

        Returns:
            coupling: (B, L, K) phase adjustments from inter-channel coupling
        """
        C_base = self._base_coupling()               # (K, K)
        C_dyn = self._dynamic_coupling(z)            # (B, L, K, K)

        # Combined coupling: broadcast C_base to (B, L, K, K) + C_dyn
        C_total = C_base.unsqueeze(0).unsqueeze(0) + C_dyn  # (B, L, K, K)

        # Phase differences: sin(φ_j - φ_k) for all pairs
        phi_j = phi.unsqueeze(-1)                    # (B, L, K, 1)
        phi_k = phi.unsqueeze(-2)                    # (B, L, 1, K)
        sin_diff = torch.sin(phi_j - phi_k)          # (B, L, K, K)
        # sin_diff[..., j, k] = sin(φ_j - φ_k)

        # Contract: (1/K) Σ_j C_total[k,j] * sin(φ_j - φ_k)
        # The 1/K factor is the standard Kuramoto mean-field normalization:
        # it makes coupling strength independent of the number of oscillators K,
        # preventing coupling force from scaling linearly with K.
        coupling = torch.einsum("blkj,bljk->blk", C_total, sin_diff) / self.K

        return coupling

    def _local_spatial_coupling(self, phi: torch.Tensor) -> torch.Tensor:
        """Local spatial coupling via depthwise causal Conv1d.

        Channel k at position i couples with channel k at position i-1.
        Causal: left-padded convolution with kernel_size=2.

        Args:
            phi: (B, L, K) current phases

        Returns:
            coupling: (B, L, K) phase adjustments from spatial coupling
        """
        # Conv1d expects (B, C, L) — reshape phases
        phi_t = phi.transpose(1, 2)              # (B, K, L)
        phi_padded = F.pad(phi_t, (1, 0))        # (B, K, L+1) left-pad
        spatial_out = self.spatial_conv(phi_padded)  # (B, K, L)
        spatial_out = spatial_out.transpose(1, 2)    # (B, L, K)

        return self.kappa_local * spatial_out

    def _global_mean_field_coupling(self, phi: torch.Tensor) -> torch.Tensor:
        """Global causal mean-field coupling.

        For oscillator channel k at position i:
            mean_field_k[i] = circular_mean(φ_k[0..i])
            Δφ_k[i] = κ_global · sin(mean_field_k[i] - φ_k[i])

        Computed efficiently via cumsum with sin/cos decomposition for
        proper circular mean (atan2). This is the global coordination
        mechanism that replaces attention.

        Args:
            phi: (B, L, K) current phases

        Returns:
            coupling: (B, L, K) phase adjustments from global mean-field
        """
        # Causal cumulative mean: mean of φ_k[0..i] for each position i
        # Use sin/cos decomposition for proper circular mean
        sin_phi = torch.sin(phi)                            # (B, L, K)
        cos_phi = torch.cos(phi)                            # (B, L, K)
        cum_sin = torch.cumsum(sin_phi, dim=1)              # (B, L, K)
        cum_cos = torch.cumsum(cos_phi, dim=1)              # (B, L, K)
        L = phi.shape[1]
        counts = torch.arange(1, L + 1, device=phi.device, dtype=phi.dtype)
        counts = counts.unsqueeze(0).unsqueeze(-1)          # (1, L, 1)
        mean_sin = cum_sin / counts                         # (B, L, K)
        mean_cos = cum_cos / counts                         # (B, L, K)

        # Circular mean phase
        mean_phase = torch.atan2(mean_sin, mean_cos)        # (B, L, K)

        # Coupling: pull toward mean field
        coupling = self.kappa_global * torch.sin(mean_phase - phi)
        return coupling

    def _phase_to_gates(
        self, phi: torch.Tensor, relevance: torch.Tensor
    ) -> torch.Tensor:
        """Convert oscillator phases to d_model-dimensional update gate.

        Steps:
            1. Per-channel gate: σ(α_k · cos(φ_k) + β_k) → (B, L, K)
            2. Modulate by relevance: gate_k * relevance_k → (B, L, K)
            3. 2-layer MLP: σ(Linear(SiLU(Linear(active)))) → (B, L, D)

        Args:
            phi: (B, L, K) current phases
            relevance: (B, L, K) per-channel relevance ∈ (0, 1)

        Returns:
            osc_gate: (B, L, d_model) temporal update gate ∈ (0, 1)
        """
        # Per-channel gates from phase
        channel_gates = torch.sigmoid(
            self.alpha.unsqueeze(0).unsqueeze(0) * torch.cos(phi) +
            self.beta.unsqueeze(0).unsqueeze(0)
        )  # (B, L, K)

        # Modulate by relevance
        active_gates = relevance * channel_gates      # (B, L, K)

        # 2-layer MLP projection to d_model
        h = F.silu(self.gate_up(active_gates))        # (B, L, H)
        osc_gate = torch.sigmoid(self.gate_down(h))   # (B, L, D) ∈ (0, 1)
        return osc_gate

    def _compute_readout(self, z: torch.Tensor) -> torch.Tensor:
        """Compute per-position readout probability.

        2-layer MLP: z → RMSNorm → Linear → SiLU → Linear → sigmoid

        Args:
            z: (B, L, d_model) current cell states

        Returns:
            readout_prob: (B, L) per-position readout weight ∈ (0, 1)
        """
        h = self.readout_norm(z)                      # (B, L, D)
        h = F.silu(self.readout_up(h))                # (B, L, H)
        readout = torch.sigmoid(self.readout_down(h))  # (B, L, 1)
        return readout.squeeze(-1)                     # (B, L)

    def _extract_slow_context(
        self, z: torch.Tensor, omega: torch.Tensor, relevance: torch.Tensor
    ) -> torch.Tensor:
        """Extract slow-frequency-filtered context for the kernel.

        Replaces z_H: provides a frequency-filtered view of z based on
        which oscillator channels are running slowly.

        Method:
            1. Compute per-channel "slowness": σ(threshold - log(freq))
            2. Weight by relevance (only active channels contribute)
            3. Aggregate mean slowness per position → scalar gate
            4. Apply to 2-layer MLP projection of z

        Args:
            z: (B, L, d_model) current cell states
            omega: (B, L, K) current frequencies
            relevance: (B, L, K) per-channel relevance

        Returns:
            slow_ctx: (B, L, d_model) slow-frequency filtered context
        """
        # Normalize frequencies to [0, 1] range for stable gating
        log_omega = torch.log(omega + 1e-6)                  # (B, L, K)

        # Per-channel slowness: σ(threshold - log_freq) — lower freq → higher weight
        slowness = torch.sigmoid(
            self.slow_threshold - log_omega
        )  # (B, L, K)

        # Weight by relevance (only active oscillators contribute)
        slow_weights = slowness * relevance                   # (B, L, K)

        # Aggregate weight per position: mean over K channels
        slow_scale = slow_weights.mean(dim=-1, keepdim=True)  # (B, L, 1)

        # 2-layer MLP projection
        h = self.slow_ctx_norm(z)                             # (B, L, D)
        h = F.silu(self.slow_ctx_up(h))                       # (B, L, H)
        slow_ctx = slow_scale * self.slow_ctx_down(h)         # (B, L, D)

        return slow_ctx

    def step(
        self,
        z: torch.Tensor,
        phi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single oscillator tick: update phases and produce outputs.

        This is the core method called once per NCA step. It:
        1. Computes state-dependent frequencies and relevance
        2. Advances phases via Kuramoto dynamics (3 coupling terms)
           with state-dependent coupling C(z)
        3. Produces oscillator-derived gates, readout weight, slow context

        Args:
            z:   (B, L, d_model) current cell states
            phi: (B, L, K) current oscillator phases

        Returns:
            phi_new:      (B, L, K) updated phases
            osc_gate:     (B, L, d_model) temporal update gate ∈ (0, 1)
            readout_prob: (B, L) readout accumulation weight ∈ (0, 1)
            slow_ctx:     (B, L, d_model) slow-frequency-filtered context
        """
        # ── 1. State-dependent frequencies and relevance ─────────────
        omega = self._compute_frequencies(z)      # (B, L, K)
        relevance = self._compute_relevance(z)    # (B, L, K)

        # ── 2. Kuramoto phase update ─────────────────────────────────
        # Natural frequency term
        d_phi = omega                              # (B, L, K)

        # Inter-channel coupling with state-dependent C(z)
        d_phi = d_phi + self._inter_channel_coupling(phi, z)

        # Local spatial coupling (optional)
        if self.enable_spatial:
            d_phi = d_phi + self._local_spatial_coupling(phi)

        # Global mean-field coupling (optional)
        if self.enable_mean_field:
            d_phi = d_phi + self._global_mean_field_coupling(phi)

        # Update phases (mod 2π to prevent unbounded growth)
        phi_new = (phi + d_phi) % (2 * math.pi)

        # ── 3. Produce outputs from updated phases ───────────────────
        osc_gate = self._phase_to_gates(phi_new, relevance)     # (B, L, D)
        readout_prob = self._compute_readout(z)                  # (B, L)
        slow_ctx = self._extract_slow_context(z, omega, relevance)  # (B, L, D)

        return phi_new, osc_gate, readout_prob, slow_ctx

    def freq_diversity_loss(
        self, omega: torch.Tensor, relevance: torch.Tensor
    ) -> torch.Tensor:
        """Frequency diversity regularization.

        Penalizes active oscillators for having similar frequencies.
        Computed as the negative weighted standard deviation of frequencies:
            L_freq = -mean_{B,L}( weighted_std(ω, weights=relevance) )

        This encourages the model to spread active oscillators across
        distinct frequencies, preventing K channels from collapsing to
        a single effective timescale.

        Args:
            omega: (B, L, K) natural frequencies
            relevance: (B, L, K) per-channel relevance ∈ (0, 1)

        Returns:
            loss: scalar, negative (minimizing = maximizing diversity)
        """
        # Weighted mean: Σ(w_k * ω_k) / Σ(w_k)
        w_sum = relevance.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, L, 1)
        w_mean = (relevance * omega).sum(dim=-1, keepdim=True) / w_sum  # (B, L, 1)

        # Weighted variance: Σ(w_k * (ω_k - mean)^2) / Σ(w_k)
        w_var = (relevance * (omega - w_mean).pow(2)).sum(dim=-1) / w_sum.squeeze(-1)
        w_std = (w_var + 1e-8).sqrt()  # (B, L)

        # Saturating frequency diversity reward: -log(1 + std)
        # Using log1p instead of raw -std prevents reward hacking:
        # with unbounded -std, the optimizer can push frequencies apart
        # indefinitely to reduce total loss (freq_div grew to -1053 in
        # the first training run, contributing more loss reduction than
        # actual CE improvement). log1p saturates: O(log(std)) growth
        # means diminishing returns for extreme frequency spread.
        # Combined with frequency clamping (ω < π), max diversity
        # reward ≈ -log(1 + π/2) ≈ -0.94 per sample — negligible.
        return -torch.log1p(w_std).mean()

    def compute_freq_diversity(
        self, z: torch.Tensor
    ) -> torch.Tensor:
        """Convenience: compute freq diversity loss from state z.

        Args:
            z: (B, L, d_model) current cell states

        Returns:
            loss: scalar frequency diversity loss
        """
        omega = self._compute_frequencies(z)
        relevance = self._compute_relevance(z)
        return self.freq_diversity_loss(omega, relevance)

    def effective_K(self, z: torch.Tensor, threshold: float = 0.5) -> float:
        """Compute mean active oscillator channels per position.

        Monitoring metric: how many of the K channels are "active"
        (relevance > threshold) on average.

        Args:
            z: (B, L, d_model) current cell states
            threshold: relevance threshold for "active"

        Returns:
            mean_active: mean number of active channels per position
        """
        with torch.no_grad():
            relevance = self._compute_relevance(z)
            active = (relevance > threshold).float()
            return active.sum(dim=-1).mean().item()

    def get_diagnostics(self, z: torch.Tensor, phi: torch.Tensor) -> dict[str, float]:
        """Compute diagnostic metrics for logging.

        Args:
            z: (B, L, d_model)
            phi: (B, L, K)

        Returns:
            dict with oscillator diagnostics
        """
        with torch.no_grad():
            omega = self._compute_frequencies(z)
            relevance = self._compute_relevance(z)
            C_base = self._base_coupling()
            C_dyn = self._dynamic_coupling(z)

            return {
                "effective_K": self.effective_K(z),
                "freq_std": omega.std(dim=-1).mean().item(),
                "freq_mean": omega.mean().item(),
                "coupling_base_norm": C_base.norm().item(),
                "coupling_dynamic_norm": C_dyn.norm(dim=(-1, -2)).mean().item(),
                "coupling_dynamic_max": C_dyn.abs().max().item(),
                "relevance_mean": relevance.mean().item(),
                "kappa_local": (
                    self.kappa_local.item() if self.enable_spatial else 0.0
                ),
                "kappa_global": (
                    self.kappa_global.item() if self.enable_mean_field else 0.0
                ),
            }
