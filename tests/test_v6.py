"""Comprehensive unit tests for AENCA V6: Oscillator-Controlled Adaptive NCA.

Tests cover:
  1. Configuration validation (oscillator params, derived properties)
  2. CoupledOscillatorNetwork (phase update, coupling, gates, readout, slow_ctx)
  3. Self-regulation (relevance gates, freq diversity, effective K)
  4. Global mean-field coupling (causal cumsum, circular mean)
  5. Fast kernel (shape, gating, zero-init)
  6. SwiGLU block (residual connection)
  7. Full model forward/backward pass
  8. Per-position halting (readout accumulation, remaining mass)
  9. Zero-init cascade (model approx identity at initialization)
  10. Gradient engagement (loss decreases with optimization)
  11. Model registration and profile discovery
  12. Deep supervision (aux losses at power-of-2 steps)
  13. Edge cases (minimal step counts, batch=1, deterministic eval)
  14. Oscillator diagnostics
"""

from __future__ import annotations

import math
import sys
import pytest
import torch
import torch.nn as nn

# Ensure project is importable
sys.path.insert(0, ".")

from ncalm.models.v6.config import V6Config
from ncalm.models.v6.kernel import FastKernel, SwiGLUBlock
from ncalm.models.v6.oscillator import CoupledOscillatorNetwork
from ncalm.models.v6.model import AENCAv6Model


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def tiny_config() -> V6Config:
    """Minimal config for fast CPU tests."""
    return V6Config(
        d_model=64, d_msg=16, d_ff=128, d_pe=16, d_slow_ff=128,
        n_heads=4, n_trunk=2, kernel_radius=1,
        n_osc_channels=4, t_max=8, t_min=1,
        readout_halt_eps=0.01,
        osc_base_freq=0.5, osc_freq_spread=2.0,
        osc_coupling_init=0.1,
        enable_spatial_coupling=True, enable_mean_field=True,
        lambda_freq_reg=0.001, checkpoint_segment_size=4,
        deep_supervision_steps=(2, 4, 8),
        vocab_size=200, max_seq_len=32, seq_len=32,
        mos_components=1, tie_embeddings=True,
        lambda_reg=0.01, lambda_aux=0.1, lambda_agc=0.02, dropout=0.0,
    )


@pytest.fixture
def tiny_config_dict(tiny_config: V6Config) -> dict:
    """Config as dict for model constructor."""
    import dataclasses
    return {f.name: getattr(tiny_config, f.name) for f in dataclasses.fields(tiny_config)}


@pytest.fixture
def tiny_model(tiny_config_dict: dict) -> AENCAv6Model:
    """Tiny V6 model for CPU testing."""
    return AENCAv6Model(tiny_config_dict)


@pytest.fixture
def tiny_batch(tiny_config: V6Config) -> torch.Tensor:
    """Random batch of token IDs."""
    return torch.randint(0, tiny_config.vocab_size, (2, tiny_config.seq_len))


@pytest.fixture
def tiny_oscillator(tiny_config: V6Config) -> CoupledOscillatorNetwork:
    """Standalone oscillator for unit testing."""
    return CoupledOscillatorNetwork(tiny_config)


# ═══════════════════════════════════════════════════════════════════════
# 1. Configuration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestV6Config:
    def test_total_steps_equals_t_max(self):
        cfg = V6Config(t_max=64)
        assert cfg.total_steps == 64
        assert cfg.K == 64

    def test_d_kernel_in(self):
        cfg = V6Config(d_model=768, d_pe=64)
        assert cfg.d_kernel_in == 768 + 768 + 64

    def test_d_head(self):
        cfg = V6Config(d_msg=96, n_heads=6)
        assert cfg.d_head == 16

    def test_residual_scale(self):
        cfg = V6Config(n_trunk=4)
        assert abs(cfg.residual_scale - 1.0 / math.sqrt(4)) < 1e-6

    def test_gate_bias_init(self):
        cfg = V6Config()
        bias = cfg.gate_bias_init(256)
        assert abs(bias - (-math.log(255))) < 1e-6
        gate = torch.sigmoid(torch.tensor(bias)).item()
        assert abs(gate - 1.0 / 256.0) < 0.001

    def test_gate_bias_init_edge(self):
        cfg = V6Config()
        assert cfg.gate_bias_init(1) == 0.0

    def test_oscillator_defaults(self):
        cfg = V6Config()
        assert cfg.n_osc_channels == 8
        assert cfg.t_max == 256
        assert cfg.t_min == 1
        assert cfg.osc_base_freq == 0.5
        assert cfg.enable_spatial_coupling is True
        assert cfg.enable_mean_field is True
        assert cfg.enable_attention is False

    def test_deep_supervision_steps_are_powers_of_two(self):
        cfg = V6Config()
        for s in cfg.deep_supervision_steps:
            assert s & (s - 1) == 0, f"{s} is not a power of 2"


# ═══════════════════════════════════════════════════════════════════════
# 2. CoupledOscillatorNetwork Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCoupledOscillatorNetwork:
    def test_init_phase_shape(self, tiny_oscillator):
        phi = tiny_oscillator.init_phase(2, 16, torch.device("cpu"))
        assert phi.shape == (2, 16, 4)
        assert (phi >= 0).all() and (phi < 2 * math.pi).all()

    def test_init_phase_deterministic(self, tiny_oscillator):
        phi1 = tiny_oscillator.init_phase(2, 16, torch.device("cpu"), deterministic=True)
        phi2 = tiny_oscillator.init_phase(2, 16, torch.device("cpu"), deterministic=True)
        assert torch.allclose(phi1, phi2)

    def test_init_phase_stochastic_varies(self, tiny_oscillator):
        phi1 = tiny_oscillator.init_phase(2, 16, torch.device("cpu"), deterministic=False)
        phi2 = tiny_oscillator.init_phase(2, 16, torch.device("cpu"), deterministic=False)
        assert not torch.allclose(phi1, phi2)

    def test_step_output_shapes(self, tiny_oscillator, tiny_config):
        B, L, D, K = 2, 16, tiny_config.d_model, tiny_config.n_osc_channels
        z = torch.randn(B, L, D)
        phi = tiny_oscillator.init_phase(B, L, torch.device("cpu"))
        phi_new, osc_gate, readout_prob, slow_ctx = tiny_oscillator.step(z, phi)
        assert phi_new.shape == (B, L, K)
        assert osc_gate.shape == (B, L, D)
        assert readout_prob.shape == (B, L)
        assert slow_ctx.shape == (B, L, D)

    def test_osc_gate_bounded(self, tiny_oscillator, tiny_config):
        z = torch.randn(2, 16, tiny_config.d_model)
        phi = tiny_oscillator.init_phase(2, 16, torch.device("cpu"))
        _, osc_gate, _, _ = tiny_oscillator.step(z, phi)
        assert (osc_gate >= 0).all() and (osc_gate <= 1).all()

    def test_readout_prob_bounded(self, tiny_oscillator, tiny_config):
        z = torch.randn(2, 16, tiny_config.d_model)
        phi = tiny_oscillator.init_phase(2, 16, torch.device("cpu"))
        _, _, readout_prob, _ = tiny_oscillator.step(z, phi)
        assert (readout_prob >= 0).all() and (readout_prob <= 1).all()

    def test_phase_wraps_to_2pi(self, tiny_oscillator, tiny_config):
        z = torch.randn(2, 16, tiny_config.d_model)
        phi = tiny_oscillator.init_phase(2, 16, torch.device("cpu"))
        phi_new, _, _, _ = tiny_oscillator.step(z, phi)
        assert (phi_new >= 0).all()
        assert (phi_new < 2 * math.pi + 1e-5).all()

    def test_antisymmetric_coupling_matrix(self, tiny_oscillator):
        C = tiny_oscillator._base_coupling()
        assert torch.allclose(C + C.T, torch.zeros_like(C), atol=1e-7)
        assert torch.allclose(C.diag(), torch.zeros(C.shape[0]), atol=1e-7)

    def test_frequencies_positive(self, tiny_oscillator, tiny_config):
        z = torch.randn(2, 16, tiny_config.d_model)
        omega = tiny_oscillator._compute_frequencies(z)
        assert (omega > 0).all()

    def test_relevance_bounded(self, tiny_oscillator, tiny_config):
        z = torch.randn(2, 16, tiny_config.d_model)
        rel = tiny_oscillator._compute_relevance(z)
        assert (rel >= 0).all() and (rel <= 1).all()

    def test_step_is_differentiable(self, tiny_oscillator, tiny_config):
        z = torch.randn(2, 16, tiny_config.d_model, requires_grad=True)
        phi = tiny_oscillator.init_phase(2, 16, torch.device("cpu"))
        phi_new, osc_gate, readout_prob, slow_ctx = tiny_oscillator.step(z, phi)
        loss = osc_gate.sum() + readout_prob.sum() + slow_ctx.sum()
        loss.backward()
        assert z.grad is not None
        assert z.grad.abs().sum() > 0


# ═══════════════════════════════════════════════════════════════════════
# 3. Self-Regulation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSelfRegulation:
    def test_initial_relevance_mostly_active(self, tiny_oscillator, tiny_config):
        z = torch.randn(4, 16, tiny_config.d_model) * 0.1
        rel = tiny_oscillator._compute_relevance(z)
        mean_rel = rel.mean().item()
        assert mean_rel > 0.6, f"Initial relevance too low: {mean_rel}"

    def test_effective_K_returns_float(self, tiny_oscillator, tiny_config):
        z = torch.randn(2, 16, tiny_config.d_model)
        eff_k = tiny_oscillator.effective_K(z)
        assert isinstance(eff_k, float)
        assert 0 <= eff_k <= tiny_config.n_osc_channels

    def test_freq_diversity_loss_is_negative(self, tiny_oscillator, tiny_config):
        z = torch.randn(2, 16, tiny_config.d_model)
        loss = tiny_oscillator.compute_freq_diversity(z)
        assert loss.item() <= 0

    def test_freq_diversity_loss_differentiable(self, tiny_oscillator, tiny_config):
        z = torch.randn(2, 16, tiny_config.d_model, requires_grad=True)
        loss = tiny_oscillator.compute_freq_diversity(z)
        loss.backward()
        assert z.grad is not None

    def test_frequency_init_distinct(self, tiny_oscillator, tiny_config):
        biases = tiny_oscillator.freq_down.bias.data.clone()
        if tiny_config.n_osc_channels > 1:
            assert biases.std().item() > 0.01


# ═══════════════════════════════════════════════════════════════════════
# 4. Global Mean-Field Coupling Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGlobalMeanField:
    def test_mean_field_shape(self, tiny_oscillator, tiny_config):
        phi = torch.randn(2, 16, tiny_config.n_osc_channels)
        coupling = tiny_oscillator._global_mean_field_coupling(phi)
        assert coupling.shape == phi.shape

    def test_mean_field_is_causal(self, tiny_oscillator, tiny_config):
        K = tiny_config.n_osc_channels
        phi = torch.randn(1, 8, K, requires_grad=True)
        coupling = tiny_oscillator._global_mean_field_coupling(phi)
        for i in range(8):
            if phi.grad is not None:
                phi.grad.zero_()
            coupling[0, i, 0].backward(retain_graph=True)
            grad = phi.grad[0]
            if i < 7:
                future_grad = grad[i + 1:].abs().sum()
                assert future_grad < 1e-6, \
                    f"Mean field at pos {i} depends on future: {future_grad}"

    def test_local_spatial_coupling_shape(self, tiny_oscillator, tiny_config):
        phi = torch.randn(2, 16, tiny_config.n_osc_channels)
        coupling = tiny_oscillator._local_spatial_coupling(phi)
        assert coupling.shape == phi.shape

    def test_inter_channel_coupling_shape(self, tiny_oscillator, tiny_config):
        phi = torch.randn(2, 16, tiny_config.n_osc_channels)
        z = torch.randn(2, 16, tiny_config.d_model)
        coupling = tiny_oscillator._inter_channel_coupling(phi, z)
        assert coupling.shape == phi.shape


# ═══════════════════════════════════════════════════════════════════════
# 5. Fast Kernel Tests
# ═══════════════════════════════════════════════════════════════════════

class TestFastKernel:
    def test_output_shape(self, tiny_config):
        kernel = FastKernel(tiny_config)
        B, L, D = 2, 16, tiny_config.d_model
        P = torch.randn(B, L, D)
        slow_ctx = torch.randn(B, L, D)
        pe = torch.randn(B, L, tiny_config.d_pe)
        delta, gate = kernel(P, slow_ctx, pe)
        assert delta.shape == (B, L, D)
        assert gate.shape == (B, L, D)

    def test_gate_bounded(self, tiny_config):
        kernel = FastKernel(tiny_config)
        P = torch.randn(2, 16, tiny_config.d_model)
        slow_ctx = torch.randn(2, 16, tiny_config.d_model)
        pe = torch.randn(2, 16, tiny_config.d_pe)
        _, gate = kernel(P, slow_ctx, pe)
        assert (gate >= 0).all() and (gate <= 1).all()

    def test_zero_init_delta_near_zero(self, tiny_config):
        kernel = FastKernel(tiny_config)
        P = torch.randn(2, 16, tiny_config.d_model)
        slow_ctx = torch.randn(2, 16, tiny_config.d_model)
        pe = torch.randn(2, 16, tiny_config.d_pe)
        delta, _ = kernel(P, slow_ctx, pe)
        assert delta.abs().max() < 1e-5


# ═══════════════════════════════════════════════════════════════════════
# 6. SwiGLU Block Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSwiGLUBlock:
    def test_output_shape(self):
        block = SwiGLUBlock(128, alpha_res=0.5)
        h = torch.randn(2, 16, 128)
        out = block(h)
        assert out.shape == h.shape

    def test_residual_connection(self):
        block = SwiGLUBlock(128, alpha_res=0.5)
        nn.init.zeros_(block.W_down.weight)
        h = torch.randn(2, 16, 128)
        out = block(h)
        assert torch.allclose(out, h, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# 7. Full Model Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAENCAv6Model:
    def test_forward_train_output(self, tiny_model, tiny_batch):
        tiny_model.train()
        loss, info = tiny_model.forward_train(tiny_batch)
        assert loss.isfinite()
        assert "ce" in info
        assert "reg" in info

    def test_ce_loss_near_random(self, tiny_model, tiny_batch):
        tiny_model.train()
        _, info = tiny_model.forward_train(tiny_batch)
        random_ce = math.log(tiny_model.config.vocab_size)
        ce = info["ce"].item()
        assert abs(ce - random_ce) < 1.5

    def test_backward_pass(self, tiny_model, tiny_batch):
        tiny_model.train()
        loss, _ = tiny_model.forward_train(tiny_batch)
        loss.backward()
        grads = [
            (n, p.grad.abs().sum().item())
            for n, p in tiny_model.named_parameters()
            if p.grad is not None
        ]
        nonzero = [(n, g) for n, g in grads if g > 0]
        assert len(nonzero) > 0

    def test_get_logits_shape(self, tiny_model, tiny_batch):
        tiny_model.eval()
        logits = tiny_model.get_logits(tiny_batch)
        B, L = tiny_batch.shape
        V = tiny_model.config.vocab_size
        assert logits.shape == (B, L, V)

    def test_get_param_groups(self, tiny_model):
        groups = tiny_model.get_param_groups()
        assert len(groups) == 2
        labels = {g["label"] for g in groups}
        assert labels == {"decay", "no_decay"}
        total_params = sum(p.numel() for g in groups for p in g["params"])
        expected = sum(p.numel() for p in tiny_model.parameters() if p.requires_grad)
        assert total_params == expected

    def test_describe(self, tiny_model):
        desc = tiny_model.describe()
        assert "Oscillator" in desc["architecture"]
        assert desc["total_params"] > 0
        assert "fast_kernel_params" in desc
        assert "oscillator_params" in desc
        assert desc["n_osc_channels"] == 4
        assert desc["t_max"] == 8

    def test_extra_metrics(self, tiny_model):
        metrics = tiny_model.extra_metrics()
        assert "coupling_base_norm" in metrics
        assert "kappa_local" in metrics
        assert "kappa_global" in metrics
        assert "readout_bias" in metrics

    def test_get_phases(self, tiny_model):
        assert tiny_model.get_phases(0, 100) == "ONCA"

    def test_info_has_monitoring_metrics(self, tiny_model, tiny_batch):
        tiny_model.train()
        _, info = tiny_model.forward_train(tiny_batch)
        assert "expected_steps" in info
        assert "pct_halted_early" in info

    def test_untied_embeddings(self, tiny_config_dict):
        cfg = {**tiny_config_dict, "tie_embeddings": False}
        model = AENCAv6Model(cfg)
        batch = torch.randint(0, cfg["vocab_size"], (2, cfg["seq_len"]))
        model.train()
        loss, info = model.forward_train(batch)
        assert loss.isfinite()
        model.eval()
        logits = model.get_logits(batch)
        assert logits.shape == (2, cfg["seq_len"], cfg["vocab_size"])


# ═══════════════════════════════════════════════════════════════════════
# 8. Per-Position Halting Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPerPositionHalting:
    def test_remaining_mass_nonnegative(self, tiny_model, tiny_batch):
        tiny_model.eval()
        cfg = tiny_model.config
        B, L = tiny_batch.shape
        device = tiny_batch.device

        s = tiny_model.token_embedding(tiny_batch)
        s = s + tiny_model.state_pe(L).unsqueeze(0)
        pe = tiny_model.pos_encoder(L).unsqueeze(0).expand(B, -1, -1)

        z = s
        phi = tiny_model.oscillator.init_phase(B, L, device, deterministic=True)
        z_out = torch.zeros_like(z)
        remaining = torch.ones(B, L, device=device)

        for t in range(cfg.t_max):
            pos_active = (remaining > cfg.readout_halt_eps).float()
            if pos_active.sum() == 0:
                break
            z_cand, phi, osc_gate, readout_prob = tiny_model._nca_step(z, phi, pe)
            z = z + pos_active.unsqueeze(-1) * (z_cand - z)
            if t >= cfg.t_min:
                if t == cfg.t_max - 1:
                    eff = remaining * pos_active
                else:
                    eff = torch.min(readout_prob * pos_active, remaining)
                z_out = z_out + eff.unsqueeze(-1) * z
                remaining = remaining - eff

        assert (remaining >= -1e-6).all()

    def test_readout_accumulates_mass(self, tiny_model, tiny_batch):
        tiny_model.eval()
        cfg = tiny_model.config
        B, L = tiny_batch.shape
        device = tiny_batch.device

        s = tiny_model.token_embedding(tiny_batch)
        s = s + tiny_model.state_pe(L).unsqueeze(0)
        pe = tiny_model.pos_encoder(L).unsqueeze(0).expand(B, -1, -1)

        z = s
        phi = tiny_model.oscillator.init_phase(B, L, device, deterministic=True)
        z_out = torch.zeros_like(z)
        remaining = torch.ones(B, L, device=device)

        for t in range(cfg.t_max):
            pos_active = (remaining > cfg.readout_halt_eps).float()
            if pos_active.sum() == 0:
                break
            z_cand, phi, osc_gate, readout_prob = tiny_model._nca_step(z, phi, pe)
            z = z + pos_active.unsqueeze(-1) * (z_cand - z)
            if t >= cfg.t_min:
                if t == cfg.t_max - 1:
                    eff = remaining * pos_active
                else:
                    eff = torch.min(readout_prob * pos_active, remaining)
                z_out = z_out + eff.unsqueeze(-1) * z
                remaining = remaining - eff

        total_mass = 1.0 - remaining.mean().item()
        assert total_mass > 0.90


# ═══════════════════════════════════════════════════════════════════════
# 9. Zero-Init Cascade Verification
# ═══════════════════════════════════════════════════════════════════════

class TestZeroInitCascade:
    def test_model_starts_near_identity(self, tiny_model, tiny_batch):
        tiny_model.eval()
        cfg = tiny_model.config
        s = tiny_model.token_embedding(tiny_batch)
        L = tiny_batch.shape[1]
        s = s + tiny_model.state_pe(L).unsqueeze(0)
        logits = tiny_model.get_logits(tiny_batch)
        with torch.no_grad():
            direct_logits = tiny_model.readout(s)
        diff = (logits - direct_logits).abs().max().item()
        assert diff < 0.1, f"Model not near-identity at init: max diff = {diff}"

    def test_oscillator_gate_initial_value(self, tiny_oscillator, tiny_config):
        """Gate should start near σ(2) ≈ 0.88 (pass-through)."""
        z = torch.randn(2, 16, tiny_config.d_model) * 0.1
        phi = tiny_oscillator.init_phase(2, 16, torch.device("cpu"), deterministic=True)
        _, osc_gate, _, _ = tiny_oscillator.step(z, phi)
        mean_gate = osc_gate.mean().item()
        # σ(2) ≈ 0.88, allow ±0.15 tolerance
        assert abs(mean_gate - 0.88) < 0.15

    def test_slow_ctx_starts_near_zero(self, tiny_oscillator, tiny_config):
        z = torch.randn(2, 16, tiny_config.d_model)
        phi = tiny_oscillator.init_phase(2, 16, torch.device("cpu"))
        _, _, _, slow_ctx = tiny_oscillator.step(z, phi)
        assert slow_ctx.abs().max() < 1e-5


# ═══════════════════════════════════════════════════════════════════════
# 10. Gradient Engagement
# ═══════════════════════════════════════════════════════════════════════

class TestGradientEngagement:
    def test_more_grads_after_steps(self, tiny_config_dict):
        model = AENCAv6Model(tiny_config_dict)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = torch.randint(0, tiny_config_dict["vocab_size"],
                              (4, tiny_config_dict["seq_len"]))
        loss, _ = model.forward_train(batch)
        loss.backward()
        init_count = sum(
            1 for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        opt.step()
        opt.zero_grad()
        for _ in range(5):
            loss, _ = model.forward_train(batch)
            loss.backward()
            opt.step()
            opt.zero_grad()
        loss, _ = model.forward_train(batch)
        loss.backward()
        later_count = sum(
            1 for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert later_count >= init_count

    def test_loss_decreases(self, tiny_config_dict):
        model = AENCAv6Model(tiny_config_dict)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = torch.randint(0, tiny_config_dict["vocab_size"],
                              (4, tiny_config_dict["seq_len"]))
        loss0, _ = model.forward_train(batch)
        init_loss = loss0.item()
        for _ in range(20):
            opt.zero_grad()
            loss, _ = model.forward_train(batch)
            loss.backward()
            opt.step()
        final_loss = loss.item()
        assert final_loss < init_loss


# ═══════════════════════════════════════════════════════════════════════
# 11. Model Registration
# ═══════════════════════════════════════════════════════════════════════

class TestRegistration:
    def test_model_registered(self):
        from ncalm.model import get_model_class
        cls = get_model_class("v6")
        assert cls is AENCAv6Model

    def test_profile_exists(self):
        from ncalm.profiles import PROFILES
        assert "smoke_v6" in PROFILES
        assert "pile_smoke_v6" in PROFILES
        assert "pile_2b_v6" in PROFILES

    def test_smoke_profile_has_oscillator_keys(self):
        from ncalm.profiles import PROFILES
        prof = PROFILES["smoke_v6"]
        assert "n_osc_channels" in prof
        assert "t_max" in prof
        assert "checkpoint_segment_size" in prof
        assert "n_theta_cycles" not in prof
        assert "t_max_gamma" not in prof
        assert "halt_prior_p" not in prof

    def test_all_v6_profiles_have_oscillator_keys(self):
        from ncalm.profiles import PROFILES
        v6_keys = ["smoke_v6", "pile_smoke_v6", "pile_250m_v6",
                    "pile_250m_v6_flops", "pile_2b_v6"]
        for key in v6_keys:
            prof = PROFILES[key]
            assert "n_osc_channels" in prof, f"{key} missing n_osc_channels"
            assert "t_max" in prof, f"{key} missing t_max"
            assert "n_theta_cycles" not in prof, f"{key} still has n_theta_cycles"
            assert "halt_prior_p" not in prof, f"{key} still has halt_prior_p"


# ═══════════════════════════════════════════════════════════════════════
# 12. Deep Supervision
# ═══════════════════════════════════════════════════════════════════════

class TestDeepSupervision:
    def test_aux_loss_present_when_enabled(self, tiny_model, tiny_batch):
        tiny_model.train()
        _, info = tiny_model.forward_train(tiny_batch)
        if "aux" in info:
            assert info["aux"].isfinite()

    def test_no_aux_when_disabled(self, tiny_config_dict):
        cfg = {**tiny_config_dict, "lambda_aux": 0.0}
        model = AENCAv6Model(cfg)
        model.train()
        batch = torch.randint(0, cfg["vocab_size"], (2, cfg["seq_len"]))
        _, info = model.forward_train(batch)
        assert "aux" not in info


# ═══════════════════════════════════════════════════════════════════════
# 13. Edge Cases & Robustness
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_single_step(self):
        cfg = {
            "d_model": 64, "d_msg": 16, "d_ff": 128, "d_pe": 16,
            "d_slow_ff": 128, "n_heads": 4, "n_trunk": 1,
            "kernel_radius": 1,
            "n_osc_channels": 4, "t_max": 1, "t_min": 0,
            "checkpoint_segment_size": 1,
            "deep_supervision_steps": (),
            "vocab_size": 100, "max_seq_len": 16, "seq_len": 16,
            "mos_components": 1, "tie_embeddings": True,
            "lambda_reg": 0.01, "lambda_aux": 0.0,
            "lambda_agc": 0.02, "dropout": 0.0,
        }
        model = AENCAv6Model(cfg)
        batch = torch.randint(0, 100, (2, 16))
        model.train()
        loss, info = model.forward_train(batch)
        assert loss.isfinite()

    def test_two_steps(self):
        cfg = {
            "d_model": 64, "d_msg": 16, "d_ff": 128, "d_pe": 16,
            "d_slow_ff": 128, "n_heads": 4, "n_trunk": 1,
            "kernel_radius": 1,
            "n_osc_channels": 2, "t_max": 2, "t_min": 0,
            "checkpoint_segment_size": 2,
            "deep_supervision_steps": (),
            "vocab_size": 100, "max_seq_len": 16, "seq_len": 16,
            "mos_components": 1, "tie_embeddings": True,
            "lambda_reg": 0.01, "lambda_aux": 0.0,
            "lambda_agc": 0.02, "dropout": 0.0,
        }
        model = AENCAv6Model(cfg)
        batch = torch.randint(0, 100, (2, 16))
        model.train()
        loss, info = model.forward_train(batch)
        assert loss.isfinite()
        loss.backward()

    def test_no_spatial_coupling(self, tiny_config_dict):
        cfg = {**tiny_config_dict, "enable_spatial_coupling": False}
        model = AENCAv6Model(cfg)
        batch = torch.randint(0, cfg["vocab_size"], (2, cfg["seq_len"]))
        model.train()
        loss, _ = model.forward_train(batch)
        assert loss.isfinite()

    def test_no_mean_field(self, tiny_config_dict):
        cfg = {**tiny_config_dict, "enable_mean_field": False}
        model = AENCAv6Model(cfg)
        batch = torch.randint(0, cfg["vocab_size"], (2, cfg["seq_len"]))
        model.train()
        loss, _ = model.forward_train(batch)
        assert loss.isfinite()

    def test_no_coupling_at_all(self, tiny_config_dict):
        cfg = {
            **tiny_config_dict,
            "enable_spatial_coupling": False,
            "enable_mean_field": False,
        }
        model = AENCAv6Model(cfg)
        batch = torch.randint(0, cfg["vocab_size"], (2, cfg["seq_len"]))
        model.train()
        loss, _ = model.forward_train(batch)
        assert loss.isfinite()
        loss.backward()

    def test_batch_size_one(self, tiny_config_dict):
        model = AENCAv6Model(tiny_config_dict)
        batch = torch.randint(0, tiny_config_dict["vocab_size"],
                              (1, tiny_config_dict["seq_len"]))
        model.train()
        loss, _ = model.forward_train(batch)
        assert loss.isfinite()
        loss.backward()

    def test_deterministic_eval(self, tiny_model, tiny_batch):
        tiny_model.eval()
        logits1 = tiny_model.get_logits(tiny_batch)
        logits2 = tiny_model.get_logits(tiny_batch)
        assert torch.allclose(logits1, logits2, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# 14. Oscillator Diagnostics
# ═══════════════════════════════════════════════════════════════════════

class TestOscillatorDiagnostics:
    def test_get_diagnostics_keys(self, tiny_oscillator, tiny_config):
        z = torch.randn(2, 16, tiny_config.d_model)
        phi = tiny_oscillator.init_phase(2, 16, torch.device("cpu"))
        diag = tiny_oscillator.get_diagnostics(z, phi)
        expected_keys = {
            "effective_K", "freq_std", "freq_mean",
            "coupling_base_norm", "coupling_dynamic_norm", "coupling_dynamic_max",
            "relevance_mean",
            "kappa_local", "kappa_global",
        }
        assert expected_keys == set(diag.keys())

    def test_diagnostics_values_finite(self, tiny_oscillator, tiny_config):
        z = torch.randn(2, 16, tiny_config.d_model)
        phi = tiny_oscillator.init_phase(2, 16, torch.device("cpu"))
        diag = tiny_oscillator.get_diagnostics(z, phi)
        for k, v in diag.items():
            assert math.isfinite(v), f"Diagnostic {k} not finite: {v}"

    def test_readout_bias_init(self, tiny_config):
        """Readout should start at ~2/T_max for better mass distribution."""
        osc = CoupledOscillatorNetwork(tiny_config)
        bias = osc.readout_down.bias.item()
        readout_init = torch.sigmoid(torch.tensor(bias)).item()
        expected = 2.0 / tiny_config.t_max
        assert abs(readout_init - expected) < 0.05


# ═══════════════════════════════════════════════════════════════════════
# Run via pytest or standalone
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
