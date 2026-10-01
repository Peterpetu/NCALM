"""Comprehensive unit tests for AENCA V7.1: BDH-inspired Hebbian NCA.

Tests cover:
  1. Configuration validation (derived properties, auto-sizing d_ff_theta)
  2. HebbianKernel (shape, zero-init cascade, concept encoding sparsity)
  3. HebbianFFN (shape, zero-init, cross-modal gating)
  4. ThetaBlock (attention causality, HebbianFFN integration, zero-init)
  5. AdaptiveHalting (mass conservation, PonderNet KL, early exit)
  6. Full model forward/backward pass
  7. Full model inference (get_logits)
  8. Zero-init cascade (model approx identity at initialization)
  9. Gradient engagement (loss decreases with optimization)
  10. Parameter groups (decay / no_decay split)
  11. Model registration and profile discovery
  12. Deep supervision (aux losses at theta boundaries)
  13. Param count (matches target within tolerance)
  14. Extra metrics (monitoring outputs)
"""

from __future__ import annotations

import dataclasses
import math
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, ".")

from ncalm.models.v7.config import V7Config
from ncalm.models.v7.hebbian import HebbianKernel, HebbianFFN
from ncalm.models.v7.attention import ThetaBlock, RotaryPositionEmbedding
from ncalm.models.v7.halting import AdaptiveHalting
from ncalm.models.v7.model import AENCAv7Model


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def tiny_config() -> V7Config:
    """Minimal config for fast CPU tests."""
    return V7Config(
        d_model=64,
        d_concept=128,
        n_theta=2, t_gamma=2,
        n_heads=4, d_head=16, d_ff_theta=128,
        kernel_radius=1,
        enable_halting=True, halt_eps=0.01, lambda_halt=0.01,
        mos_components=1, tie_embeddings=True,
        vocab_size=200, max_seq_len=32, seq_len=32,
        lambda_reg=0.01, lambda_aux=0.05, lambda_agc=0.02, dropout=0.0,
    )


@pytest.fixture
def tiny_config_dict(tiny_config: V7Config) -> dict:
    """Config as flat dict for model constructor."""
    return {f.name: getattr(tiny_config, f.name) for f in dataclasses.fields(tiny_config)}


@pytest.fixture
def tiny_model(tiny_config_dict: dict) -> AENCAv7Model:
    """Tiny V7.1 model for CPU testing."""
    return AENCAv7Model(tiny_config_dict)


@pytest.fixture
def tiny_batch(tiny_config: V7Config) -> torch.Tensor:
    """Random batch of token IDs."""
    return torch.randint(0, tiny_config.vocab_size, (2, tiny_config.seq_len))


@pytest.fixture
def tiny_kernel(tiny_config: V7Config) -> HebbianKernel:
    """Tiny HebbianKernel for unit tests."""
    return HebbianKernel(tiny_config)


@pytest.fixture
def tiny_ffn(tiny_config: V7Config) -> HebbianFFN:
    """Tiny HebbianFFN for unit tests."""
    return HebbianFFN(tiny_config.d_model, tiny_config.d_ff_theta, tiny_config.dropout)


@pytest.fixture
def tiny_theta(tiny_config: V7Config) -> ThetaBlock:
    """Tiny ThetaBlock for unit tests."""
    return ThetaBlock(tiny_config, block_idx=0)


@pytest.fixture
def tiny_halting(tiny_config: V7Config) -> AdaptiveHalting:
    """Tiny AdaptiveHalting for unit tests."""
    return AdaptiveHalting(tiny_config)


# ═══════════════════════════════════════════════════════════════════════
# 1. Configuration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestV7Config:
    """Test V7Config dataclass and derived properties."""

    def test_defaults(self):
        cfg = V7Config()
        assert cfg.d_model == 768
        assert cfg.n_theta == 4
        assert cfg.t_gamma == 3
        assert cfg.d_concept == 2048

    def test_total_gamma_steps(self, tiny_config):
        assert tiny_config.total_gamma_steps == 2 * 2

    def test_total_steps(self, tiny_config):
        assert tiny_config.total_steps == 2 * (2 + 1)

    def test_K_alias(self, tiny_config):
        assert tiny_config.K == tiny_config.total_steps

    def test_residual_scale(self, tiny_config):
        expected = 1.0 / math.sqrt(2.0 * 2)
        assert abs(tiny_config.residual_scale - expected) < 1e-6

    def test_gate_bias_init(self, tiny_config):
        bias = tiny_config.gate_bias_init(tiny_config.total_gamma_steps)
        prob = 1.0 / (1.0 + math.exp(-bias))
        expected = 1.0 / tiny_config.total_gamma_steps
        assert abs(prob - expected) < 0.01

    def test_compute_d_ff_theta(self):
        cfg = V7Config()
        d_ff = cfg.compute_d_ff_theta(162_000_000)
        assert d_ff > 256
        assert d_ff < 30_000

    def test_d_concept_in_config(self):
        cfg = V7Config(d_concept=1024)
        assert cfg.d_concept == 1024


# ═══════════════════════════════════════════════════════════════════════
# 2. HebbianKernel Tests
# ═══════════════════════════════════════════════════════════════════════

class TestHebbianKernel:
    """Test HebbianKernel shape, zero-init cascade, and concept encoding."""

    def test_output_shape(self, tiny_kernel, tiny_config):
        B, L, d = 2, 32, tiny_config.d_model
        P = torch.randn(B, L, d)
        z = torch.randn(B, L, d)
        delta = tiny_kernel(P, z)
        assert delta.shape == (B, L, d)

    def test_zero_init_cascade(self, tiny_kernel, tiny_config):
        """W_dec is zero-init -> initial delta should be near-zero."""
        B, L, d = 2, 32, tiny_config.d_model
        P = torch.randn(B, L, d)
        z = torch.randn(B, L, d)
        delta = tiny_kernel(P, z)
        assert delta.abs().max().item() < 1e-5, "HebbianKernel delta should start near zero"

    def test_concept_encoding_positive(self, tiny_kernel, tiny_config):
        """ReLU concept encodings should be non-negative."""
        import torch.nn.functional as F
        P = torch.randn(2, 16, tiny_config.d_model)
        z = torch.randn(2, 16, tiny_config.d_model)
        x = F.relu(tiny_kernel.W_enc_p(tiny_kernel.norm_p(P)))
        y = F.relu(tiny_kernel.W_enc_z(tiny_kernel.norm_z(z)))
        assert x.min().item() >= 0.0
        assert y.min().item() >= 0.0

    def test_concept_encoding_sparse(self, tiny_kernel, tiny_config):
        """ReLU concept encodings should be sparse."""
        import torch.nn.functional as F
        P = torch.randn(4, 32, tiny_config.d_model)
        x = F.relu(tiny_kernel.W_enc_p(tiny_kernel.norm_p(P)))
        sparsity = (x == 0).float().mean().item()
        assert sparsity > 0.1, f"Expected sparse concepts, got sparsity={sparsity:.2%}"

    def test_hebbian_product_gating(self, tiny_kernel, tiny_config):
        """If one encoding is zero, output should be zero (Hebbian gate)."""
        B, L, d = 1, 8, tiny_config.d_model
        P = torch.zeros(B, L, d)
        z = torch.randn(B, L, d)
        delta = tiny_kernel(P, z)
        assert delta.abs().max().item() < 1e-5

    def test_gradient_flows(self, tiny_kernel, tiny_config):
        P = torch.randn(2, 16, tiny_config.d_model, requires_grad=True)
        z = torch.randn(2, 16, tiny_config.d_model, requires_grad=True)
        delta = tiny_kernel(P, z)
        loss = delta.sum()
        loss.backward()
        for name, param in tiny_kernel.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ═══════════════════════════════════════════════════════════════════════
# 3. HebbianFFN Tests
# ═══════════════════════════════════════════════════════════════════════

class TestHebbianFFN:
    """Test HebbianFFN shape, zero-init, and cross-modal gating."""

    def test_output_shape(self, tiny_ffn, tiny_config):
        z_pre = torch.randn(2, 32, tiny_config.d_model)
        z_post = torch.randn(2, 32, tiny_config.d_model)
        out = tiny_ffn(z_pre, z_post)
        assert out.shape == (2, 32, tiny_config.d_model)

    def test_zero_init_cascade(self, tiny_ffn, tiny_config):
        """W_dec is zero-init -> initial output should be near-zero."""
        z_pre = torch.randn(2, 16, tiny_config.d_model)
        z_post = torch.randn(2, 16, tiny_config.d_model)
        out = tiny_ffn(z_pre, z_post)
        assert out.abs().max().item() < 1e-5

    def test_cross_modal_different_inputs(self, tiny_ffn, tiny_config):
        """FFN with different inputs should produce different results."""
        nn.init.xavier_normal_(tiny_ffn.W_dec.weight)
        z1 = torch.randn(2, 16, tiny_config.d_model)
        z2 = torch.randn(2, 16, tiny_config.d_model)
        out_same = tiny_ffn(z1, z1)
        out_cross = tiny_ffn(z1, z2)
        assert not torch.allclose(out_same, out_cross, atol=1e-3)

    def test_gradient_flows(self, tiny_ffn, tiny_config):
        z_pre = torch.randn(2, 16, tiny_config.d_model, requires_grad=True)
        z_post = torch.randn(2, 16, tiny_config.d_model, requires_grad=True)
        out = tiny_ffn(z_pre, z_post)
        loss = out.sum()
        loss.backward()
        assert z_pre.grad is not None
        assert z_post.grad is not None
        for name, param in tiny_ffn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_param_count_matches_swiglu(self, tiny_config):
        """HebbianFFN should have 3*d*d_ff + 2*d params."""
        d = tiny_config.d_model
        d_ff = tiny_config.d_ff_theta
        ffn = HebbianFFN(d, d_ff)
        total_params = sum(p.numel() for p in ffn.parameters())
        expected = 3 * d * d_ff + 2 * d
        assert total_params == expected, f"Got {total_params}, expected {expected}"


# ═══════════════════════════════════════════════════════════════════════
# 4. ThetaBlock Tests
# ═══════════════════════════════════════════════════════════════════════

class TestThetaBlock:
    """Test ThetaBlock attention + HebbianFFN."""

    def test_output_shape(self, tiny_theta, tiny_config):
        z = torch.randn(2, 32, tiny_config.d_model)
        out = tiny_theta(z)
        assert out.shape == z.shape

    def test_zero_init_identity(self, tiny_theta, tiny_config):
        """With zero-init output projections, block should be near-identity."""
        z = torch.randn(2, 16, tiny_config.d_model)
        out = tiny_theta(z)
        assert torch.allclose(out, z, atol=1e-5), \
            "ThetaBlock should start as near-identity"

    def test_attention_causality(self, tiny_config):
        """Attention output at position i should not depend on position i+1."""
        block = ThetaBlock(tiny_config, block_idx=0)
        nn.init.xavier_normal_(block.out_proj.weight)
        nn.init.xavier_normal_(block.ffn.W_dec.weight)

        z = torch.randn(1, 16, tiny_config.d_model, requires_grad=True)
        out = block(z)

        out[0, 5, 0].backward(retain_graph=True)
        grad = z.grad.clone()
        z.grad = None
        assert grad[0, 6:, :].abs().max().item() < 1e-5, \
            "ThetaBlock attention is not causal!"

    def test_gradient_flows(self, tiny_theta, tiny_config):
        z = torch.randn(2, 16, tiny_config.d_model, requires_grad=True)
        out = tiny_theta(z)
        loss = out.sum()
        loss.backward()
        for name, param in tiny_theta.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_has_hebbian_ffn(self, tiny_theta):
        """ThetaBlock should contain a HebbianFFN (not SwiGLU)."""
        assert hasattr(tiny_theta, "ffn")
        assert isinstance(tiny_theta.ffn, HebbianFFN)


# ═══════════════════════════════════════════════════════════════════════
# 5. RoPE Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRoPE:
    def test_cache_shape(self):
        rope = RotaryPositionEmbedding(64, max_seq_len=128)
        cos, sin = rope(64)
        assert cos.shape == (64, 64)
        assert sin.shape == (64, 64)

    def test_cache_extension(self):
        rope = RotaryPositionEmbedding(64, max_seq_len=32)
        cos, sin = rope(64)
        assert cos.shape == (64, 64)


# ═══════════════════════════════════════════════════════════════════════
# 6. Adaptive Halting Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAdaptiveHalting:
    def test_halt_prob_range(self, tiny_halting, tiny_config):
        z = torch.randn(2, 32, tiny_config.d_model)
        hp = tiny_halting.compute_halt_prob(z)
        assert hp.shape == (2, 32)
        assert hp.min().item() >= 0.0
        assert hp.max().item() <= 1.0

    def test_mass_conservation(self, tiny_halting, tiny_config):
        z = torch.randn(2, 16, tiny_config.d_model)
        z_out = torch.zeros_like(z)
        remaining = torch.ones(2, 16)
        for theta_idx in range(tiny_config.n_theta):
            is_last = (theta_idx == tiny_config.n_theta - 1)
            z_out, remaining, _ = tiny_halting.accumulate(z, z_out, remaining, is_last)
        assert remaining.abs().max().item() < 1e-5

    def test_last_cycle_absorbs_all(self, tiny_halting, tiny_config):
        z = torch.randn(2, 16, tiny_config.d_model)
        z_out = torch.zeros_like(z)
        remaining = torch.ones(2, 16) * 0.3
        z_out, remaining, _ = tiny_halting.accumulate(z, z_out, remaining, is_last=True)
        assert remaining.abs().max().item() < 1e-6

    def test_kl_loss_computable(self, tiny_halting, tiny_config):
        halt_probs = [torch.rand(2, 16) for _ in range(tiny_config.n_theta)]
        kl = tiny_halting.compute_kl_loss(halt_probs)
        assert kl.dim() == 0
        assert torch.isfinite(kl)


# ═══════════════════════════════════════════════════════════════════════
# 7. Full Model Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAENCAv7Model:
    def test_forward_train_shape(self, tiny_model, tiny_batch):
        loss, info = tiny_model.forward_train(tiny_batch)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert "ce" in info

    def test_forward_train_info_keys(self, tiny_model, tiny_batch):
        _, info = tiny_model.forward_train(tiny_batch)
        required_keys = {"ce", "expected_steps", "pct_halted_early"}
        assert required_keys.issubset(info.keys())

    def test_get_logits_shape(self, tiny_model, tiny_batch, tiny_config):
        logits = tiny_model.get_logits(tiny_batch)
        B, L = tiny_batch.shape
        assert logits.shape == (B, L, tiny_config.vocab_size)

    def test_backward_all_params_get_grad(self, tiny_model, tiny_batch):
        loss, _ = tiny_model.forward_train(tiny_batch)
        loss.backward()
        no_grad_params = []
        for name, param in tiny_model.named_parameters():
            if param.requires_grad and param.grad is None:
                no_grad_params.append(name)
        assert len(no_grad_params) == 0, \
            f"Parameters without gradients: {no_grad_params}"

    def test_zero_init_near_identity(self, tiny_model, tiny_config):
        batch = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.seq_len))
        loss, info = tiny_model.forward_train(batch)
        expected_ce = math.log(tiny_config.vocab_size)
        actual_ce = info["ce"].item()
        assert abs(actual_ce - expected_ce) / expected_ce < 0.25, \
            f"Initial CE={actual_ce:.2f}, expected ~{expected_ce:.2f}"

    def test_gradient_engagement(self, tiny_model, tiny_config):
        model = tiny_model
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch = torch.randint(0, tiny_config.vocab_size, (4, tiny_config.seq_len))
        losses = []
        for _ in range(5):
            loss, _ = model.forward_train(batch)
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
        assert losses[-1] < losses[0], \
            f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_describe(self, tiny_model):
        desc = tiny_model.describe()
        assert "architecture" in desc
        assert "total_params" in desc
        assert "v7.1" in desc["architecture"].lower()
        assert desc["total_params"] > 0
        assert "d_concept" in desc
        assert "hebbian_kernel_params" in desc

    def test_get_param_groups(self, tiny_model):
        groups = tiny_model.get_param_groups()
        assert len(groups) == 2
        labels = {g["label"] for g in groups}
        assert labels == {"decay", "no_decay"}
        for g in groups:
            if g["label"] == "no_decay":
                assert g["weight_decay"] == 0.0

    def test_extra_metrics(self, tiny_model):
        metrics = tiny_model.extra_metrics()
        assert "kernel_dec_norm" in metrics
        assert "kernel_enc_p_norm" in metrics
        assert "kernel_enc_z_norm" in metrics
        assert "halt_bias" in metrics
        assert "theta0_attn_norm" in metrics
        assert "theta0_ffn_dec_norm" in metrics

    def test_get_phases(self, tiny_model):
        assert tiny_model.get_phases(0, 1000) == "HNCA"

    def test_has_hebbian_kernel(self, tiny_model):
        assert hasattr(tiny_model, "hebbian_kernel")
        assert isinstance(tiny_model.hebbian_kernel, HebbianKernel)
        assert not hasattr(tiny_model, "slim_kernel")
        assert not hasattr(tiny_model, "css")


# ═══════════════════════════════════════════════════════════════════════
# 8. Halting Behavior Tests
# ═══════════════════════════════════════════════════════════════════════

class TestHaltingBehavior:
    def test_no_halting_mode(self, tiny_config):
        cfg_dict = {f.name: getattr(tiny_config, f.name)
                    for f in dataclasses.fields(tiny_config)}
        cfg_dict["enable_halting"] = False
        model = AENCAv7Model(cfg_dict)
        batch = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.seq_len))
        loss, info = model.forward_train(batch)
        assert torch.isfinite(loss)

    def test_single_theta_cycle(self, tiny_config):
        cfg_dict = {f.name: getattr(tiny_config, f.name)
                    for f in dataclasses.fields(tiny_config)}
        cfg_dict["n_theta"] = 1
        cfg_dict["d_ff_theta"] = 128
        model = AENCAv7Model(cfg_dict)
        batch = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.seq_len))
        loss, info = model.forward_train(batch)
        assert torch.isfinite(loss)


# ═══════════════════════════════════════════════════════════════════════
# 9. Deep Supervision Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDeepSupervision:
    def test_aux_loss_present(self, tiny_model, tiny_batch):
        _, info = tiny_model.forward_train(tiny_batch)
        if tiny_model.config.n_theta > 1 and tiny_model.config.lambda_aux > 0:
            assert "aux" in info, "Aux loss should be present"

    def test_aux_loss_zero_lambda(self, tiny_config):
        cfg_dict = {f.name: getattr(tiny_config, f.name)
                    for f in dataclasses.fields(tiny_config)}
        cfg_dict["lambda_aux"] = 0.0
        model = AENCAv7Model(cfg_dict)
        batch = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.seq_len))
        _, info = model.forward_train(batch)
        assert "aux" not in info


# ═══════════════════════════════════════════════════════════════════════
# 10. Registration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRegistration:
    def test_model_registered(self):
        from ncalm.model import get_model_class
        cls = get_model_class("v7")
        assert cls is AENCAv7Model

    def test_profiles_exist(self):
        from ncalm.profiles import PROFILES
        assert "smoke_v7" in PROFILES
        assert "pile_250m_v7" in PROFILES
        assert "pile_2b_v7" in PROFILES

    def test_smoke_profile_creates_model(self):
        from ncalm.profiles import PROFILES
        profile = PROFILES["smoke_v7"]
        model = AENCAv7Model(profile)
        assert model is not None
        desc = model.describe()
        assert desc["total_params"] > 0


# ═══════════════════════════════════════════════════════════════════════
# 11. Param Count Tests
# ═══════════════════════════════════════════════════════════════════════

class TestParamCount:
    def test_smoke_param_count(self):
        from ncalm.profiles import PROFILES
        model = AENCAv7Model(PROFILES["smoke_v7"])
        total = sum(p.numel() for p in model.parameters())
        assert total < 20_000_000, f"Smoke model too large: {total:,}"
        assert total > 100_000, f"Smoke model too small: {total:,}"

    def test_pile_250m_auto_size(self):
        from ncalm.profiles import PROFILES
        profile = PROFILES["pile_250m_v7"]
        model = AENCAv7Model(profile)
        total = sum(p.numel() for p in model.parameters())
        target = profile.get("target_params", 162_000_000)
        ratio = total / target
        assert 0.85 < ratio < 1.15, \
            f"Param count {total:,} not near target {target:,} (ratio={ratio:.2f})"


# ═══════════════════════════════════════════════════════════════════════
# 12. Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_batch_size_1(self, tiny_config):
        cfg_dict = {f.name: getattr(tiny_config, f.name)
                    for f in dataclasses.fields(tiny_config)}
        model = AENCAv7Model(cfg_dict)
        batch = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.seq_len))
        loss, info = model.forward_train(batch)
        assert torch.isfinite(loss)

    def test_short_sequence(self, tiny_config):
        cfg_dict = {f.name: getattr(tiny_config, f.name)
                    for f in dataclasses.fields(tiny_config)}
        model = AENCAv7Model(cfg_dict)
        batch = torch.randint(0, tiny_config.vocab_size, (2, 4))
        loss, info = model.forward_train(batch)
        assert torch.isfinite(loss)

    def test_deterministic_inference(self, tiny_model, tiny_config):
        batch = torch.randint(0, tiny_config.vocab_size, (1, 16))
        logits1 = tiny_model.get_logits(batch)
        logits2 = tiny_model.get_logits(batch)
        assert torch.allclose(logits1, logits2, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
