"""Comprehensive unit tests for AENCA V13: Morphogenetic Attention.

Tests cover:
  1. Configuration (auto-compute d_ff, validation, defaults)
  2. DepthwiseCausalConv (shape, causality, runtime dilation)
  3. MorphogenField (seed, evolve, gates, feedback, zero-init)
  4. ThetaAttention (shape, gate application, RoPE, causal mask, zero-init)
  5. HebbianFFN (shape, cross-modal gating, zero-init)
  6. AdaptiveHalting (mass conservation, KL loss)
  7. Full model forward_train (loss, info dict, shapes)
  8. Full model get_logits (inference shapes)
  9. Zero-init cascade (V13 = V7.1 behavior at init)
  10. Gradient engagement (loss decreases with optimization)
  11. Parameter groups (decay / no_decay split)
  12. Param count (matches target within tolerance for full-scale config)
  13. Extra metrics (monitoring outputs)
  14. Diversity loss (routing entropy computation)
"""

from __future__ import annotations

import dataclasses
import math
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, ".")

from ncalm.models.v13.config import V13Config
from ncalm.models.v13.morphogen import DepthwiseCausalConv, MorphogenField
from ncalm.models.v13.attention import ThetaAttention, RotaryPositionEmbedding
from ncalm.models.v13.hebbian_ffn import HebbianFFN
from ncalm.models.v13.halting import AdaptiveHalting
from ncalm.models.v13.model import AENCAv13Model


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def tiny_config() -> V13Config:
    """Minimal config for fast CPU tests."""
    return V13Config(
        d_model=64,
        d_h=16,
        n_rules=2,
        n_morph_steps=2,
        n_theta=2,
        n_heads=4,
        d_head=16,
        d_ff=128,
        vocab_size=200,
        max_seq_len=32,
        enable_halting=True,
        halt_eps=0.01,
        lambda_halt=0.01,
        tie_embeddings=True,
        mos_components=1,
        morph_dilations=(1, 2),
        morph_gate_bias=3.0,
        morph_conv_kernel=3,
        lambda_reg=0.01,
        lambda_aux=0.05,
        lambda_div=0.01,
        dropout=0.0,
        checkpoint_interval=0,
        target_params=50_000,
    )


@pytest.fixture
def tiny_config_dict(tiny_config: V13Config) -> dict:
    """Config as flat dict for model constructor."""
    return {f.name: getattr(tiny_config, f.name) for f in dataclasses.fields(tiny_config)}


@pytest.fixture
def tiny_model(tiny_config_dict: dict) -> AENCAv13Model:
    """Tiny V13 model for CPU testing."""
    return AENCAv13Model(tiny_config_dict)


@pytest.fixture
def tiny_batch(tiny_config: V13Config) -> torch.Tensor:
    """Random batch of token IDs."""
    return torch.randint(0, tiny_config.vocab_size, (2, tiny_config.max_seq_len))


@pytest.fixture
def tiny_morphogen(tiny_config: V13Config) -> MorphogenField:
    """Tiny MorphogenField for unit tests."""
    return MorphogenField(tiny_config)


@pytest.fixture
def tiny_attention(tiny_config: V13Config) -> ThetaAttention:
    """Tiny ThetaAttention for unit tests."""
    return ThetaAttention(tiny_config, block_idx=0)


@pytest.fixture
def tiny_ffn(tiny_config: V13Config) -> HebbianFFN:
    """Tiny HebbianFFN for unit tests."""
    return HebbianFFN(tiny_config.d_model, tiny_config.d_ff, tiny_config.dropout)


@pytest.fixture
def tiny_halting(tiny_config: V13Config) -> AdaptiveHalting:
    """Tiny AdaptiveHalting for unit tests."""
    return AdaptiveHalting(tiny_config)


# ═══════════════════════════════════════════════════════════════════════
# 1. Configuration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestV13Config:
    """Test V13Config dataclass and derived properties."""

    def test_defaults(self):
        cfg = V13Config()
        assert cfg.d_model == 768
        assert cfg.n_theta == 4
        assert cfg.d_h == 128
        assert cfg.n_rules == 4
        assert cfg.n_morph_steps == 4
        assert cfg.morph_dilations == (1, 2, 4, 8)
        assert cfg.morph_gate_bias == 3.0

    def test_total_steps(self, tiny_config):
        # n_theta * (n_morph_steps + 1)
        assert tiny_config.total_steps == 2 * (2 + 1)

    def test_K_alias(self, tiny_config):
        assert tiny_config.K == tiny_config.total_steps

    def test_auto_compute_d_ff(self):
        """Test d_ff auto-computation for full-scale config."""
        cfg = V13Config(
            d_model=768, d_h=128, n_rules=4, n_morph_steps=4,
            n_theta=4, n_heads=12, d_head=64, d_ff=0,
            vocab_size=50304, target_params=162_000_000,
            tie_embeddings=False,
        )
        # d_ff should be auto-computed to fill 162M target
        assert cfg.d_ff > 7000
        assert cfg.d_ff < 9000

    def test_validation_passes(self, tiny_config):
        tiny_config.validate()  # Should not raise

    def test_validation_fails_d_model_mismatch(self):
        with pytest.raises(AssertionError, match="d_model"):
            V13Config(d_model=100, n_heads=4, d_head=64, d_ff=128)

    def test_morph_dilations_list_converted(self):
        """morph_dilations should be converted from list to tuple."""
        cfg = V13Config(d_ff=128, morph_dilations=[1, 2, 3])
        assert isinstance(cfg.morph_dilations, tuple)
        assert cfg.morph_dilations == (1, 2, 3)


# ═══════════════════════════════════════════════════════════════════════
# 2. DepthwiseCausalConv Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDepthwiseCausalConv:
    """Test DepthwiseCausalConv shape, causality, and runtime dilation."""

    def test_output_shape(self):
        conv = DepthwiseCausalConv(channels=16, kernel_size=3)
        x = torch.randn(2, 10, 16)
        out = conv(x, dilation=1)
        assert out.shape == (2, 10, 16)

    def test_output_shape_with_dilation(self):
        conv = DepthwiseCausalConv(channels=16, kernel_size=3)
        x = torch.randn(2, 10, 16)
        out = conv(x, dilation=4)
        assert out.shape == (2, 10, 16)

    def test_causal_no_future_leak(self):
        """Position i should not depend on position j > i."""
        conv = DepthwiseCausalConv(channels=4, kernel_size=3)
        B, L, C = 1, 8, 4
        x = torch.randn(B, L, C, requires_grad=True)
        out = conv(x, dilation=1)

        # Check: output at position 3 should not depend on positions 4+
        out[0, 3].sum().backward()
        grad = x.grad[0]  # (L, C)

        # Positions 4, 5, 6, 7 should have zero gradient
        assert (grad[4:].abs() < 1e-7).all(), "Future positions have non-zero gradient"

    def test_causal_with_dilation(self):
        """Causality should hold even with large dilation."""
        conv = DepthwiseCausalConv(channels=4, kernel_size=3)
        B, L, C = 1, 16, 4
        x = torch.randn(B, L, C, requires_grad=True)
        out = conv(x, dilation=4)

        # Check causality at position 8
        out[0, 8].sum().backward()
        grad = x.grad[0]
        assert (grad[9:].abs() < 1e-7).all(), "Future positions have non-zero gradient with dilation"

    def test_different_dilations_same_weight(self):
        """Different dilations should use the same weight tensor."""
        conv = DepthwiseCausalConv(channels=4, kernel_size=3)
        x = torch.randn(2, 10, 4)
        out1 = conv(x, dilation=1)
        out2 = conv(x, dilation=2)
        # Outputs should differ (different effective receptive field)
        assert not torch.allclose(out1, out2, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# 3. MorphogenField Tests
# ═══════════════════════════════════════════════════════════════════════

class TestMorphogenField:
    """Test MorphogenField: seed, evolve, gates, feedback, zero-init."""

    def test_seed_shape(self, tiny_config, tiny_morphogen):
        z = torch.randn(2, 8, tiny_config.d_model)
        h = tiny_morphogen.seed(z)
        assert h.shape == (2, 8, tiny_config.d_h)

    def test_seed_bounded(self, tiny_config, tiny_morphogen):
        """Seed uses tanh, so values should be in [-1, 1]."""
        z = torch.randn(2, 8, tiny_config.d_model)
        h = tiny_morphogen.seed(z)
        assert h.min() >= -1.0
        assert h.max() <= 1.0

    def test_evolve_shape(self, tiny_config, tiny_morphogen):
        h = torch.randn(2, 8, tiny_config.d_h)
        h_out, routing = tiny_morphogen.evolve(h, n_steps=2)
        assert h_out.shape == (2, 8, tiny_config.d_h)
        assert len(routing) == 2  # one per morph step
        assert routing[0].shape == (2, 8, tiny_config.n_rules)

    def test_evolve_routing_sums_to_one(self, tiny_config, tiny_morphogen):
        h = torch.randn(2, 8, tiny_config.d_h)
        _, routing = tiny_morphogen.evolve(h, n_steps=2)
        for w in routing:
            sums = w.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_generate_gates_shape(self, tiny_config, tiny_morphogen):
        h = torch.randn(2, 8, tiny_config.d_h)
        q_gate, k_gate = tiny_morphogen.generate_gates(h)
        assert q_gate.shape == (2, 8, tiny_config.d_model)
        assert k_gate.shape == (2, 8, tiny_config.d_model)

    def test_generate_gates_range(self, tiny_config, tiny_morphogen):
        """Gates = 1 + tanh(...) should be in [0, 2]."""
        h = torch.randn(2, 8, tiny_config.d_h)
        q_gate, k_gate = tiny_morphogen.generate_gates(h)
        assert q_gate.min() >= 0.0 - 1e-6
        assert q_gate.max() <= 2.0 + 1e-6
        assert k_gate.min() >= 0.0 - 1e-6
        assert k_gate.max() <= 2.0 + 1e-6

    def test_generate_gates_at_init_equal_one(self, tiny_config, tiny_morphogen):
        """At init, W_Q_h=0, W_K_h=0 → gates = 1 + tanh(0) = 1.0."""
        h = torch.randn(2, 8, tiny_config.d_h)
        q_gate, k_gate = tiny_morphogen.generate_gates(h)
        assert torch.allclose(q_gate, torch.ones_like(q_gate), atol=1e-5)
        assert torch.allclose(k_gate, torch.ones_like(k_gate), atol=1e-5)

    def test_feedback_shape(self, tiny_config, tiny_morphogen):
        h = torch.randn(2, 8, tiny_config.d_h)
        z = torch.randn(2, 8, tiny_config.d_model)
        h_out = tiny_morphogen.update_from_content(h, z)
        assert h_out.shape == (2, 8, tiny_config.d_h)

    def test_feedback_noop_at_init(self, tiny_config, tiny_morphogen):
        """W_h_feedback=0 → feedback is zero at init."""
        h = torch.randn(2, 8, tiny_config.d_h)
        z = torch.randn(2, 8, tiny_config.d_model)
        h_out = tiny_morphogen.update_from_content(h, z)
        assert torch.allclose(h_out, h, atol=1e-5)

    def test_evolve_noop_at_init(self, tiny_config, tiny_morphogen):
        """W_h2=0 → delta_h = 0 at init; morphogen stays unchanged."""
        h = torch.randn(2, 8, tiny_config.d_h)
        h_out, _ = tiny_morphogen.evolve(h, n_steps=2)
        assert torch.allclose(h_out, h, atol=1e-5)

    def test_router_zero_init(self, tiny_morphogen):
        """W_router should be zero-initialized → uniform routing."""
        assert (tiny_morphogen.W_router.weight.abs() < 1e-8).all()


# ═══════════════════════════════════════════════════════════════════════
# 4. ThetaAttention Tests
# ═══════════════════════════════════════════════════════════════════════

class TestThetaAttention:
    """Test ThetaAttention shape, causal mask, and zero-init."""

    def test_output_shape(self, tiny_config, tiny_attention):
        z = torch.randn(2, 8, tiny_config.d_model)
        q_gate = torch.ones(2, 8, tiny_config.d_model)
        k_gate = torch.ones(2, 8, tiny_config.d_model)
        out = tiny_attention(z, q_gate, k_gate)
        assert out.shape == z.shape

    def test_zero_init_out_proj(self, tiny_attention):
        """out_proj should be zero-initialized."""
        assert (tiny_attention.out_proj.weight.abs() < 1e-8).all()

    def test_identity_at_init(self, tiny_config, tiny_attention):
        """At init, out_proj=0 → attention output is zero → input unchanged."""
        z = torch.randn(2, 8, tiny_config.d_model)
        q_gate = torch.ones(2, 8, tiny_config.d_model)
        k_gate = torch.ones(2, 8, tiny_config.d_model)
        out = tiny_attention(z, q_gate, k_gate)
        assert torch.allclose(out, z, atol=1e-5)

    def test_gate_modulates_output(self, tiny_config):
        """Non-trivial gates should change the output (after training)."""
        attn = ThetaAttention(tiny_config, block_idx=0)
        # Manually set out_proj to non-zero to see gate effect
        nn.init.xavier_normal_(attn.out_proj.weight)
        z = torch.randn(2, 8, tiny_config.d_model)
        q_gate_ones = torch.ones(2, 8, tiny_config.d_model)
        q_gate_half = 0.5 * torch.ones(2, 8, tiny_config.d_model)
        k_gate = torch.ones(2, 8, tiny_config.d_model)

        out1 = attn(z, q_gate_ones, k_gate)
        out2 = attn(z, q_gate_half, k_gate)
        assert not torch.allclose(out1, out2, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# 5. HebbianFFN Tests
# ═══════════════════════════════════════════════════════════════════════

class TestHebbianFFN:
    """Test HebbianFFN shape, cross-modal gating, and zero-init."""

    def test_output_shape(self, tiny_config, tiny_ffn):
        z_pre = torch.randn(2, 8, tiny_config.d_model)
        z_post = torch.randn(2, 8, tiny_config.d_model)
        out = tiny_ffn(z_pre, z_post)
        assert out.shape == (2, 8, tiny_config.d_model)

    def test_zero_init_decoder(self, tiny_ffn):
        """W_dec should be zero-initialized."""
        assert (tiny_ffn.W_dec.weight.abs() < 1e-8).all()

    def test_zero_output_at_init(self, tiny_config, tiny_ffn):
        """At init, W_dec=0 → output is zero."""
        z_pre = torch.randn(2, 8, tiny_config.d_model)
        z_post = torch.randn(2, 8, tiny_config.d_model)
        out = tiny_ffn(z_pre, z_post)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-8)


# ═══════════════════════════════════════════════════════════════════════
# 6. AdaptiveHalting Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAdaptiveHalting:
    """Test AdaptiveHalting mass conservation and KL loss."""

    def test_halt_prob_shape(self, tiny_config, tiny_halting):
        z = torch.randn(2, 8, tiny_config.d_model)
        hp = tiny_halting.compute_halt_prob(z)
        assert hp.shape == (2, 8)
        assert (hp >= 0).all() and (hp <= 1).all()

    def test_mass_conservation(self, tiny_config, tiny_halting):
        """Total mass consumed over all theta cycles should be ~1.0."""
        B, L, D = 2, 8, tiny_config.d_model
        z = torch.randn(B, L, D)
        z_out = torch.zeros_like(z)
        remaining = torch.ones(B, L)

        for i in range(tiny_config.n_theta):
            is_last = (i == tiny_config.n_theta - 1)
            z_out, remaining, _ = tiny_halting.accumulate(z, z_out, remaining, is_last)

        # After all cycles, remaining should be ~0
        assert (remaining.abs() < 0.01).all()

    def test_kl_loss_scalar(self, tiny_config, tiny_halting):
        halt_probs = [torch.rand(2, 8) for _ in range(tiny_config.n_theta)]
        kl = tiny_halting.compute_kl_loss(halt_probs)
        assert kl.dim() == 0  # scalar
        assert kl.item() >= 0


# ═══════════════════════════════════════════════════════════════════════
# 7. Full Model Forward Train Tests
# ═══════════════════════════════════════════════════════════════════════

class TestV13ModelForwardTrain:
    """Test full model forward_train: loss, info dict, shapes."""

    def test_forward_returns_loss_and_info(self, tiny_model, tiny_batch):
        loss, info = tiny_model.forward_train(tiny_batch)
        assert loss.dim() == 0  # scalar
        assert "ce" in info
        assert isinstance(info["ce"], torch.Tensor)

    def test_forward_loss_is_finite(self, tiny_model, tiny_batch):
        loss, _ = tiny_model.forward_train(tiny_batch)
        assert torch.isfinite(loss)

    def test_forward_ce_positive(self, tiny_model, tiny_batch):
        _, info = tiny_model.forward_train(tiny_batch)
        assert info["ce"].item() > 0

    def test_forward_backward(self, tiny_model, tiny_batch):
        """Loss should be backpropagable through all parameters."""
        loss, _ = tiny_model.forward_train(tiny_batch)
        loss.backward()

        # At least some parameters should have gradients
        has_grad = sum(1 for p in tiny_model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        total = sum(1 for p in tiny_model.parameters() if p.requires_grad)
        assert has_grad > 0, "No parameters received gradients"

    def test_info_has_expected_keys(self, tiny_model, tiny_batch):
        _, info = tiny_model.forward_train(tiny_batch)
        assert "ce" in info
        assert "expected_steps" in info
        assert "pct_halted_early" in info


# ═══════════════════════════════════════════════════════════════════════
# 8. Full Model Inference Tests
# ═══════════════════════════════════════════════════════════════════════

class TestV13ModelInference:
    """Test get_logits for inference."""

    def test_logits_shape(self, tiny_model, tiny_config):
        tokens = torch.randint(0, tiny_config.vocab_size, (2, 16))
        logits = tiny_model.get_logits(tokens)
        assert logits.shape == (2, 16, tiny_config.vocab_size)

    def test_logits_finite(self, tiny_model, tiny_config):
        tokens = torch.randint(0, tiny_config.vocab_size, (2, 16))
        logits = tiny_model.get_logits(tokens)
        assert torch.isfinite(logits).all()


# ═══════════════════════════════════════════════════════════════════════
# 9. Zero-Init Cascade
# ═══════════════════════════════════════════════════════════════════════

class TestZeroInitCascade:
    """V13 should behave like identity at initialization."""

    def test_morph_rules_zero(self, tiny_morphogen):
        """All W_h2 should be zero."""
        for k in range(len(tiny_morphogen.W_h2)):
            assert (tiny_morphogen.W_h2[k].weight.abs() < 1e-8).all()

    def test_q_k_gate_weights_zero(self, tiny_morphogen):
        """W_Q_h and W_K_h should be zero."""
        assert (tiny_morphogen.W_Q_h.weight.abs() < 1e-8).all()
        assert (tiny_morphogen.W_K_h.weight.abs() < 1e-8).all()

    def test_feedback_zero(self, tiny_morphogen):
        """W_h_feedback should be zero."""
        assert (tiny_morphogen.W_h_feedback.weight.abs() < 1e-8).all()

    def test_router_zero(self, tiny_morphogen):
        """W_router should be zero."""
        assert (tiny_morphogen.W_router.weight.abs() < 1e-8).all()

    def test_out_proj_zero(self, tiny_model):
        """All theta attention out_proj should be zero."""
        for attn in tiny_model.theta_attns:
            assert (attn.out_proj.weight.abs() < 1e-8).all()

    def test_ffn_dec_zero(self, tiny_model):
        """All theta FFN decoders should be zero."""
        for ffn in tiny_model.theta_ffns:
            assert (ffn.W_dec.weight.abs() < 1e-8).all()

    def test_gate_bias_positive(self, tiny_model):
        """Bilinear gate bias should be +3.0."""
        assert torch.allclose(
            tiny_model.morphogen.gate_bias,
            torch.full_like(tiny_model.morphogen.gate_bias, 3.0),
        )


# ═══════════════════════════════════════════════════════════════════════
# 10. Gradient Engagement
# ═══════════════════════════════════════════════════════════════════════

class TestGradientEngagement:
    """Loss should decrease with a few optimization steps."""

    def test_loss_decreases(self, tiny_model, tiny_batch):
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            loss, _ = tiny_model.forward_train(tiny_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease (not strictly monotonic, but overall trend)
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses}"


# ═══════════════════════════════════════════════════════════════════════
# 11. Parameter Groups
# ═══════════════════════════════════════════════════════════════════════

class TestParamGroups:
    """Test parameter group split for optimizer."""

    def test_groups_cover_all_params(self, tiny_model):
        groups = tiny_model.get_param_groups()
        grouped_params = set()
        for g in groups:
            for p in g["params"]:
                grouped_params.add(id(p))

        all_params = set(id(p) for p in tiny_model.parameters() if p.requires_grad)
        assert grouped_params == all_params

    def test_no_decay_has_weight_decay_zero(self, tiny_model):
        groups = tiny_model.get_param_groups()
        no_decay_group = [g for g in groups if g["label"] == "no_decay"]
        assert len(no_decay_group) == 1
        assert no_decay_group[0]["weight_decay"] == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 12. Full-Scale Param Count
# ═══════════════════════════════════════════════════════════════════════

class TestParamCount:
    """Test that auto-computed d_ff gives us ~162M params."""

    def test_full_scale_param_count(self):
        cfg_dict = {
            "d_model": 768,
            "d_h": 128,
            "n_rules": 4,
            "n_morph_steps": 4,
            "n_theta": 4,
            "n_heads": 12,
            "d_head": 64,
            "d_ff": 0,  # auto-compute
            "vocab_size": 50304,
            "max_seq_len": 1024,
            "target_params": 162_000_000,
            "tie_embeddings": False,
            "mos_components": 1,
            "enable_halting": True,
        }
        model = AENCAv13Model(cfg_dict)
        total = sum(p.numel() for p in model.parameters())

        # Should be within 1% of 162M
        assert abs(total - 162_000_000) / 162_000_000 < 0.01, (
            f"Total params {total:,} not within 1% of 162M"
        )


# ═══════════════════════════════════════════════════════════════════════
# 13. Extra Metrics
# ═══════════════════════════════════════════════════════════════════════

class TestExtraMetrics:
    """Test extra_metrics returns expected monitoring data."""

    def test_extra_metrics_keys(self, tiny_model):
        metrics = tiny_model.extra_metrics()
        assert "morph_Wqh_norm" in metrics
        assert "morph_Wkh_norm" in metrics
        assert "morph_router_norm" in metrics
        assert "morph_feedback_norm" in metrics
        assert "halt_bias" in metrics

    def test_extra_metrics_per_rule(self, tiny_model, tiny_config):
        metrics = tiny_model.extra_metrics()
        for k in range(tiny_config.n_rules):
            assert f"morph_rule{k}_norm" in metrics

    def test_extra_metrics_per_theta(self, tiny_model, tiny_config):
        metrics = tiny_model.extra_metrics()
        for i in range(tiny_config.n_theta):
            assert f"theta{i}_attn_norm" in metrics
            assert f"theta{i}_ffn_dec_norm" in metrics

    def test_zero_init_metrics_are_zero(self, tiny_model):
        """At init, zero-init weights should have norm ~0."""
        metrics = tiny_model.extra_metrics()
        assert metrics["morph_Wqh_norm"] < 1e-6
        assert metrics["morph_Wkh_norm"] < 1e-6
        assert metrics["morph_router_norm"] < 1e-6
        assert metrics["morph_feedback_norm"] < 1e-6


# ═══════════════════════════════════════════════════════════════════════
# 14. Describe
# ═══════════════════════════════════════════════════════════════════════

class TestDescribe:
    """Test describe() method."""

    def test_describe_keys(self, tiny_model):
        desc = tiny_model.describe()
        assert desc["architecture"] == "AENCA v13 (Morphogenetic Attention)"
        assert "total_params" in desc
        assert "morphogen_params" in desc
        assert "d_ff" in desc

    def test_describe_total_params(self, tiny_model):
        desc = tiny_model.describe()
        total = sum(p.numel() for p in tiny_model.parameters())
        assert desc["total_params"] == total


# ═══════════════════════════════════════════════════════════════════════
# 15. Model Registration
# ═══════════════════════════════════════════════════════════════════════

class TestModelRegistration:
    """Test model is registered and discoverable."""

    def test_registered(self):
        from ncalm.model import get_model_class
        cls = get_model_class("v13")
        assert cls is AENCAv13Model

    def test_profile_exists(self):
        from ncalm.profiles import PROFILES
        assert "smoke_v13" in PROFILES
        assert "pile_250m_v13" in PROFILES
        assert "pile_2b_v13" in PROFILES


# ═══════════════════════════════════════════════════════════════════════
# 16. Diversity Loss Sign
# ═══════════════════════════════════════════════════════════════════════

class TestDiversityLossSign:
    """Verify diversity loss correctly encourages uniform routing."""

    def test_uniform_routing_has_lower_loss_than_collapsed(self):
        """Uniform routing should produce LOWER div_loss than collapsed.

        div_loss = Σ w·log(w+ε) (negative entropy, always ≤ 0).
        Uniform: most negative. Collapsed: near 0.
        Since lambda_div > 0, minimizing total_loss minimizes div_loss
        → prefers uniform (more negative) → encourages diversity.
        """
        K = 4
        # Uniform routing: w = [0.25, 0.25, 0.25, 0.25]
        w_uniform = torch.full((1, 1, K), 1.0 / K)
        div_uniform = (w_uniform * (w_uniform + 1e-8).log()).sum(dim=-1).mean()

        # Collapsed routing: w = [1, 0, 0, 0]
        w_collapsed = torch.zeros(1, 1, K)
        w_collapsed[0, 0, 0] = 1.0
        div_collapsed = (w_collapsed * (w_collapsed + 1e-8).log()).sum(dim=-1).mean()

        # Uniform MUST have lower (more negative) div_loss than collapsed
        assert div_uniform < div_collapsed, (
            f"Diversity loss sign error: uniform={div_uniform:.4f} should be < "
            f"collapsed={div_collapsed:.4f}"
        )
        # Uniform should be negative, collapsed should be near 0
        assert div_uniform < 0, f"Uniform div_loss should be negative, got {div_uniform:.4f}"
        assert div_collapsed > -0.01, f"Collapsed div_loss should be ~0, got {div_collapsed:.4f}"

    def test_diversity_loss_in_forward_train(self, tiny_model, tiny_batch):
        """The 'div' key in info should be negative (negative entropy)."""
        _, info = tiny_model.forward_train(tiny_batch)
        if "div" in info:
            assert info["div"].item() <= 0, (
                f"Diversity loss should be ≤ 0 (negative entropy), got {info['div'].item()}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
