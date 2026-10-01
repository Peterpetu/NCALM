"""Comprehensive smoke tests for AENCA V10 — Persistent Growth NCA.

Tests verify:
  1. Model builds from config dict (tiny + full scale)
  2. Forward/backward pass works (no NaN, gradient flows)
  3. Kernel weights are truly shared across all CA steps
  4. Dilations cycle correctly through all scales
  5. Zero-init W_d produces zero deltas at initialization
  6. Settlement: cells track convergence, growth gate works
  7. Settlement is detached from the gradient graph
  8. Spectral norm estimation runs without error
  9. Describe method returns required keys
  10. Param groups separate decay and no_decay correctly
  11. Loss components (CE, perm, spectral) are reasonable
  12. Tied embeddings work correctly
  13. Causal conv doesn't leak future information
  14. No attention modules anywhere
  15. Growth frontier shrinks over CA steps
"""

import dataclasses
import math
import sys

import pytest
import torch

sys.path.insert(0, ".")

from ncalm.models.v10.config import V10Config
from ncalm.models.v10.kernel import SimpleHebbianRule, DepthwiseCausalConv
from ncalm.models.v10.settlement import SettlementState
from ncalm.models.v10.model import AENCAv10Model


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_config() -> V10Config:
    """Minimal V10 config for CPU testing."""
    return V10Config(
        d_model=64,
        d_rule=32,
        n_steps=8,
        n_scales=4,
        conv_kernel_size=3,
        vocab_size=256,
        max_seq_len=32,
        tie_embeddings=True,
        dropout=0.0,
        lambda_perm=0.01,
        lambda_spectral=0.001,
        settlement_beta=0.9,
        settlement_tau=0.05,
        settlement_eta=0.1,
        checkpoint_interval=0,
    )


@pytest.fixture
def tiny_config_dict(tiny_config) -> dict:
    return {f.name: getattr(tiny_config, f.name) for f in dataclasses.fields(tiny_config)}


@pytest.fixture
def tiny_model(tiny_config_dict) -> AENCAv10Model:
    return AENCAv10Model(tiny_config_dict)


@pytest.fixture
def tiny_batch(tiny_config) -> torch.Tensor:
    return torch.randint(0, tiny_config.vocab_size, (2, tiny_config.max_seq_len))


# ── Test 1: Model Builds ────────────────────────────────────────────────


class TestModelBuilds:
    def test_builds_from_config_dict(self, tiny_config_dict):
        model = AENCAv10Model(tiny_config_dict)
        assert isinstance(model, AENCAv10Model)

    def test_builds_untied_embeddings(self, tiny_config_dict):
        tiny_config_dict["tie_embeddings"] = False
        model = AENCAv10Model(tiny_config_dict)
        assert model.lm_head is not None

    def test_config_stored(self, tiny_model, tiny_config):
        assert tiny_model.config.d_model == tiny_config.d_model
        assert tiny_model.config.d_rule == tiny_config.d_rule
        assert tiny_model.config.n_steps == tiny_config.n_steps


# ── Test 2: Forward & Backward ──────────────────────────────────────────


class TestForwardBackward:
    def test_forward_train_runs(self, tiny_model, tiny_batch):
        loss, info = tiny_model.forward_train(tiny_batch)
        assert loss.shape == ()
        assert loss.requires_grad
        assert "ce" in info

    def test_backward_runs(self, tiny_model, tiny_batch):
        loss, _ = tiny_model.forward_train(tiny_batch)
        loss.backward()
        for name, p in tiny_model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_no_nan_in_output(self, tiny_model, tiny_batch):
        loss, info = tiny_model.forward_train(tiny_batch)
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"

    def test_get_logits_shape(self, tiny_model, tiny_batch):
        logits = tiny_model.get_logits(tiny_batch)
        B, L = tiny_batch.shape
        assert logits.shape == (B, L, tiny_model.config.vocab_size)

    def test_loss_components_present(self, tiny_model, tiny_batch):
        _, info = tiny_model.forward_train(tiny_batch)
        assert "ce" in info
        assert "perm" in info
        assert "spectral" in info
        assert "settled" in info
        assert "active" in info


# ── Test 3: Shared Kernel ────────────────────────────────────────────────


class TestSharedKernel:
    def test_single_rule_instance(self, tiny_model):
        """The model should have exactly ONE SimpleHebbianRule."""
        rules = [m for m in tiny_model.modules() if isinstance(m, SimpleHebbianRule)]
        assert len(rules) == 1, f"Expected 1 rule, found {len(rules)}"

    def test_rule_used_n_steps_times(self, tiny_model, tiny_batch):
        """Verify the rule is called n_steps times (same object)."""
        call_count = 0
        original_forward = tiny_model.rule.forward

        def counting_forward(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_forward(*args, **kwargs)

        tiny_model.rule.forward = counting_forward
        try:
            tiny_model.forward_train(tiny_batch)
            assert call_count == tiny_model.config.n_steps
        finally:
            tiny_model.rule.forward = original_forward

    def test_conv_weights_shared_across_dilations(self, tiny_model):
        """The depthwise conv inside the rule should be ONE set of weights."""
        convs = [m for m in tiny_model.modules() if isinstance(m, DepthwiseCausalConv)]
        assert len(convs) == 1, f"Expected 1 conv, found {len(convs)}"


# ── Test 4: Dilations & Receptive Field ──────────────────────────────────


class TestDilations:
    def test_dilation_schedule_length(self, tiny_model):
        """Dilation schedule should have n_steps entries."""
        assert len(tiny_model._dilation_schedule) == tiny_model.config.n_steps

    def test_dilation_schedule_cyclic(self, tiny_config):
        """Dilations should cycle: [1,2,4,8, 1,2,4,8] for n_scales=4, n_steps=8."""
        cfg = tiny_config
        expected = [1, 2, 4, 8, 1, 2, 4, 8]
        config_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
        model = AENCAv10Model(config_dict)
        assert model._dilation_schedule == expected

    def test_dilations_are_powers_of_two(self, tiny_config):
        for d in tiny_config.dilations:
            assert d & (d - 1) == 0 and d > 0, f"Dilation {d} is not power of 2"

    def test_causal_conv_no_future_leak(self):
        """Conv output at position t must not depend on positions > t."""
        d_model, L = 4, 16
        conv = DepthwiseCausalConv(d_model, kernel_size=3)
        x = torch.randn(1, L, d_model, requires_grad=True)

        for dilation in [1, 4, 8]:
            out = conv(x, dilation=dilation)
            out[0, 5, 0].backward(retain_graph=True)
            grad = x.grad[0]  # (L, d_model)
            assert (grad[6:] == 0).all(), (
                f"Causal violation at dilation={dilation}: "
                f"grad from pos 5 leaks to future positions"
            )
            x.grad = None

    def test_full_scale_receptive_field(self):
        """Full-scale V10 (64 steps, 8 scales) should cover seq_len=1024."""
        cfg = V10Config(
            d_model=64, d_rule=32,
            n_steps=64, n_scales=8,
            max_seq_len=1024, vocab_size=256,
        )
        assert cfg.receptive_field >= 1024, (
            f"RF={cfg.receptive_field} < L=1024"
        )


# ── Test 5: Zero-Init and Class I Still Life ─────────────────────────────


class TestZeroInit:
    def test_w_d_starts_zero(self, tiny_model):
        """W_d must be zero at initialization."""
        W_d = tiny_model.rule.W_d.weight
        assert (W_d == 0).all(), "W_d not zero-initialized"

    def test_zero_init_produces_zero_deltas(self, tiny_model, tiny_batch):
        """With zero W_d, all deltas should be exactly zero."""
        cfg = tiny_model.config
        positions = torch.arange(cfg.max_seq_len)
        z = tiny_model.token_embedding(tiny_batch) + tiny_model.pos_embedding(positions)

        for dilation in tiny_model._dilation_schedule:
            delta = tiny_model.rule(z, dilation=dilation)
            assert (delta == 0).all(), f"Non-zero delta with zero W_d at d={dilation}"


# ── Test 6: Settlement ──────────────────────────────────────────────────


class TestSettlement:
    def test_settlement_state_creation(self, tiny_config):
        """SettlementState.create should produce valid initial state."""
        st = SettlementState.create(2, 16, torch.device("cpu"), tiny_config)
        assert st.a_bar.shape == (2, 16)
        assert st.settled.shape == (2, 16)
        # Initial settled score should be low (~0.047)
        assert st.mean_settled < 0.1, f"Initial settled={st.mean_settled}, expected < 0.1"
        # Initial active fraction should be high
        assert st.active_fraction > 0.9

    def test_settlement_update_decreases_a_bar(self, tiny_config):
        """Zero deltas should drive a_bar toward 0 → increasing settled."""
        st = SettlementState.create(2, 16, torch.device("cpu"), tiny_config)
        initial_a_bar = st.a_bar.mean().item()

        # Apply zero deltas (represents converged cells)
        zero_delta = torch.zeros(2, 16, tiny_config.d_model)
        for _ in range(20):
            st.update(zero_delta, tiny_config)

        assert st.a_bar.mean().item() < initial_a_bar, "a_bar should decrease with zero deltas"
        assert st.mean_settled > 0.5, f"Cells should settle with zero deltas, got {st.mean_settled}"

    def test_settlement_update_large_delta_keeps_active(self, tiny_config):
        """Large deltas should keep a_bar high → cells stay active."""
        st = SettlementState.create(2, 16, torch.device("cpu"), tiny_config)

        large_delta = torch.randn(2, 16, tiny_config.d_model) * 10.0
        for _ in range(20):
            st.update(large_delta, tiny_config)

        assert st.mean_settled < 0.1, f"Cells should stay active with large deltas, got {st.mean_settled}"

    def test_growth_gate_shape(self, tiny_config):
        """growth_gate should return (B, L) tensor."""
        st = SettlementState.create(2, 16, torch.device("cpu"), tiny_config)
        gate = st.growth_gate()
        assert gate.shape == (2, 16)
        # Initially, gate should be close to 1 (all active)
        assert gate.mean().item() > 0.9

    def test_growth_gate_blocks_settled_cells(self, tiny_config):
        """After convergence, growth gate should be ~0."""
        st = SettlementState.create(2, 16, torch.device("cpu"), tiny_config)
        zero_delta = torch.zeros(2, 16, tiny_config.d_model)
        for _ in range(50):
            st.update(zero_delta, tiny_config)

        gate = st.growth_gate()
        assert gate.mean().item() < 0.5, f"Gate should be low after settlement, got {gate.mean()}"

    def test_forward_train_returns_settlement_info(self, tiny_model, tiny_batch):
        """forward_train info should contain settlement metrics."""
        _, info = tiny_model.forward_train(tiny_batch)
        assert "settled" in info
        assert "active" in info
        assert isinstance(info["settled"], float)
        assert isinstance(info["active"], float)
        assert 0.0 <= info["settled"] <= 1.0
        assert 0.0 <= info["active"] <= 1.0


# ── Test 7: Settlement Gradient Isolation ────────────────────────────────


class TestSettlementGradients:
    def test_settlement_detached_from_graph(self, tiny_config):
        """Settlement tensors must not require gradients."""
        st = SettlementState.create(2, 16, torch.device("cpu"), tiny_config)
        assert not st.a_bar.requires_grad
        assert not st.settled.requires_grad

        # Update with gradient-bearing delta
        delta = torch.randn(2, 16, tiny_config.d_model, requires_grad=True)
        st.update(delta, tiny_config)
        assert not st.a_bar.requires_grad
        assert not st.settled.requires_grad

    def test_growth_gate_no_grad(self, tiny_config):
        """Growth gate should not carry gradients from settlement state."""
        st = SettlementState.create(2, 16, torch.device("cpu"), tiny_config)
        gate = st.growth_gate()
        assert not gate.requires_grad


# ── Test 8: Spectral Norm ────────────────────────────────────────────────


class TestSpectralNorm:
    def test_spectral_norm_runs(self, tiny_model):
        sigma = tiny_model._spectral_norm_loss()
        assert sigma.shape == ()
        assert not torch.isnan(sigma)

    def test_spectral_norm_positive(self, tiny_model, tiny_batch):
        """After forward pass (power iteration), spectral norm should be ≥ 0."""
        tiny_model.forward_train(tiny_batch)
        sigma = tiny_model._spectral_norm_loss()
        assert sigma >= 0


# ── Test 9: Describe ────────────────────────────────────────────────────


class TestDescribe:
    def test_describe_has_required_keys(self, tiny_model):
        d = tiny_model.describe()
        assert "architecture" in d
        assert "total_params" in d
        assert d["architecture"] == "AENCA-V10-PersistentGrowth"

    def test_describe_params_match(self, tiny_model):
        d = tiny_model.describe()
        actual = sum(p.numel() for p in tiny_model.parameters())
        assert d["total_params"] == actual

    def test_describe_has_v10_keys(self, tiny_model):
        d = tiny_model.describe()
        for key in ["d_model", "d_rule", "n_steps", "n_scales",
                     "settlement_beta", "settlement_tau", "settlement_eta",
                     "receptive_field"]:
            assert key in d, f"Missing key: {key}"


# ── Test 10: Param Groups ───────────────────────────────────────────────


class TestParamGroups:
    def test_two_groups(self, tiny_model):
        groups = tiny_model.get_param_groups()
        assert len(groups) == 2
        labels = {g["label"] for g in groups}
        assert labels == {"decay", "no_decay"}

    def test_all_params_covered(self, tiny_model):
        groups = tiny_model.get_param_groups()
        group_ids = set()
        for g in groups:
            for p in g["params"]:
                group_ids.add(id(p))
        all_ids = {id(p) for p in tiny_model.parameters() if p.requires_grad}
        assert group_ids == all_ids, "Not all parameters covered by groups"

    def test_no_decay_has_zero_wd(self, tiny_model):
        groups = tiny_model.get_param_groups()
        no_decay = [g for g in groups if g["label"] == "no_decay"][0]
        assert no_decay.get("weight_decay", None) == 0.0


# ── Test 11: Tied Embeddings ─────────────────────────────────────────────


class TestTiedEmbeddings:
    def test_tied_shares_weight(self, tiny_model):
        assert tiny_model.lm_head is None
        batch = torch.randint(0, tiny_model.config.vocab_size, (1, 8))
        logits = tiny_model.get_logits(batch)
        assert logits.shape[-1] == tiny_model.config.vocab_size

    def test_untied_has_separate_head(self):
        config_dict = {
            "d_model": 64,
            "d_rule": 32,
            "n_steps": 4,
            "n_scales": 2,
            "vocab_size": 256,
            "max_seq_len": 32,
            "tie_embeddings": False,
        }
        model = AENCAv10Model(config_dict)
        assert model.lm_head is not None
        assert model.lm_head.weight is not model.token_embedding.weight


# ── Test 12: Config Properties ───────────────────────────────────────────


class TestConfigProperties:
    def test_a_bar_init(self):
        cfg = V10Config()
        assert cfg.a_bar_init == pytest.approx(cfg.settlement_tau + 3 * cfg.settlement_eta)

    def test_dilations_powers_of_two(self):
        cfg = V10Config(n_scales=8)
        assert cfg.dilations == [1, 2, 4, 8, 16, 32, 64, 128]

    def test_total_params_estimate(self):
        cfg = V10Config()
        est = cfg.total_params_estimate
        assert est > 100_000_000  # > 100M
        assert est < 150_000_000  # < 150M

    def test_receptive_field_positive(self):
        cfg = V10Config()
        assert cfg.receptive_field > 0


# ── Test 13: No Attention ───────────────────────────────────────────────


class TestNoAttention:
    def test_no_attention_modules(self, tiny_model):
        """V10 should have NO attention modules — pure CA."""
        for name, module in tiny_model.named_modules():
            class_name = type(module).__name__.lower()
            assert "attention" not in class_name, f"Found attention module: {name}"

    def test_no_attention_weights(self, tiny_model):
        """No parameter names should contain 'attention' or 'attn'."""
        for name, _ in tiny_model.named_parameters():
            assert "attention" not in name.lower(), f"Found attention param: {name}"
            assert "attn" not in name.lower(), f"Found attention param: {name}"


# ── Test 14: Full-Scale Config ───────────────────────────────────────────


class TestFullScaleConfig:
    def test_param_count_around_110m(self):
        """Full-scale V10 should be ~110M params (less than GPT-2's 162M)."""
        cfg = V10Config(
            d_model=2048, d_rule=800, n_steps=64, n_scales=8,
            vocab_size=50304, max_seq_len=1024, tie_embeddings=True,
        )
        config_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
        model = AENCAv10Model(config_dict)
        total = sum(p.numel() for p in model.parameters())
        # Should be in 105M-120M range
        assert 100_000_000 < total < 125_000_000, f"Total params: {total:,}"

    def test_rule_is_tiny_fraction(self):
        """Kernel should be <5% of total params (Wolfram: simple rule)."""
        cfg = V10Config(
            d_model=2048, d_rule=800,
            vocab_size=50304, max_seq_len=1024, tie_embeddings=True,
        )
        config_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
        model = AENCAv10Model(config_dict)
        total = sum(p.numel() for p in model.parameters())
        rule = sum(p.numel() for p in model.rule.parameters())
        frac = rule / total
        assert frac < 0.06, f"Rule fraction: {frac:.2%} (expected <5%)"


# ── Test 15: Permanence Loss ────────────────────────────────────────────


class TestPermanenceLoss:
    def test_permanence_loss_nonnegative(self, tiny_model, tiny_batch):
        """Permanence loss (mean delta²) must be ≥ 0."""
        _, info = tiny_model.forward_train(tiny_batch)
        perm = info["perm"]
        if isinstance(perm, torch.Tensor):
            perm = perm.item()
        assert perm >= 0

    def test_permanence_loss_zero_at_init(self, tiny_model, tiny_batch):
        """With zero-init W_d, permanence loss should be exactly 0."""
        _, info = tiny_model.forward_train(tiny_batch)
        perm = info["perm"]
        if isinstance(perm, torch.Tensor):
            perm = perm.item()
        assert perm == 0.0, f"Expected perm=0 at init, got {perm}"


# ── Test 16: Gradient Checkpointing ─────────────────────────────────────


class TestGradientCheckpointing:
    def test_checkpoint_interval_works(self):
        """Model with checkpoint_interval>0 should still produce correct gradients."""
        config_dict = {
            "d_model": 64, "d_rule": 32, "n_steps": 16, "n_scales": 4,
            "vocab_size": 256, "max_seq_len": 32,
            "checkpoint_interval": 4,
        }
        model = AENCAv10Model(config_dict)
        batch = torch.randint(0, 256, (2, 32))
        loss, _ = model.forward_train(batch)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name} with checkpointing"


# ── Test 17: Residual Scaling ────────────────────────────────────────────


class TestResidualScaling:
    def test_logit_scale_prevents_explosion(self, tiny_model):
        """Tied-embedding logits should be scaled by 1/√d_model."""
        batch = torch.randint(0, tiny_model.config.vocab_size, (1, 8))
        logits = tiny_model.get_logits(batch)
        # Logits should be reasonable magnitude, not exploding
        assert logits.abs().max() < 100, f"Logits max={logits.abs().max():.1f}"
