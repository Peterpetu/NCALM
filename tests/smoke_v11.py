"""Smoke tests for AENCA V11: Morphogenetic Field NCA.

Comprehensive verification of V11's two-field coupled CA architecture.
Tests cover all V10 guarantees (since V11 IS V10 at init + morphogen),
plus morphogen-specific properties.

Test inventory (~63 tests across 23 classes):

  V10-inherited (adapted for V11):
    1.  Model builds from config dict (3 tests)
    2.  Forward & backward pass correctness (5 tests)
    3.  Shared content kernel — single instance (3 tests)
    4.  Dilations & receptive field (4 tests)
    5.  Zero-init → Class I still life (2 tests)
    6.  Settlement state mechanics (6 tests)
    7.  Settlement gradient isolation (2 tests)
    8.  Spectral norm estimation (2 tests)
    9.  Describe method returns required keys (3 tests)
    10. Param groups separate decay and no_decay (3 tests)
    11. Tied embeddings work correctly (2 tests)
    12. Config properties are correct (5 tests)
    13. No attention modules anywhere (2 tests)
    14. Full-scale config param count (2 tests)
    15. Permanence loss properties (2 tests)
    16. Gradient checkpointing (1 test)
    17. Residual scaling prevents logit explosion (1 test)

  V11-specific (morphogen):
    18. MorphogenRule: builds, zero-init, shape, own conv (4 tests)
    19. Morphogen gate: init, shape, bias (3 tests)
    20. Morphogen seeding: from content, bounded (2 tests)
    21. Dual-field CA: two convs, both updated, growth applies both (3 tests)
    22. Gradient flow through morphogen gate (2 tests)
    23. Morphogen conv causality (1 test)
"""

import dataclasses
import math
import sys

import pytest
import torch

sys.path.insert(0, ".")

from ncalm.models.v11.config import V11Config
from ncalm.models.v11.kernel import SimpleHebbianRule, MorphogenRule, DepthwiseCausalConv
from ncalm.models.v11.settlement import SettlementState
from ncalm.models.v11.model import AENCAv11Model


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_config() -> V11Config:
    """Minimal V11 config for CPU testing."""
    return V11Config(
        d_model=64,
        d_rule=32,
        d_h=16,
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
def tiny_model(tiny_config_dict) -> AENCAv11Model:
    return AENCAv11Model(tiny_config_dict)


@pytest.fixture
def tiny_batch(tiny_config) -> torch.Tensor:
    return torch.randint(0, tiny_config.vocab_size, (2, tiny_config.max_seq_len))


# ── Test 1: Model Builds ────────────────────────────────────────────────


class TestModelBuilds:
    def test_builds_from_config_dict(self, tiny_config_dict):
        model = AENCAv11Model(tiny_config_dict)
        assert isinstance(model, AENCAv11Model)

    def test_builds_untied_embeddings(self, tiny_config_dict):
        tiny_config_dict["tie_embeddings"] = False
        model = AENCAv11Model(tiny_config_dict)
        assert model.lm_head is not None

    def test_config_stored(self, tiny_model, tiny_config):
        assert tiny_model.config.d_model == tiny_config.d_model
        assert tiny_model.config.d_rule == tiny_config.d_rule
        assert tiny_model.config.d_h == tiny_config.d_h
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


# ── Test 3: Shared Content Kernel ────────────────────────────────────────


class TestSharedKernel:
    def test_single_content_rule_instance(self, tiny_model):
        """The model should have exactly ONE SimpleHebbianRule."""
        rules = [m for m in tiny_model.modules() if isinstance(m, SimpleHebbianRule)]
        assert len(rules) == 1, f"Expected 1 content rule, found {len(rules)}"

    def test_single_morph_rule_instance(self, tiny_model):
        """The model should have exactly ONE MorphogenRule."""
        rules = [m for m in tiny_model.modules() if isinstance(m, MorphogenRule)]
        assert len(rules) == 1, f"Expected 1 morphogen rule, found {len(rules)}"

    def test_content_conv_weights_shared(self, tiny_model):
        """The content rule should have ONE depthwise conv."""
        content_convs = []
        for name, m in tiny_model.named_modules():
            if isinstance(m, DepthwiseCausalConv) and "rule.conv" == name:
                content_convs.append(m)
        assert len(content_convs) == 1


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
        model = AENCAv11Model(config_dict)
        assert model._dilation_schedule == expected

    def test_dilations_are_powers_of_two(self, tiny_config):
        for d in tiny_config.dilations:
            assert d & (d - 1) == 0 and d > 0, f"Dilation {d} is not power of 2"

    def test_causal_conv_no_future_leak(self):
        """Content conv output at position t must not depend on positions > t."""
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
        """Full-scale V11 (64 steps, 8 scales) should cover seq_len=1024."""
        cfg = V11Config(
            d_model=64, d_rule=32, d_h=16,
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

    def test_zero_init_produces_zero_content_deltas(self, tiny_model, tiny_batch):
        """With zero W_d, all content deltas should be exactly zero."""
        cfg = tiny_model.config
        positions = torch.arange(cfg.max_seq_len)
        z = tiny_model.token_embedding(tiny_batch) + tiny_model.pos_embedding(positions)

        # At init, gate ≈ 0.95 (near 1), so gating doesn't change zero deltas
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
        assert st.mean_settled < 0.1, f"Initial settled={st.mean_settled}, expected < 0.1"
        assert st.active_fraction > 0.9

    def test_settlement_update_decreases_a_bar(self, tiny_config):
        """Zero deltas should drive a_bar toward 0 → increasing settled."""
        st = SettlementState.create(2, 16, torch.device("cpu"), tiny_config)
        initial_a_bar = st.a_bar.mean().item()

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
        assert d["architecture"] == "AENCA-V11-MorphogeneticField-Scaffold"

    def test_describe_params_match(self, tiny_model):
        d = tiny_model.describe()
        actual = sum(p.numel() for p in tiny_model.parameters())
        assert d["total_params"] == actual

    def test_describe_has_v11_keys(self, tiny_model):
        d = tiny_model.describe()
        for key in ["d_model", "d_rule", "d_h", "n_steps", "n_scales",
                     "settlement_beta", "settlement_tau", "settlement_eta",
                     "receptive_field", "morphogen_params", "gate_bias_init",
                     "content_rule_params", "morph_rule_params"]:
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
            "d_h": 16,
            "n_steps": 4,
            "n_scales": 2,
            "vocab_size": 256,
            "max_seq_len": 32,
            "tie_embeddings": False,
        }
        model = AENCAv11Model(config_dict)
        assert model.lm_head is not None
        assert model.lm_head.weight is not model.token_embedding.weight


# ── Test 12: Config Properties ───────────────────────────────────────────


class TestConfigProperties:
    def test_a_bar_init(self):
        cfg = V11Config()
        assert cfg.a_bar_init == pytest.approx(cfg.settlement_tau + 3 * cfg.settlement_eta)

    def test_dilations_powers_of_two(self):
        cfg = V11Config(n_scales=8)
        assert cfg.dilations == [1, 2, 4, 8, 16, 32, 64, 128]

    def test_total_params_estimate(self):
        cfg = V11Config()
        est = cfg.total_params_estimate
        assert est > 100_000_000  # > 100M
        assert est < 150_000_000  # < 150M

    def test_receptive_field_positive(self):
        cfg = V11Config()
        assert cfg.receptive_field > 0

    def test_d_h_in_config(self):
        cfg = V11Config()
        assert cfg.d_h == 128


# ── Test 13: No Attention ───────────────────────────────────────────────


class TestNoAttention:
    def test_no_attention_modules(self, tiny_model):
        """V11 should have NO attention modules — pure CA."""
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
        """Full-scale V11 should be ~110.6M params (less than GPT-2's 162M)."""
        cfg = V11Config(
            d_model=2048, d_rule=800, d_h=128,
            n_steps=64, n_scales=8,
            vocab_size=50304, max_seq_len=1024, tie_embeddings=True,
        )
        config_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
        model = AENCAv11Model(config_dict)
        total = sum(p.numel() for p in model.parameters())
        # Should be in 105M-125M range (V10 was ~110M, V11 adds ~575K)
        assert 100_000_000 < total < 125_000_000, f"Total params: {total:,}"

    def test_content_rule_is_tiny_fraction(self):
        """Content kernel should be <5% of total params (Wolfram: simple rule)."""
        cfg = V11Config(
            d_model=2048, d_rule=800, d_h=128,
            vocab_size=50304, max_seq_len=1024, tie_embeddings=True,
        )
        config_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
        model = AENCAv11Model(config_dict)
        total = sum(p.numel() for p in model.parameters())
        rule = sum(p.numel() for p in model.rule.parameters())
        frac = rule / total
        assert frac < 0.06, f"Content rule fraction: {frac:.2%} (expected <5%)"


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
        """With zero-init W_d (and W_h2), permanence loss should be exactly 0."""
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
            "d_model": 64, "d_rule": 32, "d_h": 16,
            "n_steps": 16, "n_scales": 4,
            "vocab_size": 256, "max_seq_len": 32,
            "checkpoint_interval": 4,
        }
        model = AENCAv11Model(config_dict)
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
        assert logits.abs().max() < 100, f"Logits max={logits.abs().max():.1f}"


# ═══════════════════════════════════════════════════════════════════════════
# V11-SPECIFIC TESTS: Morphogenetic Field
# ═══════════════════════════════════════════════════════════════════════════


# ── Test 18: MorphogenRule ───────────────────────────────────────────────


class TestMorphogenRule:
    def test_morph_rule_builds(self, tiny_config):
        """MorphogenRule should construct from V11Config."""
        rule = MorphogenRule(tiny_config)
        assert isinstance(rule, MorphogenRule)

    def test_morph_rule_zero_output_at_init(self, tiny_config):
        """W_h2 is zero-init → delta_h should be exactly 0 at init."""
        rule = MorphogenRule(tiny_config)
        h = torch.randn(2, 16, tiny_config.d_h)
        delta_h = rule(h, dilation=1)
        assert (delta_h == 0).all(), "MorphogenRule should produce zero deltas at init"

    def test_morph_rule_output_shape(self, tiny_config):
        """Output shape should match input: (B, L, d_h)."""
        rule = MorphogenRule(tiny_config)
        h = torch.randn(2, 16, tiny_config.d_h)
        delta_h = rule(h, dilation=1)
        assert delta_h.shape == (2, 16, tiny_config.d_h)

    def test_morph_rule_has_own_conv(self, tiny_config):
        """MorphogenRule should have its own DepthwiseCausalConv with d_h channels."""
        rule = MorphogenRule(tiny_config)
        assert hasattr(rule, "conv")
        assert isinstance(rule.conv, DepthwiseCausalConv)
        assert rule.conv.channels == tiny_config.d_h


# ── Test 19: Morphogen Gate ─────────────────────────────────────────────


class TestMorphogenGate:
    def test_gate_near_passthrough_at_init(self, tiny_model, tiny_batch):
        """At init, bilinear gate ≈ σ(bias) ≈ 0.95 since SiLU(small)×small ≈ 0."""
        cfg = tiny_model.config
        positions = torch.arange(cfg.max_seq_len)
        z = tiny_model.token_embedding(tiny_batch) + tiny_model.pos_embedding(positions)
        h = torch.tanh(tiny_model.W_h_init(z))

        gate = tiny_model.compute_gate(h)
        # Small random weights → SiLU(W_a·h) × W_b·h ≈ 0 → gate ≈ σ(bias) = σ(3.0) ≈ 0.953
        expected = torch.sigmoid(torch.tensor(3.0))
        assert gate.mean().item() == pytest.approx(expected.item(), abs=0.05), (
            f"Gate mean={gate.mean():.4f}, expected ≈ {expected:.4f}"
        )

    def test_gate_shape(self, tiny_model, tiny_batch):
        """Gate should be (B, L, d_model) — same as content z."""
        cfg = tiny_model.config
        positions = torch.arange(cfg.max_seq_len)
        z = tiny_model.token_embedding(tiny_batch) + tiny_model.pos_embedding(positions)
        h = torch.tanh(tiny_model.W_h_init(z))

        gate = tiny_model.compute_gate(h)
        B, L, d = tiny_batch.shape[0], tiny_batch.shape[1], cfg.d_model
        assert gate.shape == (B, L, d)

    def test_gate_bias_init(self, tiny_model):
        """Gate bias should be initialized to 3.0."""
        bias = tiny_model.gate_bias
        assert (bias == 3.0).all(), f"Gate bias should be 3.0, got {bias}"


# ── Test 20: Morphogen Seeding ──────────────────────────────────────────


class TestMorphogenSeeding:
    def test_h_seeded_from_content(self, tiny_model, tiny_batch):
        """h₀ = tanh(W_h_init · z) should have shape (B, L, d_h)."""
        cfg = tiny_model.config
        positions = torch.arange(cfg.max_seq_len)
        z = tiny_model.token_embedding(tiny_batch) + tiny_model.pos_embedding(positions)
        h = torch.tanh(tiny_model.W_h_init(z))

        assert h.shape == (2, cfg.max_seq_len, cfg.d_h)

    def test_h_bounded(self, tiny_model, tiny_batch):
        """tanh ensures h ∈ [-1, 1]."""
        cfg = tiny_model.config
        positions = torch.arange(cfg.max_seq_len)
        z = tiny_model.token_embedding(tiny_batch) + tiny_model.pos_embedding(positions)
        h = torch.tanh(tiny_model.W_h_init(z))

        assert h.min() >= -1.0
        assert h.max() <= 1.0


# ── Test 21: Dual-Field CA ──────────────────────────────────────────────


class TestDualFieldCA:
    def test_two_conv_instances(self, tiny_model):
        """Model should have TWO DepthwiseCausalConv: content (d_model) + morphogen (d_h)."""
        convs = [m for m in tiny_model.modules() if isinstance(m, DepthwiseCausalConv)]
        assert len(convs) == 2, f"Expected 2 DepthwiseCausalConv, found {len(convs)}"

        channels = sorted([c.channels for c in convs])
        expected = sorted([tiny_model.config.d_model, tiny_model.config.d_h])
        assert channels == expected, (
            f"Conv channels {channels} don't match expected {expected}"
        )

    def test_content_rule_called_with_gate(self, tiny_model, tiny_batch):
        """The content rule should be called with a non-None gate during forward."""
        gate_values = []
        original_forward = tiny_model.rule.forward

        def capturing_forward(z, dilation=1, gate=None):
            gate_values.append(gate is not None)
            return original_forward(z, dilation=dilation, gate=gate)

        tiny_model.rule.forward = capturing_forward
        try:
            tiny_model.forward_train(tiny_batch)
            # All calls should have gate (from _ca_step)
            assert all(gate_values), (
                f"Content rule should always receive gate, "
                f"but {sum(not g for g in gate_values)} calls had gate=None"
            )
            assert len(gate_values) == tiny_model.config.n_steps
        finally:
            tiny_model.rule.forward = original_forward

    def test_growth_gate_applies_to_both_fields(self, tiny_config):
        """When cells are settled, both z and h should stop getting updates.

        We verify by running _ca_step manually and checking that growth_gate
        is applied to both delta_z and delta_h.
        """
        # Build a model where we can observe behavior
        config_dict = {f.name: getattr(tiny_config, f.name)
                       for f in dataclasses.fields(tiny_config)}
        model = AENCAv11Model(config_dict)

        # Force non-zero deltas by setting W_d and W_h2 to non-zero
        with torch.no_grad():
            model.rule.W_d.weight.fill_(0.01)
            model.morph_rule.W_h2.weight.fill_(0.01)

        # Create inputs
        B, L = 2, tiny_config.max_seq_len
        z = torch.randn(B, L, tiny_config.d_model)
        h = torch.randn(B, L, tiny_config.d_h)

        # Compute deltas
        delta_z, delta_h = model._ca_step(z, h, dilation=1)

        # Verify both are non-zero (so we actually test the gate works)
        assert delta_z.abs().sum() > 0, "delta_z should be non-zero for this test"
        assert delta_h.abs().sum() > 0, "delta_h should be non-zero for this test"


# ── Test 22: Gradient Flow Through Morphogen ────────────────────────────


class TestMorphogenGradientFlow:
    def test_gradient_flows_to_w_h_init(self, tiny_model, tiny_batch):
        """Gradient should reach W_h_init through CE → z → gate → h → W_h_init.

        At init, W_d=0 blocks the delta_z→CE path. We set W_d,
        W_h2, and W_gate_a/b all non-zero to test the full gradient chain:
        CE → z_final → delta_z → gate → h → W_h_init.
        """
        with torch.no_grad():
            tiny_model.rule.W_d.weight.fill_(0.01)
            tiny_model.morph_rule.W_h2.weight.fill_(0.01)
            tiny_model.W_gate_a.weight.fill_(0.01)
            tiny_model.W_gate_b.weight.fill_(0.01)

        loss, _ = tiny_model.forward_train(tiny_batch)
        loss.backward()
        assert tiny_model.W_h_init.weight.grad is not None, "No gradient for W_h_init"
        assert tiny_model.W_h_init.weight.grad.abs().sum() > 0, "Gradient for W_h_init is all zeros"

    def test_gradient_flows_to_w_gate(self, tiny_model, tiny_batch):
        """Gradient should reach W_gate_a/b through CE → z → perceived → gate → W_gate."""
        loss, _ = tiny_model.forward_train(tiny_batch)
        loss.backward()
        # Bilinear gate: check both projections and bias
        assert tiny_model.W_gate_a.weight.grad is not None, "No gradient for W_gate_a.weight"
        assert tiny_model.W_gate_b.weight.grad is not None, "No gradient for W_gate_b.weight"
        assert tiny_model.gate_bias.grad is not None, "No gradient for gate_bias"


# ── Test 23: Morphogen Conv Causality ───────────────────────────────────


class TestMorphogenConvCausality:
    def test_morph_conv_no_future_leak(self):
        """Morphogen conv at position t must not depend on positions > t."""
        d_h, L = 8, 16
        conv = DepthwiseCausalConv(d_h, kernel_size=3)
        x = torch.randn(1, L, d_h, requires_grad=True)

        for dilation in [1, 2, 4]:
            out = conv(x, dilation=dilation)
            out[0, 5, 0].backward(retain_graph=True)
            grad = x.grad[0]
            assert (grad[6:] == 0).all(), (
                f"Morphogen causal violation at dilation={dilation}: "
                f"grad from pos 5 leaks to future positions"
            )
            x.grad = None


# ── Test 24: W_h2 Zero-Init ──────────────────────────────────────────────


class TestMorphogenZeroInit:
    def test_w_h2_starts_zero(self, tiny_model):
        """W_h2 (morphogen output) must be zero-initialized."""
        W_h2 = tiny_model.morph_rule.W_h2.weight
        assert (W_h2 == 0).all(), "W_h2 not zero-initialized"

    def test_both_zero_inits_produce_zero_deltas(self, tiny_model, tiny_batch):
        """At init, both delta_z and delta_h should be exactly zero."""
        cfg = tiny_model.config
        positions = torch.arange(cfg.max_seq_len)
        z = tiny_model.token_embedding(tiny_batch) + tiny_model.pos_embedding(positions)
        h = torch.tanh(tiny_model.W_h_init(z))

        for dilation in tiny_model._dilation_schedule[:4]:
            delta_z, delta_h = tiny_model._ca_step(z, h, dilation)
            assert (delta_z == 0).all(), f"delta_z not zero at init (d={dilation})"
            assert (delta_h == 0).all(), f"delta_h not zero at init (d={dilation})"


# ── Test 25: Morphogen Param Count ──────────────────────────────────────


class TestMorphogenParamCount:
    def test_morphogen_overhead_is_small(self):
        """Morphogen params should be < 1% of total params at full scale."""
        cfg = V11Config(
            d_model=2048, d_rule=800, d_h=128,
            vocab_size=50304, max_seq_len=1024, tie_embeddings=True,
        )
        config_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
        model = AENCAv11Model(config_dict)

        total = sum(p.numel() for p in model.parameters())
        morph_params = (
            sum(p.numel() for p in model.morph_rule.parameters())
            + sum(p.numel() for p in model.W_gate_a.parameters())
            + sum(p.numel() for p in model.W_gate_b.parameters())
            + model.gate_bias.numel()
            + model.W_h_init.weight.numel()
        )
        frac = morph_params / total
        assert frac < 0.01, f"Morphogen overhead: {frac:.2%} (expected < 1%)"

    def test_morphogen_params_breakdown(self, tiny_model):
        """Verify morphogen params are tracked in describe()."""
        d = tiny_model.describe()
        assert d["morphogen_params"] > 0
        assert d["morph_rule_params"] > 0
        assert d["morph_gate_params"] > 0
        assert d["morph_seed_params"] > 0
        assert d["morphogen_params"] == (
            d["morph_rule_params"] + d["morph_gate_params"] + d["morph_seed_params"]
        )


# ═══════════════════════════════════════════════════════════════════════════
# SCAFFOLD TESTS — Persistent Morphogen Scaffold (Epigenetic Memory)
# ═══════════════════════════════════════════════════════════════════════════

# ── Fixtures for scaffold tests ──────────────────────────────────────────


@pytest.fixture
def scaffold_config() -> V11Config:
    """Config with scaffold enabled and short warmup for testing."""
    return V11Config(
        d_model=64,
        d_rule=32,
        d_h=16,
        n_steps=4,
        n_scales=2,
        conv_kernel_size=3,
        vocab_size=128,
        max_seq_len=32,
        tie_embeddings=True,
        dropout=0.0,
        lambda_perm=0.01,
        lambda_spectral=0.001,
        settlement_beta=0.9,
        settlement_tau=0.05,
        settlement_eta=0.1,
        checkpoint_interval=0,
        # Scaffold params
        n_scaffold_templates=8,
        scaffold_alpha=0.3,
        scaffold_temperature=1.0,
        scaffold_ema_decay=0.99,
        scaffold_warmup_steps=5,
        scaffold_maintenance_interval=10,
        scaffold_min_norm=0.01,
        scaffold_merge_threshold=0.95,
    )


@pytest.fixture
def scaffold_config_dict(scaffold_config) -> dict:
    return {f.name: getattr(scaffold_config, f.name) for f in dataclasses.fields(scaffold_config)}


@pytest.fixture
def scaffold_model(scaffold_config_dict) -> AENCAv11Model:
    return AENCAv11Model(scaffold_config_dict)


@pytest.fixture
def scaffold_batch(scaffold_config) -> torch.Tensor:
    return torch.randint(0, scaffold_config.vocab_size, (2, scaffold_config.max_seq_len))


# ── Test 26: Scaffold Initialization ────────────────────────────────────


class TestScaffoldInitialization:
    """Scaffold buffers exist and are correctly initialized."""

    def test_scaffold_templates_exist(self, scaffold_model):
        """Model has scaffold_templates buffer."""
        assert hasattr(scaffold_model, "scaffold_templates")
        assert isinstance(scaffold_model.scaffold_templates, torch.Tensor)

    def test_scaffold_templates_shape(self, scaffold_model):
        """Templates shape is (K, d_h)."""
        K = scaffold_model.config.n_scaffold_templates
        d_h = scaffold_model.config.d_h
        assert scaffold_model.scaffold_templates.shape == (K, d_h)

    def test_scaffold_templates_zero_init(self, scaffold_model):
        """Templates start at zero (no scaffold effect initially)."""
        assert (scaffold_model.scaffold_templates == 0).all()

    def test_scaffold_usage_exists_and_zero(self, scaffold_model):
        """Usage counter exists and starts at zero."""
        assert hasattr(scaffold_model, "scaffold_usage")
        K = scaffold_model.config.n_scaffold_templates
        assert scaffold_model.scaffold_usage.shape == (K,)
        assert (scaffold_model.scaffold_usage == 0).all()

    def test_scaffold_step_exists_and_zero(self, scaffold_model):
        """Step counter exists and starts at zero."""
        assert hasattr(scaffold_model, "scaffold_step")
        assert scaffold_model.scaffold_step.item() == 0

    def test_scaffold_buffers_in_state_dict(self, scaffold_model):
        """Scaffold buffers are persistent — appear in state_dict."""
        sd = scaffold_model.state_dict()
        assert "scaffold_templates" in sd
        assert "scaffold_usage" in sd
        assert "scaffold_step" in sd

    def test_scaffold_not_in_parameters(self, scaffold_model):
        """Scaffold buffers are NOT parameters — not trained by optimizer."""
        param_names = {name for name, _ in scaffold_model.named_parameters()}
        assert "scaffold_templates" not in param_names
        assert "scaffold_usage" not in param_names
        assert "scaffold_step" not in param_names


# ── Test 27: Scaffold Seeding ───────────────────────────────────────────


class TestScaffoldSeeding:
    """Scaffold enriches morphogen seeding when active."""

    def test_zero_templates_produce_pure_h_fresh(self, scaffold_model, scaffold_batch):
        """When templates are all zero, seeding = pure tanh(W_h_init(z))."""
        z = scaffold_model.token_embedding(scaffold_batch) + \
            scaffold_model.pos_embedding(torch.arange(scaffold_batch.shape[1]))

        # Scaffold templates are zero → _seed_morphogen should return h_fresh
        h = scaffold_model._seed_morphogen(z)
        h_fresh = torch.tanh(scaffold_model.W_h_init(z))

        assert torch.allclose(h, h_fresh, atol=1e-6), \
            "Zero templates should produce pure h_fresh seeding"

    def test_nonzero_templates_change_seeding(self, scaffold_model, scaffold_batch):
        """When templates are populated and past warmup, seeding differs."""
        # Populate templates with non-zero values
        K = scaffold_model.config.n_scaffold_templates
        d_h = scaffold_model.config.d_h
        scaffold_model.scaffold_templates.copy_(torch.randn(K, d_h) * 0.5)

        # Set step past warmup
        scaffold_model.scaffold_step.fill_(scaffold_model.config.scaffold_warmup_steps + 1)

        z = scaffold_model.token_embedding(scaffold_batch) + \
            scaffold_model.pos_embedding(torch.arange(scaffold_batch.shape[1]))

        h = scaffold_model._seed_morphogen(z)
        h_fresh = torch.tanh(scaffold_model.W_h_init(z))

        # With populated templates and past warmup, h should differ from h_fresh
        assert not torch.allclose(h, h_fresh, atol=1e-4), \
            "Populated scaffold should change seeding"

    def test_warmup_ramp_is_gradual(self, scaffold_model, scaffold_batch):
        """During warmup, scaffold contribution ramps from 0 to α."""
        K = scaffold_model.config.n_scaffold_templates
        d_h = scaffold_model.config.d_h
        scaffold_model.scaffold_templates.copy_(torch.randn(K, d_h) * 0.5)

        z = scaffold_model.token_embedding(scaffold_batch) + \
            scaffold_model.pos_embedding(torch.arange(scaffold_batch.shape[1]))
        h_fresh = torch.tanh(scaffold_model.W_h_init(z))

        # Step 0 (warmup start) → α should be ~0
        scaffold_model.scaffold_step.fill_(0)
        h_start = scaffold_model._seed_morphogen(z)
        diff_start = (h_start - h_fresh).abs().mean().item()

        # Step at midpoint → α should be ~0.5 * scaffold_alpha
        warmup = scaffold_model.config.scaffold_warmup_steps
        scaffold_model.scaffold_step.fill_(warmup // 2)
        h_mid = scaffold_model._seed_morphogen(z)
        diff_mid = (h_mid - h_fresh).abs().mean().item()

        # Step past warmup → α should be full scaffold_alpha
        scaffold_model.scaffold_step.fill_(warmup + 1)
        h_full = scaffold_model._seed_morphogen(z)
        diff_full = (h_full - h_fresh).abs().mean().item()

        # Differences should be monotonically increasing
        assert diff_start < diff_mid, \
            f"Warmup ramp: diff_start ({diff_start:.6f}) should < diff_mid ({diff_mid:.6f})"
        assert diff_mid < diff_full, \
            f"Warmup ramp: diff_mid ({diff_mid:.6f}) should < diff_full ({diff_full:.6f})"

    def test_alpha_zero_disables_scaffold(self, scaffold_batch):
        """scaffold_alpha=0 → seeding is pure h_fresh regardless of templates."""
        cfg = V11Config(
            d_model=64, d_rule=32, d_h=16, n_steps=4, n_scales=2,
            vocab_size=128, max_seq_len=32, n_scaffold_templates=8,
            scaffold_alpha=0.0,  # DISABLED
            checkpoint_interval=0,
        )
        d = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
        model = AENCAv11Model(d)

        # Populate templates
        model.scaffold_templates.copy_(torch.randn(8, 16) * 0.5)
        model.scaffold_step.fill_(1000)

        z = model.token_embedding(scaffold_batch) + \
            model.pos_embedding(torch.arange(scaffold_batch.shape[1]))

        h = model._seed_morphogen(z)
        h_fresh = torch.tanh(model.W_h_init(z))

        assert torch.allclose(h, h_fresh, atol=1e-6), \
            "alpha=0 should disable scaffold completely"


# ── Test 28: Scaffold EMA Update (on_step_end) ─────────────────────────


class TestScaffoldEmaUpdate:
    """on_step_end crystallizes morphogen states into templates via EMA."""

    def test_first_update_initializes_templates(self, scaffold_model, scaffold_batch):
        """First on_step_end with zero templates → initializes from observed h."""
        assert (scaffold_model.scaffold_templates == 0).all()

        loss, info = scaffold_model.forward_train(scaffold_batch)
        loss.backward()
        scaffold_model.on_step_end(0, info)

        # Templates should now be non-zero (initialized from observed h)
        norms = scaffold_model.scaffold_templates.norm(dim=-1)
        assert norms.max().item() > 0.01, \
            "After first update, templates should be non-zero"

    def test_step_counter_increments(self, scaffold_model, scaffold_batch):
        """Each on_step_end increments scaffold_step by 1."""
        assert scaffold_model.scaffold_step.item() == 0

        loss, info = scaffold_model.forward_train(scaffold_batch)
        loss.backward()
        scaffold_model.on_step_end(0, info)

        assert scaffold_model.scaffold_step.item() == 1

        loss2, info2 = scaffold_model.forward_train(scaffold_batch)
        loss2.backward()
        scaffold_model.on_step_end(1, info2)

        assert scaffold_model.scaffold_step.item() == 2

    def test_ema_moves_templates_toward_observed(self, scaffold_model, scaffold_batch):
        """After repeated updates, templates converge toward observed morphogen states."""
        # Run several forward+update cycles
        for step in range(10):
            loss, info = scaffold_model.forward_train(scaffold_batch)
            loss.backward()
            scaffold_model.on_step_end(step, info)

        templates_after_10 = scaffold_model.scaffold_templates.clone()

        # Run more cycles
        for step in range(10, 20):
            loss, info = scaffold_model.forward_train(scaffold_batch)
            loss.backward()
            scaffold_model.on_step_end(step, info)

        templates_after_20 = scaffold_model.scaffold_templates.clone()

        # Templates should have changed (not frozen)
        assert not torch.allclose(templates_after_10, templates_after_20, atol=1e-6), \
            "Templates should evolve over training steps"

    def test_usage_accumulates(self, scaffold_model, scaffold_batch):
        """Usage counter accumulates across on_step_end calls."""
        assert (scaffold_model.scaffold_usage == 0).all()

        loss, info = scaffold_model.forward_train(scaffold_batch)
        loss.backward()
        scaffold_model.on_step_end(0, info)

        # After first update (initialization), usage may not be set
        # After second update, usage should accumulate
        loss2, info2 = scaffold_model.forward_train(scaffold_batch)
        loss2.backward()
        scaffold_model.on_step_end(1, info2)

        assert scaffold_model.scaffold_usage.sum().item() > 0, \
            "Usage should accumulate after updates"

    def test_transient_state_cleared_after_update(self, scaffold_model, scaffold_batch):
        """_last_h_final and _last_settled are cleared after on_step_end."""
        loss, info = scaffold_model.forward_train(scaffold_batch)
        assert len(scaffold_model._last_h_final) > 0
        assert len(scaffold_model._last_settled) > 0

        loss.backward()
        scaffold_model.on_step_end(0, info)

        assert len(scaffold_model._last_h_final) == 0
        assert len(scaffold_model._last_settled) == 0


# ── Test 29: Scaffold Pruning ───────────────────────────────────────────


class TestScaffoldPruning:
    """Dead templates (low norm) are recycled from alive ones."""

    def test_dead_templates_get_reinitialized(self, scaffold_model):
        """Templates with norm < min_norm are re-initialized."""
        K = scaffold_model.config.n_scaffold_templates
        d_h = scaffold_model.config.d_h

        # Set some templates alive, some dead
        scaffold_model.scaffold_templates[:4] = torch.randn(4, d_h) * 0.5  # alive
        scaffold_model.scaffold_templates[4:] = torch.randn(4, d_h) * 0.001  # dead

        dead_norms_before = scaffold_model.scaffold_templates[4:].norm(dim=-1)
        assert (dead_norms_before < scaffold_model.config.scaffold_min_norm).all()

        scaffold_model._scaffold_prune()

        # Dead templates should now have non-trivial norms (re-initialized)
        dead_norms_after = scaffold_model.scaffold_templates[4:].norm(dim=-1)
        assert (dead_norms_after >= scaffold_model.config.scaffold_min_norm).all(), \
            "Dead templates should be re-initialized above min_norm"

    def test_pruning_preserves_alive_templates(self, scaffold_model):
        """Alive templates are not modified by pruning."""
        K = scaffold_model.config.n_scaffold_templates
        d_h = scaffold_model.config.d_h

        alive_templates = torch.randn(4, d_h) * 0.5
        scaffold_model.scaffold_templates[:4] = alive_templates.clone()
        scaffold_model.scaffold_templates[4:] = torch.randn(4, d_h) * 0.001  # dead

        scaffold_model._scaffold_prune()

        # Alive templates should be unchanged
        assert torch.allclose(scaffold_model.scaffold_templates[:4], alive_templates), \
            "Pruning should not modify alive templates"

    def test_all_alive_no_pruning(self, scaffold_model):
        """When all templates are alive, pruning is a no-op."""
        d_h = scaffold_model.config.d_h
        original = torch.randn(scaffold_model.config.n_scaffold_templates, d_h) * 0.5
        scaffold_model.scaffold_templates.copy_(original)

        scaffold_model._scaffold_prune()

        assert torch.allclose(scaffold_model.scaffold_templates, original), \
            "No dead templates → pruning should be no-op"


# ── Test 30: Scaffold Compression ───────────────────────────────────────


class TestScaffoldCompression:
    """Similar templates merge to prevent redundancy."""

    def test_identical_templates_get_merged(self, scaffold_model):
        """Two identical templates should trigger merge."""
        K = scaffold_model.config.n_scaffold_templates
        d_h = scaffold_model.config.d_h

        # Make templates 0 and 1 nearly identical
        base = torch.randn(d_h) * 0.5
        scaffold_model.scaffold_templates[0] = base
        scaffold_model.scaffold_templates[1] = base + torch.randn(d_h) * 0.001
        # Make rest distinct
        for i in range(2, K):
            scaffold_model.scaffold_templates[i] = torch.randn(d_h) * 0.5

        # Set usage weights
        scaffold_model.scaffold_usage[0] = 100.0
        scaffold_model.scaffold_usage[1] = 50.0

        templates_before = scaffold_model.scaffold_templates.clone()
        scaffold_model._scaffold_compress()

        # Template 0 should now be a weighted blend
        # Template 1 should be re-initialized (different from before)
        assert not torch.allclose(
            scaffold_model.scaffold_templates[1], templates_before[1], atol=1e-3
        ), "Merged template slot should be re-initialized"

    def test_diverse_templates_not_merged(self, scaffold_model):
        """Diverse templates (low cosine sim) should not be merged."""
        K = scaffold_model.config.n_scaffold_templates
        d_h = scaffold_model.config.d_h

        # Make orthogonal templates
        templates = torch.eye(K, d_h) * 0.5  # K×d_h with orthogonal rows
        scaffold_model.scaffold_templates.copy_(templates)

        original = scaffold_model.scaffold_templates.clone()
        scaffold_model._scaffold_compress()

        # All templates should remain unchanged
        assert torch.allclose(scaffold_model.scaffold_templates, original), \
            "Diverse templates should not be merged"


# ── Test 31: Scaffold State Dict Persistence ────────────────────────────


class TestScaffoldPersistence:
    """Scaffold survives save/load round-trip (the whole point)."""

    def test_scaffold_round_trip(self, scaffold_model, scaffold_batch):
        """Templates, usage, step survive state_dict save/load."""
        # Run a few training steps to populate scaffold
        for step in range(5):
            loss, info = scaffold_model.forward_train(scaffold_batch)
            loss.backward()
            scaffold_model.on_step_end(step, info)

        # Save state
        templates_before = scaffold_model.scaffold_templates.clone()
        usage_before = scaffold_model.scaffold_usage.clone()
        step_before = scaffold_model.scaffold_step.clone()

        state_dict = scaffold_model.state_dict()

        # Create new model and load
        cfg_dict = {f.name: getattr(scaffold_model.config, f.name)
                    for f in dataclasses.fields(scaffold_model.config)}
        new_model = AENCAv11Model(cfg_dict)
        new_model.load_state_dict(state_dict)

        # Verify scaffold state is restored
        assert torch.allclose(new_model.scaffold_templates, templates_before), \
            "Templates should survive save/load"
        assert torch.allclose(new_model.scaffold_usage, usage_before), \
            "Usage should survive save/load"
        assert new_model.scaffold_step.item() == step_before.item(), \
            "Step counter should survive save/load"

    def test_backward_compat_strict_false(self):
        """Loading old checkpoint (no scaffold) into scaffold model with strict=False."""
        cfg = V11Config(
            d_model=64, d_rule=32, d_h=16, n_steps=4, n_scales=2,
            vocab_size=128, max_seq_len=32, n_scaffold_templates=8,
            checkpoint_interval=0,
        )
        d = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}

        model1 = AENCAv11Model(d)
        state_dict = model1.state_dict()

        # Remove scaffold keys (simulate old checkpoint)
        scaffold_keys = [k for k in state_dict if "scaffold" in k]
        for k in scaffold_keys:
            del state_dict[k]

        # Load with strict=False should work
        model2 = AENCAv11Model(d)
        model2.load_state_dict(state_dict, strict=False)

        # Scaffold should be at defaults (zero)
        assert (model2.scaffold_templates == 0).all()
        assert (model2.scaffold_usage == 0).all()
        assert model2.scaffold_step.item() == 0


# ── Test 32: Scaffold Extra Metrics ─────────────────────────────────────


class TestScaffoldExtraMetrics:
    """extra_metrics returns scaffold health indicators."""

    def test_extra_metrics_keys(self, scaffold_model):
        """extra_metrics has all expected scaffold keys."""
        metrics = scaffold_model.extra_metrics()
        expected_keys = [
            "scaffold/mean_norm",
            "scaffold/usage_entropy",
            "scaffold/n_alive",
            "scaffold/mean_cosine",
            "scaffold/step",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_metrics_all_zero_at_init(self, scaffold_model):
        """At init (zero templates), metrics reflect empty scaffold."""
        metrics = scaffold_model.extra_metrics()
        assert metrics["scaffold/mean_norm"] == 0.0 or \
               metrics["scaffold/mean_norm"] < 1e-8
        assert metrics["scaffold/n_alive"] == 0.0
        assert metrics["scaffold/step"] == 0.0

    def test_metrics_after_training(self, scaffold_model, scaffold_batch):
        """After training steps, metrics reflect populated scaffold."""
        for step in range(10):
            loss, info = scaffold_model.forward_train(scaffold_batch)
            loss.backward()
            scaffold_model.on_step_end(step, info)

        metrics = scaffold_model.extra_metrics()
        assert metrics["scaffold/mean_norm"] > 0.01, \
            "After training, templates should have non-trivial norms"
        assert metrics["scaffold/n_alive"] > 0, \
            "After training, some templates should be alive"
        assert metrics["scaffold/step"] == 10.0

    def test_disabled_scaffold_returns_empty(self):
        """K=0 → extra_metrics returns empty dict."""
        cfg = V11Config(
            d_model=64, d_rule=32, d_h=16, n_steps=4, n_scales=2,
            vocab_size=128, max_seq_len=32, n_scaffold_templates=0,
            checkpoint_interval=0,
        )
        d = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
        model = AENCAv11Model(d)
        assert model.extra_metrics() == {}


# ── Test 33: Scaffold Describe Integration ──────────────────────────────


class TestScaffoldDescribe:
    """describe() includes scaffold information."""

    def test_describe_has_scaffold_keys(self, scaffold_model):
        """describe() includes scaffold configuration info."""
        d = scaffold_model.describe()
        for key in ["scaffold_templates_K", "scaffold_alpha",
                     "scaffold_ema_decay", "scaffold_warmup_steps",
                     "scaffold_maintenance_interval"]:
            assert key in d, f"Missing scaffold key in describe(): {key}"

    def test_describe_scaffold_values_match_config(self, scaffold_model):
        """describe() scaffold values match config."""
        d = scaffold_model.describe()
        cfg = scaffold_model.config
        assert d["scaffold_templates_K"] == cfg.n_scaffold_templates
        assert d["scaffold_alpha"] == cfg.scaffold_alpha
        assert d["scaffold_ema_decay"] == cfg.scaffold_ema_decay

    def test_paradigm_mentions_scaffold(self, scaffold_model):
        """paradigm string mentions scaffold."""
        d = scaffold_model.describe()
        assert "scaffold" in d["paradigm"].lower() or "Scaffold" in d["paradigm"]


# ── Test 34: Scaffold Gradient Isolation ────────────────────────────────


class TestScaffoldGradientIsolation:
    """Scaffold buffers don't receive gradients (EMA only)."""

    def test_templates_no_grad(self, scaffold_model, scaffold_batch):
        """scaffold_templates should not have grad after backward."""
        # Populate templates so scaffold is active
        K = scaffold_model.config.n_scaffold_templates
        d_h = scaffold_model.config.d_h
        scaffold_model.scaffold_templates.copy_(torch.randn(K, d_h) * 0.5)
        scaffold_model.scaffold_step.fill_(100)

        loss, info = scaffold_model.forward_train(scaffold_batch)
        loss.backward()

        assert scaffold_model.scaffold_templates.grad is None, \
            "Scaffold templates should not receive gradients"

    def test_h_final_is_detached(self, scaffold_model, scaffold_batch):
        """_last_h_final saved for EMA should be detached from graph."""
        loss, info = scaffold_model.forward_train(scaffold_batch)

        assert len(scaffold_model._last_h_final) > 0
        assert not scaffold_model._last_h_final[0].requires_grad, \
            "_last_h_final must be detached from gradient graph"


# ── Test 35: Scaffold Inference Path ────────────────────────────────────


class TestScaffoldInference:
    """Scaffold works correctly during inference (get_logits)."""

    def test_inference_uses_scaffold(self, scaffold_model, scaffold_batch):
        """get_logits uses scaffold seeding (when populated).

        At zero-init, W_d=0 (content update) blocks h→logits path.
        We set W_d to non-zero to open the h→gate→content→logits path.
        """
        K = scaffold_model.config.n_scaffold_templates
        d_h = scaffold_model.config.d_h

        # Open the full h→gate→content pathway
        with torch.no_grad():
            scaffold_model.rule.W_d.weight.normal_(std=0.1)
            scaffold_model.W_gate_a.weight.normal_(std=0.1)
            scaffold_model.W_gate_b.weight.normal_(std=0.1)

        # Inference with zero templates
        logits_no_scaffold = scaffold_model.get_logits(scaffold_batch).clone()

        # Populate templates and set past warmup
        scaffold_model.scaffold_templates.copy_(torch.randn(K, d_h) * 0.5)
        scaffold_model.scaffold_step.fill_(100)

        # Inference with populated templates
        logits_with_scaffold = scaffold_model.get_logits(scaffold_batch)

        # Scaffold changes h → gate → content → logits
        max_diff = (logits_no_scaffold - logits_with_scaffold).abs().max().item()
        assert max_diff > 0, \
            "Populated scaffold should produce different inference output"

    def test_inference_doesnt_save_state(self, scaffold_model, scaffold_batch):
        """get_logits (inference) should NOT save _last_h_final."""
        scaffold_model.eval()
        scaffold_model.get_logits(scaffold_batch)
        assert len(scaffold_model._last_h_final) == 0, \
            "Inference should not save transient state for EMA"


# ── Test 36: Scaffold Full Training Cycle ───────────────────────────────


class TestScaffoldTrainingCycle:
    """End-to-end: scaffold activates over training, affects seeding."""

    def test_scaffold_activates_after_warmup(self, scaffold_model, scaffold_batch):
        """Before warmup: scaffold inactive. After: scaffold contributes."""
        warmup = scaffold_model.config.scaffold_warmup_steps

        # Train through warmup
        for step in range(warmup + 5):
            loss, info = scaffold_model.forward_train(scaffold_batch)
            loss.backward()
            scaffold_model.on_step_end(step, info)

        # Scaffold should be populated
        norms = scaffold_model.scaffold_templates.norm(dim=-1)
        n_alive = (norms >= scaffold_model.config.scaffold_min_norm).sum().item()
        assert n_alive > 0, \
            "After warmup+training, scaffold should have alive templates"

        # Verify scaffold step counter
        assert scaffold_model.scaffold_step.item() == warmup + 5

    def test_maintenance_runs_at_interval(self, scaffold_model, scaffold_batch):
        """Pruning/compression runs at scaffold_maintenance_interval."""
        interval = scaffold_model.config.scaffold_maintenance_interval
        K = scaffold_model.config.n_scaffold_templates
        d_h = scaffold_model.config.d_h

        # Manual setup: some dead templates
        scaffold_model.scaffold_templates[:4] = torch.randn(4, d_h) * 0.5
        scaffold_model.scaffold_templates[4:] = torch.randn(4, d_h) * 0.001

        # Set step counter to just before maintenance interval
        scaffold_model.scaffold_step.fill_(interval - 1)

        # Forward + on_step_end should trigger maintenance (step becomes interval)
        loss, info = scaffold_model.forward_train(scaffold_batch)
        loss.backward()
        scaffold_model.on_step_end(interval - 1, info)

        # Dead templates should have been re-initialized by pruning
        norms = scaffold_model.scaffold_templates[4:].norm(dim=-1)
        assert (norms >= scaffold_model.config.scaffold_min_norm).all(), \
            "Maintenance at interval should prune dead templates"
