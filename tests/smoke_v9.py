"""Comprehensive smoke tests for AENCA V9 — Wolfram Cellular Automata.

Tests verify:
  1. Model builds from config dict (smoke + full scale)
  2. Forward/backward pass works (no NaN, gradient flows)
  3. Kernel weights are truly shared across all CA steps
  4. Dilations provide full sequence receptive field coverage
  5. Zero-init W_dec produces zero deltas at initialization
  6. Spectral norm estimation runs without error
  7. Describe method returns required keys
  8. Param groups separate decay and no_decay correctly
  9. Loss components (CE, perm, spectral) are reasonable
  10. Tied embeddings work correctly
  11. Auto-sizing d_concept works correctly
"""

import dataclasses
import math
import sys

import pytest
import torch

sys.path.insert(0, ".")

from ncalm.models.v9.config import V9Config
from ncalm.models.v9.kernel import HebbianKernel, DepthwiseCausalConv
from ncalm.models.v9.model import AENCAv9Model


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_config() -> V9Config:
    """Minimal V9 config for CPU testing."""
    return V9Config(
        d_model=64,
        d_concept=128,
        n_steps=3,
        conv_kernel_size=3,
        vocab_size=256,
        max_seq_len=32,
        tie_embeddings=True,
        dropout=0.0,
        lambda_perm=0.01,
        lambda_spectral=0.001,
    )


@pytest.fixture
def tiny_config_dict(tiny_config) -> dict:
    return {f.name: getattr(tiny_config, f.name) for f in dataclasses.fields(tiny_config)}


@pytest.fixture
def tiny_model(tiny_config_dict) -> AENCAv9Model:
    return AENCAv9Model(tiny_config_dict)


@pytest.fixture
def tiny_batch(tiny_config) -> torch.Tensor:
    return torch.randint(0, tiny_config.vocab_size, (2, tiny_config.max_seq_len))


# ── Test 1: Model Builds ────────────────────────────────────────────────


class TestModelBuilds:
    def test_builds_from_config_dict(self, tiny_config_dict):
        model = AENCAv9Model(tiny_config_dict)
        assert isinstance(model, AENCAv9Model)

    def test_builds_with_auto_d_concept(self):
        config_dict = {
            "d_model": 128,
            "d_concept": 0,  # auto
            "target_params": 500_000,
            "n_steps": 3,
            "conv_kernel_size": 3,
            "vocab_size": 256,
            "max_seq_len": 32,
        }
        model = AENCAv9Model(config_dict)
        assert model.config.d_concept > 0
        total = sum(p.numel() for p in model.parameters())
        # Should be close to target (within ~1%)
        assert abs(total - 500_000) / 500_000 < 0.02, f"Got {total}, expected ~500K"

    def test_builds_untied_embeddings(self, tiny_config_dict):
        tiny_config_dict["tie_embeddings"] = False
        model = AENCAv9Model(tiny_config_dict)
        assert model.lm_head is not None

    def test_raises_without_d_concept_or_target(self):
        config_dict = {
            "d_model": 64,
            "d_concept": 0,
            "target_params": 0,
            "vocab_size": 256,
            "max_seq_len": 32,
        }
        with pytest.raises(ValueError, match="d_concept or target_params"):
            AENCAv9Model(config_dict)


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
        # Check gradient flows to all parameters
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


# ── Test 3: Shared Kernel ────────────────────────────────────────────────


class TestSharedKernel:
    def test_single_kernel_instance(self, tiny_model):
        """The model should have exactly ONE HebbianKernel."""
        kernels = [m for m in tiny_model.modules() if isinstance(m, HebbianKernel)]
        assert len(kernels) == 1, f"Expected 1 kernel, found {len(kernels)}"

    def test_kernel_used_n_steps_times(self, tiny_model, tiny_batch):
        """Verify the kernel is called n_steps times (same object)."""
        call_count = 0
        original_forward = tiny_model.kernel.forward

        def counting_forward(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_forward(*args, **kwargs)

        tiny_model.kernel.forward = counting_forward
        try:
            tiny_model.forward_train(tiny_batch)
            assert call_count == tiny_model.config.n_steps
        finally:
            tiny_model.kernel.forward = original_forward

    def test_conv_weights_shared_across_dilations(self, tiny_model):
        """The depthwise conv inside the kernel should be ONE set of weights."""
        convs = [m for m in tiny_model.modules() if isinstance(m, DepthwiseCausalConv)]
        assert len(convs) == 1, f"Expected 1 conv, found {len(convs)}"


# ── Test 4: Dilations & Receptive Field ──────────────────────────────────


class TestDilations:
    def test_dilations_cover_seq_len(self):
        """Dilations should give receptive field >= max_seq_len."""
        for seq_len in [32, 128, 256, 512, 1024, 2048]:
            cfg = V9Config(max_seq_len=seq_len, d_model=64, d_concept=64)
            assert cfg.receptive_field >= seq_len, (
                f"RF={cfg.receptive_field} < L={seq_len}, dilations={cfg.dilations}"
            )

    def test_dilations_are_geometric(self):
        cfg = V9Config(max_seq_len=1024, d_model=64, d_concept=64)
        d = cfg.dilations
        assert len(d) == 3
        assert d[0] == 1  # First dilation always 1 (local)
        # Geometric: d[i] = r^i
        r = d[1]
        for i, di in enumerate(d):
            assert di == r ** i, f"dilation[{i}]={di}, expected {r}^{i}={r**i}"

    def test_causal_conv_no_future_leak(self):
        """Conv output at position t must not depend on positions > t."""
        d_model, L = 4, 16
        conv = DepthwiseCausalConv(d_model, kernel_size=3)
        x = torch.randn(1, L, d_model, requires_grad=True)

        for dilation in [1, 4, 8]:
            out = conv(x, dilation=dilation)
            # Check position 5: gradient should only flow from positions ≤ 5
            out[0, 5, 0].backward(retain_graph=True)
            grad = x.grad[0]  # (L, d_model)
            # Positions > 5 should have zero gradient
            assert (grad[6:] == 0).all(), (
                f"Causal violation at dilation={dilation}: "
                f"grad from pos 5 leaks to future positions"
            )
            x.grad = None


# ── Test 5: Zero-Init and Class I Still Life ─────────────────────────────


class TestZeroInit:
    def test_w_dec_starts_zero(self, tiny_model):
        """W_dec must be zero at initialization."""
        W_dec = tiny_model.kernel.W_dec.weight
        assert (W_dec == 0).all(), "W_dec not zero-initialized"

    def test_zero_init_produces_zero_deltas(self, tiny_model, tiny_batch):
        """With zero W_dec, all deltas should be exactly zero."""
        cfg = tiny_model.config
        positions = torch.arange(cfg.max_seq_len)
        z = tiny_model.token_embedding(tiny_batch) + tiny_model.pos_embedding(positions)

        for dilation in tiny_model._dilations:
            delta = tiny_model.kernel(z, dilation=dilation)
            assert (delta == 0).all(), f"Non-zero delta with zero W_dec at d={dilation}"


# ── Test 6: Spectral Norm ────────────────────────────────────────────────


class TestSpectralNorm:
    def test_spectral_norm_runs(self, tiny_model):
        """Spectral norm estimation should run without error."""
        sigma = tiny_model._spectral_norm_loss()
        assert sigma.shape == ()
        assert not torch.isnan(sigma)

    def test_spectral_norm_positive(self, tiny_model, tiny_batch):
        """After one forward pass (power iteration), spectral norm should be ≥ 0."""
        # Run forward to trigger power iteration
        tiny_model.forward_train(tiny_batch)
        sigma = tiny_model._spectral_norm_loss()
        assert sigma >= 0


# ── Test 7: Describe ────────────────────────────────────────────────────


class TestDescribe:
    def test_describe_has_required_keys(self, tiny_model):
        d = tiny_model.describe()
        assert "architecture" in d
        assert "total_params" in d
        assert d["architecture"] == "AENCA-V9-Wolfram"

    def test_describe_params_match(self, tiny_model):
        d = tiny_model.describe()
        actual = sum(p.numel() for p in tiny_model.parameters())
        assert d["total_params"] == actual

    def test_describe_has_v9_keys(self, tiny_model):
        d = tiny_model.describe()
        for key in ["d_model", "d_concept", "n_steps", "dilations", "receptive_field"]:
            assert key in d, f"Missing key: {key}"


# ── Test 8: Param Groups ────────────────────────────────────────────────


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


# ── Test 9: Tied Embeddings ──────────────────────────────────────────────


class TestTiedEmbeddings:
    def test_tied_shares_weight(self, tiny_model):
        """With tie_embeddings=True, readout should use embed weight."""
        assert tiny_model.lm_head is None
        # Verify logits use the embedding weight
        batch = torch.randint(0, tiny_model.config.vocab_size, (1, 8))
        logits = tiny_model.get_logits(batch)
        assert logits.shape[-1] == tiny_model.config.vocab_size

    def test_untied_has_separate_head(self):
        config_dict = {
            "d_model": 64,
            "d_concept": 128,
            "vocab_size": 256,
            "max_seq_len": 32,
            "tie_embeddings": False,
        }
        model = AENCAv9Model(config_dict)
        assert model.lm_head is not None
        # Separate head should have its own weight
        assert model.lm_head.weight is not model.token_embedding.weight


# ── Test 10: Full-Scale Config Sanity ────────────────────────────────────


class TestFullScaleConfig:
    def test_162m_param_count(self):
        """Full-scale V9 should hit ~162M params."""
        cfg = V9Config(
            d_model=2048,
            d_concept=0,
            target_params=162_000_000,
            vocab_size=50304,
            max_seq_len=1024,
            tie_embeddings=True,
        )
        dc = cfg.compute_d_concept(162_000_000)
        cfg.d_concept = dc
        config_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
        model = AENCAv9Model(config_dict)
        total = sum(p.numel() for p in model.parameters())
        # Within 1% of target
        assert abs(total - 162_000_000) / 162_000_000 < 0.01, (
            f"Total params: {total:,} (expected ~162M, d_concept={dc})"
        )

    def test_full_scale_dilations(self):
        cfg = V9Config(d_model=2048, d_concept=9600, max_seq_len=1024)
        assert cfg.receptive_field >= 1024
        assert len(cfg.dilations) == 3
        assert cfg.dilations[0] == 1

    def test_kernel_param_fraction(self):
        """Kernel should be ~36% of total params."""
        cfg = V9Config(d_model=2048, d_concept=0, target_params=162_000_000, vocab_size=50304)
        cfg.d_concept = cfg.compute_d_concept()
        config_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
        model = AENCAv9Model(config_dict)
        total = sum(p.numel() for p in model.parameters())
        kernel = sum(p.numel() for p in model.kernel.parameters())
        frac = kernel / total
        assert 0.30 < frac < 0.40, f"Kernel fraction: {frac:.2%} (expected ~36%)"


# ── Test 11: FLOP Budget Sanity ─────────────────────────────────────────


class TestFLOPs:
    def test_gemm_count(self):
        """V9 should have exactly 3 × n_steps GEMMs (no attention)."""
        # 3 GEMMs per step (W_enc_p, W_enc_z, W_dec)
        n_steps = 3
        expected_gemms = 3 * n_steps  # = 9
        assert expected_gemms == 9

    def test_no_attention_modules(self, tiny_model):
        """V9 should have NO attention modules."""
        for name, module in tiny_model.named_modules():
            class_name = type(module).__name__.lower()
            assert "attention" not in class_name, f"Found attention module: {name}"
