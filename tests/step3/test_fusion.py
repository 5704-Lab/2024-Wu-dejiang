import pytest
import torch

from step3_model.fusion import (
    CBAM,
    ChannelAttention,
    MultiScaleFusion,
    SpatialAttention,
    _STAGE_NAMES,
)
from step3_model.offset_predictor import OFFSET_SPEC


def _dummy_diff_features(batch=2, height=64, width=64, device="cpu", seed=42):
    gen = torch.Generator(device=device).manual_seed(seed)
    return {
        name: torch.randn(batch, channels, height // stride, width // stride,
                          device=device, generator=gen)
        for name, channels, stride in OFFSET_SPEC
    }


class TestCBAM:
    @pytest.fixture(scope="class")
    def cbam(self):
        return CBAM(64, reduction=8, spatial_kernel=7)

    def test_shape_preserved(self, cbam):
        x = torch.randn(2, 64, 32, 32)
        y = cbam(x)
        assert y.shape == x.shape

    def test_finite(self, cbam):
        x = torch.randn(2, 64, 32, 32)
        y = cbam(x)
        assert torch.isfinite(y).all()

    def test_backward(self, cbam):
        cbam.train()
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        y = cbam(x)
        y.mean().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0

    def test_channel_attention_output_range(self):
        ca = ChannelAttention(64, reduction=16)
        x = torch.randn(2, 64, 16, 16)
        y = ca(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_spatial_attention_output_range(self):
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(2, 64, 16, 16)
        y = sa(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()


class TestMultiScaleFusion:
    @pytest.fixture(scope="class")
    def fusion(self):
        return MultiScaleFusion(hidden=32, reduction=8)

    def test_output_shape(self, fusion):
        diff = _dummy_diff_features(height=64, width=64)
        fused = fusion(diff)
        assert fused.shape == (2, 1, 800, 800)

    def test_finite_output(self, fusion):
        diff = _dummy_diff_features()
        fused = fusion(diff)
        assert torch.isfinite(fused).all()

    def test_fusion_weights_sum_to_one(self, fusion):
        w = torch.softmax(fusion.fusion_weights, dim=0)
        assert abs(w.sum().item() - 1.0) < 1e-6

    def test_missing_key_raises(self, fusion):
        diff = _dummy_diff_features()
        del diff["C4"]
        with pytest.raises(KeyError, match="C4"):
            fusion(diff)

    def test_non_finite_raises(self, fusion):
        diff = _dummy_diff_features()
        diff["C3"][0, 0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="Non-finite"):
            fusion(diff)

    def test_backward(self, fusion):
        fusion.train()
        diff = _dummy_diff_features()
        diff_with_grad = {k: v.clone().requires_grad_(True) for k, v in diff.items()}
        fused = fusion(diff_with_grad)
        fused.mean().backward()
        assert any(v.grad is not None for v in diff_with_grad.values())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_cuda_forward(self, fusion):
        fusion.cuda()
        diff = _dummy_diff_features(device="cuda")
        fused = fusion(diff)
        assert fused.is_cuda
