import pytest
import torch

from step3_model.feature_warp import WARP_SPEC, FeatureWarp


@pytest.fixture(scope="module")
def warp():
    return FeatureWarp(kernel_size=3)


def _dummy_features(batch: int = 1, height: int = 64, width: int = 64, device: str = "cpu"):
    return {
        name: torch.rand(batch, channels, height // stride, width // stride, device=device)
        for name, channels, stride in WARP_SPEC
    }


def _dummy_offsets(batch: int = 1, height: int = 64, width: int = 64, device: str = "cpu", nonzero: bool = False):
    offsets = {}
    for name, _, stride in WARP_SPEC:
        h, w = height // stride, width // stride
        if nonzero:
            off = torch.randn(batch, 18, h, w, device=device) * 0.5
            off = torch.tanh(off) * stride * 0.5  # bounded, non-zero
        else:
            off = torch.zeros(batch, 18, h, w, device=device)
        offsets[name] = off
    return offsets


def test_warped_output_keys_and_shapes(warp):
    features = _dummy_features()
    offsets = _dummy_offsets()
    with torch.inference_mode():
        warped = warp(features, offsets)
    assert list(warped) == warp.key_order
    for name, channels, stride in WARP_SPEC:
        assert warped[name].shape == (1, channels, 64 // stride, 64 // stride)


def test_zero_offset_identity(warp):
    features = _dummy_features()
    offsets = _dummy_offsets()
    with torch.inference_mode():
        warped = warp(features, offsets)
    for name, _, _ in WARP_SPEC:
        assert torch.equal(warped[name], features[name]), f"Zero-offset identity violated at {name}"


def test_nonzero_offset_finite_and_differs(warp):
    features = _dummy_features()
    offsets = _dummy_offsets(nonzero=True)
    with torch.inference_mode():
        warped = warp(features, offsets)
    for name, channels, stride in WARP_SPEC:
        assert warped[name].shape == (1, channels, 64 // stride, 64 // stride)
        assert torch.isfinite(warped[name]).all(), f"Non-finite output at {name}"
        assert not torch.equal(warped[name], features[name]), (
            f"Non-zero offsets should produce different output at {name}"
        )


def test_backward_through_feature_warp():
    warp_train = FeatureWarp(kernel_size=3).train()
    features = _dummy_features()
    features_with_grad = {k: v.clone().requires_grad_(True) for k, v in features.items()}
    offsets = _dummy_offsets(nonzero=True)
    offsets_with_grad = {k: v.clone().requires_grad_(True) for k, v in offsets.items()}
    warped = warp_train(features_with_grad, offsets_with_grad)
    loss = sum(v.mean() for v in warped.values())
    loss.backward()
    assert any(v.grad is not None for v in features_with_grad.values()), (
        "Template features should receive gradients"
    )


def test_input_contract_rejects_wrong_keys(warp):
    features = _dummy_features()
    offsets = _dummy_offsets()
    bad = {"C3": features["C3"], "C4": features["C4"]}
    with pytest.raises(ValueError, match="keys"):
        warp(bad, offsets)


def test_input_contract_rejects_shape_mismatch(warp):
    features = _dummy_features()
    offsets = _dummy_offsets()
    bad = {**features, "C3": torch.rand(1, 512, 32, 32)}
    with pytest.raises(ValueError, match="spatial mismatch"):
        warp(bad, offsets)


def test_input_contract_rejects_non_finite(warp):
    features = _dummy_features()
    offsets = _dummy_offsets()
    features["C3"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        warp(features, offsets)


def test_input_contract_dtype_device_mismatch(warp):
    features = _dummy_features()
    offsets = _dummy_offsets()
    if torch.cuda.is_available():
        offsets_cuda = {k: v.to("cuda") for k, v in _dummy_offsets().items()}
        with pytest.raises(ValueError, match="device"):
            warp(features, offsets_cuda)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_forward(warp):
    warp.cuda()
    features = _dummy_features(device="cuda")
    offsets = _dummy_offsets(device="cuda")
    with torch.inference_mode():
        warped = warp(features, offsets)
    for name, _, _ in WARP_SPEC:
        assert warped[name].is_cuda
