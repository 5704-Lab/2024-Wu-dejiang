import pytest
import torch

from step3_model.offset_predictor import OFFSET_SPEC, OffsetPredictor


@pytest.fixture(scope="module")
def predictor():
    return OffsetPredictor(zero_init=True)


def _dummy_features(batch: int = 1, height: int = 64, width: int = 64, device: str = "cpu"):
    return {
        name: torch.rand(batch, channels, height // stride, width // stride, device=device)
        for name, channels, stride in OFFSET_SPEC
    }


def test_offset_output_keys_and_shapes(predictor):
    dummy = _dummy_features()
    offsets = predictor(dummy, dummy)
    assert list(offsets) == [name for name, _, _ in OFFSET_SPEC]
    for name, _, stride in OFFSET_SPEC:
        assert offsets[name].shape == (1, 18, 64 // stride, 64 // stride)


def test_zero_init_produces_exact_zeros(predictor):
    dummy = _dummy_features()
    offsets = predictor(dummy, dummy)
    for name, _, _ in OFFSET_SPEC:
        assert torch.equal(offsets[name], torch.zeros_like(offsets[name]))


def test_non_zero_init_offset_bounded_by_stride():
    predictor = OffsetPredictor(zero_init=False)
    dummy = _dummy_features()
    with torch.inference_mode():
        offsets = predictor(dummy, dummy)
    for name, _, stride in OFFSET_SPEC:
        assert offsets[name].abs().max().item() <= stride + 1e-6


def test_backward_through_predictor():
    predictor = OffsetPredictor(zero_init=True).train()
    dummy = _dummy_features()
    # Ensure features require grad so we can test gradient flow into them
    dummy_with_grad = {k: v.clone().requires_grad_(True) for k, v in dummy.items()}
    offsets = predictor(dummy_with_grad, dummy_with_grad)
    loss = sum(v.mean() for v in offsets.values())
    loss.backward()
    # Check predictor parameters receive gradients
    assert any(param.grad is not None for param in predictor.parameters())
    # Check input features receive gradients (end-to-end training readiness)
    assert any(v.grad is not None for v in dummy_with_grad.values())


def test_input_contract_rejects_wrong_keys(predictor):
    dummy = _dummy_features()
    bad = {"C3": dummy["C3"], "C4": dummy["C4"]}
    with pytest.raises(ValueError, match="keys"):
        predictor(bad, dummy)


def test_input_contract_rejects_non_finite(predictor):
    dummy = _dummy_features()
    dummy["C3"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        predictor(dummy, dummy)


def test_shape_mismatch_rejected(predictor):
    dummy = _dummy_features()
    bad = {**dummy, "C3": torch.rand(1, 512, 32, 32)}
    with pytest.raises(ValueError, match="Shape mismatch"):
        predictor(bad, dummy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_forward(predictor):
    predictor.cuda()
    dummy = _dummy_features(device="cuda")
    with torch.inference_mode():
        offsets = predictor(dummy, dummy)
    for name, _, _ in OFFSET_SPEC:
        assert offsets[name].is_cuda