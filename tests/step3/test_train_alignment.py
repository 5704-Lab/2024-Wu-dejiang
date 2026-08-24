import pytest
import torch

from step3_model.train_alignment import (
    MicroAlignmentLoss,
    _build_pipeline,
    train_one_epoch,
    validate_one_epoch,
    build_normal_region_masks,
)
from step3_model.offset_predictor import OFFSET_SPEC
from step3_model.feature_warp import WARP_SPEC


@pytest.fixture(scope="module")
def criterion():
    return MicroAlignmentLoss(alpha_c3=1.0, alpha_c4=1.0, alpha_c5=1.0)


def _dummy_features(
    batch: int = 2, height: int = 64, width: int = 64, device: str = "cpu",
    seed: int = 42,
):
    gen = torch.Generator(device=device).manual_seed(seed)
    return {
        name: torch.rand(batch, channels, height // stride, width // stride,
                          device=device, generator=gen)
        for name, channels, stride in OFFSET_SPEC
    }


def _dummy_offsets(
    batch: int = 2, height: int = 64, width: int = 64, device: str = "cpu",
    nonzero: bool = False,
):
    offsets = {}
    for name, _, stride in OFFSET_SPEC:
        h, w = height // stride, width // stride
        if nonzero:
            off = torch.randn(batch, 18, h, w, device=device) * 0.01
        else:
            off = torch.zeros(batch, 18, h, w, device=device)
        offsets[name] = off
    return offsets


def test_loss_zero_offsets_is_zero(criterion):
    features = _dummy_features()
    assert criterion(features, features)[1]["loss_total"] == 0.0


def test_loss_nonzero_offsets_positive(criterion):
    """With random features warped by random offsets, loss should be > 0."""
    # Use two different random sets to simulate warped != query
    qf = _dummy_features(seed=1)
    warped = _dummy_features(seed=2)
    _loss, stats = criterion(warped, qf)
    assert stats["loss_total"] > 0


def test_loss_per_scale_weights():
    """Verify alpha scaling: with alpha_C3=10 others=0, only C3 contributes."""
    c = MicroAlignmentLoss(alpha_c3=10.0, alpha_c4=0.0, alpha_c5=0.0)
    qf = _dummy_features(seed=1)
    warped = _dummy_features(seed=2)
    _loss, stats = c(warped, qf)
    assert stats["loss_total"] > 0
    # With alpha_c4 = alpha_c5 = 0, total should equal 10 * C3 MSE
    expected = 10.0 * stats["loss_C3"]
    assert abs(stats["loss_total"] - expected) < 1e-4


def test_loss_positive_definite(criterion):
    """Loss should be >= 0 for any pair of feature tensors."""
    for seed in [1, 2, 3]:
        qf = _dummy_features(seed=seed)
        warped = _dummy_features(seed=seed + 10)
        _loss, stats = criterion(warped, qf)
        assert stats["loss_total"] >= 0


def test_loss_backward_through_loss():
    """Gradient flows from loss to a tensor with requires_grad."""
    qf = _dummy_features(seed=1)
    warped = {k: v.clone().requires_grad_(True) for k, v in _dummy_features(seed=2).items()}
    criterion = MicroAlignmentLoss()
    loss, _stats = criterion(warped, qf)
    loss.backward()
    assert any(v.grad is not None for v in warped.values())


def test_defect_region_is_excluded_from_alignment_loss():
    qf = _dummy_features(batch=1, height=64, width=64, seed=1)
    warped = {key: value.clone() for key, value in qf.items()}
    for value in warped.values():
        value[:, :, : value.shape[-2] // 2, : value.shape[-1] // 2] += 10
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 32.0, 32.0]]), "size": torch.tensor([64, 64])}]
    masks = build_normal_region_masks(targets, qf)
    _loss, stats = MicroAlignmentLoss()(warped, qf, masks)
    assert stats["loss_total"] == pytest.approx(0.0)
    assert stats["normal_ratio_C3"] < 1.0


def test_build_pipeline_cpu():
    backbone, predictor, warp = _build_pipeline(torch.device("cpu"))
    assert isinstance(backbone, torch.nn.Module)
    assert isinstance(predictor, torch.nn.Module)
    assert isinstance(warp, torch.nn.Module)
    # Backbone should be frozen
    assert all(not p.requires_grad for p in backbone.parameters())
    # Predictor should be trainable
    assert any(p.requires_grad for p in predictor.parameters())
    # Warp has no parameters
    assert sum(1 for _ in warp.parameters()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_build_pipeline_cuda():
    backbone, predictor, warp = _build_pipeline(torch.device("cuda"))
    assert next(backbone.parameters()).is_cuda
    assert next(predictor.parameters()).is_cuda
    assert all(not p.requires_grad for p in backbone.parameters())
    assert any(p.requires_grad for p in predictor.parameters())
