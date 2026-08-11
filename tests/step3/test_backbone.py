import pytest
import torch

from step3_model.backbone import FEATURE_STAGES, SharedSiameseBackbone

KEYS = [name for name, _, _, _ in FEATURE_STAGES]


@pytest.fixture(scope="module")
def model():
    return SharedSiameseBackbone(weights=None).eval()


def test_forward_returns_ordered_multiscale_dicts(model):
    images = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        query, template = model(images, images.clone())
    assert isinstance(query, dict) and isinstance(template, dict)
    assert list(query) == KEYS and list(template) == KEYS
    assert query["C3"].shape == (1, 512, 8, 8)
    assert query["C4"].shape == (1, 1024, 4, 4)
    assert query["C5"].shape == (1, 2048, 2, 2)
    for name in KEYS:
        assert torch.equal(query[name], template[name])


def test_feature_spec_matches_800x800_shapes(model):
    images = torch.rand(1, 3, 800, 800)
    with torch.inference_mode():
        features, _ = model(images, images)
    for name, _, channels, stride in FEATURE_STAGES:
        assert features[name].shape == (1, channels, 800 // stride, 800 // stride)


def test_scale_monotonicity(model):
    with torch.inference_mode():
        features, _ = model(torch.rand(1, 3, 800, 800), torch.rand(1, 3, 800, 800))
    c3, c4, c5 = (features[name] for name in KEYS)
    assert c3.shape[2] == 2 * c4.shape[2] == 4 * c5.shape[2]
    assert c3.shape[1] * 2 == c4.shape[1] and c4.shape[1] * 2 == c5.shape[1]


def test_backbone_has_only_one_parameter_set(model):
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert not any("query_backbone" in key or "template_backbone" in key for key in model.state_dict())


def test_normalization_does_not_mutate_or_resize(model):
    images = torch.rand(1, 3, 32, 48)
    original = images.clone()
    normalized = model.normalize(images)
    assert normalized.shape == images.shape
    assert torch.equal(images, original)
    expected = (images[:, :, 0, 0] - model.mean[:, :, 0, 0]) / model.std[:, :, 0, 0]
    assert torch.allclose(normalized[:, :, 0, 0], expected)


@pytest.mark.parametrize("bad", [
    torch.zeros(3, 32, 32),
    torch.zeros(1, 1, 32, 32),
    torch.zeros(1, 3, 32, 32, dtype=torch.int64),
])
def test_invalid_input_contract_is_rejected(model, bad):
    with pytest.raises((TypeError, ValueError)):
        model(bad, bad)


def test_non_finite_and_out_of_range_are_rejected(model):
    invalid = torch.zeros(1, 3, 32, 32)
    invalid[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        model(invalid, invalid)
    invalid = torch.full((1, 3, 32, 32), 1.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model(invalid, invalid)


def test_branch_shape_mismatch_is_rejected(model):
    with pytest.raises(ValueError, match="shapes differ"):
        model(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 64, 64))


def test_backward_reaches_all_feature_stages():
    model = SharedSiameseBackbone(weights=None).train()
    query, template = model(torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64))
    loss = sum(features[name].mean() for features in (query, template) for name in KEYS)
    loss.backward()
    for _, stage, _, _ in FEATURE_STAGES:
        grads = [parameter.grad for parameter in getattr(model.encoder, stage).parameters()]
        assert any(grad is not None and grad.abs().sum() > 0 for grad in grads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_forward():
    model = SharedSiameseBackbone(weights=None).cuda().eval()
    images = torch.rand(1, 3, 64, 64, device="cuda")
    with torch.inference_mode():
        query, template = model(images, images)
    assert all(value.is_cuda for value in list(query.values()) + list(template.values()))
