import pytest
import torch

from step3_model.backbone import SharedSiameseBackbone


@pytest.fixture(scope="module")
def model():
    return SharedSiameseBackbone(weights=None)


def test_shared_backbone_returns_single_tensor_per_branch(model):
    model.eval()
    images = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        query, template = model(images, images.clone())
    assert isinstance(query, torch.Tensor) and isinstance(template, torch.Tensor)
    assert query.shape == template.shape == (1, 2048, 2, 2)
    assert torch.equal(query, template)


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


def test_both_branches_backpropagate_to_shared_parameters():
    model = SharedSiameseBackbone(weights=None).train()
    query, template = model(torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32))
    (query.mean() + template.mean()).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_forward():
    model = SharedSiameseBackbone(weights=None).cuda().eval()
    images = torch.rand(1, 3, 64, 64, device="cuda")
    with torch.inference_mode():
        query, template = model(images, images)
    assert query.is_cuda and template.is_cuda
