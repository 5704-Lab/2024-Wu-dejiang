import inspect

import pytest
import torch

from step3_model.crop import (
    boxes_fully_contained,
    build_crops,
    crop_image_tensor,
    expand_box,
    inverse_map_boxes,
    map_boxes_to_crop,
    merge_overlapping_boxes,
)
from step3_model.run_crop import _resolve_device, _validate_config, run_crop_smoke


# ---------------------------------------------------------------------------
# expand_box tests
# ---------------------------------------------------------------------------

def test_expand_box_center_invariant():
    """Center should stay approximately the same after expansion."""
    box = torch.tensor([100.0, 100.0, 200.0, 200.0])
    x, y, w, h = expand_box(box, k=2.0, image_size=(800, 800))
    cx_crop = x + w / 2
    cy_crop = y + h / 2
    assert abs(cx_crop - 150) < 1.0
    assert abs(cy_crop - 150) < 1.0


def test_expand_box_boundary_clip():
    """Box at the edge should clip to image boundary."""
    box = torch.tensor([0.0, 0.0, 20.0, 20.0])
    x, y, w, h = expand_box(box, k=3.0, image_size=(800, 800))
    assert x >= 0
    assert y >= 0
    assert x + w <= 800
    assert y + h <= 800


def test_expand_box_minimum_size():
    """Very small box should still produce at least 2x2 crop."""
    box = torch.tensor([100.0, 100.0, 102.0, 102.0])
    x, y, w, h = expand_box(box, k=1.0, image_size=(800, 800))
    assert w >= 2
    assert h >= 2


# ---------------------------------------------------------------------------
# merge_overlapping_boxes tests
# ---------------------------------------------------------------------------

def test_merge_overlapping_boxes_merges():
    # Boxes with high overlap: IoU((10,10,50,50), (20,20,60,60)) = 30*30 / (40*40+40*40-30*30) ≈ 0.39 > 0.3
    boxes = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60], [200, 200, 250, 250]], dtype=torch.float32)
    merged = merge_overlapping_boxes(boxes, iou_threshold=0.3)
    # First two overlap → merged; third is separate
    assert merged.shape[0] == 2


def test_merge_non_overlapping_boxes():
    boxes = torch.tensor([[10, 10, 50, 50], [200, 200, 250, 250]], dtype=torch.float32)
    merged = merge_overlapping_boxes(boxes, iou_threshold=0.3)
    assert merged.shape[0] == 2


def test_merge_empty():
    boxes = torch.empty((0, 4), dtype=torch.float32)
    merged = merge_overlapping_boxes(boxes, iou_threshold=0.3)
    assert merged.shape[0] == 0


def test_merge_single_box():
    boxes = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
    merged = merge_overlapping_boxes(boxes, iou_threshold=0.3)
    assert merged.shape[0] == 1
    assert torch.allclose(merged[0], boxes[0])


# ---------------------------------------------------------------------------
# coordinate mapping tests
# ---------------------------------------------------------------------------

def test_map_boxes_to_crop():
    boxes = torch.tensor([[60, 60, 80, 80]], dtype=torch.float32)
    mapped = map_boxes_to_crop(boxes, (50, 50, 100, 100))
    assert torch.allclose(mapped[0], torch.tensor([10.0, 10.0, 30.0, 30.0]))


def test_map_boxes_to_crop_clamp():
    boxes = torch.tensor([[-10, -10, 200, 200]], dtype=torch.float32)
    mapped = map_boxes_to_crop(boxes, (0, 0, 100, 100))
    # Box extends beyond crop → clamped
    assert mapped[0, 0] >= 0 and mapped[0, 2] <= 100


def test_map_boxes_to_crop_empty():
    boxes = torch.empty((0, 4), dtype=torch.float32)
    mapped = map_boxes_to_crop(boxes, (0, 0, 100, 100))
    assert mapped.shape[0] == 0


# ---------------------------------------------------------------------------
# inverse mapping tests
# ---------------------------------------------------------------------------

def test_inverse_map_boxes():
    """Round-trip: map to crop, then inverse map back → original (for boxes fully inside)."""
    crop_rect = (50, 50, 100, 100)
    boxes = torch.tensor([[60, 60, 80, 80]], dtype=torch.float32)
    mapped = map_boxes_to_crop(boxes, crop_rect)
    restored = inverse_map_boxes(mapped, crop_rect)
    assert torch.allclose(restored[0], boxes[0])


def test_inverse_map_boxes_multiple():
    crop_rect = (100, 50, 200, 150)
    boxes = torch.tensor([[120, 70, 180, 130], [200, 100, 280, 160]], dtype=torch.float32)
    mapped = map_boxes_to_crop(boxes, crop_rect)
    restored = inverse_map_boxes(mapped, crop_rect)
    assert torch.allclose(restored, boxes)


def test_inverse_map_boxes_empty():
    boxes = torch.empty((0, 4), dtype=torch.float32)
    restored = inverse_map_boxes(boxes, (10, 20, 30, 40))
    assert restored.shape[0] == 0


def test_map_inverse_map_roundtrip_random():
    """Random boxes fully inside a region → roundtrip preserves coordinates."""
    crop_rect = (100, 100, 400, 300)
    x0, y0, w, h = crop_rect
    for _ in range(10):
        x1 = x0 + torch.rand(1).item() * (w - 20)
        y1 = y0 + torch.rand(1).item() * (h - 20)
        x2 = x1 + 10 + torch.rand(1).item() * min(50, w - (x1 - x0) - 10)
        y2 = y1 + 10 + torch.rand(1).item() * min(50, h - (y1 - y0) - 10)
        boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        mapped = map_boxes_to_crop(boxes, crop_rect)
        restored = inverse_map_boxes(mapped, crop_rect)
        assert torch.allclose(restored, boxes, atol=1e-4), f"Failed at box {boxes.tolist()}"


# ---------------------------------------------------------------------------
# crop_image_tensor tests
# ---------------------------------------------------------------------------

def test_crop_image_tensor_shapes():
    img = torch.rand(3, 256, 256)
    cropped = crop_image_tensor(img, (50, 50, 100, 80))
    assert cropped.shape == (3, 80, 100)


# ---------------------------------------------------------------------------
# boxes_fully_contained tests
# ---------------------------------------------------------------------------

def test_boxes_fully_contained():
    boxes = torch.tensor([[60, 60, 80, 80], [200, 200, 300, 300]], dtype=torch.float32)
    idx = boxes_fully_contained(boxes, (50, 50, 100, 100))
    assert idx.tolist() == [0]  # only first box inside


def test_boxes_fully_contained_empty():
    boxes = torch.empty((0, 4), dtype=torch.float32)
    idx = boxes_fully_contained(boxes, (0, 0, 100, 100))
    assert idx.numel() == 0


# ---------------------------------------------------------------------------
# build_crops integration tests
# ---------------------------------------------------------------------------

def test_build_crops_with_synthetic_data():
    """Synthetic 128x128 image with one candidate box and one GT box."""
    query = torch.rand(3, 128, 128)
    template = torch.rand(3, 128, 128)
    candidates = torch.tensor([[40, 40, 60, 60]], dtype=torch.float32)
    gt = {
        "boxes": torch.tensor([[45, 45, 55, 55]], dtype=torch.float32),
        "labels": torch.tensor([0], dtype=torch.int64),
    }
    crops = build_crops(query, template, candidates, gt, expand_k=2.0, merge_iou=0.3, image_size=(128, 128))
    assert len(crops) >= 1
    crop = crops[0]
    assert "query_crop" in crop
    assert "template_crop" in crop
    assert crop["query_crop"].shape == crop["template_crop"].shape


def test_build_crops_empty_candidates():
    query = torch.rand(3, 128, 128)
    template = torch.rand(3, 128, 128)
    candidates = torch.empty((0, 4), dtype=torch.float32)
    gt = {"boxes": torch.empty((0, 4)), "labels": torch.empty(0, dtype=torch.int64)}
    crops = build_crops(query, template, candidates, gt)
    assert len(crops) == 0


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------

def valid_config():
    return {
        "schema_version": "step3-crop-config-v1",
        "roadmap_step": "4-A",
        "crop": {"expand_k": 1.8, "merge_iou_threshold": 0.3},
        "run": {"purpose": "crop_smoke", "split": "val", "selection": "ordered_prefix",
                "sample_count": 40, "batch_size": 1, "device": "cpu", "num_workers": 0, "inference_dtype": "float32"},
    }


def test_crop_config_contract():
    _validate_config(valid_config())


def test_crop_config_rejects_wrong_schema():
    config = valid_config()
    config["roadmap_step"] = "3.5-C"
    with pytest.raises(ValueError, match="schema"):
        _validate_config(config)


def test_crop_config_rejects_nonpositive_expand_k():
    config = valid_config()
    config["crop"]["expand_k"] = 0
    with pytest.raises(ValueError, match="expand_k"):
        _validate_config(config)


def test_device_resolution_cpu():
    assert _resolve_device("cpu").type == "cpu"


def test_explicit_cuda_fails_when_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        _resolve_device("cuda")


def test_crop_runner_constructs_fusion_before_loading_checkpoint():
    """回归检查：Fusion 必须先实例化，之后才能向实例加载监督训练权重。"""
    source = inspect.getsource(run_crop_smoke)
    construct_position = source.index("fusion = MultiScaleFusion(")
    load_position = source.index("load_model_checkpoint(fusion")
    assert construct_position < load_position
