import numpy as np
import pytest
import torch

from step3_model.proposals import (
    _heatmap_to_mask,
    _mask_to_boxes,
    evaluate_recall,
    postprocess_batch,
    postprocess_heatmap,
)
from step3_model.run_proposals import _resolve_device, _validate_config


# ---------------------------------------------------------------------------
# Postprocessing unit tests
# ---------------------------------------------------------------------------

def test_postprocess_heatmap_empty_heatmap():
    """All-zero heatmap → no boxes."""
    hm = np.zeros((200, 200), dtype=np.float32)
    boxes = postprocess_heatmap(hm, adaptive_block=11)
    assert boxes.shape[0] == 0
    assert boxes.shape == (0, 4)


def test_postprocess_heatmap_synthetic_blob():
    """Synthetic bright rectangle → one box with reasonable IoU."""
    hm = np.zeros((256, 256), dtype=np.float32)
    hm[50:80, 60:120] = 1.0  # bright blob at (60,50)→(120,80)
    boxes = postprocess_heatmap(hm, min_area=16, max_aspect_ratio=10, adaptive_block=11)
    assert boxes.shape[0] >= 1, f"Expected at least 1 box, got {boxes.shape[0]}"
    # The detected box should have reasonable overlap with the synthetic blob
    gt = np.array([[60, 50, 120, 80]], dtype=np.float32)
    from torchvision.ops import box_iou
    ious = box_iou(torch.from_numpy(boxes), torch.from_numpy(gt))
    assert ious.max().item() > 0.3, f"Expected IoU > 0.3, got {ious.max().item()}"


def test_mask_to_boxes_min_area_filtering():
    """Small regions are filtered by min_area."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:18, 10:18] = 255  # area = 64
    mask[30:34, 30:34] = 255  # area = 16, too small
    boxes = _mask_to_boxes(mask, min_area=32.0)
    # Should only get the 8x8 box, not the 4x4
    assert boxes.shape[0] == 1


def test_mask_to_boxes_aspect_ratio_filtering():
    """Very wide/narrow boxes are filtered."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:50, 10:90] = 255  # 80x10, aspect 8:1
    mask[60:70, 30:50] = 255  # 20x10, aspect 2:1
    boxes = _mask_to_boxes(mask, min_area=8, max_aspect_ratio=5.0)
    # The 80x10 should be filtered, 20x10 kept
    assert boxes.shape[0] == 1


def test_postprocess_batch():
    """Batch of 2 identical heatmaps → 2 result tensors."""
    hm = torch.zeros(2, 1, 128, 128)
    hm[:, :, 30:50, 40:70] = 0.8
    results = postprocess_batch(hm, min_area=32, adaptive_block=11)
    assert len(results) == 2
    assert all(isinstance(r, torch.Tensor) for r in results)
    assert all(r.shape[1] == 4 for r in results)


# ---------------------------------------------------------------------------
# Recall evaluation unit tests
# ---------------------------------------------------------------------------

def _make_gt(boxes_list, labels_list):
    """Build gt_targets list from boxes/labels."""
    targets = []
    for boxes, labels in zip(boxes_list, labels_list):
        targets.append({
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        })
    return targets


def test_evaluate_recall_perfect():
    """GT boxes exactly matched → recall@0.5 = 1.0."""
    pred = [torch.tensor([[10, 10, 50, 50], [60, 60, 90, 90]], dtype=torch.float32)]
    gt = _make_gt([[[10, 10, 50, 50], [60, 60, 90, 90]]], [[0, 1]])
    stats = evaluate_recall(pred, gt, (0.5,))
    assert stats["recall@0.5"] == 1.0


def test_evaluate_recall_no_candidates():
    """No candidate boxes → recall = 0."""
    pred = [torch.empty((0, 4), dtype=torch.float32)]
    gt = _make_gt([[[10, 10, 50, 50]]], [[0]])
    stats = evaluate_recall(pred, gt, (0.5,))
    assert stats["recall@0.5"] == 0.0


def test_evaluate_recall_partial():
    """1 of 2 GT boxes matched → recall@0.5 = 0.5."""
    pred = [torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)]
    gt = _make_gt([[[10, 10, 50, 50], [200, 200, 300, 300]]], [[0, 1]])
    stats = evaluate_recall(pred, gt, (0.5,))
    assert stats["recall@0.5"] == 0.5


def test_evaluate_recall_no_gt():
    """Sample with zero GT boxes should not crash."""
    pred = [torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)]
    gt = _make_gt([np.empty((0, 4), dtype=np.float32)], [np.empty(0, dtype=np.int64)])
    stats = evaluate_recall(pred, gt, (0.5,))
    assert stats["total_gt_boxes"] == 0
    assert stats["recall@0.5"] == 0.0


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------

def valid_config():
    return {
        "schema_version": "step3-proposals-config-v1",
        "roadmap_step": "3.5-C",
        "proposals": {"min_area": 64, "max_aspect_ratio": 5.0, "adaptive_block": 51, "adaptive_c": -5, "morph_kernel": 5, "max_area_ratio": 0.8},
        "evaluation": {"iou_thresholds": [0.3, 0.5, 0.7]},
        "run": {"purpose": "proposals_smoke", "split": "val", "selection": "ordered_prefix",
                "sample_count": 40, "batch_size": 1, "device": "cpu", "num_workers": 0, "inference_dtype": "float32"},
    }


def test_proposals_config_contract():
    _validate_config(valid_config())


def test_proposals_config_rejects_wrong_schema():
    config = valid_config()
    config["roadmap_step"] = "3.5-B"
    with pytest.raises(ValueError, match="schema"):
        _validate_config(config)


def test_proposals_config_rejects_nonpositive_min_area():
    config = valid_config()
    config["proposals"]["min_area"] = 0
    with pytest.raises(ValueError, match="min_area"):
        _validate_config(config)


def test_proposals_config_rejects_bad_iou_thresholds():
    config = valid_config()
    config["evaluation"]["iou_thresholds"] = [0.0, 1.0]
    with pytest.raises(ValueError, match="IoU"):
        _validate_config(config)


def test_device_resolution_cpu():
    assert _resolve_device("cpu").type == "cpu"


def test_explicit_cuda_fails_when_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        _resolve_device("cuda")
