import pytest
import torch

from step3_model.coco_exporter import export_coco_dataset
from step3_model.inference_bridge import _nms_detections, pad_image_to_size
from step3_model.run_detection import _dummy_rtdetr, _validate_config


# ---------------------------------------------------------------------------
# pad_image_to_size tests
# ---------------------------------------------------------------------------

def test_pad_image_no_padding_needed():
    img = torch.rand(3, 800, 800)
    padded = pad_image_to_size(img, (800, 800))
    assert torch.equal(padded, img)


def test_pad_image_adds_padding():
    img = torch.ones(3, 200, 300)
    padded = pad_image_to_size(img, (800, 800))
    assert padded.shape == (3, 800, 800)
    # Original region unchanged
    assert torch.equal(padded[:, :200, :300], img)
    # Padding area is fill value (0)
    assert padded[:, 200:, :].sum() == 0
    assert padded[:, :, 300:].sum() == 0


def test_pad_image_fill_value():
    img = torch.ones(3, 100, 100)
    padded = pad_image_to_size(img, (800, 800), fill=0.5)
    assert torch.equal(padded[:, :100, :100], img)
    assert abs(padded[0, 100, 100] - 0.5) < 1e-6


def test_pad_image_various_sizes():
    for h, w in [(50, 50), (800, 100), (300, 800)]:
        img = torch.rand(3, h, w)
        padded = pad_image_to_size(img, (800, 800))
        assert padded.shape == (3, 800, 800)


# ---------------------------------------------------------------------------
# NMS tests
# ---------------------------------------------------------------------------

def test_nms_detections_empty():
    result = _nms_detections([])
    assert result == []


def test_nms_detections_single():
    dets = [{"box": [10, 10, 50, 50], "score": 0.9, "class": 0}]
    result = _nms_detections(dets)
    assert len(result) == 1


def test_nms_detections_duplicate():
    dets = [
        {"box": [10, 10, 50, 50], "score": 0.9, "class": 0},
        {"box": [12, 12, 52, 52], "score": 0.8, "class": 0},  # heavily overlapping
    ]
    result = _nms_detections(dets, iou_threshold=0.5)
    assert len(result) == 1
    assert result[0]["score"] == 0.9


# ---------------------------------------------------------------------------
# Dummy RT-DETR tests
# ---------------------------------------------------------------------------

def test_dummy_rtdetr_returns_empty():
    img = torch.rand(1, 3, 800, 800)
    result = _dummy_rtdetr(img)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (0, 6)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def valid_config():
    return {
        "schema_version": "step3-detection-config-v1",
        "run": {"device": "cpu", "purpose": "detection_smoke", "sample_count": 4},
    }


def test_config_contract():
    _validate_config(valid_config())


def test_config_rejects_bad_device():
    config = valid_config()
    config["run"]["device"] = "tpu"
    with pytest.raises(ValueError, match="device"):
        _validate_config(config)
