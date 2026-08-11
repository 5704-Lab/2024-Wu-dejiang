from pathlib import Path

import pytest

from step3_model.run_backbone import _resolve_device, _validate_config


def valid_config():
    return {
        "schema_version": "step3-backbone-config-v1",
        "roadmap_step": "3.3-A",
        "model": {
            "architecture": "resnet50", "weights": "none", "output_stage": "layer4",
            "output_channels": 2048, "output_stride": 32, "multi_scale_output": False,
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        },
        "run": {"purpose": "backbone_smoke", "split": "val", "selection": "ordered_prefix",
                "sample_count": 2, "batch_size": 1, "device": "cpu", "inference_dtype": "float32"},
    }


def test_backbone_config_contract():
    config = valid_config()
    _validate_config(config)
    config["model"]["multi_scale_output"] = True
    with pytest.raises(ValueError, match="multi_scale_output"):
        _validate_config(config)


def test_backbone_config_rejects_test_split():
    config = valid_config()
    config["run"]["split"] = "test"
    with pytest.raises(ValueError, match="smoke contract"):
        _validate_config(config)


def test_backbone_config_rejects_unknown_weights():
    config = valid_config()
    config["model"]["weights"] = "latest"
    with pytest.raises(ValueError, match="weights"):
        _validate_config(config)


def test_device_resolution_cpu():
    assert _resolve_device("cpu").type == "cpu"


def test_explicit_cuda_fails_when_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        _resolve_device("cuda")
