from pathlib import Path

import pytest

from step3_model.run_offset import _resolve_device, _validate_config


def valid_config():
    return {
        "schema_version": "step3-offset-config-v1",
        "roadmap_step": "3.4-A",
        "model": {
            "architecture": "offset_predictor", "kernel_size": 3, "hidden_ratio": 4,
            "zero_init": True, "offset_semantics": "pixel_sampling_offsets",
            "max_offset_per_stage": [8, 16, 32],
        },
        "run": {"purpose": "offset_smoke", "split": "val", "selection": "ordered_prefix",
                "sample_count": 2, "batch_size": 1, "device": "cpu", "inference_dtype": "float32",
                "train_started": False, "dcn_started": False},
    }


def test_offset_config_contract():
    config = valid_config()
    _validate_config(config)
    config["model"]["zero_init"] = False
    with pytest.raises(ValueError, match="zero_init"):
        _validate_config(config)


def test_offset_config_rejects_training():
    config = valid_config()
    config["run"]["train_started"] = True
    with pytest.raises(ValueError, match="train"):
        _validate_config(config)


def test_offset_config_rejects_dcn():
    config = valid_config()
    config["run"]["dcn_started"] = True
    with pytest.raises(ValueError, match="DCN"):
        _validate_config(config)


def test_offset_config_rejects_wrong_max_offset():
    config = valid_config()
    config["model"]["max_offset_per_stage"] = [16, 32, 64]
    with pytest.raises(ValueError, match="max_offset"):
        _validate_config(config)


def test_device_resolution_cpu():
    assert _resolve_device("cpu").type == "cpu"


def test_explicit_cuda_fails_when_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        _resolve_device("cuda")