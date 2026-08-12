import pytest

from step3_model.run_feature_warp import _resolve_device, _validate_config


def valid_config():
    return {
        "schema_version": "step3-feature-warp-config-v1",
        "roadmap_step": "3.4-B",
        "model": {
            "architecture": "feature_warp", "kernel_size": 3,
            "identity_kernel": True, "warp_input": "template_features",
        },
        "run": {
            "purpose": "feature_warp_smoke", "split": "val",
            "selection": "ordered_prefix", "sample_count": 2,
            "batch_size": 1, "device": "cpu", "inference_dtype": "float32",
            "dcn_warp_started": True, "train_started": False,
        },
    }


def test_feature_warp_config_contract():
    _validate_config(valid_config())


def test_feature_warp_config_rejects_wrong_schema():
    config = valid_config()
    config["roadmap_step"] = "3.4-A"
    with pytest.raises(ValueError, match="schema"):
        _validate_config(config)
    config = valid_config()
    config["schema_version"] = "step3-offset-config-v1"
    with pytest.raises(ValueError, match="schema"):
        _validate_config(config)


def test_feature_warp_config_rejects_wrong_architecture():
    config = valid_config()
    config["model"]["architecture"] = "offset_predictor"
    with pytest.raises(ValueError, match="architecture"):
        _validate_config(config)


def test_feature_warp_config_rejects_non_identity_kernel():
    config = valid_config()
    config["model"]["identity_kernel"] = False
    with pytest.raises(ValueError, match="identity_kernel"):
        _validate_config(config)


def test_feature_warp_config_rejects_wrong_kernel_size():
    config = valid_config()
    config["model"]["kernel_size"] = 5
    with pytest.raises(ValueError, match="kernel_size"):
        _validate_config(config)


def test_feature_warp_config_rejects_training():
    config = valid_config()
    config["run"]["train_started"] = True
    with pytest.raises(ValueError, match="train"):
        _validate_config(config)


def test_feature_warp_config_requires_dcn_warp_started():
    config = valid_config()
    config["run"]["dcn_warp_started"] = False
    with pytest.raises(ValueError, match="dcn_warp"):
        _validate_config(config)


def test_feature_warp_config_rejects_test_split():
    config = valid_config()
    config["run"]["split"] = "test"
    with pytest.raises(ValueError, match="smoke contract"):
        _validate_config(config)


def test_device_resolution_cpu():
    assert _resolve_device("cpu").type == "cpu"


def test_explicit_cuda_fails_when_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        _resolve_device("cuda")
