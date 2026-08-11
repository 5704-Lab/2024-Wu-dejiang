import copy

import pytest

from step3_model.backbone import FEATURE_STAGES
from step3_model.run_backbone import _resolve_device, _validate_config


def _feature_stages():
    return [
        {"name": name, "stage": stage, "channels": channels, "stride": stride}
        for name, stage, channels, stride in FEATURE_STAGES
    ]


def valid_config():
    return {
        "schema_version": "step3-backbone-config-v1",
        "roadmap_step": "3.3-B",
        "model": {
            "architecture": "resnet50", "weights": "none",
            "multi_scale_output": True, "feature_stages": _feature_stages(),
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        },
        "run": {"purpose": "backbone_smoke", "split": "val", "selection": "ordered_prefix",
                "sample_count": 2, "batch_size": 1, "device": "cpu", "inference_dtype": "float32"},
    }


def test_backbone_config_contract():
    _validate_config(valid_config())


def test_backbone_config_rejects_single_scale_and_old_step():
    config = valid_config()
    config["model"]["multi_scale_output"] = False
    with pytest.raises(ValueError, match="multi_scale_output"):
        _validate_config(config)
    config = valid_config()
    config["roadmap_step"] = "3.3-A"
    with pytest.raises(ValueError, match="schema"):
        _validate_config(config)


@pytest.mark.parametrize("mutate", [
    lambda stages: stages.reverse(),
    lambda stages: stages.__setitem__(0, {**stages[0], "stride": 16}),
    lambda stages: stages.pop(),
    lambda stages: stages.__setitem__(2, {**stages[2], "channels": 512}),
])
def test_feature_stages_must_match_canonical(mutate):
    config = valid_config()
    mutate(config["model"]["feature_stages"])
    with pytest.raises(ValueError, match="feature_stages"):
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
