import pytest

from step3_model.run_train_alignment import _resolve_device, _validate_config


def valid_config():
    return {
        "schema_version": "step3-training-config-v1",
        "roadmap_step": "3.4-C",
        "model": {"kernel_size": 3, "hidden_ratio": 4, "zero_init": True},
        "training": {
            "epochs": 20, "batch_size": 8, "num_workers": 0,
            "device": "cpu", "lr": 0.001, "weight_decay": 0.0,
            "scheduler": "cosine", "validation_interval": 1,
        },
        "loss": {"per_scale_weights": {"C3": 1.0, "C4": 1.0, "C5": 1.0}},
    }


def test_train_config_contract():
    _validate_config(valid_config())


def test_train_config_rejects_wrong_schema():
    config = valid_config()
    config["roadmap_step"] = "3.4-B"
    with pytest.raises(ValueError, match="schema"):
        _validate_config(config)


def test_train_config_rejects_zero_epochs():
    config = valid_config()
    config["training"]["epochs"] = 0
    with pytest.raises(ValueError, match="epochs"):
        _validate_config(config)


def test_train_config_rejects_negative_lr():
    config = valid_config()
    config["training"]["lr"] = -0.001
    with pytest.raises(ValueError, match="lr"):
        _validate_config(config)


def test_train_config_rejects_missing_loss_weight():
    config = valid_config()
    del config["loss"]["per_scale_weights"]["C4"]
    with pytest.raises(ValueError, match="per_scale_weight"):
        _validate_config(config)


def test_train_config_rejects_wrong_device():
    config = valid_config()
    config["training"]["device"] = "tpu"
    with pytest.raises(ValueError, match="device"):
        _validate_config(config)


def test_train_config_rejects_nonzero_init():
    config = valid_config()
    config["model"]["zero_init"] = False
    with pytest.raises(ValueError, match="zero_init"):
        _validate_config(config)


def test_train_config_rejects_wrong_scheduler():
    config = valid_config()
    config["training"]["scheduler"] = "step"
    with pytest.raises(ValueError, match="scheduler"):
        _validate_config(config)


def test_device_resolution_cpu():
    assert _resolve_device("cpu").type == "cpu"


def test_explicit_cuda_fails_when_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        _resolve_device("cuda")
