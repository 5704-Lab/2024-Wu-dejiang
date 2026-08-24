from pathlib import Path

import pytest
import torch

from step3_model.model_io import load_model_checkpoint


def test_checkpoint_contract_is_required(tmp_path: Path):
    source = torch.nn.Linear(2, 1)
    path = tmp_path / "model.pt"
    torch.save({"model_state_dict": source.state_dict(), "training_contract": "new-v1"}, path)
    target = torch.nn.Linear(2, 1)
    load_model_checkpoint(target, path, torch.device("cpu"), "test", "new-v1")
    for left, right in zip(source.parameters(), target.parameters()):
        assert torch.equal(left, right)


def test_checkpoint_contract_rejects_stale_weights(tmp_path: Path):
    path = tmp_path / "model.pt"
    torch.save({"model_state_dict": torch.nn.Linear(2, 1).state_dict()}, path)
    with pytest.raises(ValueError, match="contract mismatch"):
        load_model_checkpoint(torch.nn.Linear(2, 1), path, torch.device("cpu"), "test", "new-v1")
