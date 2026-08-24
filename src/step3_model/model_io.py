"""Strict checkpoint loading shared by training, evaluation, and demo code."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def load_model_checkpoint(module: nn.Module, path: Path, device: torch.device, component: str,
                          expected_contract: str | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required {component} checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if expected_contract is not None and checkpoint.get("training_contract") != expected_contract:
        raise ValueError(
            f"{component} checkpoint training contract mismatch: expected {expected_contract}, "
            f"got {checkpoint.get('training_contract')!r}"
        )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"{component} checkpoint has no model_state_dict: {path}")
    module.load_state_dict(state, strict=True)
    return checkpoint
