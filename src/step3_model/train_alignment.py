"""Micro-alignment loss and training loops for Step 3.4-C."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .backbone import SharedSiameseBackbone
from .feature_warp import FeatureWarp
from .offset_predictor import OFFSET_SPEC, OffsetPredictor

_STAGE_NAMES = [name for name, _, _ in OFFSET_SPEC]


class MicroAlignmentLoss(nn.Module):
    """Per-scale weighted MSE between warped template features and query features.

    loss = sum_{scale} alpha[scale] * MSE(warped[scale], query[scale])

    Each scale's MSE is already normalized by its own C×H×W elements, so
    the alphas control relative importance across scales.
    """

    def __init__(self, alpha_c3: float = 1.0, alpha_c4: float = 1.0, alpha_c5: float = 1.0) -> None:
        super().__init__()
        self.alphas = {"C3": alpha_c3, "C4": alpha_c4, "C5": alpha_c5}

    def forward(
        self,
        warped: dict[str, torch.Tensor],
        query_features: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = torch.tensor(0.0, device=query_features["C3"].device)
        stats: dict[str, float] = {}
        for key in _STAGE_NAMES:
            mse = F.mse_loss(warped[key], query_features[key])
            weighted = self.alphas[key] * mse
            total = total + weighted
            stats[f"loss_{key}"] = float(mse.item())
        stats["loss_total"] = float(total.item())
        return total, stats


def _build_pipeline(
    device: torch.device,
    kernel_size: int = 3,
    hidden_ratio: int = 4,
    zero_init: bool = True,
) -> tuple[SharedSiameseBackbone, OffsetPredictor, FeatureWarp]:
    backbone = SharedSiameseBackbone("IMAGENET1K_V2").to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    predictor = OffsetPredictor(
        kernel_size=kernel_size, hidden_ratio=hidden_ratio, zero_init=zero_init,
    ).to(device)
    warp = FeatureWarp(kernel_size=kernel_size).to(device).eval()
    return backbone, predictor, warp


def train_one_epoch(
    backbone: SharedSiameseBackbone,
    predictor: OffsetPredictor,
    warp: FeatureWarp,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    criterion: MicroAlignmentLoss,
    device: torch.device,
) -> dict[str, float]:
    predictor.train()
    total_loss = 0.0
    per_scale_losses = {"C3": 0.0, "C4": 0.0, "C5": 0.0}
    count = 0
    for batch in loader:
        query = batch["query"].to(device)
        template = batch["template"].to(device)
        with torch.no_grad():
            qf, tf = backbone(query, template)
        offsets = predictor(qf, tf)
        warped = warp(tf, offsets)
        loss, stats = criterion(warped, qf)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += stats["loss_total"]
        for key in _STAGE_NAMES:
            per_scale_losses[key] += stats[f"loss_{key}"]
        count += 1
    n = max(count, 1)
    return {"train_loss": total_loss / n, **{f"train_loss_{k}": v / n for k, v in per_scale_losses.items()}}


@torch.no_grad()
def validate_one_epoch(
    backbone: SharedSiameseBackbone,
    predictor: OffsetPredictor,
    warp: FeatureWarp,
    loader: Any,
    criterion: MicroAlignmentLoss,
    device: torch.device,
) -> dict[str, float]:
    predictor.eval()
    total_loss = 0.0
    per_scale_losses = {"C3": 0.0, "C4": 0.0, "C5": 0.0}
    count = 0
    for batch in loader:
        query = batch["query"].to(device)
        template = batch["template"].to(device)
        qf, tf = backbone(query, template)
        offsets = predictor(qf, tf)
        warped = warp(tf, offsets)
        _loss, stats = criterion(warped, qf)
        total_loss += stats["loss_total"]
        for key in _STAGE_NAMES:
            per_scale_losses[key] += stats[f"loss_{key}"]
        count += 1
    n = max(count, 1)
    return {"val_loss": total_loss / n, **{f"val_loss_{k}": v / n for k, v in per_scale_losses.items()}}
