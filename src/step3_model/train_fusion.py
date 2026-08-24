"""Supervised heatmap targets and losses for multi-scale fusion training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def build_defect_heatmaps(targets: list[dict[str, torch.Tensor]], height: int = 800, width: int = 800,
                          device: torch.device | None = None) -> torch.Tensor:
    heatmaps = torch.zeros((len(targets), 1, height, width), dtype=torch.float32, device=device)
    for index, target in enumerate(targets):
        source_h, source_w = [int(v) for v in target["size"].tolist()]
        for box in target["boxes"]:
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            left = max(0, min(width, int(x1 * width / source_w)))
            top = max(0, min(height, int(y1 * height / source_h)))
            right = max(0, min(width, int((x2 * width / source_w) + 0.999999)))
            bottom = max(0, min(height, int((y2 * height / source_h) + 0.999999)))
            if right > left and bottom > top:
                heatmaps[index, 0, top:bottom, left:right] = 1.0
    return heatmaps


class FusionHeatmapLoss(nn.Module):
    def __init__(self, positive_weight: float = 10.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.positive_weight = float(positive_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        positive_weight = torch.tensor(self.positive_weight, device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), pos_weight=positive_weight)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        total = bce + self.dice_weight * dice
        return total, {"loss_total": float(total.item()), "loss_bce": float(bce.item()), "loss_dice": float(dice.item())}
