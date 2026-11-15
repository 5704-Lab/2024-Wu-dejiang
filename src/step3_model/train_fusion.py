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


def build_defect_heatmaps_with_weights(
    targets: list[dict[str, torch.Tensor]],
    class_positive_weights: dict[int, float] | None = None,
    height: int = 800,
    width: int = 800,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build defect heatmaps plus per-pixel positive weights for class-aware BCE."""
    class_positive_weights = class_positive_weights or {}
    heatmaps = torch.zeros((len(targets), 1, height, width), dtype=torch.float32, device=device)
    weights = torch.ones((len(targets), 1, height, width), dtype=torch.float32, device=device)
    for index, target in enumerate(targets):
        source_h, source_w = [int(v) for v in target["size"].tolist()]
        for box, label in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            left = max(0, min(width, int(x1 * width / source_w)))
            top = max(0, min(height, int(y1 * height / source_h)))
            right = max(0, min(width, int((x2 * width / source_w) + 0.999999)))
            bottom = max(0, min(height, int((y2 * height / source_h) + 0.999999)))
            if right > left and bottom > top:
                heatmaps[index, 0, top:bottom, left:right] = 1.0
                class_id = int(label.item())
                positive_weight = float(class_positive_weights.get(class_id, 1.0))
                weights[index, 0, top:bottom, left:right] = positive_weight
    return heatmaps, weights


class FusionHeatmapLoss(nn.Module):
    def __init__(self, positive_weight: float = 10.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.positive_weight = float(positive_weight)
        self.dice_weight = float(dice_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pixel_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        targets = targets.to(logits.dtype)
        if pixel_weights is not None:
            if pixel_weights.shape != logits.shape:
                raise ValueError(
                    f"pixel_weights shape {pixel_weights.shape} must match logits {logits.shape}"
                )
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, weight=pixel_weights.to(logits.dtype), reduction="mean"
            )
        else:
            positive_weight = torch.tensor(self.positive_weight, device=logits.device, dtype=logits.dtype)
            bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weight)
        probabilities = torch.sigmoid(logits)
        if pixel_weights is not None:
            weighted_targets = targets * pixel_weights.to(logits.dtype)
            intersection = (probabilities * weighted_targets).sum(dim=(1, 2, 3))
            denominator = ((probabilities + targets) * pixel_weights.to(logits.dtype)).sum(dim=(1, 2, 3))
        else:
            intersection = (probabilities * targets).sum(dim=(1, 2, 3))
            denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        total = bce + self.dice_weight * dice
        return total, {"loss_total": float(total.item()), "loss_bce": float(bce.item()), "loss_dice": float(dice.item())}
