"""Candidate box post-processing and recall evaluation for Step 3.5-C."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou


def _heatmap_to_mask(
    heatmap: np.ndarray,
    adaptive_block: int = 51,
    adaptive_c: int = -5,
    morph_kernel: int = 5,
) -> np.ndarray:
    """Convert a continuous heatmap to a binary mask via adaptive threshold + morph close.

    Args:
        heatmap: [H, W] float32 or float64, unnormalized.
        adaptive_block: block size for adaptiveThreshold (must be odd).
        adaptive_c: constant subtracted from the mean.
        morph_kernel: kernel size for morphological close.

    Returns:
        [H, W] uint8 binary mask (0 or 255).
    """
    # Normalize to [0, 255]
    vmin, vmax = float(heatmap.min()), float(heatmap.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(heatmap, dtype=np.uint8)
    normalized = np.clip((heatmap - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

    # Gaussian adaptive thresholding
    block = max(3, adaptive_block)
    if block % 2 == 0:
        block += 1
    binary = cv2.adaptiveThreshold(
        normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, adaptive_c,
    )

    # Morphological close: merge nearby regions
    if morph_kernel >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


def _mask_to_boxes(
    mask: np.ndarray,
    min_area: float = 64.0,
    max_aspect_ratio: float = 5.0,
    max_area_ratio: float = 0.8,
) -> np.ndarray:
    """Extract axis-aligned bounding boxes from a binary mask.

    Args:
        mask: [H, W] uint8 binary (0 or 255).
        min_area: minimum box area in pixels.
        max_aspect_ratio: maximum width/height ratio (or inverse).
        max_area_ratio: maximum box area as fraction of image area (0→1).

    Returns:
        [K, 4] float32 xyxy boxes.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    max_area_px = max_area_ratio * h * w
    boxes: list[list[float]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < min_area or area > max_area_px:
            continue
        aspect = bw / bh if bh > 0 else 0
        if aspect > max_aspect_ratio or (aspect > 0 and 1.0 / aspect > max_aspect_ratio):
            continue
        boxes.append([float(x), float(y), float(x + bw), float(y + bh)])
    if not boxes:
        return np.empty((0, 4), dtype=np.float32)
    return np.array(boxes, dtype=np.float32)


def postprocess_heatmap(
    heatmap: np.ndarray,
    min_area: float = 64.0,
    max_aspect_ratio: float = 5.0,
    adaptive_block: int = 51,
    adaptive_c: int = -5,
    morph_kernel: int = 5,
) -> np.ndarray:
    """Convert a single heatmap [H, W] to candidate xyxy boxes [K, 4].

    Pipeline: normalize → adaptive threshold → morph close → findContours → boundingRect → filter.
    """
    mask = _heatmap_to_mask(heatmap, adaptive_block, adaptive_c, morph_kernel)
    return _mask_to_boxes(mask, min_area, max_aspect_ratio)


def postprocess_batch(
    heatmaps: torch.Tensor,
    min_area: float = 64.0,
    max_aspect_ratio: float = 5.0,
    adaptive_block: int = 51,
    adaptive_c: int = -5,
    morph_kernel: int = 5,
) -> list[torch.Tensor]:
    """Convert [B, 1, H, W] enhanced heatmaps to per-sample candidate box lists.

    Returns:
        List of [K_i, 4] tensors (xyxy, float32), one per batch element.
    """
    results: list[torch.Tensor] = []
    for idx in range(heatmaps.shape[0]):
        hm = heatmaps[idx, 0].detach().cpu().numpy()
        boxes_np = postprocess_heatmap(hm, min_area, max_aspect_ratio, adaptive_block, adaptive_c, morph_kernel)
        results.append(torch.from_numpy(boxes_np))
    return results


def evaluate_recall(
    pred_boxes_list: list[torch.Tensor],
    gt_targets_list: list[dict[str, torch.Tensor]],
    iou_thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> dict[str, float | int]:
    """Compute per-GT recall: fraction of GT boxes matched by at least one candidate.

    Args:
        pred_boxes_list: [K_i, 4] xyxy per sample.
        gt_targets_list: target dicts (containing "boxes" [N_i, 4] xyxy, "labels" [N_i]).
        iou_thresholds: IoU thresholds for matching.

    Returns:
        Dict with recall@{t}, total_candidate_count, total_gt_count, etc.
    """
    total_gt = 0
    matched = {t: 0 for t in iou_thresholds}
    candidate_count = 0

    for pred, target in zip(pred_boxes_list, gt_targets_list):
        gt_boxes = target["boxes"]  # [N, 4] xyxy
        if gt_boxes.numel() == 0:
            continue
        total_gt += gt_boxes.shape[0]
        candidate_count += pred.shape[0]

        if pred.shape[0] == 0:
            continue

        ious = box_iou(pred, gt_boxes)  # [K, N]
        best_iou_per_gt = ious.max(dim=0).values  # [N]
        for t in iou_thresholds:
            matched[t] += int((best_iou_per_gt >= t).sum().item())

    return {
        **{f"recall@{t}": matched[t] / max(total_gt, 1) for t in iou_thresholds},
        "total_gt_boxes": total_gt,
        "total_candidate_boxes": int(candidate_count),
        "candidates_per_gt": candidate_count / max(total_gt, 1),
    }
