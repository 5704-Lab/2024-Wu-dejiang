"""Context cropping with box expansion and coordinate mapping for Step 4-A."""

from __future__ import annotations

import torch
from torchvision.ops import box_iou


def expand_box(
    box_xyxy: torch.Tensor,
    k: float = 1.8,
    image_size: tuple[int, int] = (800, 800),
) -> tuple[int, int, int, int]:
    """Expand a single [4] box by factor k around its center, clipped to image bounds.

    Args:
        box_xyxy: [x1, y1, x2, y2] float or int
        k: expansion factor (width × k, height × k)
        image_size: (width, height) of the source image

    Returns:
        (x, y, w, h) crop rectangle, guaranteed within [0, 0, W, H].
    """
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = (x2 - x1) * k / 2.0, (y2 - y1) * k / 2.0
    left, right = int(cx - hw), int(cx + hw)
    top, bottom = int(cy - hh), int(cy + hh)

    # Clip to image bounds
    w_img, h_img = image_size
    left = max(0, min(left, w_img - 1))
    right = max(left + 2, min(right, w_img))
    top = max(0, min(top, h_img - 1))
    bottom = max(top + 2, min(bottom, h_img))

    w = right - left
    h = bottom - top
    return (left, top, w, h)


def merge_overlapping_boxes(
    boxes: torch.Tensor,
    iou_threshold: float = 0.3,
) -> torch.Tensor:
    """Greedy merge of overlapping boxes by IoU threshold.

    Args:
        boxes: [N, 4] xyxy
        iou_threshold: merge pairs with IoU > this

    Returns:
        [M, 4] merged boxes.
    """
    if boxes.shape[0] <= 1:
        return boxes.clone()

    remaining = boxes.clone()
    merged: list[torch.Tensor] = []

    while remaining.shape[0] > 0:
        current = remaining[0:1]  # [1, 4]
        keep_mask = torch.ones(remaining.shape[0], dtype=torch.bool)

        # Find all boxes that overlap with current
        ious = box_iou(current, remaining)[0]  # [N]
        overlap_mask = ious > iou_threshold
        overlap_mask[0] = True  # include current itself

        if overlap_mask.sum() > 1:
            group = remaining[overlap_mask]
            x1 = group[:, 0].min()
            y1 = group[:, 1].min()
            x2 = group[:, 2].max()
            y2 = group[:, 3].max()
            merged.append(torch.tensor([x1, y1, x2, y2], dtype=boxes.dtype, device=boxes.device))
        else:
            merged.append(current[0])

        keep_mask[overlap_mask] = False
        remaining = remaining[keep_mask]

    if not merged:
        return torch.empty((0, 4), dtype=boxes.dtype, device=boxes.device)
    return torch.stack(merged)


def map_boxes_to_crop(
    boxes: torch.Tensor,
    crop_rect: tuple[int, int, int, int],
) -> torch.Tensor:
    """Map boxes from image coordinates to crop-local coordinates.

    Args:
        boxes: [N, 4] xyxy in image space
        crop_rect: (x, y, w, h) of crop window in image space

    Returns:
        [N, 4] xyxy in crop space (clamped to [0, w, 0, h]).
    """
    if boxes.numel() == 0:
        return boxes.clone()
    x0, y0, w, h = crop_rect
    shifted = boxes.clone()
    shifted[:, [0, 2]] -= x0
    shifted[:, [1, 3]] -= y0
    shifted[:, 0].clamp_(0, w)
    shifted[:, 1].clamp_(0, h)
    shifted[:, 2].clamp_(0, w)
    shifted[:, 3].clamp_(0, h)
    return shifted


def crop_image_tensor(
    image: torch.Tensor,
    crop_rect: tuple[int, int, int, int],
) -> torch.Tensor:
    """Crop a [C, H, W] tensor by (x, y, w, h) rect.

    Returns [C, h, w] tensor (shares storage with input — use .clone() if needed).
    """
    x, y, w, h = crop_rect
    return image[..., y:y + h, x:x + w]


def boxes_fully_contained(
    boxes: torch.Tensor,
    crop_rect: tuple[int, int, int, int],
) -> torch.Tensor:
    """Return indices of boxes fully inside the crop window.

    A box is fully contained if x1 >= x0, y1 >= y0, x2 <= x0+w, y2 <= y0+h.
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    x0, y0, w, h = crop_rect
    contained = (
        (boxes[:, 0] >= x0) & (boxes[:, 1] >= y0) &
        (boxes[:, 2] <= x0 + w) & (boxes[:, 3] <= y0 + h)
    )
    return contained.nonzero(as_tuple=False).squeeze(1)


def build_crops(
    query_img: torch.Tensor,
    template_img: torch.Tensor,
    candidate_boxes: torch.Tensor,
    gt_target: dict[str, torch.Tensor],
    expand_k: float = 1.8,
    merge_iou: float = 0.3,
    image_size: tuple[int, int] = (800, 800),
) -> list[dict[str, object]]:
    """Build context crops from candidate boxes.

    Args:
        query_img: [3, 800, 800] float32 [0,1]
        template_img: [3, 800, 800] float32 [0,1]
        candidate_boxes: [K, 4] xyxy from proposals.py
        gt_target: target dict with "boxes" [N,4], "labels" [N]
        expand_k: box expansion factor
        merge_iou: IoU threshold for merging overlapping candidates
        image_size: (W, H) of source images

    Returns:
        List of crop dicts, each containing query_crop, template_crop,
        gt_boxes (in crop coords), gt_labels, crop_rect, etc.
    """
    if candidate_boxes.numel() == 0:
        return []

    # Merge overlapping candidate boxes
    merged = merge_overlapping_boxes(candidate_boxes, merge_iou)

    crops: list[dict[str, object]] = []
    gt_boxes = gt_target["boxes"]
    gt_labels = gt_target["labels"]

    for idx, box in enumerate(merged):
        rect = expand_box(box, k=expand_k, image_size=image_size)
        q_crop = crop_image_tensor(query_img, rect)
        t_crop = crop_image_tensor(template_img, rect)
        crop_gt = map_boxes_to_crop(gt_boxes, rect)
        # Filter GT boxes: only keep those fully inside the crop
        contained_idx = boxes_fully_contained(gt_boxes, rect)
        valid_gt = crop_gt[contained_idx] if contained_idx.numel() > 0 else crop_gt[:0]
        valid_labels = gt_labels[contained_idx] if contained_idx.numel() > 0 else torch.empty(0, dtype=torch.int64)

        crops.append({
            "query_crop": q_crop,
            "template_crop": t_crop,
            "gt_boxes": valid_gt,
            "gt_labels": valid_labels,
            "crop_rect": list(rect),  # [x, y, w, h]
            "crop_index": idx,
            "num_candidate_boxes_in_crop": int(contained_idx.numel()),
        })

    return crops


def crop_statistics(
    all_crops: list[list[dict]],
    all_gt_targets: list[dict],
) -> dict[str, float | int]:
    """Compute aggregate statistics over all samples' crops.

    Returns:
        Dict with total_crops, crops_with_gt, empty_crops, gt_coverage, etc.
    """
    total_crops = 0
    crops_with_gt = 0
    total_gt = 0
    covered_gt = 0
    sizes: list[int] = []

    for sample_crops, target in zip(all_crops, all_gt_targets):
        total_crops += len(sample_crops)
        gt_count = target["boxes"].shape[0]
        total_gt += gt_count

        gt_matched = torch.zeros(gt_count, dtype=torch.bool)
        for crop in sample_crops:
            _, _, w, h = crop["crop_rect"]
            sizes.append(w * h)
            n = int(crop["num_candidate_boxes_in_crop"])
            if n > 0:
                crops_with_gt += 1
            # Simple heuristic: a GT is "covered" if at least one crop contains it
            # For precise coverage, check each GT against each crop rect
            if gt_count > 0 and "crop_rect" in crop:
                rx, ry, rw, rh = crop["crop_rect"]
                boxes = target["boxes"]
                inside = (
                    (boxes[:, 0] >= rx) & (boxes[:, 1] >= ry) &
                    (boxes[:, 2] <= rx + rw) & (boxes[:, 3] <= ry + rh)
                )
                gt_matched = gt_matched | inside

        covered_gt += int(gt_matched.sum().item())

    return {
        "total_crops": total_crops,
        "crops_with_gt_boxes": crops_with_gt,
        "empty_crops": total_crops - crops_with_gt,
        "total_gt_boxes": total_gt,
        "covered_gt_boxes": covered_gt,
        "gt_coverage": covered_gt / max(total_gt, 1),
        "crop_area_mean": sum(sizes) / max(len(sizes), 1),
        "crop_area_min": min(sizes) if sizes else 0,
        "crop_area_max": max(sizes) if sizes else 0,
    }
