"""Inference bridge: Siamese screening -> crop -> pad -> RT-DETR -> inverse map."""

from __future__ import annotations

import torch
from torchvision.ops import box_iou

from .crop import build_crops, inverse_map_boxes
from .fusion import MultiScaleFusion
from .proposals import postprocess_batch


def pad_image_to_size(
    image: torch.Tensor,
    target_size: tuple[int, int] = (800, 800),
    fill: float = 0.0,
) -> torch.Tensor:
    """Pad a [C, h, w] tensor to target_size (W, H) with constant fill value.

    Padding is added to the right and bottom edges only.
    """
    _, h, w = image.shape
    target_w, target_h = target_size
    pw = max(0, target_w - w)
    ph = max(0, target_h - h)
    if pw == 0 and ph == 0:
        return image
    return torch.nn.functional.pad(image, (0, pw, 0, ph), value=fill)


def run_inference(
    query_img: torch.Tensor,
    template_img: torch.Tensor,
    backbone: torch.nn.Module,
    predictor: torch.nn.Module,
    warp: torch.nn.Module,
    fusion: MultiScaleFusion,
    rtdetr_model: object,  # callable that takes [1,3,800,800] and returns [K,6] xyxy-score-cls
    proposals_cfg: dict,
    image_size: tuple[int, int] = (800, 800),
    device: torch.device | None = None,
    pair_aware: bool = False,
) -> list[dict]:
    """End-to-end inference: Siamese screening -> context crops -> RT-DETR.

    Args:
        query_img: [3, H, W] float32 [0,1]
        template_img: [3, H, W] float32 [0,1]
        backbone, predictor, warp, fusion: Siamese pipeline modules
        rtdetr_model: RT-DETR model or any callable mapping [1,3,800,800] -> [K,6] tensor
        proposals_cfg: postprocessing parameters
        image_size: (W, H) of source images
        device: torch device

    Returns:
        List of detection dicts, each with boxes (image coords), scores, classes.
    """
    if device is None:
        device = next(backbone.parameters()).device

    qi = query_img.unsqueeze(0).to(device)
    ti = template_img.unsqueeze(0).to(device)

    with torch.inference_mode():
        qf, tf = backbone(qi, ti)
        offsets = predictor(qf, tf)
        warped = warp(tf, offsets)
        diff_features = {key: (warped[key] - qf[key]).abs() for key in ["C3", "C4", "C5"]}
        enhanced = fusion(diff_features)

        candidate_boxes = postprocess_batch(
            enhanced,
            min_area=float(proposals_cfg["min_area"]),
            max_aspect_ratio=float(proposals_cfg["max_aspect_ratio"]),
            adaptive_block=int(proposals_cfg.get("adaptive_block", 51)),
            adaptive_c=int(proposals_cfg.get("adaptive_c", -5)),
            morph_kernel=int(proposals_cfg.get("morph_kernel", 5)),
            heatmap_method=str(proposals_cfg.get("heatmap_method", "probability_threshold")),
            probability_threshold=float(proposals_cfg.get("probability_threshold", 0.5)),
            max_area_ratio=float(proposals_cfg.get("max_area_ratio", 0.8)),
        )[0]  # single sample

        if candidate_boxes.numel() == 0:
            return []

        # Build crops (contextual expansion, no GT target needed)
        crops = build_crops(
            query_img, template_img, candidate_boxes,
            {"boxes": torch.empty((0, 4)), "labels": torch.empty(0, dtype=torch.int64)},
            expand_k=float(proposals_cfg.get("expand_k", 1.8)),
            merge_iou=float(proposals_cfg.get("merge_iou", 0.3)),
            image_size=image_size,
        )

        all_detections: list[dict] = []
        for crop in crops:
            q_crop = crop["query_crop"]  # [3, h, w]
            padded = pad_image_to_size(q_crop, image_size)

            # Run RT-DETR on the padded crop
            if pair_aware:
                t_crop = crop["template_crop"]
                t_padded = pad_image_to_size(t_crop, image_size)
                dets = rtdetr_model(
                    padded.unsqueeze(0).to(device),
                    t_padded.unsqueeze(0).to(device),
                )
            else:
                dets = rtdetr_model(padded.unsqueeze(0).to(device))
            if isinstance(dets, torch.Tensor) and dets.numel() > 0 and dets.shape[1] >= 6:
                # dets: [K, 6] = [x1, y1, x2, y2, score, class]
                crop_boxes = dets[:, :4].cpu()
                scores = dets[:, 4].cpu()
                classes = dets[:, 5].cpu().long()

                # Map back to image coordinates
                crop_rect = tuple(crop["crop_rect"])  # (x, y, w, h)
                global_boxes = inverse_map_boxes(crop_boxes, crop_rect)

                for idx in range(global_boxes.shape[0]):
                    all_detections.append({
                        "box": global_boxes[idx].tolist(),
                        "score": float(scores[idx]),
                        "class": int(classes[idx]),
                        "source_crop_rect": list(crop_rect),
                    })

        # Simple IoU-based NMS to remove duplicate detections from overlapping crops
        return _nms_detections(all_detections, iou_threshold=0.5)


def _nms_detections(
    detections: list[dict],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """Apply class-aware IoU NMS to a list of detection dicts."""
    if len(detections) <= 1:
        return detections

    from torchvision.ops import nms

    by_class: dict[int, list[tuple[int, dict]]] = {}
    for index, detection in enumerate(detections):
        by_class.setdefault(int(detection["class"]), []).append((index, detection))

    keep_indices: list[int] = []
    for same_class in by_class.values():
        boxes = torch.tensor([item["box"] for _, item in same_class], dtype=torch.float32)
        scores = torch.tensor([item["score"] for _, item in same_class], dtype=torch.float32)
        for index in nms(boxes, scores, iou_threshold):
            keep_indices.append(same_class[int(index)][0])

    keep = sorted(keep_indices)
    return [detections[index] for index in keep]


def torchvision_ops_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Wrapper for torchvision.ops.nms (resolves by score descending)."""
    from torchvision.ops import nms
    return nms(boxes, scores, iou_threshold)
