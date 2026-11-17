"""Analyze end-to-end false negatives and determine the failing stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import box_iou

from ultralytics import RTDETR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from step1_audit.paths import write_json
from step2_dataset.dataset import PairedQueryTemplateDataset
from step2_dataset.index import build_dataset_index, load_config as load_step2_config
from step3_model.backbone import SharedSiameseBackbone
from step3_model.end_to_end_eval import (
    _build_rtdetr_callable,
    _load_frozen_proposal_parameters,
    _resolve_device,
    load_e2e_config,
)
from step3_model.feature_warp import FeatureWarp
from step3_model.fusion import MultiScaleFusion
from step3_model.inference_bridge import run_inference
from step3_model.model_io import load_model_checkpoint
from step3_model.offset_predictor import OffsetPredictor


def _expand_rect(box: torch.Tensor, expand_k: float, image_size: tuple[int, int] = (800, 800)) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = (x2 - x1) * expand_k / 2.0, (y2 - y1) * expand_k / 2.0
    left = max(0, int(cx - hw))
    top = max(0, int(cy - hh))
    right = min(image_size[0], int(cx + hw))
    bottom = min(image_size[1], int(cy + hh))
    return left, top, right - left, bottom - top


def _candidate_covers(candidate_boxes: torch.Tensor, gt_box: torch.Tensor, expand_k: float) -> bool:
    if candidate_boxes.numel() == 0:
        return False
    for candidate in candidate_boxes:
        rect = _expand_rect(candidate, expand_k)
        x, y, w, h = rect
        gx1, gy1, gx2, gy2 = [float(v) for v in gt_box.tolist()]
        if gx1 >= x and gy1 >= y and gx2 <= x + w and gy2 <= y + h:
            return True
    return False


def _matched_gt_mask(detections: list[dict], target: dict) -> torch.Tensor:
    gt_boxes = target["boxes"]
    gt_labels = target["labels"]
    matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        label = int(detection["class"])
        box = torch.tensor(detection["box"], dtype=torch.float32).reshape(1, 4)
        candidates = (gt_labels == label).nonzero(as_tuple=False).squeeze(1)
        if candidates.numel() == 0:
            continue
        ious = box_iou(box, gt_boxes[candidates])[0]
        best_local = int(ious.argmax().item())
        best_global = int(candidates[best_local].item())
        if float(ious[best_local]) >= 0.5 and not matched[best_global]:
            matched[best_global] = True
    return matched


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = np.clip(image.detach().cpu().permute(1, 2, 0).numpy() * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _render_fn(
    query: torch.Tensor,
    template: torch.Tensor,
    gt_box: list[float],
    class_name: str,
    covered: bool,
    output_path: Path,
) -> None:
    query_pil = _tensor_to_pil(query)
    template_pil = _tensor_to_pil(template)
    x1, y1, x2, y2 = [int(v) for v in gt_box]
    box = [x1 - 20, y1 - 20, x2 + 20, y2 + 20]
    box = [max(0, box[0]), max(0, box[1]), min(800, box[2]), min(800, box[3])]
    q_crop = query_pil.crop(box)
    t_crop = template_pil.crop(box)
    canvas = Image.new("RGB", (q_crop.width + t_crop.width + 8, max(q_crop.height, t_crop.height)), (255, 255, 255))
    canvas.paste(q_crop, (0, 0))
    canvas.paste(t_crop, (q_crop.width + 8, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, q_crop.width - 1, q_crop.height - 1], outline=(255, 0, 0), width=3)
    draw.text((4, 4), f"FN {class_name} covered={covered}", fill=(255, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze end-to-end false negatives")
    parser.add_argument("--config", type=Path, default=Path("configs/step3_end_to_end_eval.yaml"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--classes", nargs="+", default=["broke", "dirty", "double", "miss"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    config = load_e2e_config(args.config)
    split = args.split
    device = _resolve_device(config["run"]["device"])
    step2 = load_step2_config(Path(config["_step2_config"]))
    dataset = PairedQueryTemplateDataset(build_dataset_index(step2), split, step2, training=False)
    max_samples = int(args.max_samples) if int(args.max_samples) > 0 else len(dataset)
    class_names = list(step2["classes"])
    classes = set(args.classes)

    backbone = SharedSiameseBackbone("IMAGENET1K_V2").to(device).eval()
    predictor = OffsetPredictor(
        kernel_size=int(config["model"]["kernel_size"]),
        hidden_ratio=int(config["model"]["hidden_ratio"]),
        zero_init=bool(config["model"]["zero_init"]),
        max_offset=float(config["model"].get("max_offset", 3.0)),
    ).to(device).eval()
    warp = FeatureWarp(kernel_size=int(config["model"]["kernel_size"])).to(device).eval()
    fusion = MultiScaleFusion(
        hidden=int(config["fusion"]["hidden"]),
        reduction=int(config["fusion"]["reduction"]),
    ).to(device).eval()
    load_model_checkpoint(predictor, Path(config["_predictor_checkpoint"]), device, "OffsetPredictor", "defect-masked-alignment-v1")
    load_model_checkpoint(fusion, Path(config["_fusion_checkpoint"]), device, "MultiScaleFusion", "supervised-fusion-heatmap-v1")

    proposal_parameters = _load_frozen_proposal_parameters(Path(config["_proposal_frozen_parameters"]))
    proposals_cfg = {
        "heatmap_method": "probability_threshold",
        "probability_threshold": float(proposal_parameters["probability_threshold"]),
        "morph_kernel": int(proposal_parameters["morph_kernel"]),
        "min_area": float(proposal_parameters["min_area"]),
        "max_aspect_ratio": float(proposal_parameters["max_aspect_ratio"]),
        "max_area_ratio": float(proposal_parameters.get("max_area_ratio", 0.8)),
        "expand_k": float(proposal_parameters.get("expand_k", 1.8)),
        "merge_iou": float(proposal_parameters.get("merge_iou_threshold", 0.3)),
    }
    rtdetr = RTDETR(str(Path(config["_rtdetr_checkpoint"])))
    rtdetr_callable = _build_rtdetr_callable(rtdetr, config, device)

    out_root = Path("artifacts/analysis/false_negatives") / split
    out_root.mkdir(parents=True, exist_ok=True)
    fn_rows = []
    per_class_counts = {name: 0 for name in class_names}

    with torch.inference_mode():
        for index in range(max_samples):
            sample = dataset[index]
            query = sample["query"].to(device)
            template = sample["template"].to(device)
            detections, candidate_boxes = run_inference(
                query, template,
                backbone, predictor, warp, fusion,
                rtdetr_model=rtdetr_callable,
                proposals_cfg=proposals_cfg,
                image_size=(800, 800),
                device=device,
                pair_aware=True,
                return_proposals=True,
            )
            target = sample["query_target"]
            matched = _matched_gt_mask(detections, target)
            for gt_index in range(target["boxes"].shape[0]):
                if matched[gt_index]:
                    continue
                class_id = int(target["labels"][gt_index].item())
                class_name = class_names[class_id]
                per_class_counts[class_name] += 1
                gt_box = target["boxes"][gt_index]
                covered = _candidate_covers(candidate_boxes, gt_box, float(proposals_cfg["expand_k"]))
                if class_name in classes:
                    fn_rows.append({
                        "sample_id": sample["meta"]["sample_id"],
                        "class_id": class_id,
                        "class_name": class_name,
                        "box": [float(v) for v in gt_box.tolist()],
                        "candidate_covered": covered,
                    })
            if (index + 1) % 50 == 0 or index + 1 == max_samples:
                print(f"processed {index + 1}/{max_samples}", flush=True)

    selected = sorted(fn_rows, key=lambda row: row["class_name"])
    for class_name in classes:
        rows = [row for row in selected if row["class_name"] == class_name][: args.limit]
        for row in rows:
            sample_id = row["sample_id"]
            records = dataset.index.records_by_split[split]
            record = records[[r.sample_id for r in records].index(sample_id)]
            with Image.open(record.query_path) as image:
                query_tensor = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
            with Image.open(record.template_path) as image:
                template_tensor = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
            output = out_root / class_name / f"{sample_id}.png"
            _render_fn(query_tensor, template_tensor, row["box"], class_name, row["candidate_covered"], output)

    summary = {
        "split": split,
        "processed_sample_count": max_samples,
        "per_class_false_negative_counts": per_class_counts,
        "false_negative_rows": selected,
    }
    write_json(out_root / "false_negative_summary.json", summary)
    print(json.dumps({
        "split": split,
        "processed_sample_count": max_samples,
        "per_class_false_negative_counts": per_class_counts,
        "covered_by_candidate_count": sum(1 for row in selected if row["candidate_covered"]),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
