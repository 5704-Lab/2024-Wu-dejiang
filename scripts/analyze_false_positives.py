"""Analyze end-to-end false positives for selected defect classes."""

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


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = np.clip(image.detach().cpu().permute(1, 2, 0).numpy() * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _crop_pil(image: Image.Image, box: list[float], expand: float = 1.5) -> Image.Image:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = (x2 - x1) * expand / 2.0, (y2 - y1) * expand / 2.0
    left = max(0, int(cx - hw))
    top = max(0, int(cy - hh))
    right = min(image.width, int(cx + hw))
    bottom = min(image.height, int(cy + hh))
    return image.crop((left, top, right, bottom))


def _render_fp(
    query: torch.Tensor,
    template: torch.Tensor,
    detection: dict,
    output_path: Path,
) -> None:
    query_pil = _tensor_to_pil(query)
    template_pil = _tensor_to_pil(template)
    query_crop = _crop_pil(query_pil, detection["box"])
    template_crop = _crop_pil(template_pil, detection["box"])
    gap = 8
    canvas = Image.new(
        "RGB",
        (query_crop.width + template_crop.width + gap, max(query_crop.height, template_crop.height)),
        (255, 255, 255),
    )
    canvas.paste(query_crop, (0, 0))
    canvas.paste(template_crop, (query_crop.width + gap, 0))
    draw = ImageDraw.Draw(canvas)
    class_label = detection.get("class", detection.get("class_name", "unknown"))
    score = float(detection.get("score", 0.0))
    draw.rectangle(
        [0, 0, query_crop.width - 1, query_crop.height - 1],
        outline=(255, 0, 0),
        width=3,
    )
    draw.text(
        (4, 4),
        f"FP class={class_label} score={score:.3f}",
        fill=(255, 0, 0),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _match_detections(detections: list[dict], target: dict) -> list[dict]:
    gt_boxes = target["boxes"]
    gt_labels = target["labels"]
    matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    labeled = []
    ordered = sorted(detections, key=lambda item: item["score"], reverse=True)
    for detection in ordered:
        label = int(detection["class"])
        box = torch.tensor(detection["box"], dtype=torch.float32).reshape(1, 4)
        candidates = (gt_labels == label).nonzero(as_tuple=False).squeeze(1)
        best_iou = 0.0
        best_global = -1
        if candidates.numel() > 0:
            ious = box_iou(box, gt_boxes[candidates])[0]
            best_local = int(ious.argmax().item())
            best_iou = float(ious[best_local])
            best_global = int(candidates[best_local].item())
        is_true_positive = best_iou >= 0.5 and best_global >= 0 and not matched[best_global]
        if is_true_positive:
            matched[best_global] = True
        labeled.append({**detection, "is_false_positive": not is_true_positive, "best_iou": best_iou})
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze end-to-end false positives")
    parser.add_argument("--config", type=Path, default=Path("configs/step3_end_to_end_eval.yaml"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--classes", nargs="+", default=["double", "dirty"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    args = parser.parse_args()

    config = load_e2e_config(args.config)
    split = args.split
    device = _resolve_device(config["run"]["device"])
    step2 = load_step2_config(Path(config["_step2_config"]))
    dataset = PairedQueryTemplateDataset(build_dataset_index(step2), split, step2, training=False)
    max_samples = int(args.max_samples) if int(args.max_samples) > 0 else len(dataset)
    classes = set(args.classes)
    class_names = list(step2["classes"])

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
    load_model_checkpoint(
        predictor, Path(config["_predictor_checkpoint"]), device, "OffsetPredictor",
        "defect-masked-alignment-v1",
    )
    load_model_checkpoint(
        fusion, Path(config["_fusion_checkpoint"]), device, "MultiScaleFusion",
        "supervised-fusion-heatmap-v1",
    )

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

    out_root = Path("artifacts/analysis/false_positives") / split
    out_root.mkdir(parents=True, exist_ok=True)
    false_positive_rows = []
    per_class_counts = {name: 0 for name in class_names}

    with torch.inference_mode():
        for index in range(max_samples):
            sample = dataset[index]
            query = sample["query"].to(device)
            template = sample["template"].to(device)
            detections = run_inference(
                query, template,
                backbone, predictor, warp, fusion,
                rtdetr_model=rtdetr_callable,
                proposals_cfg=proposals_cfg,
                image_size=(800, 800),
                device=device,
                pair_aware=True,
            )
            labeled = _match_detections(detections, sample["query_target"])
            for detection in labeled:
                if not detection["is_false_positive"]:
                    continue
                class_id = int(detection["class"])
                if class_id >= len(class_names):
                    continue
                class_name = class_names[class_id]
                per_class_counts[class_name] += 1
                if class_name in classes:
                    row = {
                        "sample_id": sample["meta"]["sample_id"],
                        "class_id": class_id,
                        "class_name": class_name,
                        "score": float(detection["score"]),
                        "box": detection["box"],
                        "best_iou": float(detection["best_iou"]),
                    }
                    false_positive_rows.append(row)
            if (index + 1) % 50 == 0 or index + 1 == max_samples:
                print(f"processed {index + 1}/{max_samples}", flush=True)

    selected = sorted(false_positive_rows, key=lambda row: row["score"], reverse=True)
    for class_name in classes:
        class_rows = [row for row in selected if row["class_name"] == class_name][: args.limit]
        for row in class_rows:
            sample_id = row["sample_id"]
            records = dataset.index.records_by_split[split]
            sample = records[[r.sample_id for r in records].index(sample_id)]
            query = sample.query_path
            template = sample.template_path
            with Image.open(query) as image:
                query_tensor = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
            with Image.open(template) as image:
                template_tensor = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
            output = out_root / class_name / f"{sample_id}_{row['score']:.3f}.png"
            _render_fp(query_tensor, template_tensor, row, output)

    summary = {
        "split": split,
        "processed_sample_count": max_samples,
        "per_class_false_positive_counts": per_class_counts,
        "analyzed_classes": sorted(classes),
        "false_positive_rows": selected[:500],
    }
    write_json(out_root / "false_positive_summary.json", summary)
    print(json.dumps({
        "split": split,
        "processed_sample_count": max_samples,
        "per_class_false_positive_counts": per_class_counts,
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
