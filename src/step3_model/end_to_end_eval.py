"""End-to-end evaluation: Siamese screening -> crops -> RT-DETR -> inverse map."""

from __future__ import annotations

import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torchvision.ops import box_iou
from ultralytics import RTDETR

from step1_audit.hashing import sha256_file
from step1_audit.paths import resolved, validate_roots, write_json
from step2_dataset.collate import build_dataloader
from step2_dataset.dataset import PairedQueryTemplateDataset
from step2_dataset.index import build_dataset_index, load_config as load_step2_config

from .backbone import SharedSiameseBackbone
from .feature_warp import FeatureWarp
from .fusion import MultiScaleFusion
from .inference_bridge import run_inference
from .model_io import load_model_checkpoint
from .offset_predictor import OffsetPredictor
from .test_eval import _load_frozen_proposal_parameters

_EVIDENCE_KEYS = (
    "alignment_validation_report",
    "alignment_frozen_parameters",
    "alignment_validation_manifest",
    "alignment_test_report",
    "alignment_test_manifest",
)


def load_e2e_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    workspace = path.resolve().parent.parent
    config["_config_path"] = str(path.resolve())
    config["_workspace"] = str(workspace)
    config["_output_root"] = str(resolved(config["output"]["root"], workspace))
    config["_dataset_root"] = str(resolved(config["inputs"]["dataset_root"], workspace))
    config["_step2_config"] = str(resolved(config["inputs"]["step2_config"], workspace))
    config["_predictor_checkpoint"] = str(resolved(config["inputs"]["predictor_checkpoint"], workspace))
    config["_fusion_checkpoint"] = str(resolved(config["inputs"]["fusion_checkpoint"], workspace))
    config["_proposal_frozen_parameters"] = str(
        resolved(config["inputs"]["proposal_frozen_parameters"], workspace)
    )
    config["_rtdetr_checkpoint"] = str(resolved(config["inputs"]["rtdetr_checkpoint"], workspace))
    for key in _EVIDENCE_KEYS:
        config[f"_{key}"] = str(resolved(config["inputs"][key], workspace))
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "step3-e2e-eval-config-v1":
        raise ValueError("Invalid end-to-end evaluation config schema")
    run = config.get("run", {})
    if run.get("split") not in {"val", "test"}:
        raise ValueError("end-to-end evaluation split must be val or test")
    if int(run.get("batch_size", 0)) <= 0:
        raise ValueError("batch_size must be positive")
    if run.get("device") not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unsupported device")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _build_rtdetr_callable(model: RTDETR, config: dict[str, Any], device: torch.device):
    imgsz = int(config["detection"]["imgsz"])
    conf = float(config["detection"]["conf"])
    iou = float(config["detection"]["iou"])
    diff_scale = float(config["detection"].get("diff_scale", 3.0))

    def _to_gray(tensor: torch.Tensor) -> np.ndarray:
        gray = (
            0.299 * tensor[0]
            + 0.587 * tensor[1]
            + 0.114 * tensor[2]
        )
        return np.clip(np.rint(gray.detach().cpu().numpy() * 255.0), 0, 255).astype(np.float32)

    def callable_model(query_padded: torch.Tensor, template_padded: torch.Tensor) -> torch.Tensor:
        query_gray = _to_gray(query_padded[0])
        template_gray = _to_gray(template_padded[0])
        diff = np.clip(np.abs(query_gray - template_gray) * diff_scale, 0, 255)
        # `model.predict` assumes BGR numpy input and converts it to RGB before
        # forward.  The paired model was trained on channel order [query, template,
        # diff], so feed [diff, template, query] (BGR) to let the predictor convert
        # it back to the trained order.
        image = np.stack([diff, template_gray, query_gray], axis=2).astype(np.uint8)
        results = model.predict(
            source=image,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=str(device),
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return torch.empty((0, 6), dtype=torch.float32)
        boxes = results[0].boxes.xyxy.cpu()
        scores = results[0].boxes.conf.cpu().unsqueeze(1)
        classes = results[0].boxes.cls.cpu().unsqueeze(1)
        return torch.cat([boxes, scores, classes], dim=1)

    return callable_model


def _detection_metrics(
    all_predictions: list[list[dict]],
    all_targets: list[dict],
    class_names: list[str],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute greedy class-aware precision/recall at a fixed IoU threshold."""
    per_class: dict[int, dict[str, int | float]] = {
        index: {"gt": 0, "tp": 0, "fp": 0}
        for index, _ in enumerate(class_names)
    }
    total_gt = total_pred = 0

    for predictions, target in zip(all_predictions, all_targets):
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        total_gt += int(gt_boxes.shape[0])
        total_pred += len(predictions)

        for label in range(len(class_names)):
            per_class[label]["gt"] = int(per_class[label]["gt"]) + int(
                (gt_labels == label).sum().item()
            )

        # Track which GT boxes have already been matched.
        matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
        prediction_rows = [
            (float(item["score"]), item)
            for item in predictions
        ]
        prediction_rows.sort(key=lambda row: row[0], reverse=True)

        for _score, item in prediction_rows:
            label = int(item["class"])
            box = torch.tensor(item["box"], dtype=torch.float32).reshape(1, 4)
            if label < 0 or label >= len(class_names):
                per_class.setdefault(label, {"gt": 0, "tp": 0, "fp": 0})
                per_class[label]["fp"] = int(per_class[label]["fp"]) + 1
                continue

            candidate_indices = (gt_labels == label).nonzero(as_tuple=False).squeeze(1)
            if candidate_indices.numel() == 0:
                per_class[label]["fp"] = int(per_class[label]["fp"]) + 1
                continue

            candidate_boxes = gt_boxes[candidate_indices]
            ious = box_iou(box, candidate_boxes)[0]
            best_local = int(ious.argmax().item())
            best_global = int(candidate_indices[best_local].item())
            if float(ious[best_local]) >= iou_threshold and not matched[best_global]:
                matched[best_global] = True
                per_class[label]["tp"] = int(per_class[label]["tp"]) + 1
            else:
                per_class[label]["fp"] = int(per_class[label]["fp"]) + 1

    class_rows = []
    micro_tp = sum(int(per_class[index]["tp"]) for index in range(len(class_names)))
    micro_fp = sum(int(per_class[index]["fp"]) for index in range(len(class_names)))
    micro_fn = total_gt - micro_tp
    for index, name in enumerate(class_names):
        row = per_class[index]
        gt = int(row["gt"])
        tp = int(row["tp"])
        fp = int(row["fp"])
        fn = gt - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        class_rows.append({
            "class_id": index,
            "class_name": name,
            "gt_boxes": gt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    micro_precision = micro_tp / max(micro_tp + micro_fp, 1)
    micro_recall = micro_tp / max(micro_tp + micro_fn, 1)
    micro_f1 = 2 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-9)
    return {
        "iou_threshold": iou_threshold,
        "total_gt_boxes": total_gt,
        "total_predictions": total_pred,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "per_class": class_rows,
    }


def run_e2e_eval(config: dict[str, Any], split: str | None = None) -> dict[str, Any]:
    _validate_config(config)
    split = split or str(config["run"]["split"])
    if split not in {"val", "test"}:
        raise ValueError("end-to-end evaluation split must be val or test")

    dataset_root = Path(config["_dataset_root"])
    output_root = Path(config["_output_root"])
    validate_roots(dataset_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    step2 = load_step2_config(Path(config["_step2_config"]))
    index = build_dataset_index(step2)
    dataset = PairedQueryTemplateDataset(index, split, step2, training=False)
    if len(dataset) != 400:
        raise ValueError(f"Unexpected {split} sample count: {len(dataset)} != 400")
    loader = build_dataloader(
        dataset,
        batch_size=int(config["run"]["batch_size"]),
        num_workers=int(config["run"]["num_workers"]),
        shuffle=False,
    )
    device = _resolve_device(config["run"]["device"])

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

    all_predictions: list[list[dict]] = []
    all_targets: list[dict] = []
    per_sample: list[dict] = []

    with torch.inference_mode():
        for batch in loader:
            query = batch["query"][0].to(device)
            template = batch["template"][0].to(device)
            detections = run_inference(
                query, template,
                backbone, predictor, warp, fusion,
                rtdetr_model=rtdetr_callable,
                proposals_cfg=proposals_cfg,
                image_size=(800, 800),
                device=device,
                pair_aware=True,
            )
            target = batch["query_targets"][0]
            all_predictions.append(detections)
            all_targets.append(target)
            per_sample.append({
                "sample_id": batch["meta"][0]["sample_id"],
                "gt_box_count": int(target["boxes"].shape[0]),
                "detection_count": len(detections),
            })

    metrics = _detection_metrics(
        all_predictions, all_targets, list(step2["classes"]),
        iou_threshold=float(config["detection"]["iou_eval"]),
    )
    report = {
        "schema_version": "step3-e2e-eval-report-v1",
        "roadmap_step": "6",
        "split": split,
        "processed_sample_count": len(per_sample),
        "selected_device": str(device),
        "detection_statistics": metrics,
        "per_sample": per_sample,
        "checkpoint_sha256": {
            "predictor": sha256_file(Path(config["_predictor_checkpoint"])),
            "fusion": sha256_file(Path(config["_fusion_checkpoint"])),
            "rtdetr": sha256_file(Path(config["_rtdetr_checkpoint"])),
            "proposal_frozen_parameters": sha256_file(Path(config["_proposal_frozen_parameters"])),
        },
    }
    json.dumps(report, allow_nan=False)
    report_path = output_root / f"e2e_{split}_report.json"
    write_json(report_path, report)
    manifest = {
        "schema_version": "step3-e2e-eval-manifest-v1",
        "roadmap_step": "6",
        "split": split,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "selected_device": str(device),
        },
        "config_sha256": sha256_file(Path(config["_config_path"])),
        "model_checkpoints": report["checkpoint_sha256"],
        "artifacts": {report_path.name: sha256_file(report_path)},
    }
    write_json(output_root / f"e2e_{split}_manifest.json", manifest)
    return report
