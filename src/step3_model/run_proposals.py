"""Real-data smoke validation for candidate box proposals and recall evaluation (Step 3.5-C)."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torchvision
import yaml
from torch.utils.data import Subset

from step1_audit.hashing import sha256_file, snapshot_digest, snapshot_tree
from step1_audit.paths import resolved, validate_roots, write_json
from step2_dataset.collate import build_dataloader
from step2_dataset.dataset import PairedQueryTemplateDataset
from step2_dataset.index import build_dataset_index, load_config as load_step2_config

from .backbone import SharedSiameseBackbone
from .feature_warp import FeatureWarp
from .fusion import MultiScaleFusion, _STAGE_NAMES
from .offset_predictor import OFFSET_SPEC, OffsetPredictor
from .model_io import load_model_checkpoint
from .proposals import evaluate_recall, postprocess_batch

_EVIDENCE_KEYS = (
    "alignment_validation_report", "alignment_frozen_parameters",
    "alignment_validation_manifest", "alignment_test_report", "alignment_test_manifest",
)


def load_proposals_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    workspace = path.resolve().parent.parent
    config["_config_path"] = str(path.resolve())
    config["_workspace"] = str(workspace)
    config["_output_root"] = str(resolved(config["output"]["root"], workspace))
    config["_dataset_root"] = str(resolved(config["inputs"]["dataset_root"]))
    config["_step2_config"] = str(resolved(config["inputs"]["step2_config"], workspace))
    config["_predictor_checkpoint"] = str(resolved(config["inputs"]["predictor_checkpoint"], workspace))
    config["_fusion_checkpoint"] = str(resolved(config["inputs"]["fusion_checkpoint"], workspace))
    for key in _EVIDENCE_KEYS:
        config[f"_{key}"] = str(resolved(config["inputs"][key], workspace))
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("roadmap_step") != "3.5-C" or config.get("schema_version") != "step3-proposals-config-v1":
        raise ValueError("Invalid Step 3.5-C config schema")
    run = config.get("run", {})
    required_run = (run.get("purpose"), run.get("split"), run.get("selection"))
    if required_run != ("proposals_smoke", "val", "ordered_prefix"):
        raise ValueError("Invalid proposals smoke contract")
    if int(run.get("sample_count", 0)) <= 0 or int(run.get("batch_size", 0)) <= 0:
        raise ValueError("Smoke sample and batch counts must be positive")
    if run.get("device") not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unsupported smoke device")
    proposals = config.get("proposals", {})
    if float(proposals.get("min_area", 0)) <= 0:
        raise ValueError("min_area must be positive")
    iou_thresholds = config.get("evaluation", {}).get("iou_thresholds", [0.5])
    if not all(0 < t < 1 for t in iou_thresholds):
        raise ValueError("IoU thresholds must be in (0, 1)")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _evidence_hashes(config: dict[str, Any]) -> dict[str, str]:
    return {key: sha256_file(Path(config[f"_{key}"])) for key in _EVIDENCE_KEYS}


def run_proposals_smoke(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    dataset_root, output_root = Path(config["_dataset_root"]), Path(config["_output_root"])
    validate_roots(dataset_root, output_root)
    sample_dir = Path(config["_alignment_validation_report"]).parent.resolve()
    if output_root.resolve() == sample_dir:
        raise ValueError("Proposals output must not overwrite Step 3.2 evidence")
    before, evidence_before = snapshot_tree(dataset_root), _evidence_hashes(config)
    source_digest = snapshot_digest(before)

    step2 = load_step2_config(Path(config["_step2_config"]))
    index = build_dataset_index(step2)
    dataset = PairedQueryTemplateDataset(index, "val", step2, training=False)
    count = int(config["run"]["sample_count"])
    selected = Subset(dataset, list(range(count)))
    loader = build_dataloader(selected, batch_size=int(config["run"]["batch_size"]),
                              num_workers=int(config["run"]["num_workers"]), shuffle=False)
    device = _resolve_device(config["run"]["device"])
    backbone = SharedSiameseBackbone("IMAGENET1K_V2").to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
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
    load_model_checkpoint(predictor, Path(config["_predictor_checkpoint"]), device, "OffsetPredictor",
                          "defect-masked-alignment-v1")
    load_model_checkpoint(fusion, Path(config["_fusion_checkpoint"]), device, "MultiScaleFusion",
                          "supervised-fusion-heatmap-v1")

    proposals_cfg = config["proposals"]
    iou_thresholds = tuple(config["evaluation"]["iou_thresholds"])

    all_pred_boxes: list[torch.Tensor] = []
    all_gt_targets: list[dict] = []
    sample_ids: list[str] = []
    per_sample_stats: list[dict] = []

    with torch.inference_mode():
        for batch in loader:
            query = batch["query"].to(device)
            template = batch["template"].to(device)
            qf, tf = backbone(query, template)
            offsets = predictor(qf, tf)
            warped = warp(tf, offsets)
            diff_features = {key: (warped[key] - qf[key]).abs() for key in [n for n, _, _ in OFFSET_SPEC]}
            enhanced = torch.sigmoid(fusion(diff_features))  # supervised defect probability

            pred_boxes = postprocess_batch(
                enhanced,
                min_area=float(proposals_cfg["min_area"]),
                max_aspect_ratio=float(proposals_cfg["max_aspect_ratio"]),
                adaptive_block=int(proposals_cfg["adaptive_block"]),
                adaptive_c=int(proposals_cfg["adaptive_c"]),
                morph_kernel=int(proposals_cfg["morph_kernel"]),
            )
            gt_targets = batch["query_targets"]  # template_targets is identical

            all_pred_boxes.extend(pred_boxes)
            all_gt_targets.extend(gt_targets)
            for i, meta in enumerate(batch["meta"]):
                sid = meta["sample_id"]
                sample_ids.append(sid)
                gt_count = gt_targets[i]["boxes"].shape[0]
                pred_count = pred_boxes[i].shape[0]
                per_sample_stats.append({
                    "sample_id": sid,
                    "gt_box_count": gt_count,
                    "candidate_box_count": pred_count,
                    "enhanced_min": float(enhanced[i].min().item()),
                    "enhanced_max": float(enhanced[i].max().item()),
                })

    recall_stats = evaluate_recall(all_pred_boxes, all_gt_targets, iou_thresholds)

    verified = len(sample_ids) == count

    if before != snapshot_tree(dataset_root):
        raise RuntimeError("Read-only dataset changed during proposals smoke")
    if evidence_before != _evidence_hashes(config):
        raise RuntimeError("Step 3.2 evidence changed during proposals smoke")
    if not verified:
        raise RuntimeError("Proposals smoke verification failed")

    contract = {
        "schema_version": "step3-proposals-contract-v1", "roadmap_step": "3.5-C",
        "postprocessing": "adaptive_threshold + morph_close + findContours + boundingRect",
        "parameters": {
            "min_area": float(proposals_cfg["min_area"]),
            "max_aspect_ratio": float(proposals_cfg["max_aspect_ratio"]),
            "max_area_ratio": float(proposals_cfg.get("max_area_ratio", 0.8)),
            "adaptive_block": int(proposals_cfg["adaptive_block"]),
            "adaptive_c": int(proposals_cfg["adaptive_c"]),
            "morph_kernel": int(proposals_cfg["morph_kernel"]),
        },
        "evaluation": {"iou_thresholds": list(iou_thresholds), "metric": "recall@IoU"},
    }
    report = {
        "schema_version": "step3-proposals-smoke-v1", "roadmap_step": "3.5-C",
        "verified": True, "processed_sample_count": len(sample_ids),
        "expected_sample_count": count, "selected_device": str(device),
        "recall_statistics": recall_stats, "per_sample": per_sample_stats,
        "checkpoint_sha256": {"predictor": sha256_file(Path(config["_predictor_checkpoint"])),
                              "fusion": sha256_file(Path(config["_fusion_checkpoint"]))},
    }
    json.dumps(contract, allow_nan=False); json.dumps(report, allow_nan=False)
    output_root.mkdir(parents=True, exist_ok=True)
    contract_path, report_path = output_root / "proposals_contract.json", output_root / "smoke_report.json"
    write_json(contract_path, contract); write_json(report_path, report)
    manifest = {
        "schema_version": "step3-proposals-run-manifest-v1", "roadmap_step": "3.5-C",
        "command": "proposals-smoke", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "cuda_available": torch.cuda.is_available(),
                     "selected_device": str(device),
                     "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "inputs": {"config_sha256": sha256_file(Path(config["_config_path"])),
                   "step2_config_sha256": sha256_file(Path(config["_step2_config"])),
                   "source_snapshot_digest": source_digest, "step3_2_evidence_sha256": evidence_before},
        "model_checkpoints": report["checkpoint_sha256"],
        "artifacts": {"proposals_contract.json": sha256_file(contract_path),
                      "smoke_report.json": sha256_file(report_path)},
        "controls": {"source_read_only_verified": True, "alignment_evidence_integrity_verified": True,
                     "trained_weights_loaded": True, "candidate_proposals_started": True, "step_3_5_c_started": True},
    }
    write_json(output_root / "run_manifest.json", manifest)
    return report
