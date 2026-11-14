"""Full Test-split evaluation for the frozen Siamese proposal/crop pipeline.

This module is intentionally separate from the validation smoke runners.
It evaluates the complete 400-sample Test split exactly once, using proposal
and crop parameters that were frozen on Validation.  No parameter search,
threshold tuning, or refreezing happens here.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torchvision
import yaml

from step1_audit.hashing import sha256_file, snapshot_digest, snapshot_tree
from step1_audit.paths import resolved, validate_roots, write_json
from step2_dataset.collate import build_dataloader
from step2_dataset.dataset import PairedQueryTemplateDataset
from step2_dataset.index import build_dataset_index, load_config as load_step2_config

from .backbone import SharedSiameseBackbone
from .crop import build_crops, crop_statistics
from .feature_warp import FeatureWarp
from .fusion import MultiScaleFusion
from .model_io import load_model_checkpoint
from .offset_predictor import OFFSET_SPEC, OffsetPredictor
from .proposals import evaluate_recall, postprocess_batch

_EVIDENCE_KEYS = (
    "alignment_validation_report",
    "alignment_frozen_parameters",
    "alignment_validation_manifest",
    "alignment_test_report",
    "alignment_test_manifest",
)


def load_test_eval_config(path: Path) -> dict[str, Any]:
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
    for key in _EVIDENCE_KEYS:
        config[f"_{key}"] = str(resolved(config["inputs"][key], workspace))
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "step3-test-eval-config-v1":
        raise ValueError("Invalid Step 3 Test evaluation config schema")
    run = config.get("run", {})
    if run.get("purpose") != "test_evaluation":
        raise ValueError("Invalid Test evaluation purpose")
    if run.get("split") != "test" or run.get("selection") != "full_split":
        raise ValueError("Test evaluation must use the full Test split")
    if int(run.get("expected_sample_count", 0)) != 400:
        raise ValueError("Test evaluation expects 400 samples")
    if run.get("device") not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unsupported device")
    if int(run.get("batch_size", 0)) <= 0:
        raise ValueError("batch_size must be positive")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _evidence_hashes(config: dict[str, Any]) -> dict[str, str]:
    return {key: sha256_file(Path(config[f"_{key}"])) for key in _EVIDENCE_KEYS}


def _load_frozen_proposal_parameters(path: Path) -> dict[str, Any]:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("schema_version") != "step3-proposal-parameters-frozen-v1":
        raise ValueError("Proposal frozen parameters have an invalid schema")
    if frozen.get("split_used") != "val":
        raise ValueError("Proposal frozen parameters were not derived from Validation")
    if frozen.get("test_used_for_tuning") is not False:
        raise ValueError("Proposal frozen parameters may not use Test for tuning")
    if frozen.get("heatmap_method") != "probability_threshold":
        raise ValueError("Only probability_threshold frozen parameters are supported")
    parameters = frozen.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Frozen parameters are missing")
    return parameters


def run_test_eval(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    dataset_root = Path(config["_dataset_root"])
    output_root = Path(config["_output_root"])
    validate_roots(dataset_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    before = snapshot_tree(dataset_root)
    evidence_before = _evidence_hashes(config)
    source_digest = snapshot_digest(before)
    proposal_parameters = _load_frozen_proposal_parameters(Path(config["_proposal_frozen_parameters"]))

    step2 = load_step2_config(Path(config["_step2_config"]))
    index = build_dataset_index(step2)
    dataset = PairedQueryTemplateDataset(index, "test", step2, training=False)
    expected_count = int(config["run"]["expected_sample_count"])
    if len(dataset) != expected_count:
        raise ValueError(f"Unexpected Test sample count: {len(dataset)} != {expected_count}")

    loader = build_dataloader(
        dataset,
        batch_size=int(config["run"]["batch_size"]),
        num_workers=int(config["run"]["num_workers"]),
        shuffle=False,
    )
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

    load_model_checkpoint(
        predictor, Path(config["_predictor_checkpoint"]), device, "OffsetPredictor",
        "defect-masked-alignment-v1",
    )
    load_model_checkpoint(
        fusion, Path(config["_fusion_checkpoint"]), device, "MultiScaleFusion",
        "supervised-fusion-heatmap-v1",
    )

    probability_threshold = float(proposal_parameters["probability_threshold"])
    morph_kernel = int(proposal_parameters["morph_kernel"])
    min_area = float(proposal_parameters["min_area"])
    max_aspect_ratio = float(proposal_parameters["max_aspect_ratio"])
    max_area_ratio = float(proposal_parameters.get("max_area_ratio", 0.8))
    expand_k = float(proposal_parameters.get("expand_k", 1.8))
    merge_iou = float(proposal_parameters.get("merge_iou_threshold", 0.3))

    all_pred_boxes: list[torch.Tensor] = []
    all_gt_targets: list[dict] = []
    all_crops: list[list[dict]] = []
    sample_ids: list[str] = []
    per_sample_stats: list[dict] = []

    with torch.inference_mode():
        for batch in loader:
            query = batch["query"].to(device)
            template = batch["template"].to(device)
            qf, tf = backbone(query, template)
            offsets = predictor(qf, tf)
            warped = warp(tf, offsets)
            diff_features = {
                key: (warped[key] - qf[key]).abs()
                for key in [name for name, _, _ in OFFSET_SPEC]
            }
            enhanced = torch.sigmoid(fusion(diff_features))
            pred_boxes = postprocess_batch(
                enhanced,
                min_area=min_area,
                max_aspect_ratio=max_aspect_ratio,
                morph_kernel=morph_kernel,
                heatmap_method="probability_threshold",
                probability_threshold=probability_threshold,
                max_area_ratio=max_area_ratio,
            )
            gt_targets = batch["query_targets"]
            all_pred_boxes.extend(pred_boxes)
            all_gt_targets.extend(gt_targets)

            for i, meta in enumerate(batch["meta"]):
                sample_id = meta["sample_id"]
                sample_ids.append(sample_id)
                sample_crops = build_crops(
                    query[i], template[i], pred_boxes[i], gt_targets[i],
                    expand_k=expand_k, merge_iou=merge_iou,
                )
                all_crops.append(sample_crops)
                per_sample_stats.append({
                    "sample_id": sample_id,
                    "gt_box_count": int(gt_targets[i]["boxes"].shape[0]),
                    "candidate_box_count": int(pred_boxes[i].shape[0]),
                    "crop_count": len(sample_crops),
                })

    recall_stats = evaluate_recall(
        all_pred_boxes, all_gt_targets, tuple(config["evaluation"]["iou_thresholds"])
    )
    crop_stats = crop_statistics(all_crops, all_gt_targets)

    if before != snapshot_tree(dataset_root):
        raise RuntimeError("Read-only dataset changed during Test evaluation")
    if evidence_before != _evidence_hashes(config):
        raise RuntimeError("Step 3.2 evidence changed during Test evaluation")
    if len(sample_ids) != expected_count:
        raise RuntimeError(f"Test evaluation processed {len(sample_ids)} samples, expected {expected_count}")

    report = {
        "schema_version": "step3-test-eval-report-v1",
        "roadmap_step": "3.5-C-test",
        "verified": True,
        "scope": {
            "split": "test",
            "selection": "full_split",
            "expected_sample_count": expected_count,
            "processed_sample_count": len(sample_ids),
            "unique_sample_count": len(set(sample_ids)),
            "complete": len(sample_ids) == expected_count,
            "tuning_performed": False,
            "parameters_refrozen": False,
        },
        "selected_device": str(device),
        "recall_statistics": recall_stats,
        "crop_statistics": crop_stats,
        "per_sample": per_sample_stats,
        "frozen_proposal_parameters": {
            "path": config["_proposal_frozen_parameters"],
            "sha256": sha256_file(Path(config["_proposal_frozen_parameters"])),
            "parameters": proposal_parameters,
        },
        "checkpoint_sha256": {
            "predictor": sha256_file(Path(config["_predictor_checkpoint"])),
            "fusion": sha256_file(Path(config["_fusion_checkpoint"])),
        },
    }
    json.dumps(report, allow_nan=False)
    report_path = output_root / "test_eval_report.json"
    write_json(report_path, report)

    manifest = {
        "schema_version": "step3-test-eval-manifest-v1",
        "roadmap_step": "3.5-C-test",
        "command": "test-eval",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(),
            "selected_device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "inputs": {
            "config_sha256": sha256_file(Path(config["_config_path"])),
            "step2_config_sha256": sha256_file(Path(config["_step2_config"])),
            "source_snapshot_digest": source_digest,
            "step3_2_evidence_sha256": evidence_before,
        },
        "model_checkpoints": report["checkpoint_sha256"],
        "artifacts": {"test_eval_report.json": sha256_file(report_path)},
        "controls": {
            "source_read_only_verified": True,
            "alignment_evidence_integrity_verified": True,
            "trained_weights_loaded": True,
            "test_tuning_performed": False,
            "test_parameters_refrozen": False,
        },
    }
    write_json(output_root / "run_manifest.json", manifest)
    return report
