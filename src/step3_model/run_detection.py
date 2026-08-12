"""COCO export and inference bridge smoke runner for Step 5-A."""

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
from .coco_exporter import export_coco_dataset
from .feature_warp import FeatureWarp
from .fusion import MultiScaleFusion
from .inference_bridge import pad_image_to_size, run_inference
from .offset_predictor import OffsetPredictor

_EVIDENCE_KEYS = (
    "alignment_validation_report", "alignment_frozen_parameters",
    "alignment_validation_manifest", "alignment_test_report", "alignment_test_manifest",
)


def _dummy_rtdetr(padded: torch.Tensor) -> torch.Tensor:
    """Dummy RT-DETR returning empty detections."""
    return torch.empty((0, 6), device=padded.device)


def load_detection_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    workspace = path.resolve().parent.parent
    config["_config_path"] = str(path.resolve())
    config["_workspace"] = str(workspace)
    config["_output_root"] = str(resolved(config["output"]["root"], workspace))
    config["_dataset_root"] = str(resolved(config["inputs"]["dataset_root"]))
    config["_step2_config"] = str(resolved(config["inputs"]["step2_config"], workspace))
    for key in _EVIDENCE_KEYS:
        config[f"_{key}"] = str(resolved(config["inputs"][key], workspace))
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "step3-detection-config-v1":
        raise ValueError("Invalid Step 5-A config schema")
    run = config.get("run", {})
    if run.get("device") not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unsupported smoke device")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _evidence_hashes(config: dict[str, Any]) -> dict[str, str]:
    return {key: sha256_file(Path(config[f"_{key}"])) for key in _EVIDENCE_KEYS}


def run_coco_export(config: dict[str, Any]) -> dict[str, Any]:
    """Export COCO datasets for train/val/test splits."""
    _validate_config(config)
    dataset_root = Path(config["_dataset_root"])
    output_root = Path(config["_output_root"])
    validate_roots(dataset_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    step2 = load_step2_config(Path(config["_step2_config"]))
    index = build_dataset_index(step2)

    summaries = {}
    for split in ("train", "val", "test"):
        output_path = output_root / f"coco_{split}.json"
        summary = export_coco_dataset(index, dataset_root, output_path, split=split)
        summaries[split] = summary

    manifest = {
        "schema_version": "step3-coco-export-manifest-v1", "roadmap_step": "5-A",
        "command": "export-coco", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summaries": summaries,
    }
    write_json(output_root / "coco_export_manifest.json", manifest)
    return manifest


def run_detection_smoke(config: dict[str, Any]) -> dict[str, Any]:
    """Smoke test: zero-init predictor + dummy RT-DETR, verify pipeline shapes."""
    _validate_config(config)
    dataset_root, output_root = Path(config["_dataset_root"]), Path(config["_output_root"])
    validate_roots(dataset_root, output_root)
    before, evidence_before = snapshot_tree(dataset_root), _evidence_hashes(config)
    source_digest = snapshot_digest(before)

    step2 = load_step2_config(Path(config["_step2_config"]))
    index = build_dataset_index(step2)
    dataset = PairedQueryTemplateDataset(index, "val", step2, training=False)
    count = int(config.get("run", {}).get("sample_count", 4))
    selected = Subset(dataset, list(range(count)))
    loader = build_dataloader(selected, batch_size=1, num_workers=0, shuffle=False)
    device = _resolve_device(config.get("run", {}).get("device", "auto"))

    backbone = SharedSiameseBackbone("IMAGENET1K_V2").to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    predictor = OffsetPredictor(kernel_size=3, hidden_ratio=4, zero_init=True).to(device).eval()
    warp = FeatureWarp(kernel_size=3).to(device).eval()
    fusion = MultiScaleFusion(hidden=128, reduction=16).to(device).eval()

    proposals_cfg = config.get("proposals", {
        "min_area": 64, "max_aspect_ratio": 5.0,
        "adaptive_block": 51, "adaptive_c": -5, "morph_kernel": 5,
        "expand_k": 1.8, "merge_iou": 0.3,
    })

    results = []
    with torch.inference_mode():
        for batch in loader:
            query = batch["query"].to(device)
            template = batch["template"].to(device)
            detections = run_inference(
                query[0], template[0],
                backbone, predictor, warp, fusion,
                rtdetr_model=_dummy_rtdetr,
                proposals_cfg=proposals_cfg,
                device=device,
            )
            results.append({
                "sample_id": batch["meta"][0]["sample_id"],
                "detection_count": len(detections),
            })

    verified = len(results) == count

    if before != snapshot_tree(dataset_root):
        raise RuntimeError("Read-only dataset changed")
    if evidence_before != _evidence_hashes(config):
        raise RuntimeError("Step 3.2 evidence changed")
    if not verified:
        raise RuntimeError("Detection smoke verification failed")

    report = {
        "schema_version": "step3-detection-smoke-v1", "roadmap_step": "5-A",
        "verified": True, "selected_device": str(device),
        "sample_count": count, "results": results,
    }
    json.dumps(report, allow_nan=False)
    write_json(output_root / "detection_smoke_report.json", report)

    manifest = {
        "schema_version": "step3-detection-run-manifest-v1", "roadmap_step": "5-A",
        "command": "detection-smoke", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "cuda_available": torch.cuda.is_available(),
                     "selected_device": str(device)},
        "inputs": {"config_sha256": sha256_file(Path(config["_config_path"])),
                   "source_snapshot_digest": source_digest},
        "controls": {"source_read_only_verified": True, "detection_bridge_started": True},
    }
    write_json(output_root / "detection_manifest.json", manifest)
    return report
