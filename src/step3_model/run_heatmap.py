"""Real-data smoke validation for multi-scale feature difference heatmaps (Step 3.5-A)."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
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
from .heatmap import _imwrite, compute_diff_maps, diff_statistics, render_heatmap_panel, upsample_diff_maps
from .offset_predictor import OFFSET_SPEC, OffsetPredictor

_EVIDENCE_KEYS = (
    "alignment_validation_report", "alignment_frozen_parameters",
    "alignment_validation_manifest", "alignment_test_report", "alignment_test_manifest",
)


def load_heatmap_config(path: Path) -> dict[str, Any]:
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
    if config.get("roadmap_step") != "3.5-A" or config.get("schema_version") != "step3-heatmap-config-v1":
        raise ValueError("Invalid Step 3.5-A config schema")
    run = config.get("run", {})
    required_run = (run.get("purpose"), run.get("split"), run.get("selection"))
    if required_run != ("heatmap_smoke", "val", "ordered_prefix"):
        raise ValueError("Invalid heatmap smoke contract")
    if int(run.get("sample_count", 0)) <= 0 or int(run.get("batch_size", 0)) <= 0:
        raise ValueError("Smoke sample and batch counts must be positive")
    if run.get("device") not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unsupported smoke device")
    heatmap = config.get("heatmap", {})
    if str(heatmap.get("colormap", "inferno")).lower() != "inferno":
        raise ValueError("Unsupported colormap")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _evidence_hashes(config: dict[str, Any]) -> dict[str, str]:
    return {key: sha256_file(Path(config[f"_{key}"])) for key in _EVIDENCE_KEYS}


def _tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convert [3, H, W] float32 [0,1] tensor to [H, W, 3] uint8 RGB."""
    img = tensor.squeeze(0).cpu().numpy()
    img = np.clip(img.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    return img


def run_heatmap_smoke(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    dataset_root, output_root = Path(config["_dataset_root"]), Path(config["_output_root"])
    validate_roots(dataset_root, output_root)
    sample_dir = Path(config["_alignment_validation_report"]).parent.resolve()
    if output_root.resolve() == sample_dir:
        raise ValueError("Heatmap output must not overwrite Step 3.2 evidence")
    before, evidence_before = snapshot_tree(dataset_root), _evidence_hashes(config)
    source_digest = snapshot_digest(before)

    step2 = load_step2_config(Path(config["_step2_config"]))
    index = build_dataset_index(step2)
    dataset = PairedQueryTemplateDataset(index, "val", step2, training=False)
    count = int(config["heatmap"]["sample_count"])
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
    ).to(device).eval()
    warp = FeatureWarp(kernel_size=int(config["model"]["kernel_size"])).to(device).eval()

    predicted_names = [name for name, _, _ in OFFSET_SPEC]
    batches = []
    sample_ids = []
    png_dir = output_root / "panels"
    png_dir.mkdir(parents=True, exist_ok=True)
    colormap = cv2.COLORMAP_INFERNO

    with torch.inference_mode():
        for batch in loader:
            query = batch["query"].to(device)
            template = batch["template"].to(device)
            qf, tf = backbone(query, template)
            offsets = predictor(qf, tf)
            warped = warp(tf, offsets)

            diff_maps = compute_diff_maps(warped, qf, eps=float(config["heatmap"]["eps"]))
            upsampled = upsample_diff_maps(diff_maps, target_size=(800, 800))
            stats = diff_statistics(diff_maps)

            # Render panel for each sample in batch (batch_size=1 typically)
            for index, meta in enumerate(batch["meta"]):
                sample_id = meta["sample_id"]
                q_img = _tensor_to_rgb(query[index:index + 1])
                t_img = _tensor_to_rgb(template[index:index + 1])
                # Extract single-sample diff maps
                single_diff = {k: upsampled[k][np.newaxis, ...] if upsampled[k].ndim == 2
                               else upsampled[k] for k in predicted_names}
                # Fix: each diff_map is already single-sample, just use as-is
                render_heatmap_panel(
                    q_img, t_img, upsampled,
                    png_dir / f"{sample_id}_heatmap.png",
                    sample_id=sample_id,
                    colormap=colormap,
                )

            expected_shapes = {
                name: [1, channels, 800 // stride, 800 // stride]
                for name, channels, stride in OFFSET_SPEC
            }
            batches.append({
                "sample_ids": [meta["sample_id"] for meta in batch["meta"]],
                "input_shape": list(query.shape),
                "warped_shapes": {name: list(warped[name].shape) for name in predicted_names},
                "diff_map_shapes": {name: list(diff_maps[name].shape) for name in predicted_names},
                "dtype": str(warped[predicted_names[0]].dtype),
                "device": str(warped[predicted_names[0]].device),
                "finite": all(np.isfinite(diff_maps[name]).all() for name in predicted_names),
                "diff_statistics": stats,
            })
            sample_ids.extend(meta["sample_id"] for meta in batch["meta"])

    verified = (
        len(sample_ids) == count
        and all(batch["finite"] for batch in batches)
        and all(batch["warped_shapes"] == expected_shapes for batch in batches)
    )

    if before != snapshot_tree(dataset_root):
        raise RuntimeError("Read-only dataset changed during heatmap smoke")
    if evidence_before != _evidence_hashes(config):
        raise RuntimeError("Step 3.2 evidence changed during heatmap smoke")
    if not verified:
        raise RuntimeError("Heatmap smoke verification failed")

    contract = {
        "schema_version": "step3-heatmap-contract-v1", "roadmap_step": "3.5-A",
        "architecture": "feature_diff_heatmap", "colormap": "inferno",
        "scales": ["C3", "C4", "C5"],
        "diff_metric": "relative_l2",
        "diff_formula": "||warped - query||_2 / (||query||_2 + eps)",
        "upsampling": "bilinear", "target_size": [800, 800],
        "key_order": ["C3", "C4", "C5"],
    }
    report = {
        "schema_version": "step3-heatmap-smoke-v1", "roadmap_step": "3.5-A",
        "verified": True, "processed_sample_count": len(sample_ids),
        "expected_sample_count": count, "ordered_validation_sample_ids": sample_ids,
        "selected_device": str(device), "batches": batches,
        "png_count": len(list(png_dir.glob("*.png"))),
    }
    json.dumps(contract, allow_nan=False); json.dumps(report, allow_nan=False)
    contract_path, report_path = output_root / "heatmap_contract.json", output_root / "smoke_report.json"
    write_json(contract_path, contract); write_json(report_path, report)
    manifest = {
        "schema_version": "step3-heatmap-run-manifest-v1", "roadmap_step": "3.5-A",
        "command": "heatmap-smoke", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "cuda_available": torch.cuda.is_available(),
                     "selected_device": str(device),
                     "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "inputs": {"config_sha256": sha256_file(Path(config["_config_path"])),
                   "step2_config_sha256": sha256_file(Path(config["_step2_config"])),
                   "source_snapshot_digest": source_digest, "step3_2_evidence_sha256": evidence_before},
        "artifacts": {"heatmap_contract.json": sha256_file(contract_path),
                      "smoke_report.json": sha256_file(report_path)},
        "controls": {"source_read_only_verified": True, "alignment_evidence_integrity_verified": True,
                     "feature_heatmap_started": True, "step_3_5_a_started": True},
    }
    write_json(output_root / "run_manifest.json", manifest)
    return report
