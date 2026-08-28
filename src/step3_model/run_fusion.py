"""Real-data smoke validation for multi-scale fusion attention (Step 3.5-B)."""

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
from .fusion import MultiScaleFusion, _STAGE_NAMES
from .heatmap import _imwrite, render_heatmap_panel
from .offset_predictor import OFFSET_SPEC, OffsetPredictor
from .model_io import load_model_checkpoint

_EVIDENCE_KEYS = (
    "alignment_validation_report", "alignment_frozen_parameters",
    "alignment_validation_manifest", "alignment_test_report", "alignment_test_manifest",
)


def load_fusion_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    workspace = path.resolve().parent.parent
    config["_config_path"] = str(path.resolve())
    config["_workspace"] = str(workspace)
    config["_output_root"] = str(resolved(config["output"]["root"], workspace))
    config["_dataset_root"] = str(resolved(config["inputs"]["dataset_root"], workspace))
    config["_step2_config"] = str(resolved(config["inputs"]["step2_config"], workspace))
    config["_predictor_checkpoint"] = str(resolved(config["inputs"]["predictor_checkpoint"], workspace))
    config["_fusion_checkpoint"] = str(resolved(config["inputs"]["fusion_checkpoint"], workspace))
    for key in _EVIDENCE_KEYS:
        config[f"_{key}"] = str(resolved(config["inputs"][key], workspace))
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("roadmap_step") != "3.5-B" or config.get("schema_version") != "step3-fusion-config-v1":
        raise ValueError("Invalid Step 3.5-B config schema")
    run = config.get("run", {})
    required_run = (run.get("purpose"), run.get("split"), run.get("selection"))
    if required_run != ("fusion_smoke", "val", "ordered_prefix"):
        raise ValueError("Invalid fusion smoke contract")
    if int(run.get("sample_count", 0)) <= 0 or int(run.get("batch_size", 0)) <= 0:
        raise ValueError("Smoke sample and batch counts must be positive")
    if run.get("device") not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unsupported smoke device")
    fusion = config.get("fusion", {})
    if int(fusion.get("hidden", 0)) <= 0 or int(fusion.get("reduction", 0)) <= 0:
        raise ValueError("fusion hidden/reduction must be positive")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _evidence_hashes(config: dict[str, Any]) -> dict[str, str]:
    return {key: sha256_file(Path(config[f"_{key}"])) for key in _EVIDENCE_KEYS}


def _tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.squeeze(0).cpu().numpy()
    img = np.clip(img.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    return img


def _render_enhanced_panel(
    query_img: np.ndarray,
    template_img: np.ndarray,
    enhanced: np.ndarray,
    output_path: Path,
    sample_id: str = "",
) -> None:
    """Render 3-column panel: query | template | enhanced_heatmap."""
    h, w = template_img.shape[:2]
    gap, title_h = 4, 36
    canvas = np.full((h + title_h, w * 3 + gap * 2, 3), 248, dtype=np.uint8)
    canvas[:h, :w] = query_img
    canvas[:h, w + gap:2 * w + gap] = template_img
    norm_max = float(enhanced.max()) or 1.0
    colored = cv2.applyColorMap(
        (np.clip(enhanced / norm_max, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO,
    )
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    canvas[:h, 2 * w + 2 * gap:3 * w + 2 * gap] = colored
    for text, x in zip(["QUERY", "TEMPLATE", "ENHANCED"], [0, w + gap, 2 * w + 2 * gap]):
        cv2.putText(canvas, text, (x + 4, h + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (44, 44, 44), 1, cv2.LINE_AA)
    if sample_id:
        cv2.putText(canvas, sample_id, (8, h + title_h - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (128, 128, 128), 1, cv2.LINE_AA)
    _imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def run_fusion_smoke(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    dataset_root, output_root = Path(config["_dataset_root"]), Path(config["_output_root"])
    validate_roots(dataset_root, output_root)
    sample_dir = Path(config["_alignment_validation_report"]).parent.resolve()
    if output_root.resolve() == sample_dir:
        raise ValueError("Fusion output must not overwrite Step 3.2 evidence")
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
    load_model_checkpoint(predictor, Path(config["_predictor_checkpoint"]), device, "OffsetPredictor",
                          "defect-masked-alignment-v1")
    load_model_checkpoint(fusion, Path(config["_fusion_checkpoint"]), device, "MultiScaleFusion",
                          "supervised-fusion-heatmap-v1")
    warp = FeatureWarp(kernel_size=int(config["model"]["kernel_size"])).to(device).eval()
    fusion = MultiScaleFusion(
        hidden=int(config["fusion"]["hidden"]),
        reduction=int(config["fusion"]["reduction"]),
    ).to(device).eval()

    panels_dir = output_root / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    predicted_names = [name for name, _, _ in OFFSET_SPEC]
    batches = []
    sample_ids = []

    with torch.inference_mode():
        for batch in loader:
            query = batch["query"].to(device)
            template = batch["template"].to(device)
            qf, tf = backbone(query, template)
            offsets = predictor(qf, tf)
            warped = warp(tf, offsets)

            # Compute diff features (raw tensor difference, not scalar)
            diff_features = {key: (warped[key] - qf[key]).abs() for key in predicted_names}
            enhanced = torch.sigmoid(fusion(diff_features))  # [B, 1, 800, 800]

            for name in predicted_names:
                if not torch.isfinite(warped[name]).all():
                    raise RuntimeError(f"Non-finite warped features at {name}")
            if not torch.isfinite(enhanced).all():
                raise RuntimeError("Non-finite enhanced heatmap")

            batch_stats = {
                "sample_ids": [meta["sample_id"] for meta in batch["meta"]],
                "input_shape": list(query.shape),
                "warped_shapes": {name: list(warped[name].shape) for name in predicted_names},
                "enhanced_shape": list(enhanced.shape),
                "enhanced_min": float(enhanced.min().item()),
                "enhanced_max": float(enhanced.max().item()),
                "enhanced_mean": float(enhanced.mean().item()),
                "fusion_weights": fusion.fusion_weights.softmax(dim=0).tolist(),
                "dtype": str(enhanced.dtype),
                "device": str(enhanced.device),
                "finite": True,
            }
            batches.append(batch_stats)
            sample_ids.extend(meta["sample_id"] for meta in batch["meta"])

            # Render comparison: enhanced heatmap
            for idx, meta in enumerate(batch["meta"]):
                q_img = _tensor_to_rgb(query[idx:idx + 1])
                t_img = _tensor_to_rgb(template[idx:idx + 1])
                enh_np = enhanced[idx, 0].cpu().numpy()
                _render_enhanced_panel(
                    q_img, t_img, enh_np,
                    panels_dir / f"{meta['sample_id']}_enhanced.png",
                    sample_id=meta["sample_id"],
                )

    expected_shapes = {
        name: [1, channels, 800 // stride, 800 // stride]
        for name, channels, stride in OFFSET_SPEC
    }
    verified = (
        len(sample_ids) == count
        and all(batch["finite"] for batch in batches)
        and all(batch["warped_shapes"] == expected_shapes for batch in batches)
        and all(batch["enhanced_shape"] == [1, 1, 800, 800] for batch in batches)
    )

    if before != snapshot_tree(dataset_root):
        raise RuntimeError("Read-only dataset changed during fusion smoke")
    if evidence_before != _evidence_hashes(config):
        raise RuntimeError("Step 3.2 evidence changed during fusion smoke")
    if not verified:
        raise RuntimeError("Fusion smoke verification failed")

    trainable_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    contract = {
        "schema_version": "step3-fusion-contract-v1", "roadmap_step": "3.5-B",
        "architecture": "cbam_per_scale_fusion",
        "components": {
            "per_scale": [
                {"scale": name, "input_channels": channels, "hidden": int(config["fusion"]["hidden"])}
                for name, channels, _ in OFFSET_SPEC
            ],
            "channel_attention": "SE",
            "spatial_attention": {"kernel": 7, "method": "mean_max_concat"},
            "fusion": "learned_softmax_weighted_sum",
        },
        "trainable_parameter_count": trainable_params,
        "key_order": _STAGE_NAMES,
    }
    report = {
        "schema_version": "step3-fusion-smoke-v1", "roadmap_step": "3.5-B",
        "verified": True, "processed_sample_count": len(sample_ids),
        "expected_sample_count": count, "ordered_validation_sample_ids": sample_ids,
        "selected_device": str(device), "batches": batches,
        "trainable_parameter_count": trainable_params,
        "png_count": len(list(panels_dir.glob("*.png"))),
    }
    json.dumps(contract, allow_nan=False); json.dumps(report, allow_nan=False)
    contract_path, report_path = output_root / "fusion_contract.json", output_root / "smoke_report.json"
    write_json(contract_path, contract); write_json(report_path, report)
    manifest = {
        "schema_version": "step3-fusion-run-manifest-v1", "roadmap_step": "3.5-B",
        "command": "fusion-smoke", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "cuda_available": torch.cuda.is_available(),
                     "selected_device": str(device),
                     "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "inputs": {"config_sha256": sha256_file(Path(config["_config_path"])),
                   "step2_config_sha256": sha256_file(Path(config["_step2_config"])),
                   "source_snapshot_digest": source_digest, "step3_2_evidence_sha256": evidence_before},
        "artifacts": {"fusion_contract.json": sha256_file(contract_path),
                      "smoke_report.json": sha256_file(report_path)},
        "controls": {"source_read_only_verified": True, "alignment_evidence_integrity_verified": True,
                     "fusion_attention_started": True, "step_3_5_b_started": True},
    }
    write_json(output_root / "run_manifest.json", manifest)
    return report
