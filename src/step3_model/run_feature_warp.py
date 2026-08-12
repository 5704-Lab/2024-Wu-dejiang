"""Real-data smoke validation for the DCN Feature Warp (Step 3.4-B)."""

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

from .backbone import FEATURE_STAGES, SharedSiameseBackbone
from .feature_warp import WARP_SPEC, FeatureWarp
from .offset_predictor import OFFSET_SPEC, OffsetPredictor

_EVIDENCE_KEYS = (
    "alignment_validation_report", "alignment_frozen_parameters",
    "alignment_validation_manifest", "alignment_test_report", "alignment_test_manifest",
)


def load_feature_warp_config(path: Path) -> dict[str, Any]:
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
    model, run = config.get("model", {}), config.get("run", {})
    if config.get("roadmap_step") != "3.4-B" or config.get("schema_version") != "step3-feature-warp-config-v1":
        raise ValueError("Invalid Step 3.4-B config schema")
    if model.get("architecture") != "feature_warp":
        raise ValueError("Invalid feature warp architecture")
    if int(model.get("kernel_size", 0)) != 3:
        raise ValueError("kernel_size must be 3 for Step 3.4-B")
    if model.get("identity_kernel") is not True:
        raise ValueError("identity_kernel must be true for Step 3.4-B")
    if model.get("warp_input") != "template_features":
        raise ValueError("warp_input must be template_features")
    required_run = (run.get("purpose"), run.get("split"), run.get("selection"), run.get("inference_dtype"))
    if required_run != ("feature_warp_smoke", "val", "ordered_prefix", "float32"):
        raise ValueError("Invalid feature warp smoke contract")
    if int(run.get("sample_count", 0)) <= 0 or int(run.get("batch_size", 0)) <= 0:
        raise ValueError("Smoke sample and batch counts must be positive")
    if run.get("device") not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unsupported smoke device")
    if run.get("dcn_warp_started") is not True:
        raise ValueError("3.4-B must have dcn_warp_started=true")
    if run.get("train_started") is not False:
        raise ValueError("3.4-B must not start training")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _evidence_hashes(config: dict[str, Any]) -> dict[str, str]:
    return {key: sha256_file(Path(config[f"_{key}"])) for key in _EVIDENCE_KEYS}


def run_feature_warp_smoke(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    dataset_root, output_root = Path(config["_dataset_root"]), Path(config["_output_root"])
    validate_roots(dataset_root, output_root)
    if output_root.resolve() == (Path(config["_alignment_validation_report"]).parent.resolve()):
        raise ValueError("Feature warp output must not overwrite Step 3.2 evidence")
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
        kernel_size=3, hidden_ratio=4, zero_init=True,
    ).to(device).eval()
    warp = FeatureWarp(kernel_size=3).to(device).eval()

    batches = []
    sample_ids = []
    stage_names = [name for name, _, _, _ in FEATURE_STAGES]
    predicted_names = [name for name, _, _ in OFFSET_SPEC]
    warp_names = [name for name, _, _ in WARP_SPEC]
    with torch.inference_mode():
        for batch in loader:
            query, template = batch["query"].to(device), batch["template"].to(device)
            qf, tf = backbone(query, template)
            offsets = predictor(qf, tf)

            # Zero-offset identity check
            warped_zero = warp(tf, offsets)
            if list(warped_zero) != warp_names:
                raise RuntimeError("Warp keys differ from WARP_SPEC")
            for name in warp_names:
                if not torch.isfinite(warped_zero[name]).all():
                    raise RuntimeError(f"Warped features non-finite at {name}")
            zero_identity_verified = all(
                torch.equal(warped_zero[name], tf[name]) for name in warp_names
            )

            # Non-zero offset sanity: synthetic small offsets
            synthetic_offsets = {}
            for name, _, stride in OFFSET_SPEC:
                h, w = offsets[name].shape[2:]
                syn = torch.randn(1, 18, h, w, device=device) * 0.5
                syn = torch.tanh(syn) * stride * 0.3
                synthetic_offsets[name] = syn
            warped_nonzero = warp(tf, synthetic_offsets)
            nonzero_finite = all(
                torch.isfinite(warped_nonzero[name]).all() for name in warp_names
            )
            nonzero_differs = all(
                not torch.equal(warped_nonzero[name], tf[name]) for name in warp_names
            )

            expected_shapes = {
                name: [1, channels, 800 // stride, 800 // stride]
                for name, channels, stride in WARP_SPEC
            }
            max_abs_diff = {
                name: float((warped_zero[name] - tf[name]).abs().max().item())
                for name in warp_names
            }
            batches.append({
                "sample_ids": [meta["sample_id"] for meta in batch["meta"]],
                "input_shape": list(query.shape),
                "template_feature_shapes": {name: list(tf[name].shape) for name in stage_names},
                "warped_feature_shapes": {name: list(warped_zero[name].shape) for name in warp_names},
                "offset_shapes": {name: list(offsets[name].shape) for name in predicted_names},
                "dtype": str(warped_zero[warp_names[0]].dtype),
                "device": str(warped_zero[warp_names[0]].device),
                "finite": True,
                "zero_offset_identity_verified": zero_identity_verified,
                "max_abs_diff_per_scale": max_abs_diff,
                "non_zero_offset_finite": nonzero_finite,
                "non_zero_offset_differs_template": nonzero_differs,
            })
            sample_ids.extend(meta["sample_id"] for meta in batch["meta"])

    verified = (
        len(sample_ids) == count
        and all(batch["template_feature_shapes"] == batch["warped_feature_shapes"] for batch in batches)
        and all(batch["finite"] for batch in batches)
        and all(batch["zero_offset_identity_verified"] for batch in batches)
        and all(batch["non_zero_offset_finite"] for batch in batches)
        and all(batch["non_zero_offset_differs_template"] for batch in batches)
    )

    if before != snapshot_tree(dataset_root):
        raise RuntimeError("Read-only dataset changed during feature warp smoke")
    if evidence_before != _evidence_hashes(config):
        raise RuntimeError("Step 3.2 evidence changed during feature warp smoke")
    if not verified:
        raise RuntimeError("Feature warp smoke verification failed")

    contract = {
        "schema_version": "step3-feature-warp-contract-v1", "roadmap_step": "3.4-B",
        "architecture": "feature_warp", "kernel_size": 3, "identity_kernel": True,
        "warp_input": "template_features",
        "warp_output_shapes": {
            name: ["B", channels, f"H/{stride}", f"W/{stride}"]
            for name, channels, stride in WARP_SPEC
        },
        "dcn": {
            "function": "torchvision.ops.deform_conv2d",
            "stride": 1, "padding": 1, "dilation": 1, "kernel_size": 3,
            "offset_layout_per_scale": ["B", 18, "H", "W"],
        },
        "zero_offset_identity": True,
        "key_order": warp_names,
    }
    report = {
        "schema_version": "step3-feature-warp-smoke-v1", "roadmap_step": "3.4-B",
        "verified": True, "processed_sample_count": len(sample_ids),
        "expected_sample_count": count, "ordered_validation_sample_ids": sample_ids,
        "selected_device": str(device), "batches": batches,
        "dcn_warp_started": True, "train_started": False,
    }
    json.dumps(contract, allow_nan=False); json.dumps(report, allow_nan=False)
    output_root.mkdir(parents=True, exist_ok=True)
    contract_path, report_path = output_root / "feature_warp_contract.json", output_root / "smoke_report.json"
    write_json(contract_path, contract); write_json(report_path, report)
    manifest = {
        "schema_version": "step3-feature-warp-run-manifest-v1", "roadmap_step": "3.4-B",
        "command": "feature-warp-smoke", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "cuda_available": torch.cuda.is_available(),
                     "selected_device": str(device),
                     "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "inputs": {"config_sha256": sha256_file(Path(config["_config_path"])),
                   "step2_config_sha256": sha256_file(Path(config["_step2_config"])),
                   "source_snapshot_digest": source_digest, "step3_2_evidence_sha256": evidence_before},
        "artifacts": {"feature_warp_contract.json": sha256_file(contract_path),
                      "smoke_report.json": sha256_file(report_path)},
        "controls": {"source_read_only_verified": True, "alignment_evidence_integrity_verified": True,
                     "train_started": False, "dcn_warp_started": True,
                     "feature_warp_started": True, "step_3_4_c_started": False},
    }
    write_json(output_root / "run_manifest.json", manifest)
    return report
