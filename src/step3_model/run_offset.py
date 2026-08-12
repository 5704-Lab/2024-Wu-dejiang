"""Real-data smoke validation for the multi-scale Offset Predictor."""

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
from .offset_predictor import OFFSET_SPEC, OffsetPredictor

_EVIDENCE_KEYS = (
    "alignment_validation_report", "alignment_frozen_parameters",
    "alignment_validation_manifest", "alignment_test_report", "alignment_test_manifest",
)


def load_offset_config(path: Path) -> dict[str, Any]:
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
    if config.get("roadmap_step") != "3.4-A" or config.get("schema_version") != "step3-offset-config-v1":
        raise ValueError("Invalid Step 3.4-A config schema")
    if model.get("architecture") != "offset_predictor":
        raise ValueError("Invalid offset predictor architecture")
    if int(model.get("kernel_size", 0)) <= 0 or int(model.get("hidden_ratio", 0)) <= 0:
        raise ValueError("kernel_size and hidden_ratio must be positive")
    if model.get("zero_init") is not True:
        raise ValueError("zero_init must be true for Step 3.4-A")
    if model.get("offset_semantics") != "pixel_sampling_offsets":
        raise ValueError("offset_semantics must be pixel_sampling_offsets")
    if model.get("max_offset_per_stage") != [stride for _, _, stride in OFFSET_SPEC]:
        raise ValueError("max_offset_per_stage must match the per-scale strides")
    required_run = (run.get("purpose"), run.get("split"), run.get("selection"), run.get("inference_dtype"))
    if required_run != ("offset_smoke", "val", "ordered_prefix", "float32"):
        raise ValueError("Invalid offset smoke contract")
    if int(run.get("sample_count", 0)) <= 0 or int(run.get("batch_size", 0)) <= 0:
        raise ValueError("Smoke sample and batch counts must be positive")
    if run.get("device") not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unsupported smoke device")
    if run.get("train_started") is not False:
        raise ValueError("3.4-A must not start training")
    if run.get("dcn_started") is not False:
        raise ValueError("3.4-A must not start DCN")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _evidence_hashes(config: dict[str, Any]) -> dict[str, str]:
    return {key: sha256_file(Path(config[f"_{key}"])) for key in _EVIDENCE_KEYS}


def run_offset_smoke(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    dataset_root, output_root = Path(config["_dataset_root"]), Path(config["_output_root"])
    validate_roots(dataset_root, output_root)
    if output_root.resolve() == (Path(config["_alignment_validation_report"]).parent.resolve()):
        raise ValueError("Offset output must not overwrite Step 3.2 evidence")
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
    ).to(device).eval()

    batches = []
    sample_ids = []
    stage_names = [name for name in FEATURE_STAGES]
    predicted_names = [name for name, _, _ in OFFSET_SPEC]
    with torch.inference_mode():
        for batch in loader:
            query, template = batch["query"].to(device), batch["template"].to(device)
            qf, tf = backbone(query, template)
            offsets = predictor(qf, tf)
            if list(offsets) != predicted_names:
                raise RuntimeError("Offset keys differ from OFFSET_SPEC")
            for name in predicted_names:
                if not torch.isfinite(offsets[name]).all():
                    raise RuntimeError(f"Offset non-finite at {name}")
            expected_shapes = {
                name: [1, 18, 800 // stride, 800 // stride]
                for name, _, stride in OFFSET_SPEC
            }
            if {name: list(v.shape) for name, v in offsets.items()} != expected_shapes:
                raise RuntimeError(f"Unexpected offset shapes: {offsets}")
            batches.append({
                "sample_ids": [meta["sample_id"] for meta in batch["meta"]],
                "input_shape": list(query.shape),
                "offset_shapes": {name: list(v.shape) for name, v in offsets.items()},
                "offset_max_abs": {name: float(v.abs().max().item()) for name, v in offsets.items()},
                "offset_mean": {name: float(v.mean().item()) for name, v in offsets.items()},
                "offset_std": {name: float(v.std().item()) for name, v in offsets.items()},
                "dtype": str(offsets[predicted_names[0]].dtype),
                "device": str(offsets[predicted_names[0]].device), "finite": True,
            })
            sample_ids.extend(meta["sample_id"] for meta in batch["meta"])
    max_abs_all = {
        name: max(batch["offset_max_abs"][name] for batch in batches) for name in predicted_names
    }
    parameter_count = sum(parameter.numel() for parameter in predictor.parameters())
    verified = (
        len(sample_ids) == count
        and all(batch["offset_shapes"] == expected_shapes for batch in batches)
        and all(batch["finite"] for batch in batches)
        and all(value == 0 for value in max_abs_all.values())
    )

    if before != snapshot_tree(dataset_root):
        raise RuntimeError("Read-only dataset changed during offset smoke")
    if evidence_before != _evidence_hashes(config):
        raise RuntimeError("Step 3.2 evidence changed during offset smoke")
    if not verified:
        raise RuntimeError("Offset predictor smoke verification failed")

    contract = {
        "schema_version": "step3-offset-contract-v1", "roadmap_step": "3.4-A",
        "architecture": "offset_predictor", "kernel_size": 3, "hidden_ratio": 4,
        "zero_init": True, "offset_semantics": "pixel_sampling_offsets",
        "stage_spec": {name: {"in_channels": channels, "stride": stride, "out_channels": 18}
                       for name, channels, stride in OFFSET_SPEC},
        "key_order": predicted_names,
        "dcn_expected_offset_layout": [1, 18, "H", "W"],
    }
    report = {
        "schema_version": "step3-offset-smoke-v1", "roadmap_step": "3.4-A",
        "verified": True, "processed_sample_count": len(sample_ids),
        "expected_sample_count": count, "ordered_validation_sample_ids": sample_ids,
        "selected_device": str(device), "batches": batches,
        "offset_max_abs_per_scale": max_abs_all, "parameter_count": parameter_count,
        "train_started": False, "dcn_started": False,
    }
    json.dumps(contract, allow_nan=False); json.dumps(report, allow_nan=False)
    output_root.mkdir(parents=True, exist_ok=True)
    contract_path, report_path = output_root / "offset_contract.json", output_root / "smoke_report.json"
    write_json(contract_path, contract); write_json(report_path, report)
    manifest = {
        "schema_version": "step3-offset-run-manifest-v1", "roadmap_step": "3.4-A",
        "command": "offset-smoke", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "cuda_available": torch.cuda.is_available(),
                     "selected_device": str(device)},
        "inputs": {"config_sha256": sha256_file(Path(config["_config_path"])),
                   "step2_config_sha256": sha256_file(Path(config["_step2_config"])),
                   "source_snapshot_digest": source_digest, "step3_2_evidence_sha256": evidence_before},
        "artifacts": {"offset_contract.json": sha256_file(contract_path),
                      "smoke_report.json": sha256_file(report_path)},
        "controls": {"source_read_only_verified": True, "alignment_evidence_integrity_verified": True,
                     "train_started": False, "dcn_started": False, "feature_warp_started": False,
                     "step_3_4_b_started": False},
    }
    write_json(output_root / "run_manifest.json", manifest)
    return report