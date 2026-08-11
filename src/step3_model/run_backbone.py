"""Real-data smoke validation for the shared Siamese backbone."""

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
from torchvision.models import ResNet50_Weights

from step1_audit.hashing import sha256_file, snapshot_digest, snapshot_tree
from step1_audit.paths import resolved, validate_roots, write_json
from step2_dataset.collate import build_dataloader
from step2_dataset.dataset import PairedQueryTemplateDataset
from step2_dataset.index import build_dataset_index, load_config as load_step2_config

from .backbone import FEATURE_STAGES, SharedSiameseBackbone

_EVIDENCE_KEYS = (
    "alignment_validation_report", "alignment_frozen_parameters",
    "alignment_validation_manifest", "alignment_test_report", "alignment_test_manifest",
)


def load_backbone_config(path: Path) -> dict[str, Any]:
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
    if config.get("roadmap_step") != "3.3-B" or config.get("schema_version") != "step3-backbone-config-v1":
        raise ValueError("Invalid Step 3.3-B config schema")
    if model.get("architecture") != "resnet50":
        raise ValueError("Invalid backbone architecture")
    if model.get("multi_scale_output") is not True:
        raise ValueError("multi_scale_output must be true for Step 3.3-B")
    declared = [
        (item["name"], item["stage"], item["channels"], item["stride"])
        for item in model.get("feature_stages", [])
    ]
    if tuple(declared) != FEATURE_STAGES:
        raise ValueError(f"feature_stages must exactly match {FEATURE_STAGES}")
    if model.get("weights") not in {"IMAGENET1K_V2", "none", None}:
        raise ValueError("Unsupported backbone weights")
    if model.get("normalization", {}).get("mean") != [0.485, 0.456, 0.406] or model.get("normalization", {}).get("std") != [0.229, 0.224, 0.225]:
        raise ValueError("Backbone normalization must match ImageNet")
    required_run = (run.get("purpose"), run.get("split"), run.get("selection"), run.get("inference_dtype"))
    if required_run != ("backbone_smoke", "val", "ordered_prefix", "float32"):
        raise ValueError("Invalid backbone smoke contract")
    if int(run.get("sample_count", 0)) <= 0 or int(run.get("batch_size", 0)) <= 0:
        raise ValueError("Smoke sample and batch counts must be positive")
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


def _weight_provenance(name: str | None) -> dict[str, Any]:
    if name != "IMAGENET1K_V2":
        return {"name": "none", "url": None, "checkpoint_sha256": None}
    weights = ResNet50_Weights.IMAGENET1K_V2
    filename = Path(weights.url).name
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / filename
    return {
        "name": name,
        "url": weights.url,
        "checkpoint_filename": filename,
        "checkpoint_cached": checkpoint.is_file(),
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
    }


def run_backbone_smoke(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    dataset_root, output_root = Path(config["_dataset_root"]), Path(config["_output_root"])
    validate_roots(dataset_root, output_root)
    if output_root.resolve() == (Path(config["_alignment_validation_report"]).parent.resolve()):
        raise ValueError("Backbone output must not overwrite Step 3.2 evidence")
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
    model = SharedSiameseBackbone(config["model"].get("weights")).to(device).eval()
    hook_calls = 0

    def count_call(_module, _inputs, _output):
        nonlocal hook_calls
        hook_calls += 1

    handle = model.encoder.register_forward_hook(count_call)
    batches = []
    sample_ids = []
    stage_names = [name for name, _, _, _ in FEATURE_STAGES]
    with torch.inference_mode():
        for batch in loader:
            query, template = batch["query"].to(device), batch["template"].to(device)
            qf, tf = model(query, template)
            for features in (qf, tf):
                if list(features) != stage_names:
                    raise RuntimeError("Backbone feature keys differ from FEATURE_STAGES")
                for name in stage_names:
                    if not torch.isfinite(features[name]).all():
                        raise RuntimeError(f"Backbone produced non-finite features at {name}")
            identical_q, identical_t = model(query, query)
            per_scale_difference = {
                name: float((identical_q[name] - identical_t[name]).abs().max().item())
                for name in stage_names
            }
            feature_stats = {
                name: {
                    "query_mean": float(qf[name].mean().item()),
                    "query_std": float(qf[name].std().item()),
                    "template_mean": float(tf[name].mean().item()),
                    "template_std": float(tf[name].std().item()),
                }
                for name in stage_names
            }
            batches.append({
                "sample_ids": [meta["sample_id"] for meta in batch["meta"]],
                "input_shape": list(query.shape),
                "query_feature_shapes": {name: list(qf[name].shape) for name in stage_names},
                "template_feature_shapes": {name: list(tf[name].shape) for name in stage_names},
                "dtype": str(qf[stage_names[0]].dtype),
                "device": str(qf[stage_names[0]].device), "finite": True,
                "feature_stats": feature_stats,
                "identical_input_max_abs_difference_per_scale": per_scale_difference,
            })
            sample_ids.extend(meta["sample_id"] for meta in batch["meta"])
    handle.remove()
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    sharing = {
        "single_encoder_registered": True,
        "no_branch_specific_state_keys": not any(
            "query_backbone" in key or "template_backbone" in key for key in model.state_dict()),
        "parameter_object_count": len(parameter_ids),
        "unique_parameter_object_count": len(set(parameter_ids)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "encoder_forward_calls": hook_calls,
        "expected_encoder_forward_calls": len(batches) * 4,
        "identical_inputs_equal": all(
            value == 0
            for batch in batches
            for value in batch["identical_input_max_abs_difference_per_scale"].values()
        ),
    }
    expected_shapes = {
        name: [1, channels, 800 // stride, 800 // stride]
        for name, _, channels, stride in FEATURE_STAGES
    }
    verified = (
        len(sample_ids) == count
        and all(batch["query_feature_shapes"] == expected_shapes for batch in batches)
        and all(batch["template_feature_shapes"] == expected_shapes for batch in batches)
        and sharing["no_branch_specific_state_keys"]
        and sharing["parameter_object_count"] == sharing["unique_parameter_object_count"]
        and sharing["encoder_forward_calls"] == sharing["expected_encoder_forward_calls"]
        and sharing["identical_inputs_equal"]
    )

    if before != snapshot_tree(dataset_root):
        raise RuntimeError("Read-only dataset changed during backbone smoke")
    if evidence_before != _evidence_hashes(config):
        raise RuntimeError("Step 3.2 evidence changed during backbone smoke")
    if not verified:
        raise RuntimeError("Shared backbone smoke verification failed")

    contract = {
        "schema_version": "step3-backbone-contract-v2", "roadmap_step": "3.3-B",
        "architecture": "resnet50", "weights": config["model"]["weights"],
        "input": {"shape": ["B", 3, 800, 800], "range": [0.0, 1.0], "channel_order": "RGB",
                  "normalization_owner": "model", "mean": [0.485, 0.456, 0.406],
                  "std": [0.229, 0.224, 0.225]},
        "output": {
            "multi_scale_output": True,
            "key_order": [name for name, _, _, _ in FEATURE_STAGES],
            "features": [
                {"name": name, "stage": stage, "channels": channels, "stride": stride,
                 "shape": ["B", channels, 800 // stride, 800 // stride]}
                for name, stage, channels, stride in FEATURE_STAGES
            ],
        },
        "shared_encoder": True, "trainable": True, "deferred_step": "3.4-A",
    }
    weight_info = _weight_provenance(config["model"].get("weights"))
    report = {
        "schema_version": "step3-backbone-smoke-v2", "roadmap_step": "3.3-B",
        "verified": True, "processed_sample_count": len(sample_ids), "expected_sample_count": count,
        "ordered_validation_sample_ids": sample_ids, "selected_device": str(device),
        "batches": batches, "sharing": sharing, "weights": weight_info,
        "input_policy": {"source": "raw_step2_pair_tensors", "alignment_applied": False,
                         "alignment_selection_index_available": False,
                         "alignment_evidence_modified": False,
                         "rationale": "Frozen Validation/Test selections retained original for all evaluated pairs."},
    }
    json.dumps(contract, allow_nan=False); json.dumps(report, allow_nan=False)
    output_root.mkdir(parents=True, exist_ok=True)
    contract_path, report_path = output_root / "backbone_contract.json", output_root / "smoke_report.json"
    write_json(contract_path, contract); write_json(report_path, report)
    manifest = {
        "schema_version": "step3-backbone-run-manifest-v2", "roadmap_step": "3.3-B",
        "command": "backbone-smoke", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "cuda_available": torch.cuda.is_available(),
                     "selected_device": str(device), "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "inputs": {"config_sha256": sha256_file(Path(config["_config_path"])),
                   "step2_config_sha256": sha256_file(Path(config["_step2_config"])),
                   "source_snapshot_digest": source_digest, "step3_2_evidence_sha256": evidence_before,
                   "pretrained_checkpoint_sha256": weight_info["checkpoint_sha256"]},
        "artifacts": {"backbone_contract.json": sha256_file(contract_path),
                      "smoke_report.json": sha256_file(report_path)},
        "controls": {"source_read_only_verified": True, "alignment_evidence_integrity_verified": True,
                     "alignment_recomputed": False, "alignment_evidence_modified": False,
                     "alignment_applied": False, "test_images_decoded": False,
                     "multi_scale_started": True, "detector_head_started": False,
                     "training_started": False},
    }
    write_json(output_root / "run_manifest.json", manifest)
    return report
