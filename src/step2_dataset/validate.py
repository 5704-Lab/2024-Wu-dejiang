from __future__ import annotations

import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from step1_audit.hashing import sha256_file, snapshot_digest, snapshot_tree
from step1_audit.paths import write_json

from .collate import pair_collate_fn
from .dataset import PairedQueryTemplateDataset
from .index import build_dataset_index


def run_validation(config: dict[str, Any]) -> dict[str, Any]:
    dataset_root = Path(config["_dataset_root"])
    output_root = Path(config["_output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    before = snapshot_tree(dataset_root)
    index = build_dataset_index(config)
    expected_counts = config["validation"]["expected_split_counts"]
    split_counts = {split: len(index.records(split)) for split in ("train", "val", "test")}
    if split_counts != expected_counts:
        raise RuntimeError(f"Unexpected split counts: {split_counts}")

    image_counts = Counter()
    object_counts = Counter()
    requested_sample_count = int(config["validation"].get("sample_count", sum(split_counts.values())))
    if requested_sample_count <= 0:
        raise ValueError("validation.sample_count must be positive")
    validated_sample_counts = _allocate_validation_samples(split_counts, requested_sample_count)
    for split in ("train", "val", "test"):
        records = index.records(split)
        split_sample_count = validated_sample_counts[split]
        for record in records[:split_sample_count]:
            for role, path in (("query", record.query_path), ("template", record.template_path)):
                with Image.open(path) as image:
                    if image.mode != config["image"]["mode"] or image.size != (config["image"]["width"], config["image"]["height"]):
                        raise RuntimeError(f"Invalid {role} image: {path}")
                image_counts[f"{split}_{role}"] += 1
            object_counts.update(record.class_names)

    probe = run_probe(config, index=index)
    after = snapshot_tree(dataset_root)
    if before != after:
        raise RuntimeError("Read-only dataset changed during Step 2 validation")
    report = {
        "schema_version": config["schema_version"],
        "verified": True,
        "split_counts": split_counts,
        "validated_sample_count": sum(validated_sample_counts.values()),
        "validated_sample_counts": validated_sample_counts,
        "image_counts": dict(image_counts),
        "object_counts": dict(sorted(object_counts.items())),
        "source_digest": snapshot_digest(after),
        "probe": probe,
    }
    write_json(output_root / "validation_report.json", report)
    write_json(output_root / "index_manifest.json", {
        "schema_version": config["schema_version"],
        "split_counts": split_counts,
        "step1_samples_sha256": sha256_file(Path(config["_step1_root"]) / config["inputs"]["samples"]),
        "split_sha256": {
            split: sha256_file(Path(config["_step1_root"]) / config["inputs"]["splits_dir"] / f"{split}.txt")
            for split in ("train", "val", "test")
        },
    })
    _write_run_manifest(config, output_root, report)
    return report


def _allocate_validation_samples(split_counts: dict[str, int], requested_count: int) -> dict[str, int]:
    total_count = sum(split_counts.values())
    target_count = min(requested_count, total_count)
    allocations = {
        split: min(split_counts[split], target_count * split_counts[split] // total_count)
        for split in ("train", "val", "test")
    }
    remaining = target_count - sum(allocations.values())
    for split in ("train", "val", "test"):
        available = split_counts[split] - allocations[split]
        added = min(available, remaining)
        allocations[split] += added
        remaining -= added
        if remaining == 0:
            break
    return allocations


def run_probe(config: dict[str, Any], index=None) -> dict[str, Any]:
    index = index or build_dataset_index(config)
    dataset = PairedQueryTemplateDataset(index, "train", config, training=True, epoch=0)
    count = min(int(config["validation"]["probe_samples"]), len(dataset))
    deterministic = True
    independent_targets = True
    independent_photometry = True
    changed_by_epoch = False
    samples: list[dict[str, Any]] = []
    for position in range(count):
        first = dataset[position]
        second = dataset[position]
        deterministic &= torch.equal(first["query"], second["query"]) and torch.equal(first["template"], second["template"])
        deterministic &= torch.equal(first["query_target"]["boxes"], second["query_target"]["boxes"])
        independent_targets &= first["query_target"] is not first["template_target"]
        independent_targets &= first["query_target"]["boxes"].data_ptr() != first["template_target"]["boxes"].data_ptr()
        params = first["meta"]["transform_params"]
        independent_photometry &= params["query_photometry"] != params["template_photometry"]
        dataset.set_epoch(1)
        next_epoch = dataset[position]
        changed_by_epoch |= not torch.equal(first["query"], next_epoch["query"]) or params != next_epoch["meta"]["transform_params"]
        dataset.set_epoch(0)
        samples.append({"sample_id": first["meta"]["sample_id"], "transform_params": params})
    batch = pair_collate_fn([dataset[0], dataset[1]])
    result = {
        "verified": deterministic and independent_targets and independent_photometry and changed_by_epoch,
        "deterministic_replay": deterministic,
        "independent_targets": independent_targets,
        "independent_photometry": independent_photometry,
        "epoch_changes_training_stream": changed_by_epoch,
        "batch_query_shape": list(batch["query"].shape),
        "batch_template_shape": list(batch["template"].shape),
        "samples": samples,
    }
    if not result["verified"]:
        raise RuntimeError(f"Step 2 deterministic probe failed: {result}")
    output_root = Path(config["_output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "deterministic_probe.json", result)
    return result


def _write_run_manifest(config: dict[str, Any], output_root: Path, report: dict[str, Any]) -> None:
    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except ImportError:
        torchvision_version = None
    write_json(output_root / "run_manifest.json", {
        "schema_version": config["schema_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset_root": config["_dataset_root"],
        "output_root": config["_output_root"],
        "source_digest": report["source_digest"],
    })
