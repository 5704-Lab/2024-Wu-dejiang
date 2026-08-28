from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from step1_audit.paths import is_within, read_jsonl, resolved

from . import CLASS_NAMES, CLASS_TO_ID
from .schema import DatasetIndex, SampleRecord

SPLITS = ("train", "val", "test")


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    workspace = config_path.resolve().parent.parent
    config["_workspace"] = str(workspace)
    config["_config_path"] = str(config_path.resolve())
    config["_step1_root"] = str(resolved(config["inputs"]["step1_root"], workspace))
    config["_output_root"] = str(resolved(config["output"]["root"], workspace))
    config["_dataset_root"] = str(resolved(config["inputs"]["dataset_root"], workspace))
    if tuple(config["classes"]) != CLASS_NAMES:
        raise ValueError(f"Class order must be {CLASS_NAMES}")
    dataset_root = Path(config["_dataset_root"])
    output_root = Path(config["_output_root"])
    if is_within(output_root, dataset_root):
        raise ValueError("Step 2 output must not be inside the read-only dataset")
    _cross_check_step1_root(config)
    return config


def build_dataset_index(config: dict[str, Any]) -> DatasetIndex:
    dataset_root = Path(config["_dataset_root"])
    step1_root = Path(config["_step1_root"])
    samples_path = step1_root / config["inputs"]["samples"]
    records = read_jsonl(samples_path)
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = record["sample_id"]
        if sample_id in by_id:
            raise ValueError(f"Duplicate sample_id in index: {sample_id}")
        by_id[sample_id] = record

    split_ids = {
        split: _read_split_ids(step1_root / config["inputs"]["splits_dir"] / f"{split}.txt")
        for split in SPLITS
    }
    _validate_split_membership(by_id, split_ids)
    records_by_split: dict[str, tuple[SampleRecord, ...]] = {}
    for split, ids in split_ids.items():
        parsed = tuple(_parse_record(by_id[sample_id], dataset_root) for sample_id in ids)
        records_by_split[split] = parsed
    return DatasetIndex(dataset_root=dataset_root, records_by_split=records_by_split)


def _cross_check_step1_root(config: dict[str, Any]) -> None:
    manifest_path = Path(config["_step1_root"]) / config["inputs"]["step1_manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Step 1 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if resolved(manifest["dataset_root"]) != Path(config["_dataset_root"]):
        raise ValueError("Step 2 dataset root differs from the verified Step 1 root")


def _read_split_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate sample IDs in split: {path}")
    return values


def _validate_split_membership(by_id: dict[str, dict[str, Any]], split_ids: dict[str, list[str]]) -> None:
    split_sets = {name: set(ids) for name, ids in split_ids.items()}
    if split_sets["train"] & split_sets["val"] or split_sets["train"] & split_sets["test"] or split_sets["val"] & split_sets["test"]:
        raise ValueError("Step 1 split sample IDs overlap")
    union = set().union(*split_sets.values())
    missing = set(by_id) - union
    unknown = union - set(by_id)
    if missing or unknown:
        raise ValueError(f"Split/index mismatch: missing={len(missing)}, unknown={len(unknown)}")
    source_assignments: dict[str, str] = {}
    for split, ids in split_ids.items():
        for sample_id in ids:
            source_id = by_id[sample_id]["source_id"]
            previous = source_assignments.setdefault(source_id, split)
            if previous != split:
                raise ValueError(f"Source group {source_id} crosses {previous}/{split}")


def _parse_record(record: dict[str, Any], dataset_root: Path) -> SampleRecord:
    query_path = _resolve_source_path(dataset_root, record["query_path"])
    template_path = _resolve_source_path(dataset_root, record["template_path"])
    boxes: list[tuple[float, float, float, float]] = []
    class_names: list[str] = []
    labels: list[int] = []
    for obj in record["objects"]:
        class_name = obj["class_name"]
        if class_name not in CLASS_TO_ID:
            raise ValueError(f"Unknown class {class_name!r} in {record['sample_id']}")
        box = tuple(float(value) for value in obj["bbox_xyxy"])
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError(f"Invalid box in {record['sample_id']}: {box}")
        xmin, ymin, xmax, ymax = box
        if not (0 <= xmin < xmax <= 800 and 0 <= ymin < ymax <= 800):
            raise ValueError(f"Out-of-bounds box in {record['sample_id']}: {box}")
        boxes.append(box)
        class_names.append(class_name)
        labels.append(CLASS_TO_ID[class_name])
    return SampleRecord(
        sample_id=record["sample_id"], source_id=record["source_id"], variant=record["variant"],
        query_path=query_path, template_path=template_path,
        boxes_xyxy=tuple(boxes), class_names=tuple(class_names), labels=tuple(labels),
    )


def _resolve_source_path(dataset_root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise ValueError(f"Source path must be relative: {relative}")
    path = (dataset_root / value).resolve()
    if not is_within(path, dataset_root) or not path.is_file():
        raise ValueError(f"Invalid source path: {relative}")
    return path
