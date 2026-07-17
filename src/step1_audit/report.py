from __future__ import annotations

import csv
import io
import json
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from .grouping import build_groups
from .hashing import sha256_file, sha256_text, snapshot_digest, snapshot_tree
from .paths import atomic_write_text, read_jsonl, resolved, validate_roots, write_json, write_jsonl
from .split import make_group_split, summarize_split, validate_split
from .voc_parse import parse_voc_xml


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    workspace = config_path.resolve().parent.parent
    config["_config_path"] = str(config_path.resolve())
    config["_workspace"] = str(workspace)
    config["_dataset_root"] = str(resolved(config["dataset"]["root"]))
    config["_output_root"] = str(resolved(config["output"]["root"], workspace))
    validate_roots(Path(config["_dataset_root"]), Path(config["_output_root"]))
    return config


def run_audit(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    dataset_root = Path(config["_dataset_root"])
    output_root = Path(config["_output_root"])
    paths = _dataset_paths(config, dataset_root)
    plan = {"dataset_root": str(dataset_root), "output_root": str(output_root), "inputs": {k: str(v) for k, v in paths.items()}}
    if dry_run:
        return plan

    before = snapshot_tree(dataset_root)
    inventory: list[dict[str, Any]] = []
    for role in ("query", "template", "annotation"):
        extension = "*.xml" if role == "annotation" else "*.jpg"
        for path in sorted(paths[role].glob(extension)):
            inventory.append({
                "role": role,
                "relative_path": path.relative_to(dataset_root).as_posix(),
                "sample_id": path.stem,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })

    by_role = {role: {item["sample_id"]: item for item in inventory if item["role"] == role} for role in ("query", "template", "annotation")}
    all_ids = sorted(set().union(*(set(values) for values in by_role.values())))
    samples: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    expected = config["expected"]
    allowed_classes = set(config["classes"])

    for sample_id in all_ids:
        missing = [role for role in by_role if sample_id not in by_role[role]]
        if missing:
            for role in missing:
                issues.append({"severity": "fatal", "sample_id": sample_id, "code": f"missing_{role}", "message": role})
            continue
        query_path = dataset_root / by_role["query"][sample_id]["relative_path"]
        template_path = dataset_root / by_role["template"][sample_id]["relative_path"]
        xml_path = dataset_root / by_role["annotation"][sample_id]["relative_path"]
        query_meta, query_issues = _image_metadata(query_path, sample_id, "query", expected)
        template_meta, template_issues = _image_metadata(template_path, sample_id, "template", expected)
        parsed, xml_issues = parse_voc_xml(xml_path, sample_id, allowed_classes)
        issues.extend(query_issues + template_issues + xml_issues)
        xml_size = parsed["xml_size"]
        if xml_size and query_meta.get("width") and (xml_size["width"], xml_size["height"]) != (query_meta["width"], query_meta["height"]):
            issues.append({"severity": "fatal", "sample_id": sample_id, "code": "xml_image_size_mismatch", "message": str(xml_size)})
        samples.append({
            "schema_version": config["schema_version"],
            "sample_id": sample_id,
            "source_id": None,
            "variant": None,
            "query_path": by_role["query"][sample_id]["relative_path"],
            "template_path": by_role["template"][sample_id]["relative_path"],
            "annotation_path": by_role["annotation"][sample_id]["relative_path"],
            "query_sha256": by_role["query"][sample_id]["sha256"],
            "template_sha256": by_role["template"][sample_id]["sha256"],
            "annotation_sha256": by_role["annotation"][sample_id]["sha256"],
            "query_image": query_meta,
            "template_image": template_meta,
            "xml_filename": parsed["xml_filename"],
            "xml_size": xml_size,
            "objects": parsed["objects"],
        })

    groups, group_issues = build_groups(samples, config["expected"]["variants"])
    issues.extend(group_issues)
    duplicates = _duplicate_report(samples)
    original_split = _audit_original_splits(paths["splits"], samples)
    class_counts = Counter(obj["class_name"] for sample in samples for obj in sample["objects"])
    code_counts = Counter(issue["code"] for issue in issues)
    report = {
        "schema_version": config["schema_version"],
        "sample_count": len(samples),
        "group_count": len(groups),
        "object_count": sum(class_counts.values()),
        "class_counts": dict(sorted(class_counts.items())),
        "issue_counts": dict(sorted(code_counts.items())),
        "fatal_issue_count": sum(issue["severity"] == "fatal" for issue in issues),
        "warning_count": sum(issue["severity"] == "warning" for issue in issues),
        "inventory_counts": {role: len(by_role[role]) for role in by_role},
        "original_split": original_split,
        "duplicate_summary": duplicates["summary"],
    }

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "source_snapshot_before.json", {"digest": snapshot_digest(before), "files": before})
    write_jsonl(output_root / "source_inventory.jsonl", inventory)
    write_jsonl(output_root / "samples.jsonl", samples)
    write_jsonl(output_root / "groups.jsonl", groups)
    _write_objects(output_root / "objects.csv", samples)
    _write_issues(output_root / "issues.csv", issues)
    write_json(output_root / "duplicate_report.json", duplicates)
    write_json(output_root / "audit_report.json", report)

    after = snapshot_tree(dataset_root)
    write_json(output_root / "source_snapshot_after.json", {"digest": snapshot_digest(after), "files": after})
    if before != after:
        raise RuntimeError("Read-only source dataset changed during audit")
    manifest = _manifest(config, output_root, "audit")
    write_json(output_root / "run_manifest.json", manifest)
    return report


def run_split(config: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(config["_output_root"])
    samples = read_jsonl(output_root / "samples.jsonl")
    groups = read_jsonl(output_root / "groups.jsonl")
    fatal_count = _read_fatal_count(output_root / "audit_report.json")
    if fatal_count:
        raise RuntimeError(f"Audit has {fatal_count} fatal issues; split refused")
    split_config = config["split"]
    duplicates = json.loads((output_root / "duplicate_report.json").read_text(encoding="utf-8"))
    linked_source_sets = [
        {member["source_id"] for member in duplicate_group}
        for duplicate_group in duplicates["template"]["cross_source_groups"]
    ]
    split = make_group_split(
        groups,
        split_config["ratios"],
        int(split_config["seed"]),
        int(split_config["search_iterations"]),
        config["classes"],
        linked_source_sets=linked_source_sets,
    )
    validate_split(split, groups)
    sample_by_group: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        sample_by_group[sample["source_id"]].append(sample["sample_id"])
    split_dir = output_root / "splits"
    for name, group_ids in split.items():
        atomic_write_text(split_dir / f"{name}_groups.txt", "".join(f"{group_id}\n" for group_id in group_ids))
        sample_ids = sorted(sample_id for group_id in group_ids for sample_id in sample_by_group[group_id])
        atomic_write_text(split_dir / f"{name}.txt", "".join(f"{sample_id}\n" for sample_id in sample_ids))
    summary = summarize_split(split, groups, config["classes"])
    write_json(output_root / "split_summary.json", summary)
    manifest = {
        "schema_version": config["schema_version"],
        "algorithm": split_config["algorithm_version"],
        "seed": split_config["seed"],
        "ratios": split_config["ratios"],
        "groups": split,
        "linked_template_source_sets": [sorted(values) for values in linked_source_sets],
        "summary": summary,
        "samples_digest": sha256_file(output_root / "samples.jsonl"),
        "groups_digest": sha256_file(output_root / "groups.jsonl"),
    }
    write_json(output_root / "split_manifest.json", manifest)
    return summary


def run_verify(config: dict[str, Any]) -> dict[str, Any]:
    dataset_root = Path(config["_dataset_root"])
    output_root = Path(config["_output_root"])
    before = json.loads((output_root / "source_snapshot_before.json").read_text(encoding="utf-8"))
    current = snapshot_tree(dataset_root)
    current_digest = snapshot_digest(current)
    if current_digest != before["digest"]:
        raise RuntimeError("Canonical dataset differs from the pre-audit snapshot")
    groups = read_jsonl(output_root / "groups.jsonl")
    split_manifest = json.loads((output_root / "split_manifest.json").read_text(encoding="utf-8"))
    validate_split(split_manifest["groups"], groups)
    split_of_group = {
        group_id: split_name
        for split_name, group_ids in split_manifest["groups"].items()
        for group_id in group_ids
    }
    for linked in split_manifest.get("linked_template_source_sets", []):
        assigned = {split_of_group[group_id] for group_id in linked}
        if len(assigned) != 1:
            raise RuntimeError(f"Identical templates cross new splits: {linked}")
    group_by_id = {group["source_id"]: group for group in groups}
    for split_name, ids in split_manifest["groups"].items():
        sample_lines = [line.strip() for line in (output_root / "splits" / f"{split_name}.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = sorted(sample_id for group_id in ids for sample_id in group_by_id[group_id]["sample_ids"])
        if sorted(sample_lines) != expected:
            raise RuntimeError(f"{split_name} sample list does not match group membership")
    return {"verified": True, "source_digest": current_digest, "group_count": len(groups), "split_summary": split_manifest["summary"]}


def _dataset_paths(config: dict[str, Any], root: Path) -> dict[str, Path]:
    paths = {
        "query": root / config["dataset"]["query_dir"],
        "template": root / config["dataset"]["template_dir"],
        "annotation": root / config["dataset"]["annotation_dir"],
        "splits": root / config["dataset"]["split_dir"],
    }
    for name, path in paths.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {name} directory: {path}")
    return paths


def _image_metadata(path: Path, sample_id: str, role: str, expected: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode, image_format = image.mode, image.format
    except Exception as exc:
        return {}, [{"severity": "fatal", "sample_id": sample_id, "code": f"unreadable_{role}", "message": str(exc)}]
    if (width, height) != (expected["width"], expected["height"]):
        issues.append({"severity": "fatal", "sample_id": sample_id, "code": f"unexpected_{role}_size", "message": f"{width}x{height}"})
    if mode != expected["mode"]:
        issues.append({"severity": "fatal", "sample_id": sample_id, "code": f"unexpected_{role}_mode", "message": mode})
    return {"width": width, "height": height, "mode": mode, "format": image_format}, issues


def _duplicate_report(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    for role, key in (("query", "query_sha256"), ("template", "template_sha256")):
        buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
        for sample in samples:
            buckets[sample[key]].append({"sample_id": sample["sample_id"], "source_id": sample["source_id"]})
        duplicate_groups = [members for members in buckets.values() if len(members) > 1]
        cross_source = [members for members in duplicate_groups if len({m["source_id"] for m in members}) > 1]
        result[role] = {"duplicate_groups": duplicate_groups, "cross_source_groups": cross_source}
        summary[role] = {
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_extra_files": sum(len(members) - 1 for members in duplicate_groups),
            "cross_source_group_count": len(cross_source),
        }
    result["summary"] = summary
    return result


def _audit_original_splits(split_dir: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    source_by_sample = {sample["sample_id"]: sample["source_id"] for sample in samples}
    split_sets: dict[str, set[str]] = {}
    for name in ("train", "val", "trainval", "test"):
        path = split_dir / f"{name}.txt"
        split_sets[name] = {line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    source_sets = {name: {source_by_sample[sample] for sample in values if sample in source_by_sample} for name, values in split_sets.items()}
    return {
        "sample_counts": {name: len(values) for name, values in split_sets.items()},
        "sample_overlaps": {
            "train_val": len(split_sets["train"] & split_sets["val"]),
            "train_test": len(split_sets["train"] & split_sets["test"]),
            "val_test": len(split_sets["val"] & split_sets["test"]),
        },
        "source_counts": {name: len(values) for name, values in source_sets.items()},
        "source_overlaps": {
            "train_val": len(source_sets["train"] & source_sets["val"]),
            "train_test": len(source_sets["train"] & source_sets["test"]),
            "val_test": len(source_sets["val"] & source_sets["test"]),
        },
        "trainval_matches_union": split_sets["trainval"] == split_sets["train"] | split_sets["val"],
    }


def _write_objects(path: Path, samples: list[dict[str, Any]]) -> None:
    fields = ["object_id", "sample_id", "source_id", "variant", "class_name", "xmin", "ymin", "xmax", "ymax", "width", "height", "area"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for sample in samples:
        for obj in sample["objects"]:
            xmin, ymin, xmax, ymax = obj["bbox_xyxy"]
            writer.writerow({
                "object_id": f"{sample['sample_id']}_{obj['object_index']:03d}", "sample_id": sample["sample_id"],
                "source_id": sample["source_id"], "variant": sample["variant"], "class_name": obj["class_name"],
                "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                "width": obj["width"], "height": obj["height"], "area": obj["area"],
            })
    atomic_write_text(path, buffer.getvalue())


def _write_issues(path: Path, issues: list[dict[str, str]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["severity", "sample_id", "code", "message"])
    writer.writeheader()
    writer.writerows(sorted(issues, key=lambda x: (x["severity"], x["code"], x["sample_id"])))
    atomic_write_text(path, buffer.getvalue())


def _read_fatal_count(path: Path) -> int:
    return int(json.loads(path.read_text(encoding="utf-8"))["fatal_issue_count"])


def _manifest(config: dict[str, Any], output_root: Path, command: str) -> dict[str, Any]:
    artifact_names = ["samples.jsonl", "groups.jsonl", "objects.csv", "issues.csv", "audit_report.json", "duplicate_report.json"]
    return {
        "schema_version": config["schema_version"],
        "command": command,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "dataset_root": config["_dataset_root"],
        "output_root": str(output_root),
        "config_digest": sha256_text(json.dumps(config, sort_keys=True)),
        "artifacts": {name: sha256_file(output_root / name) for name in artifact_names},
    }
