"""Validation-only orchestration and parameter freezing for Step 3.2-C."""

from __future__ import annotations

import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from step1_audit.hashing import sha256_file, sha256_text, snapshot_digest, snapshot_tree
from step1_audit.paths import read_jsonl, resolved, validate_roots, write_json

from .alignment import AlignmentResult, SIFTAligner, load_image_rgb, render_comparison

_PARAMETER_SECTIONS = ("sift", "matcher", "ransac", "geometry", "near_identity", "quality")
_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    workspace = config_path.resolve().parent.parent
    config["_config_path"] = str(config_path.resolve())
    config["_workspace"] = str(workspace)
    config["_step1_root"] = str(resolved(config["inputs"]["step1_root"], workspace))
    config["_step2_root"] = str(resolved(config["inputs"]["step2_root"], workspace))
    config["_output_root"] = str(resolved(config["output"]["root"], workspace))
    config["_dataset_root"] = str(resolved(config["inputs"]["dataset_root"], workspace))
    return config


def run_validation_freeze(config: dict[str, Any]) -> dict[str, Any]:
    """Run the complete grouped Validation split and freeze alignment parameters."""
    _validate_run_contract(config)
    dataset_root = Path(config["_dataset_root"])
    output_root = Path(config["_output_root"])
    validate_roots(dataset_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    before = snapshot_tree(dataset_root)
    source_digest = snapshot_digest(before)
    entries, provenance = _load_full_validation_split(config)
    aligner = SIFTAligner(config)
    results: list[AlignmentResult] = []
    total = len(entries)
    for position, entry in enumerate(entries, 1):
        query = load_image_rgb(entry["query_path"])
        template = load_image_rgb(entry["template_path"])
        _, report = aligner.align(query, template, entry["sample_id"])
        results.append(report)
        if position == 1 or position % 25 == 0 or position == total:
            print(f"  Validation {position:3d}/{total}: {entry['sample_id']} -> {report.status}")

    after = snapshot_tree(dataset_root)
    if before != after:
        raise RuntimeError("Read-only dataset changed during Step 3.2-C validation")
    if len(results) != total:
        raise RuntimeError(f"Incomplete Validation run: {len(results)} != {total}")

    diagnostic_records = _render_limited_diagnostics(config, entries, results, aligner)
    parameters = _effective_parameters(config)
    report = _validation_report(
        results,
        config,
        provenance,
        source_digest,
        diagnostic_records,
    )
    report_path = output_root / "alignment_report.json"
    write_json(report_path, report)
    report_sha256 = sha256_file(report_path)

    freeze = _freeze_artifact(config, provenance, source_digest, report_sha256)
    freeze_path = output_root / "alignment_parameters_frozen.json"
    write_json(freeze_path, freeze)
    freeze_sha256 = sha256_file(freeze_path)

    manifest = _run_manifest(
        config,
        provenance,
        source_digest,
        report_sha256,
        freeze_sha256,
        parameters,
    )
    write_json(output_root / "run_manifest.json", manifest)
    return report


def run_alignment_validation(config: dict[str, Any]) -> dict[str, Any]:
    """Compatibility alias for the canonical Step 3.2-C command."""
    return run_validation_freeze(config)


def _validate_run_contract(config: dict[str, Any]) -> None:
    run = config.get("run", {})
    expected = {
        "roadmap_step": config.get("roadmap_step"),
        "purpose": run.get("purpose"),
        "split": run.get("split"),
        "selection": run.get("selection"),
    }
    required = {
        "roadmap_step": "3.2-C",
        "purpose": "validation_freeze",
        "split": "val",
        "selection": "full_split",
    }
    if expected != required:
        raise ValueError(f"Invalid Step 3.2-C run contract: {expected}")
    if int(run.get("expected_sample_count", 0)) <= 0:
        raise ValueError("run.expected_sample_count must be positive")
    if int(run.get("expected_group_count", 0)) <= 0:
        raise ValueError("run.expected_group_count must be positive")


def _load_full_validation_split(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load all and only Validation records, without opening the Test split."""
    _validate_run_contract(config)
    step1_root = Path(config["_step1_root"])
    step2_root = Path(config["_step2_root"])
    dataset_root = Path(config["_dataset_root"])
    inputs = config["inputs"]
    samples_path = step1_root / inputs.get("samples", "samples.jsonl")
    val_path = step1_root / inputs.get("splits_dir", "splits") / "val.txt"
    split_manifest_path = step1_root / inputs.get("split_manifest", "split_manifest.json")
    split_summary_path = step1_root / inputs.get("split_summary", "split_summary.json")
    step2_manifest_path = step2_root / inputs.get("step2_index_manifest", "index_manifest.json")

    records = read_jsonl(samples_path)
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = record["sample_id"]
        if sample_id in by_id:
            raise ValueError(f"Duplicate sample record: {sample_id}")
        by_id[sample_id] = record

    validation_ids = [line.strip() for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_count = int(config["run"]["expected_sample_count"])
    if len(validation_ids) != expected_count:
        raise ValueError(f"Unexpected Validation sample count: {len(validation_ids)} != {expected_count}")
    if len(set(validation_ids)) != len(validation_ids):
        raise ValueError("Validation split contains duplicate sample IDs")
    unknown = [sample_id for sample_id in validation_ids if sample_id not in by_id]
    if unknown:
        raise ValueError(f"Validation split contains unknown IDs: {unknown[:5]}")

    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
    validation_groups = set(split_manifest["groups"]["val"])
    expected_groups = int(config["run"]["expected_group_count"])
    if len(validation_groups) != expected_groups:
        raise ValueError(f"Unexpected Validation group count: {len(validation_groups)} != {expected_groups}")
    if split_summary["val"]["sample_count"] != expected_count or split_summary["val"]["group_count"] != expected_groups:
        raise ValueError("Step 1 Validation summary does not match the run contract")
    wrong_groups = [sample_id for sample_id in validation_ids if by_id[sample_id]["source_id"] not in validation_groups]
    if wrong_groups:
        raise ValueError(f"Validation IDs outside Validation groups: {wrong_groups[:5]}")

    samples_hash = sha256_file(samples_path)
    val_hash = sha256_file(val_path)
    if split_manifest.get("samples_digest") != samples_hash:
        raise RuntimeError("Step 1 samples hash does not match split_manifest.json")
    step2_manifest = json.loads(step2_manifest_path.read_text(encoding="utf-8"))
    if step2_manifest.get("step1_samples_sha256") != samples_hash:
        raise RuntimeError("Step 2 samples hash does not match Step 1 samples.jsonl")
    if step2_manifest.get("split_sha256", {}).get("val") != val_hash:
        raise RuntimeError("Step 2 Validation split hash does not match Step 1 val.txt")

    entries = [{
        "sample_id": sample_id,
        "split": "val",
        "source_id": by_id[sample_id]["source_id"],
        "query_path": _resolve(dataset_root, by_id[sample_id]["query_path"]),
        "template_path": _resolve(dataset_root, by_id[sample_id]["template_path"]),
    } for sample_id in validation_ids]
    provenance = {
        "step1_samples_sha256": samples_hash,
        "step1_val_split_sha256": val_hash,
        "step1_split_manifest_sha256": sha256_file(split_manifest_path),
        "step2_validation_split_sha256": step2_manifest["split_sha256"]["val"],
        "ordered_validation_ids_sha256": sha256_text("\n".join(validation_ids) + "\n"),
        "validation_sample_count": expected_count,
        "validation_group_count": expected_groups,
    }
    return entries, provenance


def _resolve(dataset_root: Path, relative: str) -> Path:
    path = (dataset_root / relative).resolve()
    try:
        path.relative_to(dataset_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Source path escapes dataset root: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Missing source file: {path}")
    return path


def _effective_parameters(config: dict[str, Any]) -> dict[str, Any]:
    return {section: config.get(section, {}) for section in _PARAMETER_SECTIONS}


def _canonical_digest(value: dict[str, Any]) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _finite_values(values: list[Any]) -> list[float]:
    return [float(value) for value in values if value is not None and np.isfinite(value)]


def _distribution(values: list[Any]) -> dict[str, Any]:
    finite = _finite_values(values)
    if not finite:
        return {
            "count": 0,
            "missing_count": len(values),
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "quantiles": {f"p{int(q * 100):02d}": None for q in _QUANTILES},
        }
    array = np.asarray(finite, dtype=np.float64)
    quantiles = np.quantile(array, _QUANTILES, method="linear")
    return {
        "count": len(finite),
        "missing_count": len(values) - len(finite),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "quantiles": {
            f"p{int(q * 100):02d}": float(value)
            for q, value in zip(_QUANTILES, quantiles)
        },
    }


def _stages(results: list[AlignmentResult]) -> dict[str, Any]:
    total = len(results)
    quality_evaluated = sum(result.quality_evaluated for result in results)
    counts = {
        "homography_estimated": sum(result.homography_estimated for result in results),
        "geometry_accepted": sum(result.geometry_accepted for result in results),
        "geometry_rejected": sum(result.homography_estimated and not result.geometry_accepted for result in results),
        "already_aligned": sum(result.already_aligned for result in results),
        "warp_applied": sum(result.warp_applied for result in results),
        "quality_evaluated": quality_evaluated,
        "quality_improved": sum(result.quality_improved is True for result in results),
    }
    output: dict[str, Any] = {}
    for name, count in counts.items():
        denominator = quality_evaluated if name == "quality_improved" else total
        output[f"{name}_count"] = count
        output[f"{name}_rate"] = count / denominator if denominator else None
    return output


def _sample_report(result: AlignmentResult) -> dict[str, Any]:
    fields = {
        "sample_id": result.sample_id,
        "split": "val",
        "success": result.success,
        "homography_estimated": result.homography_estimated,
        "geometry_accepted": result.geometry_accepted,
        "quality_improved": result.quality_improved,
        "already_aligned": result.already_aligned,
        "warp_applied": result.warp_applied,
        "quality_evaluated": result.quality_evaluated,
        "selected_variant": result.selected_variant,
        "status": result.status,
        "reasons": result.reasons,
        "criteria": result.criteria,
    }
    for name in _metric_names():
        fields[name] = getattr(result, name)
    fields.update({
        "homography_matrix": result.homography_matrix,
        "homography_normalized": result.homography_normalized,
        "projected_query_corners": result.projected_query_corners,
        "orientation_preserved": result.orientation_preserved,
        "self_intersects": result.self_intersects,
        "error": result.error or None,
    })
    return fields


def _metric_names() -> tuple[str, ...]:
    return (
        "num_keypoints_query", "num_keypoints_template", "num_raw_matches",
        "num_inlier_matches", "inlier_ratio", "ransac_reprojection_error",
        "max_inlier_reprojection_error", "homography_condition_number",
        "projected_area_ratio", "projected_overlap_ratio", "local_scale",
        "local_scale_x", "local_scale_y", "anisotropy_ratio", "rotation_degrees",
        "perspective_strength", "max_corner_displacement_pixels", "valid_pixel_count",
        "valid_pixel_ratio", "ssim_valid_pixel_count", "ssim_valid_pixel_ratio",
        "ssim_original", "ssim_warped", "ssim_selected", "ssim_candidate_delta",
        "mutual_info_original", "mutual_info_warped", "mutual_info_selected",
        "mutual_info_candidate_delta",
    )


def _distributions(results: list[AlignmentResult]) -> dict[str, Any]:
    distributions = {}
    for name in _metric_names():
        values = [getattr(result, name) for result in results]
        if name == "rotation_degrees":
            values = [abs(value) if value is not None else None for value in values]
        distributions[name] = _distribution(values)
    return distributions


def _validation_report(
    results: list[AlignmentResult],
    config: dict[str, Any],
    provenance: dict[str, Any],
    source_digest: str,
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    statuses = Counter(result.status for result in results)
    reasons = Counter(reason for result in results for reason in result.reasons)
    criterion_failures = Counter(
        name for result in results for name, criterion in result.criteria.items()
        if criterion.get("passed") is False
    )
    status_distributions = {
        status: _distributions([result for result in results if result.status == status])
        for status in sorted(statuses)
    }
    return {
        "schema_version": "step3-alignment-validation-v1",
        "roadmap_step": "3.2-C",
        "run_type": "validation_full_split",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "split": "val",
            "selection": "full_split",
            "expected_sample_count": int(config["run"]["expected_sample_count"]),
            "processed_sample_count": len(results),
            "unique_sample_count": len({result.sample_id for result in results}),
            "group_count": int(config["run"]["expected_group_count"]),
            "complete": len(results) == int(config["run"]["expected_sample_count"]),
            "test_accessed": False,
        },
        "provenance": {**provenance, "source_snapshot_digest": source_digest},
        "effective_parameters": _effective_parameters(config),
        "parameters_sha256": _canonical_digest(_effective_parameters(config)),
        "stages": _stages(results),
        "decisions": {
            "selected_original_count": sum(result.selected_variant == "original" for result in results),
            "selected_warped_count": sum(result.selected_variant == "warped" for result in results),
            "status_counts": dict(sorted(statuses.items())),
        },
        "reason_counts": dict(sorted(reasons.items())),
        "criterion_failure_counts": dict(sorted(criterion_failures.items())),
        "distributions": _distributions(results),
        "distributions_by_status": status_distributions,
        "diagnostics": diagnostics,
        "samples": [_sample_report(result) for result in results],
        "failures": [{
            "sample_id": result.sample_id,
            "status": result.status,
            "error": result.error or None,
            "reasons": result.reasons,
        } for result in results if not result.geometry_accepted],
    }


def _diagnostic_priority(result: AlignmentResult) -> tuple[int, float, str]:
    priority = {
        "homography_not_estimated": 0,
        "geometry_rejected": 1,
        "original_retained": 2,
        "warped_selected": 3,
        "already_aligned": 4,
    }.get(result.status, 5)
    risk = result.max_inlier_reprojection_error or 0.0
    return priority, -risk, result.sample_id


def _render_limited_diagnostics(
    config: dict[str, Any],
    entries: list[dict[str, Any]],
    results: list[AlignmentResult],
    aligner: SIFTAligner,
) -> list[dict[str, Any]]:
    limit = max(0, int(config.get("diagnostics", {}).get("max_samples", 0)))
    selected = sorted(results, key=_diagnostic_priority)[:limit]
    by_id = {entry["sample_id"]: entry for entry in entries}
    output_root = Path(config["_output_root"])
    records = []
    for result in selected:
        entry = by_id[result.sample_id]
        query = load_image_rgb(entry["query_path"])
        template = load_image_rgb(entry["template_path"])
        selected_image, repeated = aligner.align(query, template, result.sample_id)
        if repeated.status != result.status or repeated.homography_matrix != result.homography_matrix:
            raise RuntimeError(f"Non-deterministic diagnostic rerun: {result.sample_id}")
        relative_dir = Path("alignment") / "val"
        paths = []
        if config.get("diagnostics", {}).get("render_comparison", True):
            path = output_root / relative_dir / f"{result.sample_id}_comparison.png"
            render_comparison(query, selected_image, template, repeated, path)
            paths.append(path.relative_to(output_root).as_posix())
        if config.get("diagnostics", {}).get("render_matches", True):
            path = output_root / relative_dir / f"{result.sample_id}_matches.png"
            aligner.render_matches(query, template, repeated, path)
            paths.append(path.relative_to(output_root).as_posix())
        records.append({
            "sample_id": result.sample_id,
            "status": result.status,
            "rank_metric": "max_inlier_reprojection_error",
            "rank_value": result.max_inlier_reprojection_error,
            "paths": paths,
        })
    return records


def _freeze_artifact(
    config: dict[str, Any],
    provenance: dict[str, Any],
    source_digest: str,
    report_sha256: str,
) -> dict[str, Any]:
    parameters = _effective_parameters(config)
    return {
        "schema_version": "step3-alignment-freeze-v1",
        "roadmap_step": "3.2-C",
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "derived_from": {
            "split": "val",
            "sample_count": provenance["validation_sample_count"],
            "group_count": provenance["validation_group_count"],
            "validation_report_sha256": report_sha256,
            "step1_samples_sha256": provenance["step1_samples_sha256"],
            "step1_val_split_sha256": provenance["step1_val_split_sha256"],
            "source_snapshot_digest": source_digest,
        },
        "parameters": parameters,
        "parameters_sha256": _canonical_digest(parameters),
        "selection": {
            "workflow": "validation_only",
            "test_used_for_tuning": False,
            "rationale": [
                {
                    "parameter": "ransac.max_inlier_reprojection_error",
                    "previous_value": 5.5,
                    "frozen_value": parameters["ransac"]["max_inlier_reprojection_error"],
                    "validation_evidence": (
                        "Three Validation samples exceeded 5.5 px only because of isolated "
                        "RANSAC inliers (6.66-8.40 px), while their mean errors were "
                        "0.049-0.087 px, inlier ratios were 0.953-0.977, condition numbers "
                        "were 1.0001-1.0014, and projected overlap was at least 0.99927."
                    ),
                    "safety_intent": (
                        "Accept globally stable near-identity homographies while retaining "
                        "a finite maximum-error guard above the observed Validation maximum."
                    ),
                }
            ],
        },
        "intended_next_use": "Step 3.2-D Test alignment evaluation",
    }


def _run_manifest(
    config: dict[str, Any],
    provenance: dict[str, Any],
    source_digest: str,
    report_sha256: str,
    freeze_sha256: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "step3-run-manifest-v1",
        "roadmap_step": "3.2-C",
        "command": "validate-freeze",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pyyaml": yaml.__version__,
            "platform": platform.platform(),
        },
        "scope": {
            "split": "val",
            "sample_count": provenance["validation_sample_count"],
            "group_count": provenance["validation_group_count"],
            "complete": True,
            "test_accessed": False,
        },
        "inputs": {
            **provenance,
            "source_snapshot_digest": source_digest,
            "config_sha256": sha256_file(Path(config["_config_path"])),
            "parameters_sha256": _canonical_digest(parameters),
        },
        "artifacts": {
            "alignment_report.json": report_sha256,
            "alignment_parameters_frozen.json": freeze_sha256,
        },
        "source_read_only_verified": True,
    }
