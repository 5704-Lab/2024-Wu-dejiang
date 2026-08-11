"""Real-data orchestration and schema-v4 reporting for Step 3 alignment."""

from __future__ import annotations

import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from step1_audit.paths import read_jsonl, resolved, validate_roots, write_json

from .alignment import AlignmentResult, SIFTAligner, load_image_rgb, render_comparison, render_diff


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    workspace = config_path.resolve().parent.parent
    config["_workspace"] = str(workspace)
    config["_step1_root"] = str(resolved(config["inputs"]["step1_root"], workspace))
    config["_output_root"] = str(resolved(config["output"]["root"], workspace))
    config["_dataset_root"] = str(resolved(config["inputs"]["dataset_root"]))
    return config


def run_alignment_validation(config: dict[str, Any]) -> dict[str, Any]:
    """Align configured samples, select the safer variant, and write a v4 report."""
    dataset_root = Path(config["_dataset_root"])
    output_root = Path(config["_output_root"])
    validate_roots(dataset_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    samples = _load_validation_samples(config)
    aligner = SIFTAligner(config)
    results: list[AlignmentResult] = []

    for entry in samples:
        sample_id, split = entry["sample_id"], entry["split"]
        query = load_image_rgb(entry["query_path"])
        template = load_image_rgb(entry["template_path"])
        selected, report = aligner.align(query, template, sample_id)
        results.append(report)
        out_dir = output_root / "alignment" / split
        render_comparison(query, selected, template, report, out_dir / f"{sample_id}_comparison.png")
        render_diff(query, template, "Diff: original Query", out_dir / f"{sample_id}_diff_before.png",
                    report.valid_mask)
        render_diff(selected, template, f"Diff: selected {report.selected_variant}",
                    out_dir / f"{sample_id}_diff_after.png", report.valid_mask)
        aligner.render_matches(query, template, report, out_dir / f"{sample_id}_matches.png")
        print(
            f"  {sample_id:20s}  {report.status:24s}  "
            f"inliers={report.num_inlier_matches:4d} ({(report.inlier_ratio or 0):.1%})  "
            f"selected={report.selected_variant}"
        )

    summary = _summarize(results, defaultdict(list), samples, config)
    write_json(output_root / "alignment_report.json", summary)
    return summary


def _load_validation_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    step1_root, dataset_root = Path(config["_step1_root"]), Path(config["_dataset_root"])
    records = read_jsonl(step1_root / "samples.jsonl")
    by_id = {record["sample_id"]: record for record in records}
    requested = int(config["validation"]["sample_count"])
    if requested < 1:
        raise ValueError("validation.sample_count must be at least 1")
    split_ids = {
        split: [line.strip() for line in (step1_root / "splits" / f"{split}.txt").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        for split in ("train", "val", "test")
    }
    count = min(requested, sum(len(ids) for ids in split_ids.values()))
    quotas = _allocate_split_counts(split_ids, count)
    rng = np.random.default_rng(int(config["validation"].get("seed", 0)))
    selected = []
    for split, ids in split_ids.items():
        chosen = rng.choice(ids, size=quotas[split], replace=False).tolist() if quotas[split] else []
        for sample_id in chosen:
            record = by_id[sample_id]
            selected.append({
                "sample_id": sample_id,
                "split": split,
                "query_path": _resolve(dataset_root, record["query_path"]),
                "template_path": _resolve(dataset_root, record["template_path"]),
            })
    return selected


def _allocate_split_counts(split_ids: dict[str, list[str]], count: int) -> dict[str, int]:
    nonempty = [split for split, ids in split_ids.items() if ids]
    quotas = {split: 0 for split in split_ids}
    if count == 0 or not nonempty:
        return quotas
    if count >= len(nonempty):
        for split in nonempty:
            quotas[split] = 1
        remaining = count - len(nonempty)
    else:
        remaining = count
    total = sum(len(split_ids[split]) for split in nonempty)
    targets = {split: count * len(split_ids[split]) / total for split in nonempty}
    while remaining:
        candidates = [split for split in nonempty if quotas[split] < len(split_ids[split])]
        if not candidates:
            break
        split = max(candidates, key=lambda name: (targets[name] - quotas[name], len(split_ids[name])))
        quotas[split] += 1
        remaining -= 1
    return quotas


def _resolve(dataset_root: Path, relative: str) -> Path:
    path = (dataset_root / relative).resolve()
    try:
        path.relative_to(dataset_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Source path escapes dataset root: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Missing source file: {path}")
    return path


def _mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _stages(results: list[AlignmentResult]) -> dict[str, Any]:
    total = len(results)
    counts = {
        "homography_estimated": sum(result.homography_estimated for result in results),
        "geometry_accepted": sum(result.geometry_accepted for result in results),
        "geometry_rejected": sum(result.homography_estimated and not result.geometry_accepted for result in results),
        "already_aligned": sum(result.already_aligned for result in results),
        "warp_applied": sum(result.warp_applied for result in results),
        "quality_evaluated": sum(result.quality_evaluated for result in results),
        "quality_improved": sum(result.quality_improved is True for result in results),
    }
    output: dict[str, Any] = {}
    for name, count in counts.items():
        denominator = counts["quality_evaluated"] if name == "quality_improved" else total
        output[f"{name}_count"] = count
        output[f"{name}_rate"] = _rate(count, denominator)
    return output


def _diagnostics(results: list[AlignmentResult]) -> dict[str, Any]:
    metrics = {
        "inlier_ratio": [r.inlier_ratio for r in results],
        "mean_inlier_reprojection_error": [r.ransac_reprojection_error for r in results],
        "max_inlier_reprojection_error": [r.max_inlier_reprojection_error for r in results],
        "condition_number": [r.homography_condition_number for r in results],
        "projected_area_ratio": [r.projected_area_ratio for r in results],
        "projected_overlap_ratio": [r.projected_overlap_ratio for r in results],
        "local_scale": [r.local_scale for r in results],
        "abs_rotation_degrees": [abs(r.rotation_degrees) if r.rotation_degrees is not None else None for r in results],
        "perspective_strength": [r.perspective_strength for r in results],
        "max_corner_displacement_pixels": [r.max_corner_displacement_pixels for r in results],
        "valid_pixel_ratio": [r.valid_pixel_ratio if r.warp_applied else None for r in results],
    }
    output = {f"{name}_mean": _mean(values) for name, values in metrics.items()}
    output["mean_inliers"] = _mean([float(result.num_inlier_matches) for result in results])
    output["projected_overlap_ratio_min"] = min(
        (value for value in metrics["projected_overlap_ratio"] if value is not None), default=None
    )
    evaluated = [result for result in results if result.quality_evaluated]
    selected = [result for result in results if result.ssim_selected is not None]
    output.update({
        "metrics_valid_count": sum(result.metrics_valid for result in results),
        "ssim_original_mean": _mean([r.ssim_original for r in selected]),
        "ssim_warped_mean": _mean([r.ssim_warped for r in evaluated]),
        "ssim_candidate_delta_mean": _mean([r.ssim_candidate_delta for r in evaluated]),
        "ssim_selected_mean": _mean([r.ssim_selected for r in selected]),
        "mi_original_mean": _mean([r.mutual_info_original for r in selected]),
        "mi_warped_mean": _mean([r.mutual_info_warped for r in evaluated]),
        "mi_candidate_delta_mean": _mean([r.mutual_info_candidate_delta for r in evaluated]),
        "mi_selected_mean": _mean([r.mutual_info_selected for r in selected]),
    })
    return output


def _sample_report(result: AlignmentResult, split: str | None) -> dict[str, Any]:
    return {
        "sample_id": result.sample_id,
        "split": split,
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
        "num_keypoints_query": result.num_keypoints_query,
        "num_keypoints_template": result.num_keypoints_template,
        "num_ratio_matches": result.num_raw_matches,
        "num_inlier_matches": result.num_inlier_matches,
        "inlier_ratio": result.inlier_ratio,
        "ransac_reprojection_error": result.ransac_reprojection_error,
        "max_inlier_reprojection_error": result.max_inlier_reprojection_error,
        "homography_matrix": result.homography_matrix,
        "homography_normalized": result.homography_normalized,
        "homography_condition_number": result.homography_condition_number,
        "projected_query_corners": result.projected_query_corners,
        "projected_area_ratio": result.projected_area_ratio,
        "orientation_preserved": result.orientation_preserved,
        "self_intersects": result.self_intersects,
        "projected_overlap_ratio": result.projected_overlap_ratio,
        "local_scale": result.local_scale,
        "local_scale_x": result.local_scale_x,
        "local_scale_y": result.local_scale_y,
        "anisotropy_ratio": result.anisotropy_ratio,
        "rotation_degrees": result.rotation_degrees,
        "perspective_strength": result.perspective_strength,
        "max_corner_displacement_pixels": result.max_corner_displacement_pixels,
        "valid_pixel_count": result.valid_pixel_count,
        "valid_pixel_ratio": result.valid_pixel_ratio,
        "ssim_valid_pixel_count": result.ssim_valid_pixel_count,
        "ssim_valid_pixel_ratio": result.ssim_valid_pixel_ratio,
        "metrics_valid": result.metrics_valid,
        "ssim_original": result.ssim_original,
        "ssim_warped": result.ssim_warped,
        "ssim_selected": result.ssim_selected,
        "ssim_candidate_delta": result.ssim_candidate_delta,
        "mutual_info_original": result.mutual_info_original,
        "mutual_info_warped": result.mutual_info_warped,
        "mutual_info_selected": result.mutual_info_selected,
        "mutual_info_candidate_delta": result.mutual_info_candidate_delta,
        "error": result.error or None,
    }


def _threshold_snapshot(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    return {name: config.get(name, {}) for name in ("ransac", "geometry", "near_identity", "quality")}


def _summarize(results: list[AlignmentResult], metrics_by_split: dict[str, list[dict[str, float]]],
               samples: list[dict[str, Any]] | None = None,
               config: dict[str, Any] | None = None) -> dict[str, Any]:
    del metrics_by_split
    split_by_id = {sample["sample_id"]: sample["split"] for sample in (samples or [])}
    grouped: dict[str, list[AlignmentResult]] = defaultdict(list)
    for result in results:
        split = split_by_id.get(result.sample_id)
        if split:
            grouped[split].append(result)
    reason_counts = Counter(reason for result in results for reason in result.reasons)
    criterion_failures = Counter(
        name for result in results for name, criterion in result.criteria.items()
        if criterion.get("passed") is False
    )
    statuses = Counter(result.status for result in results)
    failures = [result for result in results if not result.geometry_accepted]
    return {
        "schema_version": "step3-alignment-v4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "configuration": _threshold_snapshot(config),
        "total_samples": len(results),
        "success_count": sum(result.success for result in results),
        "failure_count": len(failures),
        "success_rate": _rate(sum(result.success for result in results), len(results)),
        "stages": _stages(results),
        "decisions": {
            "selected_original_count": sum(result.selected_variant == "original" for result in results),
            "selected_warped_count": sum(result.selected_variant == "warped" for result in results),
            "status_counts": dict(sorted(statuses.items())),
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "criterion_failure_counts": dict(sorted(criterion_failures.items())),
        "by_split": {
            split: {"count": len(entries), "stages": _stages(entries), "overall": _diagnostics(entries)}
            for split, entries in grouped.items()
        },
        "overall": _diagnostics(results),
        "samples": [_sample_report(result, split_by_id.get(result.sample_id)) for result in results],
        "failures": [{
            "sample_id": result.sample_id,
            "status": result.status,
            "error": result.error,
            "reasons": result.reasons,
        } for result in failures],
    }
