"""Orchestration for SIFT+RANSAC alignment validation on real paired images."""

from __future__ import annotations

import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from step1_audit.paths import read_jsonl, resolved, validate_roots, write_json

from .alignment import (
    AlignmentResult,
    SIFTAligner,
    load_image_rgb,
    render_comparison,
    render_diff,
)


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
    """Align configured samples and compute before/after similarity metrics."""
    dataset_root = Path(config["_dataset_root"])
    output_root = Path(config["_output_root"])
    validate_roots(dataset_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    samples = _load_validation_samples(config)
    aligner = SIFTAligner(config)

    results: list[AlignmentResult] = []
    metrics_by_split: dict[str, list[dict[str, float]]] = defaultdict(list)

    for entry in samples:
        sample_id = entry["sample_id"]
        split = entry["split"]
        query = load_image_rgb(entry["query_path"])
        template = load_image_rgb(entry["template_path"])

        warped, report = aligner.align(query, template, sample_id)
        results.append(report)

        metrics_by_split[split].append({
            "ssim_before": report.ssim_before,
            "ssim_after": report.ssim_after,
            "mutual_info_before": report.mutual_info_before,
            "mutual_info_after": report.mutual_info_after,
            "num_inliers": report.num_inlier_matches,
            "reprojection_error": report.ransac_reprojection_error,
            "valid_pixel_ratio": report.valid_pixel_ratio,
            "ssim_valid_pixel_ratio": report.ssim_valid_pixel_ratio,
            "metrics_valid": report.metrics_valid,
        })

        out_dir = output_root / "alignment" / split
        render_comparison(
            query,
            warped,
            template,
            report,
            out_dir / f"{sample_id}_comparison.png",
        )
        render_diff(
            query,
            template,
            "Masked diff BEFORE alignment",
            out_dir / f"{sample_id}_diff_before.png",
            report.valid_mask,
        )
        render_diff(
            warped,
            template,
            "Masked diff AFTER alignment",
            out_dir / f"{sample_id}_diff_after.png",
            report.valid_mask,
        )
        aligner.render_matches(
            query,
            template,
            report,
            out_dir / f"{sample_id}_matches.png",
        )

        print(
            f"  {sample_id:20s}  inliers={report.num_inlier_matches:4d}  "
            f"SSIM {report.ssim_before:.3f}->{report.ssim_after:.3f}  "
            f"MI {report.mutual_info_before:.3f}->{report.mutual_info_after:.3f}  "
            f"{'OK' if report.success else 'FAIL'}"
        )

    summary = _summarize(results, metrics_by_split, samples)
    write_json(output_root / "alignment_report.json", summary)
    return summary


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _load_validation_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Select exactly sample_count records across splits using a fixed seed."""
    step1_root = Path(config["_step1_root"])
    dataset_root = Path(config["_dataset_root"])
    records = read_jsonl(step1_root / "samples.jsonl")
    by_id = {record["sample_id"]: record for record in records}

    requested = int(config["validation"]["sample_count"])
    if requested < 1:
        raise ValueError("validation.sample_count must be at least 1")

    split_ids: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        split_path = step1_root / "splits" / f"{split}.txt"
        split_ids[split] = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    total_available = sum(len(ids) for ids in split_ids.values())
    count = min(requested, total_available)
    quotas = _allocate_split_counts(split_ids, count)
    rng = np.random.default_rng(int(config["validation"].get("seed", 0)))

    selected: list[dict[str, Any]] = []
    for split, ids in split_ids.items():
        quota = quotas[split]
        chosen = rng.choice(ids, size=quota, replace=False).tolist() if quota else []
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
    """Allocate a total count proportionally while representing each split when possible."""
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

    total_available = sum(len(split_ids[split]) for split in nonempty)
    targets = {
        split: count * len(split_ids[split]) / total_available for split in nonempty
    }
    while remaining:
        candidates = [
            split for split in nonempty if quotas[split] < len(split_ids[split])
        ]
        if not candidates:
            break
        split = max(
            candidates,
            key=lambda name: (targets[name] - quotas[name], len(split_ids[name])),
        )
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


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _summarize(
    results: list[AlignmentResult],
    metrics_by_split: dict[str, list[dict[str, float]]],
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    successes = [result for result in results if result.success]
    failures = [result for result in results if not result.success]

    per_split: dict[str, Any] = {}
    for split, entries in metrics_by_split.items():
        if not entries:
            continue
        valid_entries = [entry for entry in entries if entry["metrics_valid"]]
        per_split[split] = {
            "count": len(entries),
            "metrics_valid_count": len(valid_entries),
            "ssim_before_mean": _mean([entry["ssim_before"] for entry in valid_entries]),
            "ssim_after_mean": _mean([entry["ssim_after"] for entry in valid_entries]),
            "ssim_delta_mean": _mean([
                entry["ssim_after"] - entry["ssim_before"] for entry in valid_entries
            ]),
            "mi_before_mean": _mean([entry["mutual_info_before"] for entry in valid_entries]),
            "mi_after_mean": _mean([entry["mutual_info_after"] for entry in valid_entries]),
            "mi_delta_mean": _mean([
                entry["mutual_info_after"] - entry["mutual_info_before"] for entry in valid_entries
            ]),
            "valid_pixel_ratio_mean": _mean([
                entry["valid_pixel_ratio"] for entry in entries
            ]),
            "valid_pixel_ratio_min": min(
                (entry["valid_pixel_ratio"] for entry in entries),
                default=None,
            ),
            "ssim_valid_pixel_ratio_mean": _mean([
                entry["ssim_valid_pixel_ratio"] for entry in entries
            ]),
            "mean_inliers": _mean([entry["num_inliers"] for entry in entries]),
            "mean_reprojection_error": _mean([
                entry["reprojection_error"] for entry in entries
            ]),
        }

    split_by_id = {
        sample["sample_id"]: sample["split"] for sample in (samples or [])
    }
    sample_reports = [
        {
            "sample_id": result.sample_id,
            "split": split_by_id.get(result.sample_id),
            "success": result.success,
            "num_keypoints_query": result.num_keypoints_query,
            "num_keypoints_template": result.num_keypoints_template,
            "num_ratio_matches": result.num_raw_matches,
            "num_inlier_matches": result.num_inlier_matches,
            "homography_matrix": result.homography_matrix,
            "ransac_reprojection_error": result.ransac_reprojection_error,
            "valid_pixel_count": result.valid_pixel_count,
            "valid_pixel_ratio": result.valid_pixel_ratio,
            "ssim_valid_pixel_count": result.ssim_valid_pixel_count,
            "ssim_valid_pixel_ratio": result.ssim_valid_pixel_ratio,
            "metrics_valid": result.metrics_valid,
            "ssim_before": result.ssim_before,
            "ssim_after": result.ssim_after,
            "ssim_delta": result.ssim_after - result.ssim_before,
            "mutual_info_before": result.mutual_info_before,
            "mutual_info_after": result.mutual_info_after,
            "mutual_info_delta": result.mutual_info_after - result.mutual_info_before,
            "error": result.error or None,
        }
        for result in results
    ]

    metric_results = [result for result in results if result.metrics_valid]
    return {
        "schema_version": "step3-alignment-v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "total_samples": len(results),
        "success_count": len(successes),
        "failure_count": len(failures),
        "success_rate": len(successes) / max(len(results), 1),
        "by_split": per_split,
        "overall": {
            "metrics_valid_count": len(metric_results),
            "ssim_before_mean": _mean([result.ssim_before for result in metric_results]),
            "ssim_after_mean": _mean([result.ssim_after for result in metric_results]),
            "ssim_delta_mean": _mean([
                result.ssim_after - result.ssim_before for result in metric_results
            ]),
            "mi_before_mean": _mean([result.mutual_info_before for result in metric_results]),
            "mi_after_mean": _mean([result.mutual_info_after for result in metric_results]),
            "mi_delta_mean": _mean([
                result.mutual_info_after - result.mutual_info_before for result in metric_results
            ]),
            "valid_pixel_ratio_mean": _mean([
                result.valid_pixel_ratio for result in results
            ]),
            "valid_pixel_ratio_min": min(
                (result.valid_pixel_ratio for result in results),
                default=None,
            ),
            "ssim_valid_pixel_ratio_mean": _mean([
                result.ssim_valid_pixel_ratio for result in results
            ]),
            "mean_inliers": _mean([result.num_inlier_matches for result in results]),
            "mean_reprojection_error": _mean([
                result.ransac_reprojection_error for result in successes
            ]),
        },
        "samples": sample_reports,
        "failures": [
            {
                "sample_id": result.sample_id,
                "error": result.error,
                "num_keypoints_query": result.num_keypoints_query,
                "num_keypoints_template": result.num_keypoints_template,
            }
            for result in failures
        ],
    }
