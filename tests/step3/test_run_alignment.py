import json
from collections import defaultdict
from pathlib import Path

import pytest

from step1_audit.paths import validate_roots
from step3_model.alignment import AlignmentResult
from step3_model.run_alignment import _allocate_split_counts, _resolve, _summarize


def test_allocate_split_counts_uses_total_requested_count():
    split_ids = {
        "train": [f"train-{index}" for index in range(80)],
        "val": [f"val-{index}" for index in range(10)],
        "test": [f"test-{index}" for index in range(10)],
    }

    quotas = _allocate_split_counts(split_ids, 10)

    assert quotas == {"train": 8, "val": 1, "test": 1}
    assert sum(quotas.values()) == 10


def test_summary_with_no_results_is_strict_json():
    summary = _summarize([], defaultdict(list), [])

    assert summary["total_samples"] == 0
    assert summary["overall"]["mean_reprojection_error"] is None
    encoded = json.dumps(summary, allow_nan=False)
    assert "NaN" not in encoded


def test_summary_contains_auditable_sample_records():
    result = AlignmentResult(
        sample_id="sample-1",
        success=True,
        num_keypoints_query=20,
        num_keypoints_template=22,
        num_raw_matches=12,
        num_inlier_matches=10,
        ransac_reprojection_error=0.4,
        valid_pixel_count=900,
        valid_pixel_ratio=0.9,
        ssim_valid_pixel_count=800,
        ssim_valid_pixel_ratio=0.8,
        metrics_valid=True,
        ssim_before=0.8,
        ssim_after=0.85,
        mutual_info_before=1.2,
        mutual_info_after=1.3,
    )
    metrics = defaultdict(list)
    metrics["val"].append({
        "ssim_before": 0.8,
        "ssim_after": 0.85,
        "mutual_info_before": 1.2,
        "mutual_info_after": 1.3,
        "num_inliers": 10,
        "reprojection_error": 0.4,
        "valid_pixel_ratio": 0.9,
        "ssim_valid_pixel_ratio": 0.8,
        "metrics_valid": True,
    })

    summary = _summarize(
        [result],
        metrics,
        [{"sample_id": "sample-1", "split": "val"}],
    )

    assert summary["schema_version"] == "step3-alignment-v3"
    assert len(summary["samples"]) == summary["total_samples"]
    assert summary["samples"][0]["split"] == "val"
    assert summary["samples"][0]["valid_pixel_ratio"] == pytest.approx(0.9)
    assert summary["samples"][0]["metrics_valid"] is True
    assert summary["overall"]["metrics_valid_count"] == 1
    assert summary["overall"]["valid_pixel_ratio_mean"] == pytest.approx(0.9)
    assert summary["samples"][0]["ssim_delta"] == pytest.approx(0.05)


def test_output_root_cannot_be_inside_dataset(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    with pytest.raises(ValueError, match="Output root must not be inside"):
        validate_roots(dataset_root, dataset_root / "artifacts")


def test_resolve_rejects_source_outside_dataset(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"not-an-image")

    with pytest.raises(ValueError, match="escapes dataset root"):
        _resolve(dataset_root, "../outside.png")
