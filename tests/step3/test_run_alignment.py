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
    assert summary["schema_version"] == "step3-alignment-v4"
    assert summary["total_samples"] == 0
    assert summary["overall"]["mean_inlier_reprojection_error_mean"] is None
    assert summary["stages"]["quality_improved_rate"] is None
    encoded = json.dumps(summary, allow_nan=False)
    assert "NaN" not in encoded


def test_summary_contains_auditable_sample_records():
    result = AlignmentResult(
        sample_id="sample-1",
        success=True,
        homography_estimated=True,
        geometry_accepted=True,
        quality_improved=True,
        warp_applied=True,
        quality_evaluated=True,
        selected_variant="warped",
        status="warped_selected",
        reasons=["warped_candidate_selected"],
        criteria={"minimum_inlier_ratio": {
            "applicable": True, "passed": True, "value": 10 / 12,
            "operator": ">=", "threshold": 0.5,
        }},
        num_keypoints_query=20,
        num_keypoints_template=22,
        num_raw_matches=12,
        num_inlier_matches=10,
        inlier_ratio=10 / 12,
        ransac_reprojection_error=0.4,
        max_inlier_reprojection_error=0.8,
        projected_overlap_ratio=0.9,
        valid_pixel_count=900,
        valid_pixel_ratio=0.9,
        ssim_valid_pixel_count=800,
        ssim_valid_pixel_ratio=0.8,
        metrics_valid=True,
        ssim_original=0.8,
        ssim_warped=0.85,
        ssim_selected=0.85,
        mutual_info_original=1.2,
        mutual_info_warped=1.3,
        mutual_info_selected=1.3,
    )
    summary = _summarize(
        [result], defaultdict(list),
        [{"sample_id": "sample-1", "split": "val"}],
    )
    sample = summary["samples"][0]
    assert sample["split"] == "val"
    assert sample["geometry_accepted"] is True
    assert sample["quality_improved"] is True
    assert sample["ssim_candidate_delta"] == pytest.approx(0.05)
    assert summary["stages"]["quality_improved_rate"] == pytest.approx(1.0)
    assert summary["decisions"]["selected_warped_count"] == 1
    assert summary["by_split"]["val"]["stages"]["geometry_accepted_count"] == 1


def test_already_aligned_is_not_a_failure():
    result = AlignmentResult(
        sample_id="aligned",
        success=True,
        homography_estimated=True,
        geometry_accepted=True,
        already_aligned=True,
        status="already_aligned",
        reasons=["near_identity_transform"],
        ssim_original=1.0,
        ssim_selected=1.0,
        mutual_info_original=2.0,
        mutual_info_selected=2.0,
    )
    summary = _summarize([result], defaultdict(list), [])
    assert summary["failure_count"] == 0
    assert summary["stages"]["already_aligned_count"] == 1
    assert summary["stages"]["quality_evaluated_count"] == 0
    assert summary["stages"]["quality_improved_rate"] is None


def test_rejection_reasons_and_criteria_are_counted():
    result = AlignmentResult(
        sample_id="bad",
        homography_estimated=True,
        status="geometry_rejected",
        reasons=["minimum_inlier_ratio", "overlap_above_minimum"],
        criteria={
            "minimum_inlier_ratio": {"passed": False},
            "overlap_above_minimum": {"passed": False},
        },
    )
    summary = _summarize([result], defaultdict(list), [])
    assert summary["failure_count"] == 1
    assert summary["reason_counts"]["minimum_inlier_ratio"] == 1
    assert summary["criterion_failure_counts"]["overlap_above_minimum"] == 1


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
