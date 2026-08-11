from pathlib import Path

import cv2
import numpy as np
import pytest

from step3_model.alignment import (
    AlignmentResult,
    SIFTAligner,
    _decide_quality,
    _erode_valid_mask,
    _is_near_identity,
    _mutual_info,
    _normalize_homography,
    _project_corners,
    _projected_overlap_ratio,
    _quadrilateral_self_intersects,
    _signed_polygon_area,
    _ssim,
    _to_gray,
    _transform_diagnostics,
    _warp_valid_mask,
    render_diff,
)


@pytest.fixture
def white_800() -> np.ndarray:
    return np.full((800, 800, 3), 255, dtype=np.uint8)


@pytest.fixture
def black_800() -> np.ndarray:
    return np.zeros((800, 800, 3), dtype=np.uint8)


def synthetic_feature_image(size: int = 320) -> np.ndarray:
    image = np.full((size, size, 3), 245, dtype=np.uint8)
    rng = np.random.default_rng(20260810)
    for index in range(50):
        center = tuple(int(value) for value in rng.integers(15, size - 15, size=2))
        color = tuple(int(value) for value in rng.integers(0, 210, size=3))
        radius = int(rng.integers(3, 10))
        cv2.circle(image, center, radius, color, -1)
        cv2.putText(
            image,
            str(index % 10),
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return image


class TestHelpers:
    def test_to_gray_rgb(self, white_800):
        gray = _to_gray(white_800)
        assert gray.ndim == 2
        assert gray.shape == (800, 800)

    def test_to_gray_already_gray(self, white_800):
        gray = _to_gray(cv2.cvtColor(white_800, cv2.COLOR_RGB2GRAY))
        assert gray.ndim == 2

    def test_ssim_identical(self, white_800):
        assert _ssim(white_800, white_800) >= 0.99

    def test_ssim_opposite(self, white_800, black_800):
        assert _ssim(white_800, black_800) < 0.5

    def test_mutual_info_is_non_negative(self, white_800, black_800):
        assert _mutual_info(white_800, white_800) >= 0.0
        assert _mutual_info(white_800, black_800) >= 0.0

    def test_masked_ssim_ignores_invalid_region(self):
        left = np.zeros((64, 64, 3), dtype=np.uint8)
        right = left.copy()
        right[:, 32:] = 255
        mask = np.zeros((64, 64), dtype=bool)
        mask[:, :24] = True

        assert _ssim(left, right) < 0.8
        assert _ssim(left, right, mask=mask) == pytest.approx(1.0)

    def test_masked_mutual_info_ignores_invalid_region(self):
        ramp = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
        left = np.repeat(ramp[..., None], 3, axis=2)
        right_a = left.copy()
        right_b = left.copy()
        right_b[:, 32:] = np.flip(right_b[:, 32:], axis=1)
        mask = np.zeros((64, 64), dtype=bool)
        mask[:, :32] = True

        assert _mutual_info(left, right_a, mask=mask) == pytest.approx(
            _mutual_info(left, right_b, mask=mask)
        )

    def test_empty_mask_metrics_are_finite(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=bool)
        assert _ssim(image, image, mask=mask) == 0.0
        assert _mutual_info(image, image, mask=mask) == 0.0

    def test_wrong_mask_shape_is_rejected(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Mask shape"):
            _ssim(image, image, mask=np.ones((16, 16), dtype=bool))

    def test_warp_mask_and_ssim_erosion(self):
        homography = np.array(
            [[1.0, 0.0, -10.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]]
        )
        mask = _warp_valid_mask((100, 100), homography, (100, 100))
        eroded = _erode_valid_mask(mask, 11)

        assert mask.shape == (100, 100)
        assert not mask[-1, -1]
        assert mask.sum() == 90 * 95
        assert eroded.sum() < mask.sum()

    def test_homography_normalization_and_invalid_values(self):
        matrix = np.array([[2.0, 0, 4], [0, 2, 6], [0, 0, 2]])
        assert np.allclose(_normalize_homography(matrix), matrix / 2)
        assert _normalize_homography(matrix * 3) is not None
        invalid = matrix.copy()
        invalid[0, 0] = np.nan
        assert _normalize_homography(invalid) is None
        invalid[0, 0] = np.inf
        assert _normalize_homography(invalid) is None
        matrix[2, 2] = 0
        assert _normalize_homography(matrix) is None

    def test_projected_quad_area_orientation_self_intersection_and_overlap(self):
        identity = np.eye(3)
        corners = _project_corners(identity, (100, 100))
        assert corners is not None
        assert _signed_polygon_area(corners) > 0
        assert not _quadrilateral_self_intersects(corners)
        assert _projected_overlap_ratio(corners, (100, 100)) == pytest.approx(1.0)
        bow_tie = np.array([[0, 0], [99, 99], [99, 0], [0, 99]], dtype=float)
        assert _quadrilateral_self_intersects(bow_tie)

    def test_transform_diagnostics_for_known_scale_and_rotation(self):
        angle = np.deg2rad(10)
        scale = 1.2
        matrix = np.array([
            [scale * np.cos(angle), -scale * np.sin(angle), 0],
            [scale * np.sin(angle), scale * np.cos(angle), 0],
            [0, 0, 1],
        ])
        diagnostics = _transform_diagnostics(matrix, (100, 100), (100, 100))
        assert diagnostics["local_scale"] == pytest.approx(scale)
        assert diagnostics["rotation_degrees"] == pytest.approx(10)
        assert diagnostics["perspective_strength"] == pytest.approx(0)

    def test_quality_policy_requires_gain_without_regression(self):
        improved, reasons = _decide_quality(0.002, 0.0, True, 0.001, 0.01, 0.0005, 0.01)
        assert improved and reasons == ["warped_candidate_selected"]
        improved, reasons = _decide_quality(0.002, -0.02, True, 0.001, 0.01, 0.0005, 0.01)
        assert not improved and "mutual_info_regression_exceeded" in reasons
        improved, reasons = _decide_quality(0.0001, 0.001, True, 0.001, 0.01, 0.0005, 0.01)
        assert not improved and "quality_gain_below_threshold" in reasons
        assert _decide_quality(None, None, False, 0.001, 0.01, 0.0005, 0.01)[0] is False

    def test_near_identity_uses_all_limits(self):
        report = AlignmentResult(
            sample_id="near",
            max_corner_displacement_pixels=0.5,
            local_scale=1.001,
            rotation_degrees=0.05,
            perspective_strength=0.001,
        )
        assert _is_near_identity(report, 1.0, 0.005, 0.1, 0.002)
        report.max_corner_displacement_pixels = 1.01
        assert not _is_near_identity(report, 1.0, 0.005, 0.1, 0.002)


class TestSIFTAligner:
    def test_rejects_unknown_matcher_backend(self):
        with pytest.raises(ValueError, match="Unsupported matcher backend"):
            SIFTAligner({"matcher": {"backend": "flann"}})

    def test_blank_images_fail_cleanly(self, white_800):
        aligner = SIFTAligner({"ransac": {"min_inliers": 4}})
        warped, report = aligner.align(white_800, white_800, "ident")
        assert not report.success
        assert not report.homography_estimated
        assert report.status == "homography_not_estimated"
        assert "Insufficient keypoints" in report.error
        assert np.array_equal(warped, white_800)

    def test_recovers_known_translation(self):
        template = synthetic_feature_image()
        translation = np.array(
            [[1.0, 0.0, 14.0], [0.0, 1.0, 9.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        query = cv2.warpPerspective(
            template,
            translation,
            (template.shape[1], template.shape[0]),
            borderValue=(255, 255, 255),
        )
        aligner = SIFTAligner({
            "sift": {"nfeatures": 1500},
            "matcher": {"backend": "bf", "lowe_ratio": 0.75},
            "ransac": {"min_inliers": 8, "reprojection_threshold": 3.0},
        })

        _, report = aligner.align(query, template, "translated")

        assert report.success, report.error
        assert report.homography_estimated
        assert report.geometry_accepted
        assert report.warp_applied
        assert report.selected_variant == "warped"
        assert report.quality_improved is True
        assert report.num_inlier_matches >= 8
        assert len(report.inlier_match_indices) == report.num_inlier_matches
        assert report.ransac_reprojection_error < 1.5
        assert report.metrics_valid
        assert report.valid_pixel_count < template.shape[0] * template.shape[1]
        assert report.ssim_valid_pixel_count < report.valid_pixel_count
        assert report.valid_mask is not None
        assert report.ssim_valid_mask is not None
        recovered = np.asarray(report.homography_matrix)
        assert recovered[0, 2] == pytest.approx(-14.0, abs=1.5)
        assert recovered[1, 2] == pytest.approx(-9.0, abs=1.5)

    def test_identical_feature_images_skip_warp(self):
        image = synthetic_feature_image()
        aligner = SIFTAligner({
            "matcher": {"backend": "bf", "lowe_ratio": 0.75},
            "ransac": {"min_inliers": 8},
        })
        selected, report = aligner.align(image, image.copy(), "already-aligned")
        assert report.homography_estimated
        assert report.geometry_accepted
        assert report.already_aligned
        assert not report.warp_applied
        assert report.quality_improved is None
        assert report.status == "already_aligned"
        assert np.array_equal(selected, image)

    def test_result_dataclass(self):
        result = AlignmentResult(
            sample_id="test",
            success=True,
            num_keypoints_query=100,
        )
        assert result.sample_id == "test"
        assert result.success
        assert result.num_keypoints_query == 100

    def test_load_image_rgb(self, tmp_path: Path):
        from step3_model.alignment import load_image_rgb

        image = np.random.default_rng(7).integers(
            0,
            256,
            (64, 64, 3),
            dtype=np.uint8,
        )
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        loaded = load_image_rgb(path)
        assert loaded.shape == (64, 64, 3)
        assert np.array_equal(loaded, image)

    def test_render_diff_marks_invalid_pixels_gray(self, tmp_path: Path):
        query = np.zeros((32, 32, 3), dtype=np.uint8)
        template = np.full((32, 32, 3), 255, dtype=np.uint8)
        mask = np.ones((32, 32), dtype=bool)
        mask[:, :8] = False
        output = tmp_path / "masked_diff.png"

        render_diff(query, template, "masked", output, mask)

        rendered = cv2.imread(str(output))
        assert rendered is not None
        assert np.array_equal(rendered[20, 2], np.array([128, 128, 128]))
