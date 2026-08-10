from pathlib import Path

import cv2
import numpy as np
import pytest

from step3_model.alignment import (
    AlignmentResult,
    SIFTAligner,
    _mutual_info,
    _ssim,
    _to_gray,
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


class TestSIFTAligner:
    def test_rejects_unknown_matcher_backend(self):
        with pytest.raises(ValueError, match="Unsupported matcher backend"):
            SIFTAligner({"matcher": {"backend": "flann"}})

    def test_blank_images_fail_cleanly(self, white_800):
        aligner = SIFTAligner({"ransac": {"min_inliers": 4}})
        warped, report = aligner.align(white_800, white_800, "ident")
        assert not report.success
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
        assert report.num_inlier_matches >= 8
        assert len(report.inlier_match_indices) == report.num_inlier_matches
        assert report.ransac_reprojection_error < 1.5
        recovered = np.asarray(report.homography_matrix)
        assert recovered[0, 2] == pytest.approx(-14.0, abs=1.5)
        assert recovered[1, 2] == pytest.approx(-9.0, abs=1.5)

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
