"""SIFT + RANSAC alignment with geometric and quality acceptance."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _imwrite(path: Path, img: np.ndarray) -> None:
    """Write an image via imencode to handle non-ASCII paths on Windows."""
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise OSError(f"Failed to encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())


@dataclass
class AlignmentResult:
    """Per-pair alignment and acceptance report."""

    sample_id: str
    success: bool = False
    homography_estimated: bool = False
    geometry_accepted: bool = False
    quality_improved: bool | None = None
    already_aligned: bool = False
    warp_applied: bool = False
    quality_evaluated: bool = False
    selected_variant: str = "original"
    status: str = "homography_not_estimated"
    reasons: list[str] = field(default_factory=list)
    criteria: dict[str, dict[str, Any]] = field(default_factory=dict)
    num_keypoints_query: int = 0
    num_keypoints_template: int = 0
    num_raw_matches: int = 0
    num_inlier_matches: int = 0
    inlier_ratio: float | None = None
    inlier_match_indices: list[int] = field(default_factory=list, repr=False)
    valid_mask: np.ndarray | None = field(default=None, repr=False)
    ssim_valid_mask: np.ndarray | None = field(default=None, repr=False)
    homography_matrix: list[list[float]] | None = None
    homography_normalized: bool = False
    homography_condition_number: float | None = None
    ransac_reprojection_error: float | None = None
    max_inlier_reprojection_error: float | None = None
    projected_query_corners: list[list[float]] | None = None
    projected_area_ratio: float | None = None
    orientation_preserved: bool | None = None
    self_intersects: bool | None = None
    projected_overlap_ratio: float | None = None
    local_scale: float | None = None
    local_scale_x: float | None = None
    local_scale_y: float | None = None
    anisotropy_ratio: float | None = None
    rotation_degrees: float | None = None
    perspective_strength: float | None = None
    max_corner_displacement_pixels: float | None = None
    valid_pixel_count: int = 0
    valid_pixel_ratio: float = 0.0
    ssim_valid_pixel_count: int = 0
    ssim_valid_pixel_ratio: float = 0.0
    metrics_valid: bool = False
    ssim_original: float | None = None
    ssim_warped: float | None = None
    ssim_selected: float | None = None
    mutual_info_original: float | None = None
    mutual_info_warped: float | None = None
    mutual_info_selected: float | None = None
    error: str = ""

    @property
    def ssim_before(self) -> float:
        return float(self.ssim_original or 0.0)

    @property
    def ssim_after(self) -> float:
        return float(self.ssim_selected or 0.0)

    @property
    def mutual_info_before(self) -> float:
        return float(self.mutual_info_original or 0.0)

    @property
    def mutual_info_after(self) -> float:
        return float(self.mutual_info_selected or 0.0)

    @property
    def ssim_candidate_delta(self) -> float | None:
        if self.ssim_original is None or self.ssim_warped is None:
            return None
        return self.ssim_warped - self.ssim_original

    @property
    def mutual_info_candidate_delta(self) -> float | None:
        if self.mutual_info_original is None or self.mutual_info_warped is None:
            return None
        return self.mutual_info_warped - self.mutual_info_original


class SIFTAligner:
    """SIFT homography estimator with geometry and quality acceptance."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        sift_cfg = cfg.get("sift", {})
        self.sift = cv2.SIFT_create(
            nfeatures=int(sift_cfg.get("nfeatures", 2000)),
            nOctaveLayers=int(sift_cfg.get("n_octave_layers", 3)),
            contrastThreshold=float(sift_cfg.get("contrast_threshold", 0.04)),
            edgeThreshold=float(sift_cfg.get("edge_threshold", 10)),
            sigma=float(sift_cfg.get("sigma", 1.6)),
        )
        matcher_cfg = cfg.get("matcher", {})
        backend = str(matcher_cfg.get("backend", "bf")).lower()
        if backend != "bf":
            raise ValueError(f"Unsupported matcher backend: {backend}")
        self.lowe_ratio = float(matcher_cfg.get("lowe_ratio", 0.7))
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        ransac = cfg.get("ransac", {})
        self.ransac_threshold = float(ransac.get("reprojection_threshold", 5.0))
        self.ransac_max_iters = int(ransac.get("max_iterations", 2000))
        self.ransac_confidence = float(ransac.get("confidence", 0.995))
        self.min_inliers = int(ransac.get("min_inliers", 10))
        self.min_inlier_ratio = float(ransac.get("min_inlier_ratio", 0.5))
        self.max_inlier_error = float(ransac.get("max_inlier_reprojection_error", 4.0))

        geometry = cfg.get("geometry", {})
        self.normalization_epsilon = float(geometry.get("normalization_epsilon", 1e-8))
        self.max_condition_number = float(geometry.get("max_condition_number", 1e4))
        self.min_area_ratio = float(geometry.get("min_projected_area_ratio", 0.5))
        self.max_area_ratio = float(geometry.get("max_projected_area_ratio", 2.0))
        self.min_overlap_ratio = float(geometry.get("min_overlap_ratio", 0.7))
        self.min_scale = float(geometry.get("min_scale", 0.5))
        self.max_scale = float(geometry.get("max_scale", 2.0))
        self.max_anisotropy = float(geometry.get("max_anisotropy_ratio", 2.0))
        self.max_rotation = float(geometry.get("max_abs_rotation_degrees", 30.0))
        self.max_perspective = float(geometry.get("max_perspective_strength", 0.25))

        near = cfg.get("near_identity", {})
        self.near_identity_enabled = bool(near.get("enabled", True))
        self.near_max_displacement = float(near.get("max_corner_displacement_pixels", 1.0))
        self.near_max_scale_deviation = float(near.get("max_scale_deviation", 0.005))
        self.near_max_rotation = float(near.get("max_abs_rotation_degrees", 0.1))
        self.near_max_perspective = float(near.get("max_perspective_strength", 0.002))

        quality = cfg.get("quality", {})
        self.ssim_window = int(quality.get("ssim_window", 11))
        self.mi_bins = int(quality.get("mutual_info_bins", 64))
        self.erode_ssim_mask = bool(quality.get("erode_ssim_mask", True))
        self.min_valid_pixels = int(quality.get("min_valid_pixels", 1))
        self.min_ssim_improvement = float(quality.get("min_ssim_improvement", 0.001))
        self.min_mi_improvement = float(quality.get("min_mutual_info_improvement", 0.01))
        self.max_ssim_regression = float(quality.get("max_ssim_regression", 0.0005))
        self.max_mi_regression = float(quality.get("max_mutual_info_regression", 0.01))
        self._validate_config()

    def _validate_config(self) -> None:
        if self.ssim_window < 1 or self.ssim_window % 2 == 0:
            raise ValueError("quality.ssim_window must be a positive odd integer")
        if self.min_valid_pixels < 1 or self.min_inliers < 4:
            raise ValueError("minimum counts must be valid")
        if not 0 <= self.min_inlier_ratio <= 1 or not 0 <= self.min_overlap_ratio <= 1:
            raise ValueError("ratio thresholds must be in [0, 1]")
        if self.min_area_ratio > self.max_area_ratio or self.min_scale > self.max_scale:
            raise ValueError("minimum geometry thresholds must not exceed maximums")
        values = [self.max_inlier_error, self.normalization_epsilon, self.max_condition_number,
                  self.max_anisotropy, self.max_rotation, self.max_perspective,
                  self.near_max_displacement, self.near_max_scale_deviation,
                  self.near_max_rotation, self.near_max_perspective,
                  self.min_ssim_improvement, self.min_mi_improvement,
                  self.max_ssim_regression, self.max_mi_regression]
        if any(value < 0 for value in values) or self.max_condition_number < 1:
            raise ValueError("acceptance thresholds must be non-negative")

    def _criterion(self, report: AlignmentResult, name: str, passed: bool | None,
                   value: Any, operator: str, threshold: Any, applicable: bool = True) -> None:
        report.criteria[name] = {
            "applicable": applicable,
            "passed": passed if applicable else None,
            "value": value if applicable else None,
            "operator": operator,
            "threshold": threshold,
        }
        if applicable and passed is False:
            report.reasons.append(name)

    def _baseline(self, query: np.ndarray, template: np.ndarray, report: AlignmentResult) -> None:
        ssim, mi = _pair_metrics(query, template, self.ssim_window, self.mi_bins)
        report.ssim_original = report.ssim_selected = ssim
        report.mutual_info_original = report.mutual_info_selected = mi

    def align(self, query: np.ndarray, template: np.ndarray,
              sample_id: str = "") -> tuple[np.ndarray, AlignmentResult]:
        """Return the selected query image and a complete acceptance report."""
        report = AlignmentResult(sample_id=sample_id)
        if query.shape != template.shape:
            report.error = f"Shape mismatch query={query.shape} template={template.shape}"
            report.reasons.append("shape_mismatch")
            self._baseline(query, template, report)
            return query, report

        gray_query, gray_template = _to_gray(query), _to_gray(template)
        kp_q, des_q = self.sift.detectAndCompute(gray_query, None)
        kp_t, des_t = self.sift.detectAndCompute(gray_template, None)
        report.num_keypoints_query = len(kp_q) if kp_q else 0
        report.num_keypoints_template = len(kp_t) if kp_t else 0
        if des_q is None or des_t is None or len(kp_q) < 4 or len(kp_t) < 4:
            report.error = "Insufficient keypoints"
            report.reasons.append("insufficient_keypoints")
            self._baseline(query, template, report)
            return query, report

        raw = self.matcher.knnMatch(des_q, des_t, k=2)
        good = [m for pair in raw if len(pair) == 2 for m, n in [pair]
                if m.distance < self.lowe_ratio * n.distance]
        report.num_raw_matches = len(good)
        self._criterion(report, "minimum_ratio_matches", len(good) >= 4,
                        len(good), ">=", 4)
        if len(good) < 4:
            report.error = "Too few ratio matches"
            self._baseline(query, template, report)
            return query, report

        src = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_threshold,
                                    maxIters=self.ransac_max_iters,
                                    confidence=self.ransac_confidence)
        if H is None or mask is None:
            report.error = "Homography estimation returned None"
            report.reasons.append("homography_not_returned")
            self._baseline(query, template, report)
            return query, report
        report.homography_estimated = True

        shape_valid = np.asarray(H).shape == (3, 3)
        self._criterion(report, "homography_shape_valid", shape_valid,
                        list(np.asarray(H).shape), "==", [3, 3])
        finite = shape_valid and bool(np.isfinite(H).all())
        self._criterion(report, "homography_finite", finite, finite, "==", True)
        normalized = _normalize_homography(H, self.normalization_epsilon) if finite else None
        self._criterion(report, "homography_normalizable", normalized is not None,
                        normalized is not None, "==", True)
        if normalized is None:
            report.status = "geometry_rejected"
            report.error = "Homography is not finite and normalizable"
            self._baseline(query, template, report)
            return query, report
        report.homography_normalized = True
        report.homography_matrix = normalized.tolist()

        inliers = mask.ravel().astype(bool)
        report.num_inlier_matches = int(inliers.sum())
        report.inlier_ratio = report.num_inlier_matches / len(good)
        report.inlier_match_indices = np.flatnonzero(inliers).astype(int).tolist()
        self._criterion(report, "minimum_inlier_matches",
                        report.num_inlier_matches >= self.min_inliers,
                        report.num_inlier_matches, ">=", self.min_inliers)
        self._criterion(report, "minimum_inlier_ratio",
                        report.inlier_ratio >= self.min_inlier_ratio,
                        report.inlier_ratio, ">=", self.min_inlier_ratio)
        if report.num_inlier_matches:
            projected = cv2.perspectiveTransform(src, normalized).reshape(-1, 2)
            errors = np.linalg.norm(projected - dst.reshape(-1, 2), axis=1)[inliers]
            report.ransac_reprojection_error = float(errors.mean())
            report.max_inlier_reprojection_error = float(errors.max())
        self._criterion(report, "maximum_inlier_reprojection_error",
                        report.max_inlier_reprojection_error is not None and
                        report.max_inlier_reprojection_error <= self.max_inlier_error,
                        report.max_inlier_reprojection_error, "<=", self.max_inlier_error)

        h, w = template.shape[:2]
        condition = _normalized_condition_number(normalized, query.shape[:2], template.shape[:2])
        report.homography_condition_number = condition
        self._criterion(report, "condition_number_within_limit",
                        condition is not None and condition <= self.max_condition_number,
                        condition, "<=", self.max_condition_number)

        corners = _project_corners(normalized, query.shape[:2], self.normalization_epsilon)
        corners_finite = corners is not None
        self._criterion(report, "projected_corners_finite", corners_finite,
                        corners_finite, "==", True)
        if corners is not None:
            report.projected_query_corners = corners.tolist()
            source_corners = _image_corners(query.shape[:2])
            source_area = _signed_polygon_area(source_corners)
            projected_area = _signed_polygon_area(corners)
            report.projected_area_ratio = abs(projected_area / source_area)
            report.orientation_preserved = source_area * projected_area > 0
            report.self_intersects = _quadrilateral_self_intersects(corners)
            report.projected_overlap_ratio = _projected_overlap_ratio(corners, template.shape[:2])
            diagnostics = _transform_diagnostics(normalized, query.shape[:2], template.shape[:2])
            for key, value in diagnostics.items():
                setattr(report, key, value)
            report.max_corner_displacement_pixels = float(
                np.linalg.norm(corners - source_corners, axis=1).max()
            )

        self._criterion(report, "projected_area_ratio_within_range",
                        report.projected_area_ratio is not None and
                        self.min_area_ratio <= report.projected_area_ratio <= self.max_area_ratio,
                        report.projected_area_ratio, "between", [self.min_area_ratio, self.max_area_ratio])
        self._criterion(report, "orientation_preserved",
                        report.orientation_preserved is True,
                        report.orientation_preserved, "==", True)
        self._criterion(report, "projected_quadrilateral_non_self_intersecting",
                        report.self_intersects is False,
                        report.self_intersects, "==", False)
        self._criterion(report, "scale_within_range",
                        report.local_scale is not None and self.min_scale <= report.local_scale <= self.max_scale,
                        report.local_scale, "between", [self.min_scale, self.max_scale])
        self._criterion(report, "anisotropy_within_limit",
                        report.anisotropy_ratio is not None and report.anisotropy_ratio <= self.max_anisotropy,
                        report.anisotropy_ratio, "<=", self.max_anisotropy)
        self._criterion(report, "rotation_within_limit",
                        report.rotation_degrees is not None and abs(report.rotation_degrees) <= self.max_rotation,
                        abs(report.rotation_degrees) if report.rotation_degrees is not None else None,
                        "<=", self.max_rotation)
        self._criterion(report, "perspective_within_limit",
                        report.perspective_strength is not None and report.perspective_strength <= self.max_perspective,
                        report.perspective_strength, "<=", self.max_perspective)
        self._criterion(report, "overlap_above_minimum",
                        report.projected_overlap_ratio is not None and report.projected_overlap_ratio >= self.min_overlap_ratio,
                        report.projected_overlap_ratio, ">=", self.min_overlap_ratio)

        mandatory = [criterion["passed"] for name, criterion in report.criteria.items()
                     if not name.startswith("near_identity")]
        report.geometry_accepted = all(value is True for value in mandatory)
        report.success = report.geometry_accepted
        if not report.geometry_accepted:
            report.status = "geometry_rejected"
            report.error = "Homography failed geometry acceptance"
            self._baseline(query, template, report)
            return query, report

        report.already_aligned = self.near_identity_enabled and _is_near_identity(
            report, self.near_max_displacement, self.near_max_scale_deviation,
            self.near_max_rotation, self.near_max_perspective
        )
        near_checks = {
            "near_identity_corner_displacement": (
                report.max_corner_displacement_pixels,
                self.near_max_displacement,
            ),
            "near_identity_scale": (
                abs(report.local_scale - 1) if report.local_scale is not None else None,
                self.near_max_scale_deviation,
            ),
            "near_identity_rotation": (
                abs(report.rotation_degrees) if report.rotation_degrees is not None else None,
                self.near_max_rotation,
            ),
            "near_identity_perspective": (
                report.perspective_strength,
                self.near_max_perspective,
            ),
        }
        for name, (value, threshold) in near_checks.items():
            report.criteria[name] = {
                "applicable": self.near_identity_enabled,
                "passed": value is not None and value <= threshold if self.near_identity_enabled else None,
                "value": value if self.near_identity_enabled else None,
                "operator": "<=",
                "threshold": threshold,
            }
        if report.already_aligned:
            report.status = "already_aligned"
            report.reasons.append("near_identity_transform")
            self._baseline(query, template, report)
            return query, report

        warped = cv2.warpPerspective(query, normalized, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        report.warp_applied = True
        valid = _warp_valid_mask(query.shape[:2], normalized, (w, h))
        ssim_valid = _erode_valid_mask(valid, self.ssim_window) if self.erode_ssim_mask else valid.copy()
        report.valid_mask, report.ssim_valid_mask = valid, ssim_valid
        report.valid_pixel_count = int(valid.sum())
        report.valid_pixel_ratio = report.valid_pixel_count / valid.size
        report.ssim_valid_pixel_count = int(ssim_valid.sum())
        report.ssim_valid_pixel_ratio = report.ssim_valid_pixel_count / ssim_valid.size
        report.metrics_valid = (report.valid_pixel_count >= self.min_valid_pixels and
                                report.ssim_valid_pixel_count >= self.min_valid_pixels)
        if report.metrics_valid:
            report.quality_evaluated = True
            report.ssim_original = _ssim(query, template, self.ssim_window, ssim_valid)
            report.ssim_warped = _ssim(warped, template, self.ssim_window, ssim_valid)
            report.mutual_info_original = _mutual_info(query, template, self.mi_bins, valid)
            report.mutual_info_warped = _mutual_info(warped, template, self.mi_bins, valid)
        report.quality_improved, quality_reasons = _decide_quality(
            report.ssim_candidate_delta, report.mutual_info_candidate_delta,
            report.metrics_valid, self.min_ssim_improvement, self.min_mi_improvement,
            self.max_ssim_regression, self.max_mi_regression
        )
        quality_values = {
            "quality_metrics_valid": (report.metrics_valid, report.metrics_valid, "==", True),
            "ssim_regression_within_tolerance": (
                report.ssim_candidate_delta is not None and
                report.ssim_candidate_delta >= -self.max_ssim_regression,
                report.ssim_candidate_delta,
                ">=",
                -self.max_ssim_regression,
            ),
            "mutual_info_regression_within_tolerance": (
                report.mutual_info_candidate_delta is not None and
                report.mutual_info_candidate_delta >= -self.max_mi_regression,
                report.mutual_info_candidate_delta,
                ">=",
                -self.max_mi_regression,
            ),
            "minimum_quality_gain_reached": (
                report.ssim_candidate_delta is not None and
                report.mutual_info_candidate_delta is not None and
                (report.ssim_candidate_delta >= self.min_ssim_improvement or
                 report.mutual_info_candidate_delta >= self.min_mi_improvement),
                {
                    "ssim_delta": report.ssim_candidate_delta,
                    "mutual_info_delta": report.mutual_info_candidate_delta,
                },
                "any>=",
                {
                    "ssim": self.min_ssim_improvement,
                    "mutual_info": self.min_mi_improvement,
                },
            ),
        }
        for name, (passed, value, operator, threshold) in quality_values.items():
            report.criteria[name] = {
                "applicable": report.warp_applied,
                "passed": passed if report.warp_applied else None,
                "value": value if report.warp_applied else None,
                "operator": operator,
                "threshold": threshold,
            }
        report.reasons.extend(quality_reasons)
        if report.quality_improved:
            report.selected_variant = "warped"
            report.status = "warped_selected"
            report.ssim_selected = report.ssim_warped
            report.mutual_info_selected = report.mutual_info_warped
            return warped, report
        report.status = "original_retained"
        report.ssim_selected = report.ssim_original
        report.mutual_info_selected = report.mutual_info_original
        return query, report

    def render_matches(self, query: np.ndarray, template: np.ndarray,
                       report: AlignmentResult, output_path: Path) -> None:
        gray_q, gray_t = _to_gray(query), _to_gray(template)
        kp_q, des_q = self.sift.detectAndCompute(gray_q, None)
        kp_t, des_t = self.sift.detectAndCompute(gray_t, None)
        if des_q is None or des_t is None:
            return
        raw = self.matcher.knnMatch(des_q, des_t, k=2)
        good = [m for pair in raw if len(pair) == 2 for m, n in [pair]
                if m.distance < self.lowe_ratio * n.distance]
        inliers = [good[index] for index in report.inlier_match_indices if index < len(good)]
        canvas = cv2.drawMatches(query, kp_q, template, kp_t, inliers[:100], None,
                                 matchColor=(42, 120, 214), singlePointColor=(237, 161, 0),
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        _imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def render_comparison(query: np.ndarray, selected: np.ndarray, template: np.ndarray,
                      report: AlignmentResult, output_path: Path) -> None:
    gap = 8
    h, w = template.shape[:2]
    canvas = np.full((h + 64, w * 3 + gap * 2, 3), 252, dtype=np.uint8)
    canvas[:h, :w], canvas[:h, w + gap:2 * w + gap] = query, selected
    canvas[:h, 2 * w + 2 * gap:3 * w + 2 * gap] = template
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = ["QUERY", f"SELECTED: {report.selected_variant.upper()}", "TEMPLATE"]
    for text, x in zip(labels, [0, w + gap, 2 * w + 2 * gap]):
        cv2.putText(canvas, text, (x, h + 34), font, 0.52, (11, 11, 11), 1, cv2.LINE_AA)
    info = f"{report.status}  inlier={report.inlier_ratio or 0:.1%}  overlap={report.projected_overlap_ratio or 0:.1%}"
    cv2.putText(canvas, info, (8, h + 55), font, 0.42, (82, 81, 78), 1, cv2.LINE_AA)
    _imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def render_diff(query: np.ndarray, template: np.ndarray, title: str, output_path: Path,
                mask: np.ndarray | None = None) -> None:
    valid = _validate_mask(mask, query.shape[:2]) if mask is not None else None
    diff_gray = np.abs(query.astype(np.float32) - template.astype(np.float32)).mean(axis=2)
    values = diff_gray[valid] if valid is not None and valid.any() else diff_gray
    scale = max(float(values.max()) if values.size else 0.0, 1.0)
    colored = cv2.applyColorMap((np.clip(diff_gray / scale, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    if valid is not None:
        colored[~valid] = (128, 128, 128)
    cv2.putText(colored, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    _imwrite(output_path, colored)


def load_image_rgb(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _normalize_homography(H: np.ndarray, epsilon: float = 1e-8) -> np.ndarray | None:
    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(matrix[2, 2]) < epsilon:
        return None
    matrix = matrix / matrix[2, 2]
    return matrix if np.isfinite(matrix).all() else None


def _image_corners(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)


def _coordinate_normalization(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.array([[2 / w, 0, -1], [0, 2 / h, -1], [0, 0, 1]], dtype=np.float64)


def _normalized_homography(H: np.ndarray, source_shape: tuple[int, int],
                           destination_shape: tuple[int, int]) -> np.ndarray | None:
    source = _coordinate_normalization(source_shape)
    destination = _coordinate_normalization(destination_shape)
    return _normalize_homography(destination @ H @ np.linalg.inv(source))


def _normalized_condition_number(H: np.ndarray, source_shape: tuple[int, int],
                                 destination_shape: tuple[int, int]) -> float | None:
    normalized = _normalized_homography(H, source_shape, destination_shape)
    if normalized is None:
        return None
    value = float(np.linalg.cond(normalized))
    return value if math.isfinite(value) else None


def _project_corners(H: np.ndarray, shape: tuple[int, int], epsilon: float = 1e-8) -> np.ndarray | None:
    corners = _image_corners(shape)
    homogeneous = np.c_[corners, np.ones(4)] @ H.T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < epsilon):
        return None
    projected = homogeneous[:, :2] / homogeneous[:, 2, None]
    return projected if np.isfinite(projected).all() else None


def _signed_polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
                        epsilon: float = 1e-9) -> bool:
    def cross(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        first = q - p
        second = r - p
        return float(first[0] * second[1] - first[1] * second[0])
    ab_c, ab_d, cd_a, cd_b = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
    return ab_c * ab_d < -epsilon and cd_a * cd_b < -epsilon


def _quadrilateral_self_intersects(points: np.ndarray) -> bool:
    return (_segments_intersect(points[0], points[1], points[2], points[3]) or
            _segments_intersect(points[1], points[2], points[3], points[0]))


def _projected_overlap_ratio(points: np.ndarray, shape: tuple[int, int]) -> float:
    if _quadrilateral_self_intersects(points) or not cv2.isContourConvex(points.astype(np.float32)):
        return 0.0
    h, w = shape
    projected = points.copy()
    for axis, upper in ((0, w - 1), (1, h - 1)):
        projected[np.isclose(projected[:, axis], 0.0, atol=1e-6), axis] = 0.0
        projected[np.isclose(projected[:, axis], upper, atol=1e-6), axis] = upper
    destination = _image_corners(shape).astype(np.float32)
    area, _ = cv2.intersectConvexConvex(projected.astype(np.float32), destination)
    return float(area / ((w - 1) * (h - 1)))


def _transform_diagnostics(H: np.ndarray, source_shape: tuple[int, int],
                           destination_shape: tuple[int, int]) -> dict[str, float | None]:
    h, w = source_shape
    x, y = (w - 1) / 2, (h - 1) / 2
    a, b, c, d, e, f, g, k, one = H.ravel()
    denominator = g * x + k * y + one
    u, v = (a * x + b * y + c) / denominator, (d * x + e * y + f) / denominator
    J = np.array([[(a - g * u) / denominator, (b - k * u) / denominator],
                  [(d - g * v) / denominator, (e - k * v) / denominator]])
    U, singular, Vt = np.linalg.svd(J)
    rotation = U @ Vt
    if np.linalg.det(rotation) < 0:
        U[:, -1] *= -1
        rotation = U @ Vt
    normalized = _normalized_homography(H, source_shape, destination_shape)
    perspective = math.hypot(normalized[2, 0], normalized[2, 1]) if normalized is not None else None
    low, high = float(singular.min()), float(singular.max())
    return {
        "local_scale": float(math.sqrt(abs(np.linalg.det(J)))),
        "local_scale_x": high,
        "local_scale_y": low,
        "anisotropy_ratio": high / low if low > 0 else None,
        "rotation_degrees": float(math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))),
        "perspective_strength": float(perspective) if perspective is not None else None,
    }


def _is_near_identity(report: AlignmentResult, max_displacement: float,
                      max_scale_deviation: float, max_rotation: float,
                      max_perspective: float) -> bool:
    return bool(report.max_corner_displacement_pixels is not None and
                report.local_scale is not None and report.rotation_degrees is not None and
                report.perspective_strength is not None and
                report.max_corner_displacement_pixels <= max_displacement and
                abs(report.local_scale - 1) <= max_scale_deviation and
                abs(report.rotation_degrees) <= max_rotation and
                report.perspective_strength <= max_perspective)


def _decide_quality(ssim_delta: float | None, mi_delta: float | None, metrics_valid: bool,
                    min_ssim: float, min_mi: float, max_ssim_regression: float,
                    max_mi_regression: float) -> tuple[bool, list[str]]:
    if not metrics_valid or ssim_delta is None or mi_delta is None:
        return False, ["quality_metrics_invalid"]
    reasons = []
    if ssim_delta < -max_ssim_regression:
        reasons.append("ssim_regression_exceeded")
    if mi_delta < -max_mi_regression:
        reasons.append("mutual_info_regression_exceeded")
    meaningful = ssim_delta >= min_ssim or mi_delta >= min_mi
    if not meaningful:
        reasons.append("quality_gain_below_threshold")
    improved = not reasons
    return improved, ["warped_candidate_selected"] if improved else reasons


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image


def _validate_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    valid = np.asarray(mask, dtype=bool)
    if valid.shape != shape:
        raise ValueError(f"Mask shape {valid.shape} does not match image shape {shape}")
    return valid


def _warp_valid_mask(source_shape: tuple[int, int], homography: np.ndarray,
                     output_size: tuple[int, int]) -> np.ndarray:
    source = np.full(source_shape, 255, dtype=np.uint8)
    warped = cv2.warpPerspective(source, homography, output_size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped == 255


def _erode_valid_mask(mask: np.ndarray, window: int) -> np.ndarray:
    valid = np.asarray(mask, dtype=bool)
    if window <= 1:
        return valid.copy()
    return cv2.erode(valid.astype(np.uint8), np.ones((window, window), dtype=np.uint8),
                     iterations=1) > 0


def _ssim(a: np.ndarray, b: np.ndarray, window: int = 11,
          mask: np.ndarray | None = None) -> float:
    a_f, b_f = a.astype(np.float32), b.astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(window, 1.5)
    kernel = kernel @ kernel.T
    valid = _validate_mask(mask, a.shape[:2]) if mask is not None else None
    total, channels = 0.0, a_f.shape[2] if a_f.ndim == 3 else 1
    for channel in range(channels):
        x = a_f[..., channel] if a_f.ndim == 3 else a_f
        y = b_f[..., channel] if b_f.ndim == 3 else b_f
        mx, my = cv2.filter2D(x, -1, kernel), cv2.filter2D(y, -1, kernel)
        vx = cv2.filter2D(x * x, -1, kernel) - mx * mx
        vy = cv2.filter2D(y * y, -1, kernel) - my * my
        cov = cv2.filter2D(x * y, -1, kernel) - mx * my
        score = ((2 * mx * my + C1) * (2 * cov + C2) /
                 ((mx * mx + my * my + C1) * (vx + vy + C2) + 1e-10))
        if valid is not None:
            if not valid.any():
                return 0.0
            total += float(score[valid].mean())
        else:
            total += float(score.mean())
    return total / channels


def _mutual_info(a: np.ndarray, b: np.ndarray, bins: int = 64,
                 mask: np.ndarray | None = None) -> float:
    left, right = _to_gray(a), _to_gray(b)
    if mask is not None:
        valid = _validate_mask(mask, left.shape)
        if not valid.any():
            return 0.0
        left, right = left[valid], right[valid]
    histogram, _, _ = np.histogram2d(left.ravel(), right.ravel(), bins=bins,
                                     range=[[0, 256], [0, 256]])
    total = histogram.sum()
    if total <= 0:
        return 0.0
    probability = histogram / total
    pa, pb = probability.sum(axis=1), probability.sum(axis=0)
    expected = pa[:, None] * pb[None, :]
    nonzero = probability > 0
    return float(np.sum(probability[nonzero] * np.log2(probability[nonzero] / expected[nonzero])))


def _pair_metrics(query: np.ndarray, template: np.ndarray, ssim_window: int,
                  mi_bins: int, mask: np.ndarray | None = None,
                  ssim_mask: np.ndarray | None = None) -> tuple[float, float]:
    return (_ssim(query, template, ssim_window, ssim_mask if ssim_mask is not None else mask),
            _mutual_info(query, template, mi_bins, mask))
