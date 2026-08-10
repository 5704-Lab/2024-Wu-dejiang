"""SIFT + RANSAC macro geometric alignment for Query–Template pairs.

Computes a robust homography between the query and template images, then
warps the query into the template coordinate frame.  All outputs are written
to ``artifacts/step3/``; the original dataset is never modified.
"""

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
    """Per-pair alignment report."""

    sample_id: str
    success: bool
    num_keypoints_query: int = 0
    num_keypoints_template: int = 0
    num_raw_matches: int = 0
    num_inlier_matches: int = 0
    inlier_match_indices: list[int] = field(default_factory=list, repr=False)
    homography_matrix: list[list[float]] = field(default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ransac_reprojection_error: float = 0.0
    ssim_before: float = 0.0
    ssim_after: float = 0.0
    mutual_info_before: float = 0.0
    mutual_info_after: float = 0.0
    error: str = ""


class SIFTAligner:
    """SIFT-based homography estimator with RANSAC outlier rejection."""

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
        matcher_backend = str(matcher_cfg.get("backend", "bf")).lower()
        if matcher_backend != "bf":
            raise ValueError(f"Unsupported matcher backend: {matcher_backend}")
        self.lowe_ratio = float(matcher_cfg.get("lowe_ratio", 0.7))
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        ransac_cfg = cfg.get("ransac", {})
        self.ransac_threshold = float(ransac_cfg.get("reprojection_threshold", 5.0))
        self.ransac_max_iters = int(ransac_cfg.get("max_iterations", 2000))
        self.ransac_confidence = float(ransac_cfg.get("confidence", 0.995))
        self.min_inliers = int(ransac_cfg.get("min_inliers", 10))

        quality_cfg = cfg.get("quality", {})
        self.ssim_window = int(quality_cfg.get("ssim_window", 11))
        self.mi_bins = int(quality_cfg.get("mutual_info_bins", 64))

    # ------------------------------------------------------------------
    def align(
        self, query: np.ndarray, template: np.ndarray, sample_id: str = ""
    ) -> tuple[np.ndarray, AlignmentResult]:
        """Warp *query* into *template* coordinates. Returns (warped_query, report)."""
        report = AlignmentResult(sample_id=sample_id, success=False)

        if query.shape != template.shape:
            report.error = f"Shape mismatch query={query.shape} template={template.shape}"
            return query, report

        gray_query = _to_gray(query)
        gray_template = _to_gray(template)

        kp_q, des_q = self.sift.detectAndCompute(gray_query, None)
        kp_t, des_t = self.sift.detectAndCompute(gray_template, None)

        report.num_keypoints_query = len(kp_q) if kp_q else 0
        report.num_keypoints_template = len(kp_t) if kp_t else 0

        if des_q is None or des_t is None or len(kp_q) < 4 or len(kp_t) < 4:
            report.error = f"Insufficient keypoints: query={len(kp_q) if kp_q else 0}, template={len(kp_t) if kp_t else 0}"
            report.ssim_before, report.mutual_info_before = _pair_metrics(query, template, self.ssim_window, self.mi_bins)
            report.ssim_after, report.mutual_info_after = report.ssim_before, report.mutual_info_before
            return query, report

        raw_matches = self.matcher.knnMatch(des_q, des_t, k=2)
        good = [m for m, n in raw_matches if m.distance < self.lowe_ratio * n.distance]
        report.num_raw_matches = len(good)

        if len(good) < self.min_inliers:
            report.error = f"Too few good matches: {len(good)} < {self.min_inliers}"
            report.ssim_before, report.mutual_info_before = _pair_metrics(query, template, self.ssim_window, self.mi_bins)
            report.ssim_after, report.mutual_info_after = report.ssim_before, report.mutual_info_before
            return query, report

        src_pts = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC,
            self.ransac_threshold, maxIters=self.ransac_max_iters, confidence=self.ransac_confidence,
        )

        if H is None:
            report.error = "Homography estimation returned None"
            report.ssim_before, report.mutual_info_before = _pair_metrics(query, template, self.ssim_window, self.mi_bins)
            report.ssim_after, report.mutual_info_after = report.ssim_before, report.mutual_info_before
            return query, report

        inlier_mask = mask.ravel().astype(bool) if mask is not None else np.zeros(len(good), dtype=bool)
        inlier_count = int(inlier_mask.sum())
        report.num_inlier_matches = inlier_count
        report.inlier_match_indices = np.flatnonzero(inlier_mask).astype(int).tolist()
        report.homography_matrix = H.astype(float).tolist()

        if inlier_count < self.min_inliers:
            report.error = f"Inliers below threshold: {inlier_count} < {self.min_inliers}"
            report.ssim_before, report.mutual_info_before = _pair_metrics(query, template, self.ssim_window, self.mi_bins)
            report.ssim_after, report.mutual_info_after = report.ssim_before, report.mutual_info_before
            return query, report

        # Reprojection error over RANSAC inliers only.
        projected = cv2.perspectiveTransform(src_pts, H).reshape(-1, 2)
        errors = np.linalg.norm(projected - dst_pts.reshape(-1, 2), axis=1)
        report.ransac_reprojection_error = float(errors[inlier_mask].mean())

        h, w = template.shape[:2]
        warped = cv2.warpPerspective(query, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        report.ssim_before, report.mutual_info_before = _pair_metrics(query, template, self.ssim_window, self.mi_bins)
        report.ssim_after, report.mutual_info_after = _pair_metrics(warped, template, self.ssim_window, self.mi_bins)
        report.success = True
        return warped, report

    # ------------------------------------------------------------------
    def render_matches(
        self,
        query: np.ndarray,
        template: np.ndarray,
        report: AlignmentResult,
        output_path: Path,
    ) -> None:
        """Draw keypoint matches and save to *output_path*."""
        gray_q = _to_gray(query)
        gray_t = _to_gray(template)
        kp_q, des_q = self.sift.detectAndCompute(gray_q, None)
        kp_t, des_t = self.sift.detectAndCompute(gray_t, None)

        if des_q is None or des_t is None:
            return

        raw = self.matcher.knnMatch(des_q, des_t, k=2)
        good = [m for m, n in raw if m.distance < self.lowe_ratio * n.distance]
        inliers = [good[index] for index in report.inlier_match_indices if index < len(good)]

        canvas = cv2.drawMatches(
            query, kp_q, template, kp_t, inliers[:100], None,
            matchColor=(42, 120, 214), singlePointColor=(237, 161, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def render_comparison(
    query: np.ndarray,
    warped: np.ndarray,
    template: np.ndarray,
    report: AlignmentResult,
    output_path: Path,
) -> None:
    """Save a 3-panel comparison: Query | Warped | Template."""
    gap = 8
    h, w = template.shape[:2]
    canvas = np.full((h + 60, w * 3 + gap * 2, 3), 252, dtype=np.uint8)
    canvas[:h, :w] = query
    canvas[:h, w + gap : 2 * w + gap] = warped
    canvas[:h, 2 * w + 2 * gap : 3 * w + 2 * gap] = template

    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = [
        ("QUERY", (0, h + 36)),
        ("WARPED -> TEMPLATE", (w + gap, h + 36)),
        ("TEMPLATE", (2 * w + 2 * gap, h + 36)),
    ]
    for text, pos in labels:
        cv2.putText(canvas, text, pos, font, 0.55, (11, 11, 11), 1, cv2.LINE_AA)

    info = f"SSIM {report.ssim_before:.3f}->{report.ssim_after:.3f}  MI {report.mutual_info_before:.3f}->{report.mutual_info_after:.3f}"
    cv2.putText(canvas, info, (8, h + 54), font, 0.45, (82, 81, 78), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def render_diff(query: np.ndarray, template: np.ndarray, title: str, output_path: Path) -> None:
    """Save an absolute-difference heatmap."""
    diff = np.abs(query.astype(np.float32) - template.astype(np.float32))
    diff_gray = diff.mean(axis=2)
    diff_gray = np.clip(diff_gray / max(diff_gray.max(), 1.0), 0, 1)
    heatmap = (diff_gray * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    h, w = colored.shape[:2]
    canvas = colored
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title, (10, 28), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _imwrite(output_path, canvas)


def load_image_rgb(path: Path) -> np.ndarray:
    """Load image via imdecode to support non-ASCII paths on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ======================================================================
# internal helpers
# ======================================================================


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image


def _ssim(a: np.ndarray, b: np.ndarray, window: int = 11) -> float:
    """Structural similarity (scalar mean over channels)."""
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(window, 1.5)
    window_2d = kernel @ kernel.T
    window_2d = window_2d / window_2d.sum()
    ssim_sum = 0.0
    channels = a_f.shape[2] if a_f.ndim == 3 else 1
    for ch in range(channels):
        a_ch = a_f[..., ch] if a_f.ndim == 3 else a_f
        b_ch = b_f[..., ch] if b_f.ndim == 3 else b_f
        mu_a = cv2.filter2D(a_ch, -1, window_2d)
        mu_b = cv2.filter2D(b_ch, -1, window_2d)
        mu_a2, mu_b2 = mu_a * mu_a, mu_b * mu_b
        sigma_a2 = cv2.filter2D(a_ch * a_ch, -1, window_2d) - mu_a2
        sigma_b2 = cv2.filter2D(b_ch * b_ch, -1, window_2d) - mu_b2
        sigma_ab = cv2.filter2D(a_ch * b_ch, -1, window_2d) - mu_a * mu_b
        num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
        den = (mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2)
        ssim_sum += (num / (den + 1e-10)).mean()
    return float(ssim_sum / channels)


def _mutual_info(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """Mutual information estimated via 2-D histogram."""
    a_g = _to_gray(a).ravel()
    b_g = _to_gray(b).ravel()
    hist_2d, _, _ = np.histogram2d(a_g, b_g, bins=bins, range=[[0, 256], [0, 256]])
    hist_2d = hist_2d / hist_2d.sum()
    marginal_a = hist_2d.sum(axis=1)
    marginal_b = hist_2d.sum(axis=0)
    mi = 0.0
    for ia in range(bins):
        if marginal_a[ia] == 0:
            continue
        for ib in range(bins):
            if marginal_b[ib] == 0 or hist_2d[ia, ib] == 0:
                continue
            mi += hist_2d[ia, ib] * math.log2(hist_2d[ia, ib] / (marginal_a[ia] * marginal_b[ib]))
    return float(mi)


def _pair_metrics(query: np.ndarray, template: np.ndarray, ssim_window: int, mi_bins: int) -> tuple[float, float]:
    s = _ssim(query, template, ssim_window)
    m = _mutual_info(query, template, mi_bins)
    return s, m
