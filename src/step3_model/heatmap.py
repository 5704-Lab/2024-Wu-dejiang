"""Multi-scale feature difference heatmap generation for Step 3.5-A."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .offset_predictor import OFFSET_SPEC

_STAGE_NAMES = [name for name, _, _ in OFFSET_SPEC]


def _imwrite(path: Path, img: np.ndarray) -> None:
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise OSError(f"Failed to encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())


def compute_diff_maps(
    warped: dict[str, torch.Tensor],
    query_features: dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Compute relative L2 difference per scale: ||warped - query|| / (||query|| + eps)."""
    diff_maps: dict[str, np.ndarray] = {}
    for key in _STAGE_NAMES:
        w, q = warped[key], query_features[key]
        diff = torch.norm(w - q, p=2, dim=1)  # [B, H, W]
        norm = torch.norm(q, p=2, dim=1) + eps
        relative = (diff / norm).squeeze(0).detach().cpu().numpy().astype(np.float32)
        diff_maps[key] = relative
    return diff_maps


def upsample_diff_maps(
    diff_maps: dict[str, np.ndarray],
    target_size: tuple[int, int] = (800, 800),
) -> dict[str, np.ndarray]:
    """Bilinear upsample each scale's diff map to target_size."""
    upsampled: dict[str, np.ndarray] = {}
    for key in _STAGE_NAMES:
        upsampled[key] = cv2.resize(
            diff_maps[key], target_size, interpolation=cv2.INTER_LINEAR,
        )
    return upsampled


def render_heatmap_panel(
    query_img: np.ndarray,
    template_img: np.ndarray,
    diff_maps_upsampled: dict[str, np.ndarray],
    output_path: Path,
    sample_id: str = "",
    colormap: int = cv2.COLORMAP_INFERNO,
) -> None:
    """Render 5-column panel: query | template | diff_C3 | diff_C4 | diff_C5.

    Query/template are shown as RGB, diff columns as colored heatmaps.
    """
    h, w = template_img.shape[:2]
    gap = 4
    ncols = 5
    title_h = 36
    canvas = np.full((h + title_h, w * ncols + gap * (ncols - 1), 3), 248, dtype=np.uint8)

    # Query and template panels
    canvas[:h, :w] = query_img
    canvas[:h, w + gap:2 * w + gap] = template_img

    # Diff heatmap panels
    norm_max = max(float(dm.max()) for dm in diff_maps_upsampled.values()) or 1.0
    for index, key in enumerate(_STAGE_NAMES, start=2):
        dm = diff_maps_upsampled[key]
        normalized = np.clip(dm / norm_max, 0, 1)
        colored = cv2.applyColorMap(
            (normalized * 255).astype(np.uint8), colormap,
        )
        # cv2.applyColorMap returns BGR
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        x0 = index * (w + gap)
        canvas[:h, x0:x0 + w] = colored

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = ["QUERY", "TEMPLATE", "DIFF C3", "DIFF C4", "DIFF C5"]
    for index, label in enumerate(labels):
        x = index * (w + gap)
        cv2.putText(canvas, label, (x + 4, h + 22), font, 0.48, (44, 44, 44), 1, cv2.LINE_AA)
    if sample_id:
        cv2.putText(canvas, sample_id, (8, h + title_h - 4), font, 0.38, (128, 128, 128), 1, cv2.LINE_AA)

    _imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def diff_statistics(
    diff_maps: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Compute per-scale min, max, mean, std of diff maps."""
    stats: dict[str, dict[str, float]] = {}
    for key in _STAGE_NAMES:
        dm = diff_maps[key]
        stats[key] = {
            "min": float(dm.min()),
            "max": float(dm.max()),
            "mean": float(dm.mean()),
            "std": float(dm.std()),
        }
    return stats
