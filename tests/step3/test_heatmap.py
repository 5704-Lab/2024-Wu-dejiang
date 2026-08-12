import numpy as np
import pytest
import torch

from step3_model.heatmap import (
    compute_diff_maps,
    diff_statistics,
    render_heatmap_panel,
    upsample_diff_maps,
)
from step3_model.offset_predictor import OFFSET_SPEC


def _dummy_features(batch=1, height=64, width=64, seed=42):
    gen = torch.Generator().manual_seed(seed)
    return {
        name: torch.rand(batch, channels, height // stride, width // stride, generator=gen)
        for name, channels, stride in OFFSET_SPEC
    }


def test_compute_diff_maps_shapes():
    features = _dummy_features()
    diff = compute_diff_maps(features, features)
    assert list(diff) == ["C3", "C4", "C5"]
    assert diff["C3"].shape == (8, 8)  # 64//8
    assert diff["C4"].shape == (4, 4)  # 64//16
    assert diff["C5"].shape == (2, 2)  # 64//32


def test_compute_diff_maps_zero():
    """Identical features → zero diff at all scales."""
    features = _dummy_features()
    diff = compute_diff_maps(features, features)
    for key in ["C3", "C4", "C5"]:
        assert diff[key].max() == 0.0


def test_compute_diff_maps_nonzero():
    """Different features → positive diff."""
    qf = _dummy_features(seed=1)
    warped = _dummy_features(seed=2)
    diff = compute_diff_maps(warped, qf)
    for key in ["C3", "C4", "C5"]:
        assert diff[key].max() > 0
        assert np.isfinite(diff[key]).all()


def test_upsample_diff_maps():
    features = _dummy_features()
    diff = compute_diff_maps(features, features)
    up = upsample_diff_maps(diff, target_size=(64, 64))
    for key in ["C3", "C4", "C5"]:
        assert up[key].shape == (64, 64)


def test_diff_statistics():
    qf1 = _dummy_features(seed=1)
    qf2 = _dummy_features(seed=2)
    diff = compute_diff_maps(qf1, qf2)
    stats = diff_statistics(diff)
    for key in ["C3", "C4", "C5"]:
        assert "min" in stats[key]
        assert "max" in stats[key]
        assert stats[key]["max"] > 0


def test_render_heatmap_panel_creates_file(tmp_path):
    query = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    template = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    diff = {"C3": np.random.rand(64, 64).astype(np.float32),
            "C4": np.random.rand(64, 64).astype(np.float32),
            "C5": np.random.rand(64, 64).astype(np.float32)}
    output = tmp_path / "panel.png"
    render_heatmap_panel(query, template, diff, output, sample_id="test")
    assert output.is_file()
    assert output.stat().st_size > 100
