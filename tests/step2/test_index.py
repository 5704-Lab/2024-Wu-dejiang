from pathlib import Path

import pytest

from step2_dataset.index import _resolve_source_path


def test_source_path_must_stay_inside_root(tmp_path: Path):
    root = tmp_path / "dataset"
    root.mkdir()
    image = root / "image.jpg"
    image.write_bytes(b"x")
    assert _resolve_source_path(root, "image.jpg") == image.resolve()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError):
        _resolve_source_path(root, "../outside.jpg")
