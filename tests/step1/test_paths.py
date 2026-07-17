from pathlib import Path

import pytest

from step1_audit.paths import validate_roots


def test_rejects_output_inside_dataset(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    with pytest.raises(ValueError):
        validate_roots(dataset, dataset / "artifacts")


def test_allows_external_output(tmp_path: Path):
    dataset = tmp_path / "dataset"
    output = tmp_path / "workspace" / "artifacts"
    dataset.mkdir()
    validate_roots(dataset, output)
