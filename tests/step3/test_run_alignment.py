import json
from pathlib import Path

import pytest

from step1_audit.paths import validate_roots
from step3_model.alignment import AlignmentResult
from step3_model.run_alignment import (
    _canonical_digest,
    _distribution,
    _effective_parameters,
    _load_full_validation_split,
    _resolve,
    _validate_run_contract,
    _validation_report,
)


def _config(tmp_path: Path, expected_count: int = 2, expected_groups: int = 1):
    step1 = tmp_path / "step1"
    step2 = tmp_path / "step2"
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    (step1 / "splits").mkdir(parents=True)
    step2.mkdir()
    (dataset / "query").mkdir(parents=True)
    (dataset / "template").mkdir(parents=True)
    return {
        "roadmap_step": "3.2-C",
        "inputs": {
            "samples": "samples.jsonl",
            "splits_dir": "splits",
            "split_manifest": "split_manifest.json",
            "split_summary": "split_summary.json",
            "step2_index_manifest": "index_manifest.json",
        },
        "run": {
            "purpose": "validation_freeze",
            "split": "val",
            "selection": "full_split",
            "expected_sample_count": expected_count,
            "expected_group_count": expected_groups,
        },
        "_step1_root": str(step1),
        "_step2_root": str(step2),
        "_dataset_root": str(dataset),
        "_output_root": str(output),
        "sift": {}, "matcher": {}, "ransac": {}, "geometry": {},
        "near_identity": {}, "quality": {}, "diagnostics": {},
    }


def _write_fixture(config):
    from step1_audit.hashing import sha256_file
    step1 = Path(config["_step1_root"])
    step2 = Path(config["_step2_root"])
    dataset = Path(config["_dataset_root"])
    records = []
    for sample_id in ("0001_a", "0001_b"):
        query = dataset / "query" / f"{sample_id}.png"
        template = dataset / "template" / f"{sample_id}.png"
        query.write_bytes(b"query")
        template.write_bytes(b"template")
        records.append({
            "sample_id": sample_id,
            "source_id": "0001",
            "query_path": f"query/{sample_id}.png",
            "template_path": f"template/{sample_id}.png",
        })
    samples_path = step1 / "samples.jsonl"
    samples_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
    val_path = step1 / "splits" / "val.txt"
    val_path.write_text("0001_a\n0001_b\n", encoding="utf-8")
    samples_hash, val_hash = sha256_file(samples_path), sha256_file(val_path)
    (step1 / "split_manifest.json").write_text(json.dumps({
        "groups": {"val": ["0001"]}, "samples_digest": samples_hash,
    }), encoding="utf-8")
    (step1 / "split_summary.json").write_text(json.dumps({
        "val": {"sample_count": 2, "group_count": 1},
    }), encoding="utf-8")
    (step2 / "index_manifest.json").write_text(json.dumps({
        "step1_samples_sha256": samples_hash,
        "split_sha256": {"val": val_hash},
    }), encoding="utf-8")


def test_run_contract_rejects_test_or_sampling(tmp_path: Path):
    config = _config(tmp_path)
    _validate_run_contract(config)
    config["run"]["split"] = "test"
    with pytest.raises(ValueError, match="run contract"):
        _validate_run_contract(config)
    config["run"]["split"] = "val"
    config["run"]["selection"] = "sample"
    with pytest.raises(ValueError, match="run contract"):
        _validate_run_contract(config)


def test_full_validation_loader_preserves_order_and_reads_only_val(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    _write_fixture(config)
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        if path.name == "test.txt":
            raise AssertionError("Test split must not be read")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    entries, provenance = _load_full_validation_split(config)
    assert [entry["sample_id"] for entry in entries] == ["0001_a", "0001_b"]
    assert {entry["split"] for entry in entries} == {"val"}
    assert provenance["validation_sample_count"] == 2
    assert "test" not in json.dumps(provenance).lower()


def test_full_validation_loader_rejects_duplicates(tmp_path: Path):
    config = _config(tmp_path)
    _write_fixture(config)
    val_path = Path(config["_step1_root"]) / "splits" / "val.txt"
    val_path.write_text("0001_a\n0001_a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        _load_full_validation_split(config)


def test_distribution_handles_missing_values_and_is_strict_json():
    distribution = _distribution([1.0, 2.0, None, float("nan")])
    assert distribution["count"] == 2
    assert distribution["missing_count"] == 2
    assert distribution["quantiles"]["p50"] == pytest.approx(1.5)
    json.dumps(distribution, allow_nan=False)


def test_parameter_digest_ignores_diagnostics_and_changes_with_algorithm():
    first = {section: {} for section in ("sift", "matcher", "ransac", "geometry", "near_identity", "quality")}
    first["diagnostics"] = {"max_samples": 1}
    second = dict(reversed(list(first.items())))
    second["diagnostics"] = {"max_samples": 20}
    assert _canonical_digest(_effective_parameters(first)) == _canonical_digest(_effective_parameters(second))
    second["ransac"] = {"min_inlier_ratio": 0.6}
    assert _canonical_digest(_effective_parameters(first)) != _canonical_digest(_effective_parameters(second))


def test_validation_report_is_validation_only_and_complete():
    result = AlignmentResult(
        sample_id="sample-1", success=True, homography_estimated=True,
        geometry_accepted=True, already_aligned=True, status="already_aligned",
        reasons=["near_identity_transform"], ssim_original=1.0, ssim_selected=1.0,
        mutual_info_original=2.0, mutual_info_selected=2.0,
    )
    config = {
        "run": {"expected_sample_count": 1, "expected_group_count": 1},
        **{section: {} for section in ("sift", "matcher", "ransac", "geometry", "near_identity", "quality")},
    }
    report = _validation_report([result], config, {}, "digest", [])
    assert report["schema_version"] == "step3-alignment-validation-v1"
    assert report["scope"]["split"] == "val"
    assert report["scope"]["complete"] is True
    assert report["scope"]["test_accessed"] is False
    assert len(report["samples"]) == 1
    json.dumps(report, allow_nan=False)


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
