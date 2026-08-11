import json
from pathlib import Path

import pytest

from step1_audit.hashing import sha256_file
from step3_model.test_evaluation import (
    _load_full_test_split,
    _normalized_text_sha256,
    _validate_test_contract,
)


def _test_config(tmp_path: Path):
    step1, step2, dataset = tmp_path / "step1", tmp_path / "step2", tmp_path / "dataset"
    (step1 / "splits").mkdir(parents=True)
    step2.mkdir()
    (dataset / "q").mkdir(parents=True)
    (dataset / "t").mkdir(parents=True)
    return {
        "roadmap_step": "3.2-D",
        "inputs": {"samples": "samples.jsonl", "splits_dir": "splits",
                   "split_manifest": "split_manifest.json", "split_summary": "split_summary.json",
                   "step2_index_manifest": "index_manifest.json"},
        "run": {"purpose": "frozen_test_evaluation", "split": "test", "selection": "full_split",
                "expected_sample_count": 2, "expected_group_count": 1, "tuning_allowed": False,
                "refreeze_allowed": False, "comparison_mode": "descriptive_only"},
        "_step1_root": str(step1), "_step2_root": str(step2), "_dataset_root": str(dataset),
    }


def _write_test_fixture(config):
    step1, step2, dataset = Path(config["_step1_root"]), Path(config["_step2_root"]), Path(config["_dataset_root"])
    records = []
    for sample_id in ("0002_a", "0002_b"):
        (dataset / "q" / f"{sample_id}.png").write_bytes(b"q")
        (dataset / "t" / f"{sample_id}.png").write_bytes(b"t")
        records.append({"sample_id": sample_id, "source_id": "0002",
                        "query_path": f"q/{sample_id}.png", "template_path": f"t/{sample_id}.png"})
    samples = step1 / "samples.jsonl"
    samples.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    test = step1 / "splits" / "test.txt"
    test.write_text("0002_a\n0002_b\n", encoding="utf-8")
    samples_hash, test_hash = sha256_file(samples), sha256_file(test)
    (step1 / "split_manifest.json").write_text(json.dumps({
        "groups": {"train": ["0000"], "val": ["0001"], "test": ["0002"]},
        "samples_digest": samples_hash,
    }), encoding="utf-8")
    (step1 / "split_summary.json").write_text(json.dumps({
        "test": {"sample_count": 2, "group_count": 1, "object_count": 2},
    }), encoding="utf-8")
    (step2 / "index_manifest.json").write_text(json.dumps({
        "step1_samples_sha256": samples_hash, "split_sha256": {"test": test_hash},
    }), encoding="utf-8")


def test_test_contract_rejects_algorithm_overrides(tmp_path: Path):
    config = _test_config(tmp_path)
    _validate_test_contract(config)
    config["ransac"] = {"min_inliers": 1}
    with pytest.raises(ValueError, match="must not override"):
        _validate_test_contract(config)


def test_test_contract_rejects_tuning_or_refreeze(tmp_path: Path):
    config = _test_config(tmp_path)
    config["run"]["tuning_allowed"] = True
    with pytest.raises(ValueError, match="run contract"):
        _validate_test_contract(config)


def test_normalized_evidence_hash_accepts_crlf(tmp_path: Path):
    lf, crlf = tmp_path / "lf.json", tmp_path / "crlf.json"
    lf.write_bytes(b'{"x": 1}\n')
    crlf.write_bytes(b'{"x": 1}\r\n')
    assert _normalized_text_sha256(lf) == _normalized_text_sha256(crlf)
    bare = tmp_path / "bare.json"
    bare.write_bytes(b'{"x":\r1}')
    with pytest.raises(ValueError, match="bare carriage"):
        _normalized_text_sha256(bare)


def test_full_test_loader_preserves_order_and_avoids_other_split_files(tmp_path: Path, monkeypatch):
    config = _test_config(tmp_path)
    _write_test_fixture(config)
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        if path.name in {"train.txt", "val.txt"}:
            raise AssertionError("Non-Test split file accessed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    entries, provenance = _load_full_test_split(config)
    assert [entry["sample_id"] for entry in entries] == ["0002_a", "0002_b"]
    assert {entry["split"] for entry in entries} == {"test"}
    assert provenance["test_sample_count"] == 2


def test_full_test_loader_rejects_group_overlap(tmp_path: Path):
    config = _test_config(tmp_path)
    _write_test_fixture(config)
    path = Path(config["_step1_root"]) / "split_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["groups"]["val"] = ["0002"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        _load_full_test_split(config)
