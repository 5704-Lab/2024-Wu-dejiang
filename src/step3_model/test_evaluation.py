"""Frozen-parameter external Test evaluation for Step 3.2-D."""

from __future__ import annotations

import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from step1_audit.hashing import sha256_file, sha256_text, snapshot_digest, snapshot_tree
from step1_audit.paths import read_jsonl, resolved, validate_roots, write_json

from .alignment import AlignmentResult, SIFTAligner, load_image_rgb, render_comparison
from .run_alignment import _canonical_digest, _distributions, _stages

_PARAMETER_SECTIONS = {"sift", "matcher", "ransac", "geometry", "near_identity", "quality"}


def load_test_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    workspace = config_path.resolve().parent.parent
    config["_config_path"] = str(config_path.resolve())
    config["_step1_root"] = str(resolved(config["inputs"]["step1_root"], workspace))
    config["_step2_root"] = str(resolved(config["inputs"]["step2_root"], workspace))
    config["_dataset_root"] = str(resolved(config["inputs"]["dataset_root"], workspace))
    config["_output_root"] = str(resolved(config["output"]["root"], workspace))
    for key in ("frozen_parameters", "validation_report", "validation_manifest"):
        config[f"_{key}"] = str(resolved(config["inputs"][key], workspace))
    return config


def _normalized_text_sha256(path: Path) -> str:
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Evidence is not valid UTF-8: {path}") from exc
    if "\r" in text.replace("\r\n", ""):
        raise ValueError(f"Evidence contains a bare carriage return: {path}")
    return sha256_text(text.replace("\r\n", "\n"))


def _validate_test_contract(config: dict[str, Any]) -> None:
    overrides = sorted(set(config) & _PARAMETER_SECTIONS)
    if overrides:
        raise ValueError(f"Test config must not override frozen parameters: {overrides}")
    run = config.get("run", {})
    actual = (
        config.get("roadmap_step"), run.get("purpose"), run.get("split"),
        run.get("selection"), run.get("tuning_allowed"), run.get("refreeze_allowed"),
        run.get("comparison_mode"),
    )
    required = ("3.2-D", "frozen_test_evaluation", "test", "full_split", False, False, "descriptive_only")
    if actual != required:
        raise ValueError(f"Invalid Step 3.2-D run contract: {actual}")
    if int(run.get("expected_sample_count", 0)) <= 0 or int(run.get("expected_group_count", 0)) <= 0:
        raise ValueError("Test expected counts must be positive")


def _load_frozen_evidence(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_test_contract(config)
    paths = {
        "validation_report_sha256": Path(config["_validation_report"]),
        "frozen_parameters_sha256": Path(config["_frozen_parameters"]),
        "validation_manifest_sha256": Path(config["_validation_manifest"]),
    }
    pins = config["frozen_evidence"]
    hashes = {name: _normalized_text_sha256(path) for name, path in paths.items()}
    for name, value in hashes.items():
        if value != pins[name]:
            raise RuntimeError(f"Frozen evidence hash mismatch: {name}")
    validation = json.loads(paths["validation_report_sha256"].read_text(encoding="utf-8"))
    freeze = json.loads(paths["frozen_parameters_sha256"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["validation_manifest_sha256"].read_text(encoding="utf-8"))
    digest = pins["parameters_sha256"]
    if manifest.get("schema_version") != "step3-run-manifest-v1" or manifest.get("roadmap_step") != "3.2-C":
        raise RuntimeError("Invalid Validation manifest")
    if manifest.get("command") != "validate-freeze" or manifest.get("source_read_only_verified") is not True:
        raise RuntimeError("Validation manifest is not a verified freeze run")
    if manifest.get("artifacts") != {
        "alignment_report.json": pins["validation_report_sha256"],
        "alignment_parameters_frozen.json": pins["frozen_parameters_sha256"],
    } or manifest.get("inputs", {}).get("parameters_sha256") != digest:
        raise RuntimeError("Validation manifest references differ from pinned evidence")
    scope = validation.get("scope", {})
    if validation.get("schema_version") != "step3-alignment-validation-v1" or scope.get("split") != "val":
        raise RuntimeError("Invalid Validation report")
    if not scope.get("complete") or scope.get("processed_sample_count") != 400 or scope.get("test_accessed") is not False:
        raise RuntimeError("Validation report is incomplete or accessed Test")
    parameters = freeze.get("parameters", {})
    if freeze.get("schema_version") != "step3-alignment-freeze-v1" or freeze.get("status") != "frozen":
        raise RuntimeError("Invalid frozen parameter artifact")
    if freeze.get("selection", {}).get("test_used_for_tuning") is not False:
        raise RuntimeError("Frozen parameters used Test for tuning")
    if freeze.get("derived_from", {}).get("validation_report_sha256") != pins["validation_report_sha256"]:
        raise RuntimeError("Freeze does not reference the pinned Validation report")
    if set(parameters) != _PARAMETER_SECTIONS:
        raise RuntimeError("Frozen parameter sections are incomplete")
    if freeze.get("parameters_sha256") != digest or _canonical_digest(parameters) != digest:
        raise RuntimeError("Frozen parameter digest mismatch")
    if validation.get("parameters_sha256") != digest or validation.get("effective_parameters") != parameters:
        raise RuntimeError("Validation report and freeze parameters differ")
    return json.loads(json.dumps(parameters)), {
        "validation": validation,
        "hashes": hashes,
        "parameters_sha256": digest,
        "integrity_verified": True,
    }


def _resolve_source(dataset_root: Path, relative: str) -> Path:
    path = (dataset_root / relative).resolve()
    try:
        path.relative_to(dataset_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Source path escapes dataset root: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Missing source file: {path}")
    return path


def _load_full_test_split(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_test_contract(config)
    step1, step2, dataset = Path(config["_step1_root"]), Path(config["_step2_root"]), Path(config["_dataset_root"])
    inputs = config["inputs"]
    samples_path = step1 / inputs.get("samples", "samples.jsonl")
    test_path = step1 / inputs.get("splits_dir", "splits") / "test.txt"
    manifest_path = step1 / inputs.get("split_manifest", "split_manifest.json")
    summary_path = step1 / inputs.get("split_summary", "split_summary.json")
    step2_path = step2 / inputs.get("step2_index_manifest", "index_manifest.json")
    records = read_jsonl(samples_path)
    by_id = {record["sample_id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate records in samples.jsonl")
    ids = [line.strip() for line in test_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = int(config["run"]["expected_sample_count"])
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError(f"Test split must contain exactly {expected} unique IDs")
    unknown = [sample_id for sample_id in ids if sample_id not in by_id]
    if unknown:
        raise ValueError(f"Unknown Test IDs: {unknown[:5]}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = {split: set(values) for split, values in manifest["groups"].items()}
    if groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]:
        raise ValueError("Source groups overlap across splits")
    expected_groups = int(config["run"]["expected_group_count"])
    represented = {by_id[sample_id]["source_id"] for sample_id in ids}
    if len(groups["test"]) != expected_groups or represented != groups["test"]:
        raise ValueError("Test IDs do not cover exactly the expected Test groups")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["test"] != {**summary["test"], "sample_count": expected, "group_count": expected_groups}:
        raise ValueError("Step 1 Test summary differs from the run contract")
    samples_hash, test_hash = sha256_file(samples_path), sha256_file(test_path)
    index = json.loads(step2_path.read_text(encoding="utf-8"))
    if manifest.get("samples_digest") != samples_hash or index.get("step1_samples_sha256") != samples_hash:
        raise RuntimeError("Step 1/Step 2 samples provenance mismatch")
    if index.get("split_sha256", {}).get("test") != test_hash:
        raise RuntimeError("Step 2 Test split hash mismatch")
    entries = [{
        "sample_id": sample_id, "split": "test", "source_id": by_id[sample_id]["source_id"],
        "query_path": _resolve_source(dataset, by_id[sample_id]["query_path"]),
        "template_path": _resolve_source(dataset, by_id[sample_id]["template_path"]),
    } for sample_id in ids]
    return entries, {
        "step1_samples_sha256": samples_hash,
        "step1_test_split_sha256": test_hash,
        "step1_split_manifest_sha256": sha256_file(manifest_path),
        "step1_split_summary_sha256": sha256_file(summary_path),
        "step2_test_split_sha256": index["split_sha256"]["test"],
        "ordered_test_ids_sha256": sha256_text("\n".join(ids) + "\n"),
        "test_sample_count": expected, "test_group_count": expected_groups,
    }


def _sample_report(result: AlignmentResult) -> dict[str, Any]:
    data = dict(vars(result))
    data.pop("inlier_match_indices", None)
    data.pop("valid_mask", None)
    data.pop("ssim_valid_mask", None)
    data["split"] = "test"
    data["ssim_candidate_delta"] = result.ssim_candidate_delta
    data["mutual_info_candidate_delta"] = result.mutual_info_candidate_delta
    return data


def _diagnostics(config: dict[str, Any], entries: list[dict[str, Any]], results: list[AlignmentResult],
                 aligner: SIFTAligner) -> list[dict[str, Any]]:
    priority = {"homography_not_estimated": 0, "geometry_rejected": 1, "original_retained": 2,
                "warped_selected": 3, "already_aligned": 4}
    limit = int(config.get("diagnostics", {}).get("max_samples", 0))
    selected = sorted(results, key=lambda r: (priority.get(r.status, 5), -(r.max_inlier_reprojection_error or 0), r.sample_id))[:limit]
    by_id = {entry["sample_id"]: entry for entry in entries}
    root = Path(config["_output_root"]).parent / "alignment" / "test"
    records = []
    for result in selected:
        entry = by_id[result.sample_id]
        query, template = load_image_rgb(entry["query_path"]), load_image_rgb(entry["template_path"])
        selected_image, rerun = aligner.align(query, template, result.sample_id)
        if rerun.status != result.status or rerun.homography_matrix != result.homography_matrix:
            raise RuntimeError(f"Non-deterministic diagnostic rerun: {result.sample_id}")
        paths = []
        if config.get("diagnostics", {}).get("render_comparison", True):
            path = root / f"{result.sample_id}_comparison.png"
            render_comparison(query, selected_image, template, rerun, path)
            paths.append(path.as_posix())
        if config.get("diagnostics", {}).get("render_matches", True):
            path = root / f"{result.sample_id}_matches.png"
            aligner.render_matches(query, template, rerun, path)
            paths.append(path.as_posix())
        records.append({"sample_id": result.sample_id, "status": result.status,
                        "rank_metric": "max_inlier_reprojection_error",
                        "rank_value": result.max_inlier_reprojection_error, "paths": paths})
    return records


def _test_report(results: list[AlignmentResult], config: dict[str, Any], provenance: dict[str, Any],
                 source_digest: str, diagnostics: list[dict[str, Any]], frozen: dict[str, Any],
                 parameters: dict[str, Any]) -> dict[str, Any]:
    statuses = Counter(result.status for result in results)
    reasons = Counter(reason for result in results for reason in result.reasons)
    failures = Counter(name for result in results for name, criterion in result.criteria.items()
                       if criterion.get("passed") is False)
    return {
        "schema_version": "step3-alignment-test-v1", "roadmap_step": "3.2-D",
        "run_type": "external_test_full_split", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"split": "test", "selection": "full_split", "expected_sample_count": 400,
                  "processed_sample_count": len(results), "unique_sample_count": len({r.sample_id for r in results}),
                  "group_count": 40, "complete": True, "external_test_evaluation": True,
                  "parameters_loaded_from_freeze": True, "tuning_performed": False, "parameters_refrozen": False},
        "provenance": {**provenance, "source_snapshot_digest": source_digest},
        "frozen_parameters": {**frozen["hashes"], "parameters_sha256": frozen["parameters_sha256"],
                              "integrity_verified": True, "test_used_for_freezing": False},
        "effective_parameters": parameters, "parameters_sha256": frozen["parameters_sha256"],
        "stages": _stages(results),
        "decisions": {"selected_original_count": sum(r.selected_variant == "original" for r in results),
                      "selected_warped_count": sum(r.selected_variant == "warped" for r in results),
                      "status_counts": dict(sorted(statuses.items()))},
        "reason_counts": dict(sorted(reasons.items())),
        "criterion_failure_counts": dict(sorted(failures.items())),
        "distributions": _distributions(results),
        "distributions_by_status": {status: _distributions([r for r in results if r.status == status])
                                    for status in sorted(statuses)},
        "diagnostics": diagnostics, "samples": [_sample_report(result) for result in results],
        "failures": [{"sample_id": r.sample_id, "status": r.status, "error": r.error or None,
                      "reasons": r.reasons} for r in results if not r.geometry_accepted],
    }


def _compare_counts(validation: dict[str, int], test: dict[str, int]) -> dict[str, Any]:
    return {key: {"validation_count": validation.get(key, 0), "test_count": test.get(key, 0),
                  "validation_rate": validation.get(key, 0) / 400, "test_rate": test.get(key, 0) / 400,
                  "test_minus_validation_count": test.get(key, 0) - validation.get(key, 0),
                  "test_minus_validation_rate_percentage_points": (test.get(key, 0) - validation.get(key, 0)) / 4}
            for key in sorted(set(validation) | set(test))}


def _comparison(validation: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    differences = {}
    for metric in sorted(set(validation["distributions"]) | set(test["distributions"])):
        val, tst = validation["distributions"].get(metric, {}), test["distributions"].get(metric, {})
        differences[metric] = {"validation": val, "test": tst,
                               "test_minus_validation_mean": (tst.get("mean") - val.get("mean"))
                               if tst.get("mean") is not None and val.get("mean") is not None else None}
    return {
        "schema_version": "step3-alignment-val-test-comparison-v1", "roadmap_step": "3.2-D",
        "comparison_type": "descriptive_only", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {"parameter_selection_performed": False, "threshold_selection_performed": False,
                   "test_result_used_for_tuning": False, "parameters_refrozen": False,
                   "acceptance_gate_applied": False, "next_step_started": False},
        "parameters_sha256": test["parameters_sha256"],
        "status": _compare_counts(validation["decisions"]["status_counts"], test["decisions"]["status_counts"]),
        "reasons": _compare_counts(validation["reason_counts"], test["reason_counts"]),
        "criterion_failures": _compare_counts(validation["criterion_failure_counts"], test["criterion_failure_counts"]),
        "distributions": differences,
    }


def run_test_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    _validate_test_contract(config)
    dataset, output = Path(config["_dataset_root"]), Path(config["_output_root"])
    validate_roots(dataset, output)
    if output.resolve() == Path(config["_frozen_parameters"]).parent.resolve():
        raise ValueError("Test output must not overwrite Step 3.2-C artifacts")
    parameters, frozen = _load_frozen_evidence(config)
    evidence_paths = [Path(config["_validation_report"]), Path(config["_frozen_parameters"]), Path(config["_validation_manifest"])]
    evidence_before = [_normalized_text_sha256(path) for path in evidence_paths]
    before = snapshot_tree(dataset)
    source_digest = snapshot_digest(before)
    entries, provenance = _load_full_test_split(config)
    aligner = SIFTAligner(parameters)
    results = []
    for position, entry in enumerate(entries, 1):
        query, template = load_image_rgb(entry["query_path"]), load_image_rgb(entry["template_path"])
        _, result = aligner.align(query, template, entry["sample_id"])
        results.append(result)
        if position == 1 or position % 25 == 0 or position == len(entries):
            print(f"  Test {position:3d}/{len(entries)}: {entry['sample_id']} -> {result.status}")
    if [r.sample_id for r in results] != [e["sample_id"] for e in entries]:
        raise RuntimeError("Test result order differs from test.txt")
    diagnostic_records = _diagnostics(config, entries, results, aligner)
    if before != snapshot_tree(dataset):
        raise RuntimeError("Read-only dataset changed during Step 3.2-D")
    if evidence_before != [_normalized_text_sha256(path) for path in evidence_paths]:
        raise RuntimeError("Step 3.2-C evidence changed during Test evaluation")
    report = _test_report(results, config, provenance, source_digest, diagnostic_records, frozen, parameters)
    comparison = _comparison(frozen["validation"], report)
    json.dumps(report, allow_nan=False); json.dumps(comparison, allow_nan=False)
    output.mkdir(parents=True, exist_ok=True)
    report_path, comparison_path = output / "alignment_report.json", output / "val_test_comparison.json"
    write_json(report_path, report); write_json(comparison_path, comparison)
    manifest = {
        "schema_version": "step3-test-run-manifest-v1", "roadmap_step": "3.2-D", "command": "test-evaluate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
                     "pyyaml": yaml.__version__, "platform": platform.platform()},
        "scope": {"split": "test", "sample_count": 400, "group_count": 40, "complete": True},
        "inputs": {**provenance, **frozen["hashes"], "parameters_sha256": frozen["parameters_sha256"],
                   "source_snapshot_digest": source_digest, "config_sha256": sha256_file(Path(config["_config_path"]))},
        "artifacts": {"alignment_report.json": sha256_file(report_path),
                      "val_test_comparison.json": sha256_file(comparison_path)},
        "controls": {"source_read_only_verified": True, "frozen_evidence_integrity_verified": True,
                     "parameters_loaded_exactly": True, "tuning_performed": False,
                     "parameter_freeze_written": False, "validation_recomputed": False,
                     "step_3_3_a_started": False},
    }
    write_json(output / "run_manifest.json", manifest)
    return report
