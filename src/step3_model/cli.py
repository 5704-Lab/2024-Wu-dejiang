"""Step 3 CLI for alignment, backbone, and offset predictor evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_alignment import load_config, run_validation_freeze
from .run_backbone import load_backbone_config, run_backbone_smoke
from .run_offset import load_offset_config, run_offset_smoke
from .test_evaluation import load_test_config, run_test_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3 alignment and model evaluation")
    parser.add_argument("command", choices=("validate-freeze", "test-evaluate", "backbone-smoke", "offset-smoke"))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.command == "validate-freeze":
        result = run_validation_freeze(load_config(args.config or Path("configs/step3_alignment.yaml")))
        summary = {key: result[key] for key in ("schema_version", "scope", "stages", "decisions", "parameters_sha256")}
    elif args.command == "test-evaluate":
        result = run_test_evaluation(load_test_config(args.config or Path("configs/step3_alignment_test.yaml")))
        summary = {key: result[key] for key in ("schema_version", "scope", "stages", "decisions", "parameters_sha256")}
    elif args.command == "backbone-smoke":
        result = run_backbone_smoke(load_backbone_config(args.config or Path("configs/step3_backbone.yaml")))
        summary = {
            "schema_version": result["schema_version"], "verified": result["verified"],
            "selected_device": result["selected_device"],
            "processed_sample_count": result["processed_sample_count"],
            "ordered_validation_sample_ids": result["ordered_validation_sample_ids"],
            "sharing": result["sharing"], "weights": result["weights"],
        }
    else:
        result = run_offset_smoke(load_offset_config(args.config or Path("configs/step3_offset.yaml")))
        summary = {
            "schema_version": result["schema_version"], "verified": result["verified"],
            "selected_device": result["selected_device"],
            "processed_sample_count": result["processed_sample_count"],
            "ordered_validation_sample_ids": result["ordered_validation_sample_ids"],
            "offset_max_abs_per_scale": result["offset_max_abs_per_scale"],
            "parameter_count": result["parameter_count"],
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())