"""Step 3 CLI for Validation freezing and frozen Test evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_alignment import load_config, run_validation_freeze
from .test_evaluation import load_test_config, run_test_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3 macro alignment evaluation")
    parser.add_argument("command", choices=("validate-freeze", "test-evaluate"))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.command == "validate-freeze":
        config = load_config(args.config or Path("configs/step3_alignment.yaml"))
        result = run_validation_freeze(config)
    else:
        config = load_test_config(args.config or Path("configs/step3_alignment_test.yaml"))
        result = run_test_evaluation(config)
    print(json.dumps({
        "schema_version": result["schema_version"], "scope": result["scope"],
        "stages": result["stages"], "decisions": result["decisions"],
        "parameters_sha256": result["parameters_sha256"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
