"""Step 3 CLI for validation-only alignment parameter freezing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_alignment import load_config, run_validation_freeze


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3 macro alignment validation")
    parser.add_argument("command", choices=("validate-freeze",))
    parser.add_argument("--config", type=Path, default=Path("configs/step3_alignment.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_validation_freeze(config)
    print(json.dumps({
        "schema_version": result["schema_version"],
        "scope": result["scope"],
        "stages": result["stages"],
        "decisions": result["decisions"],
        "parameters_sha256": result["parameters_sha256"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
