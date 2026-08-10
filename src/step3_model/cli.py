"""Step 3 CLI — SIFT+RANSAC macro alignment validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_alignment import load_config, run_alignment_validation


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3 SIFT+RANSAC macro alignment")
    parser.add_argument("command", choices=("align",))
    parser.add_argument("--config", type=Path, default=Path("configs/step3_alignment.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    if args.command == "align":
        result = run_alignment_validation(config)

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
