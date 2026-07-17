from __future__ import annotations

import argparse
import json
from pathlib import Path

from .report import load_config, run_audit, run_split, run_verify


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Step 1 dataset audit and external indexing")
    parser.add_argument("command", choices=("audit", "split", "verify", "all"))
    parser.add_argument("--config", type=Path, default=Path("configs/step1_dataset.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "audit":
        result = run_audit(config, dry_run=args.dry_run)
    elif args.command == "split":
        if args.dry_run:
            result = {"command": "split", "output_root": config["_output_root"], "dry_run": True}
        else:
            result = run_split(config)
    elif args.command == "verify":
        if args.dry_run:
            result = {"command": "verify", "dataset_root": config["_dataset_root"], "dry_run": True}
        else:
            result = run_verify(config)
    else:
        if args.dry_run:
            result = run_audit(config, dry_run=True) | {"commands": ["audit", "split", "verify"]}
        else:
            audit = run_audit(config)
            split = run_split(config)
            verify = run_verify(config)
            result = {"audit": audit, "split": split, "verify": verify}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
