from __future__ import annotations

import argparse
import json
from pathlib import Path

from step1_audit.paths import write_json

from .index import load_config
from .validate import run_probe, run_validation
from .visualize import render_visualizations


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 2 paired Query-Template dataset pipeline")
    parser.add_argument("command", choices=("validate", "probe", "visualize", "all"))
    parser.add_argument("--config", type=Path, default=Path("configs/step2_dataset.yaml"))
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "validate":
        result = run_validation(config)
    elif args.command == "probe":
        result = run_probe(config)
    elif args.command == "visualize":
        result = render_visualizations(config, args.split, args.limit)
    else:
        validation = run_validation(config)
        visualizations = render_visualizations(config, args.split, args.limit)
        result = {"validation": validation, "visualizations": visualizations}
    output_root = Path(config["_output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "last_command.json", {"command": args.command, "result": result})
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
