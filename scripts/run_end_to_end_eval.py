"""End-to-end Siamese + RT-DETR evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from step3_model.end_to_end_eval import load_e2e_config, run_e2e_eval


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end Siamese + RT-DETR evaluation")
    parser.add_argument("--config", type=Path, default=Path("configs/step3_end_to_end_eval.yaml"))
    parser.add_argument("--split", choices=("val", "test"))
    args = parser.parse_args()
    config = load_e2e_config(args.config)
    report = run_e2e_eval(config, split=args.split)
    print(json.dumps({
        "split": report["split"],
        "processed_sample_count": report["processed_sample_count"],
        "detection_statistics": report["detection_statistics"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
