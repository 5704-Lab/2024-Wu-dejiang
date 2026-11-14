"""Full Test-split evaluation entrypoint for the frozen Siamese pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from step3_model.test_eval import load_test_eval_config, run_test_eval


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3 Test-split frozen evaluation")
    parser.add_argument("--config", type=Path, default=Path("configs/step3_test_eval.yaml"))
    args = parser.parse_args()
    config = load_test_eval_config(args.config)
    report = run_test_eval(config)
    print(json.dumps({
        "schema_version": report["schema_version"],
        "processed_sample_count": report["scope"]["processed_sample_count"],
        "recall_statistics": report["recall_statistics"],
        "crop_statistics": report["crop_statistics"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
