"""Sweep expand_k on the validation split and summarize end-to-end metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from step1_audit.paths import write_json
from step3_model.end_to_end_eval import load_e2e_config, run_e2e_eval

VALUES = [1.8, 2.5, 3.0, 3.5, 4.0, 4.5]


def main() -> None:
    config = load_e2e_config(Path("configs/step3_end_to_end_eval.yaml"))
    rows = []
    for expand_k in VALUES:
        report = run_e2e_eval(config, split="val", expand_k=expand_k)
        stats = report["detection_statistics"]
        row = {
            "expand_k": expand_k,
            "precision": stats["micro_precision"],
            "recall": stats["micro_recall"],
            "f1": stats["micro_f1"],
            "total_predictions": stats["total_predictions"],
            "total_gt_boxes": stats["total_gt_boxes"],
            "per_class_f1": {
                item["class_name"]: item["f1"]
                for item in stats["per_class"]
            },
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    output = ROOT / "artifacts/analysis/expand_k_sweep.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"split": "val", "results": rows})
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
