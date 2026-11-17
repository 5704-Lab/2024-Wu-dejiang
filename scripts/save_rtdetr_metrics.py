"""Save per-class RT-DETR val/test metrics to JSON files.

Ultralytics prints these metrics to stdout but does not persist them by default.
This script writes machine-readable JSON summaries for both the original and the
pair-aware RT-DETR baselines.
"""

from __future__ import annotations

import json
from pathlib import Path

from ultralytics import RTDETR

ROOT = Path(__file__).resolve().parents[1]

CASES = (
    {
        "name": "baseline",
        "data": ROOT / "artifacts/step5/rtdetr_dataset/dataset.yaml",
        "checkpoint": ROOT / "artifacts/step5/rtdetr_baseline/train/weights/best.pt",
        "out_dir": ROOT / "artifacts/step5/rtdetr_baseline",
        "splits": ("val", "test"),
    },
    {
        "name": "paired",
        "data": ROOT / "artifacts/step5/rtdetr_paired_dataset/dataset.yaml",
        "checkpoint": ROOT / "artifacts/step5/rtdetr_paired_baseline/train/weights/best.pt",
        "out_dir": ROOT / "artifacts/step5/rtdetr_paired_baseline",
        "splits": ("val", "test"),
    },
    {
        "name": "overfit",
        "data": ROOT / "artifacts/step5/rtdetr_overfit/dataset.yaml",
        "checkpoint": ROOT / "artifacts/step5/rtdetr_overfit/run/weights/best.pt",
        "out_dir": ROOT / "artifacts/step5/rtdetr_overfit",
        "splits": ("val",),
    },
)


def _number(value):
    try:
        return float(value)
    except Exception:
        return None


def main() -> None:
    for case in CASES:
        model = RTDETR(str(case["checkpoint"]))
        case["out_dir"].mkdir(parents=True, exist_ok=True)
        for split in case["splits"]:
            metrics = model.val(
                data=str(case["data"]),
                split=split,
                imgsz=800,
                batch=8,
                device=0,
            )
            box = metrics.box
            per_class = []
            for index, class_id in enumerate(box.ap_class_index):
                per_class.append({
                    "class_id": int(class_id),
                    "class_name": model.names[int(class_id)],
                    "images": _number(metrics.nt_per_image[int(class_id)]),
                    "instances": _number(metrics.nt_per_class[int(class_id)]),
                    "precision": _number(box.p[index]),
                    "recall": _number(box.r[index]),
                    "mAP50": _number(box.ap50[index]),
                    "mAP50_95": _number(box.ap[index]),
                })
            overall = {
                key: _number(value)
                for key, value in metrics.results_dict.items()
            }
            result = {
                "model": case["name"],
                "split": split,
                "checkpoint": str(case["checkpoint"]),
                "overall": overall,
                "per_class": per_class,
            }
            output = case["out_dir"] / f"{split}_metrics.json"
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"saved {output}")
            print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
