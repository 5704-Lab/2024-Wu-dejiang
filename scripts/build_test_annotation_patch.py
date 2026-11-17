"""Build a test-only annotation patch from human-reviewed false positives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "artifacts/analysis/false_positives/test/false_positive_summary.json"
STEP2_CONFIG = ROOT / "configs/step2_dataset.yaml"
OUTPUT = ROOT / "artifacts/step1/test_annotation_patch.jsonl"


def _iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build test annotation patch from FP summary")
    parser.add_argument("--classes", nargs="+", default=["dirty", "double"])
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=["0364_s1_test", "0364_test"],
        help="Sample IDs that must not be patched",
    )
    parser.add_argument("--dedup-iou", type=float, default=0.9)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--merge", action="store_true", help="Merge with existing patch file instead of overwriting")
    args = parser.parse_args()

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    step2 = yaml.safe_load(STEP2_CONFIG.read_text(encoding="utf-8"))
    class_names = list(step2["classes"])
    allowed = set(args.classes)
    excluded = set(args.exclude)

    patch_rows: list[dict] = []
    if args.merge and args.output.is_file():
        patch_rows = [
            json.loads(line)
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    for row in summary["false_positive_rows"]:
        if row["class_name"] not in allowed:
            continue
        sample_id = row["sample_id"]
        if sample_id in excluded:
            continue
        box = [max(0, int(round(float(v)))) for v in row["box"]]
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        duplicate = any(
            item["sample_id"] == sample_id
            and item["class_name"] == row["class_name"]
            and _iou(item["bbox_xyxy"], box) >= float(args.dedup_iou)
            for item in patch_rows
        )
        if duplicate:
            continue
        patch_rows.append({
            "sample_id": sample_id,
            "class_name": row["class_name"],
            "class_id": class_names.index(row["class_name"]),
            "bbox_xyxy": box,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in patch_rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "patched_box_count": len(patch_rows),
        "class_counts": {
            name: sum(1 for row in patch_rows if row["class_name"] == name)
            for name in sorted(allowed)
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
