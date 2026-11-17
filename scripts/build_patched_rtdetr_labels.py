"""Append reviewed test annotation patches to the paired RT-DETR label files."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "artifacts/step1/test_annotation_patch.jsonl"
LABEL_DIR = ROOT / "artifacts/step5/rtdetr_paired_dataset/labels/test"
IMAGE_SIZE = 800


def main() -> None:
    parser = argparse.ArgumentParser(description="Append patched test labels to paired RT-DETR dataset")
    parser.add_argument("--classes", nargs="+", default=None)
    args = parser.parse_args()
    selected_classes = set(args.classes) if args.classes else None
    if not PATCH.is_file():
        raise FileNotFoundError(f"缺少标注补丁文件：{PATCH}")
    patched = 0
    for line in PATCH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if selected_classes is not None and row["class_name"] not in selected_classes:
            continue
        sample_id = row["sample_id"]
        class_id = int(row["class_id"])
        x1, y1, x2, y2 = [float(v) for v in row["bbox_xyxy"]]
        cx = (x1 + x2) / 2.0 / IMAGE_SIZE
        cy = (y1 + y2) / 2.0 / IMAGE_SIZE
        w = (x2 - x1) / IMAGE_SIZE
        h = (y2 - y1) / IMAGE_SIZE
        label_path = LABEL_DIR / f"{sample_id}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"缺少标签文件：{label_path}")
        with label_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{class_id} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}\n")
        patched += 1
    print(json.dumps({"patched_label_count": patched, "label_dir": str(LABEL_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
