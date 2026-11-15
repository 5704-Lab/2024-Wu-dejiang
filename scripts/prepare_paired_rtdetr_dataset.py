"""Prepare a 3-channel pair-aware RT-DETR dataset from query/template pairs.

Channels:
    R = query grayscale
    G = template grayscale
    B = abs(query - template) * diff_scale, clipped to [0, 255]

Labels reuse the audited Step 1 object boxes and train/val/test splits.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STEP2_CONFIG = ROOT / "configs/step2_dataset.yaml"
STEP1_ROOT = ROOT / "artifacts/step1"
OUT = ROOT / "artifacts/step5/rtdetr_paired_dataset"
DIFF_SCALE = 3.0


def _load_samples() -> dict[str, dict]:
    samples = {}
    for line in (STEP1_ROOT / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        samples[record["sample_id"]] = record
    return samples


def convert_split(
    split: str,
    data_root: Path,
    samples: dict[str, dict],
    class_names: list[str],
) -> dict[str, int]:
    ids = [
        line.strip()
        for line in (STEP1_ROOT / "splits" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    image_dir = OUT / "images" / split
    label_dir = OUT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    annotation_count = 0
    for sample_id in ids:
        record = samples[sample_id]
        query_path = data_root / record["query_path"]
        template_path = data_root / record["template_path"]
        with Image.open(query_path) as image:
            query = np.asarray(image.convert("RGB").convert("L"), dtype=np.float32)
        with Image.open(template_path) as image:
            template = np.asarray(image.convert("RGB").convert("L"), dtype=np.float32)

        diff = np.abs(query - template)
        diff = np.clip(diff * DIFF_SCALE, 0, 255)
        paired = np.stack([query, template, diff], axis=2).astype(np.uint8)
        output_image = image_dir / f"{sample_id}.png"
        Image.fromarray(paired, mode="RGB").save(output_image)

        height, width = paired.shape[:2]
        lines = []
        for obj in record["objects"]:
            class_name = obj["class_name"]
            if class_name not in class_names:
                raise ValueError(f"未知类别：{class_name}")
            class_id = class_names.index(class_name)
            x1, y1, x2, y2 = [float(v) for v in obj["bbox_xyxy"]]
            cx = (x1 + x2) / 2.0 / width
            cy = (y1 + y2) / 2.0 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            lines.append(f"{class_id} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}")
            annotation_count += 1
        (label_dir / f"{sample_id}.txt").write_text("\n".join(lines), encoding="utf-8")
    return {"images": len(ids), "annotations": annotation_count}


def main() -> None:
    step2 = yaml.safe_load(STEP2_CONFIG.read_text(encoding="utf-8"))
    data_root = (ROOT / step2["inputs"]["dataset_root"]).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在：{data_root}")
    class_names = list(step2["classes"])
    samples = _load_samples()

    if OUT.exists():
        shutil.rmtree(OUT)

    stats = {split: convert_split(split, data_root, samples, class_names) for split in ("train", "val", "test")}
    data = {
        "path": str(OUT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    (OUT / "dataset.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    report = {
        "schema_version": "rtdetr-paired-dataset-v1",
        "diff_scale": DIFF_SCALE,
        "stats": stats,
    }
    (OUT / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"dataset_yaml={OUT / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
