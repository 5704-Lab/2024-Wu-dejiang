"""COCO-format dataset exporter from Step 1 samples.jsonl for RT-DETR."""

from __future__ import annotations

import json
from pathlib import Path

from step1_audit.paths import write_json
from step2_dataset.index import DatasetIndex


CLASS_NAMES = ("broke", "dirty", "double", "miss")


def export_coco_dataset(
    index: DatasetIndex,
    dataset_root: Path,
    output_path: Path,
    split: str | None = None,
) -> dict[str, object]:
    """Export dataset to COCO JSON format.

    Args:
        index: DatasetIndex with records_by_split
        dataset_root: root path where images are stored
        output_path: where to write the COCO JSON
        split: if given, only export this split (train/val/test); if None, export all

    Returns:
        Summary dict with image_count, annotation_count, split_counts.
    """
    splits = (split,) if split else ("train", "val", "test")
    categories = [
        {"id": idx, "name": name, "supercategory": "defect"}
        for idx, name in enumerate(CLASS_NAMES)
    ]

    images: list[dict] = []
    annotations: list[dict] = []
    image_id = 0
    annotation_id = 0
    split_counts: dict[str, int] = {}

    for split_name in splits:
        records = index.records(split_name)
        split_counts[split_name] = len(records)
        for record in records:
            # Use query image as the "image" for detection
            img_path = record.query_path
            images.append({
                "id": image_id,
                "file_name": str(img_path.relative_to(dataset_root)).replace("\\", "/"),
                "width": 800,
                "height": 800,
                "split": split_name,
                "sample_id": record.sample_id,
            })

            for box, label in zip(record.boxes_xyxy, record.labels):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, w, h],  # COCO format: [x, y, w, h]
                    "area": w * h,
                    "iscrowd": 0,
                })
                annotation_id += 1

            image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    write_json(output_path, coco)

    return {
        "image_count": len(images),
        "annotation_count": len(annotations),
        "split_counts": split_counts,
        "output_path": str(output_path),
    }
