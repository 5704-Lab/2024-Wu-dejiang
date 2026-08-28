"""将已审计 COCO 标注转换为 RT-DETR(Ultralytics) 可直接读取的数据集。

转换只改变标注容器格式，不改变图像像素和原始 Train/Val/Test 划分。
COCO 的 [x, y, width, height] 框会转换为 YOLO 的归一化 [cx, cy, w, h]。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STEP2_CONFIG = ROOT / "configs/step2_dataset.yaml"
COCO_ROOT = ROOT / "artifacts/step3/detection"
OUT = ROOT / "artifacts/step5/rtdetr_dataset"
NAMES = {0: "broke", 1: "dirty", 2: "double", 3: "miss"}


def convert_split(split: str) -> dict[str, int]:
    """复制一个划分的原图并生成同名 YOLO 标签，返回转换统计。"""
    source = json.loads((COCO_ROOT / f"coco_{split}.json").read_text(encoding="utf-8"))
    images = source["images"]
    anns: dict[int, list[dict]] = {}
    for ann in source["annotations"]:
        anns.setdefault(int(ann["image_id"]), []).append(ann)
    image_dir = OUT / "images" / split
    label_dir = OUT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    annotation_count = 0
    for image in images:
        src = DATA_ROOT / image["file_name"].replace("\\", "/")
        dst = image_dir / src.name
        if not src.is_file():
            raise FileNotFoundError(f"缺少图像文件：{src}")
        shutil.copy2(src, dst)
        lines: list[str] = []
        for ann in anns.get(int(image["id"]), []):
            x, y, w, h = map(float, ann["bbox"])
            if ann["category_id"] not in NAMES or w <= 0 or h <= 0:
                raise ValueError(f"非法标注：{ann}")
            lines.append(f"{ann['category_id']} {(x+w/2)/image['width']:.8f} {(y+h/2)/image['height']:.8f} {w/image['width']:.8f} {h/image['height']:.8f}")
            annotation_count += 1
        (label_dir / f"{dst.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    return {"images": len(images), "annotations": annotation_count}


def main() -> None:
    """重建完整数据目录并写入固定数据集配置。"""
    global DATA_ROOT
    # 数据根目录只从统一配置读取；相对路径始终以仓库根目录为基准解析。
    step2 = yaml.safe_load(STEP2_CONFIG.read_text(encoding="utf-8"))
    DATA_ROOT = (ROOT / step2["inputs"]["dataset_root"]).resolve()
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"配置的数据集根目录不存在：{DATA_ROOT}")
    if OUT.exists():
        shutil.rmtree(OUT)
    stats = {split: convert_split(split) for split in ("train", "val", "test")}
    data = {"path": str(OUT), "train": "images/train", "val": "images/val", "test": "images/test", "names": NAMES}
    (OUT / "dataset.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (OUT / "conversion_report.json").write_text(json.dumps({"schema_version": "rtdetr-dataset-v1", "stats": stats}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"dataset_yaml={OUT / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
