"""RT-DETR 基线小样本过拟合测试入口。

该脚本只负责把已审计的 COCO 标注转换为 Ultralytics 所需的 YOLO 标注，
然后在固定数量的小样本上训练，用于验证数据、类别映射、反向传播和权重保存。
它不是正式训练入口；正式训练应使用冻结配置中的完整数据集。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml
from ultralytics import RTDETR


ROOT = Path(__file__).resolve().parents[1]
STEP2_CONFIG = ROOT / "configs/step2_dataset.yaml"
COCO = ROOT / "artifacts/step3/detection/coco_train.json"
OUT = ROOT / "artifacts/step5/rtdetr_overfit"


def build_subset(sample_count: int = 16) -> Path:
    """生成独立的小样本 YOLO 数据集，并严格保持原始图片及标注坐标。"""
    step2 = yaml.safe_load(STEP2_CONFIG.read_text(encoding="utf-8"))
    # 与完整数据转换共用唯一数据入口，避免本机绝对路径进入可移植训练代码。
    data_root = (ROOT / step2["inputs"]["dataset_root"]).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"配置的数据集根目录不存在：{data_root}")
    coco = json.loads(COCO.read_text(encoding="utf-8"))
    anns = {}
    for ann in coco["annotations"]:
        anns.setdefault(ann["image_id"], []).append(ann)
    # 按类别轮询选样，避免顺序前缀样本只覆盖单一缺陷类别而产生无效过拟合结论。
    per_class = sample_count // len(coco["categories"])
    selected_ids: set[int] = set()
    class_counts = {category["id"]: 0 for category in coco["categories"]}
    for image in coco["images"]:
        labels = {ann["category_id"] for ann in anns.get(image["id"], [])}
        if any(class_counts[label] < per_class for label in labels):
            selected_ids.add(image["id"])
            for label in labels:
                if class_counts[label] < per_class:
                    class_counts[label] += 1
        if len(selected_ids) >= sample_count and all(count >= per_class for count in class_counts.values()):
            break
    images = [image for image in coco["images"] if image["id"] in selected_ids][:sample_count]
    if len(images) != sample_count or any(count < per_class for count in class_counts.values()):
        raise RuntimeError(f"无法构造四类均衡的小样本集合：{class_counts}")
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("images/train", "labels/train"):
        (OUT / split).mkdir(parents=True, exist_ok=True)
    for image in images:
        source = data_root / image["file_name"].replace("\\", "/")
        target = OUT / "images/train" / source.name
        shutil.copy2(source, target)
        lines = []
        for ann in anns.get(image["id"], []):
            x, y, w, h = ann["bbox"]
            lines.append(f"{ann['category_id']} {(x+w/2)/800:.8f} {(y+h/2)/800:.8f} {w/800:.8f} {h/800:.8f}")
        (OUT / "labels/train" / f"{source.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    data = {"path": str(OUT), "train": "images/train", "val": "images/train", "names": coco["categories"]}
    data["names"] = {c["id"]: c["name"] for c in coco["categories"]}
    path = OUT / "dataset.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def main() -> None:
    """执行小样本训练并输出可重新加载的 checkpoint。"""
    data = build_subset()
    checkpoint = ROOT / "rtdetr-l.pt"
    # 预训练权重是冻结实验契约的一部分，不允许在缺失时静默退化为随机初始化。
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"缺少官方 RT-DETR-L 预训练权重：{checkpoint}。"
            "请先下载 rtdetr-l.pt 放到仓库根目录，再重新执行本脚本。"
        )
    model = RTDETR(str(checkpoint))
    model.train(data=str(data), imgsz=800, epochs=80, batch=1, device=0, workers=0,
                amp=False, seed=20260828, augment=False, patience=80,
                mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
                degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
                perspective=0.0, flipud=0.0, fliplr=0.0,
                project=str(OUT), name="run", exist_ok=True, plots=True)
    print(f"overfit_output={OUT / 'run'}")


if __name__ == "__main__":
    main()
