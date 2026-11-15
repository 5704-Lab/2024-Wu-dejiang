"""Train the pair-aware 3-channel RT-DETR baseline."""

from pathlib import Path

import yaml
from ultralytics import RTDETR

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/rtdetr_paired_baseline.yaml").read_text(encoding="utf-8"))
    weights = ROOT / config["model"]
    data = ROOT / config["data"]
    if not weights.is_file():
        raise FileNotFoundError(f"缺少预训练权重：{weights}")
    if not data.is_file():
        raise FileNotFoundError(f"缺少数据配置：{data}")

    model = RTDETR(str(weights))
    train = config["train"]
    model.train(
        data=str(data),
        imgsz=config["imgsz"],
        epochs=train["epochs"],
        batch=train["batch"],
        device=train["device"],
        workers=train["workers"],
        amp=train["amp"],
        seed=train["seed"],
        pretrained=train["pretrained"],
        patience=train["patience"],
        project=str(ROOT / train["project"]),
        name=train["name"],
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
