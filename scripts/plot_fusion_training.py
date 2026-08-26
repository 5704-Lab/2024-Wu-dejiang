"""绘制 MultiScaleFusion 监督训练的 Total、BCE 与 Dice 损失曲线。

最佳 checkpoint 只保存单个 Epoch 的模型状态，因此完整曲线从相同目录下的
``training_report.json`` 读取；脚本同时核对 checkpoint 中的最佳轮次，防止图表
标记与实际部署权重不一致。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from plot_alignment_training import (
    BACKGROUND,
    HEIGHT,
    MUTED,
    TEXT,
    TRAIN,
    VALIDATION,
    WIDTH,
    _font,
    _plot_panel,
)


def render_fusion_report(report_path: Path, checkpoint_path: Path, output_path: Path) -> Path:
    """读取训练报告和最佳权重元数据，生成三面板 Fusion 损失图。"""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report.get("epochs_log", [])
    if not records:
        raise ValueError("Fusion 训练报告不包含 epochs_log，无法绘制曲线")

    # checkpoint 仅用于核对被正式采用的最佳轮次，不读取或改写模型参数。
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_epoch = int(checkpoint["epoch"])
    report_epoch = int(report["best_epoch"])
    if checkpoint_epoch != report_epoch:
        raise ValueError(
            f"最佳权重与训练报告不一致：checkpoint={checkpoint_epoch}, report={report_epoch}"
        )

    epochs = [int(item["epoch"]) for item in records]
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((60, 36), "MultiScaleFusion 训练曲线", font=_font(42, True), fill=TEXT)
    subtitle = (
        f"{len(epochs)} Epochs · 最佳 Epoch {report_epoch} · "
        f"最佳验证损失 {report['best_val_loss']:.6f} · 设备 {report.get('selected_device', 'unknown')}"
    )
    draw.text((60, 92), subtitle, font=_font(22), fill=MUTED)

    _plot_panel(
        draw, (50, 145, 1750, 535), "总损失：BCE + Dice",
        epochs,
        [
            ("训练集", [float(item["train_loss"]) for item in records], TRAIN),
            ("验证集", [float(item["val_loss"]) for item in records], VALIDATION),
        ],
        best_epoch=report_epoch,
    )
    _plot_panel(
        draw, (50, 565, 875, 1140), "BCE 像素分类损失",
        epochs,
        [
            ("训练集", [float(item["train_loss_bce"]) for item in records], TRAIN),
            ("验证集", [float(item["val_loss_bce"]) for item in records], VALIDATION),
        ],
        value_format=lambda value: f"{value:.3f}",
        best_epoch=report_epoch,
    )
    _plot_panel(
        draw, (925, 565, 1750, 1140), "Dice 区域重叠损失",
        epochs,
        [
            ("训练集", [float(item["train_loss_dice"]) for item in records], TRAIN),
            ("验证集", [float(item["val_loss_dice"]) for item in records], VALIDATION),
        ],
        value_format=lambda value: f"{value:.3f}",
        best_epoch=report_epoch,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return output_path


def main() -> int:
    """解析默认产物路径并输出 Fusion 损失曲线。"""
    parser = argparse.ArgumentParser(description="绘制 MultiScaleFusion 训练损失曲线")
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/step3/fusion_training/training_report.json"),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("artifacts/step3/fusion_training/checkpoint_best.pt"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/step3/fusion_training/fusion_loss_curves.png"),
    )
    args = parser.parse_args()
    output = render_fusion_report(args.report, args.checkpoint, args.output)
    print(f"Fusion 训练曲线已生成：{output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
