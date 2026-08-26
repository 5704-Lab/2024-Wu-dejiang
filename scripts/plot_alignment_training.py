"""将 OffsetPredictor 训练报告绘制为可复现的 PNG 曲线图。

本脚本仅依赖项目已有的 Pillow，不引入 Matplotlib。默认读取 Step 3.4-C 的
``training_report.json``，并输出总损失、分尺度验证损失和学习率三块面板。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1800, 1200
BACKGROUND = "#F7F8FA"
PANEL = "#FFFFFF"
GRID = "#DDE2E8"
TEXT = "#20252B"
MUTED = "#68717D"
TRAIN = "#2878D0"
VALIDATION = "#E45756"
SCALE_COLORS = {"C3": "#2A78D6", "C4": "#1BAF7A", "C5": "#EDA100"}


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """优先加载 Windows 中文字体；不可用时回退至 Pillow 默认字体。"""
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _plot_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    epochs: list[int],
    series: list[tuple[str, list[float], str]],
    value_format: Callable[[float], str] = lambda value: f"{value:.3f}",
    best_epoch: int | None = None,
) -> None:
    """在指定矩形中绘制多序列折线图，并自动计算纵轴范围。"""
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=18, fill=PANEL, outline="#E4E8ED", width=2)
    draw.text((left + 28, top + 20), title, font=_font(30, True), fill=TEXT)
    plot = (left + 105, top + 85, right - 40, bottom - 75)
    px1, py1, px2, py2 = plot
    values = [value for _, data, _ in series for value in data]
    low, high = min(values), max(values)
    padding = max((high - low) * 0.12, abs(high) * 0.002, 1e-8)
    low, high = low - padding, high + padding

    def x_coord(epoch: int) -> float:
        return px1 + (epoch - epochs[0]) / max(epochs[-1] - epochs[0], 1) * (px2 - px1)

    def y_coord(value: float) -> float:
        return py2 - (value - low) / max(high - low, 1e-12) * (py2 - py1)

    # 固定网格数量使不同报告生成的图具有一致版式，便于论文横向比较。
    for index in range(6):
        ratio = index / 5
        y = py2 - ratio * (py2 - py1)
        value = low + ratio * (high - low)
        draw.line((px1, y, px2, y), fill=GRID, width=1)
        label = value_format(value)
        draw.text((px1 - 16, y), label, font=_font(18), fill=MUTED, anchor="rm")
    for epoch in epochs:
        if epoch == 1 or epoch == epochs[-1] or epoch % 2 == 0:
            x = x_coord(epoch)
            draw.line((x, py1, x, py2), fill="#EEF1F4", width=1)
            draw.text((x, py2 + 14), str(epoch), font=_font(18), fill=MUTED, anchor="ma")

    if best_epoch is not None:
        x = x_coord(best_epoch)
        draw.line((x, py1, x, py2), fill="#8A63D2", width=3)
        draw.text((x + 8, py1 + 8), f"最佳 Epoch {best_epoch}", font=_font(18), fill="#7048B5")

    legend_x = right - 35
    for name, data, color in reversed(series):
        points = [(x_coord(epoch), y_coord(value)) for epoch, value in zip(epochs, data)]
        draw.line(points, fill=color, width=4, joint="curve")
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        text_width = draw.textlength(name, font=_font(18))
        legend_x -= int(text_width + 48)
        draw.line((legend_x, top + 42, legend_x + 24, top + 42), fill=color, width=4)
        draw.text((legend_x + 30, top + 42), name, font=_font(18), fill=TEXT, anchor="lm")
        legend_x -= 18

    draw.text(((px1 + px2) / 2, bottom - 23), "Epoch", font=_font(19), fill=MUTED, anchor="mm")


def render_report(report_path: Path, output_path: Path) -> Path:
    """读取训练报告并生成曲线图；报告缺少必要字段时立即报错。"""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report.get("epochs_log", [])
    if not records:
        raise ValueError("训练报告不包含 epochs_log，无法绘图")
    epochs = [int(item["epoch"]) for item in records]
    best_epoch = int(report["best_epoch"])

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((60, 36), "OffsetPredictor 训练曲线", font=_font(42, True), fill=TEXT)
    subtitle = (
        f"20 Epochs · 最佳 Epoch {best_epoch} · 最佳验证损失 {report['best_val_loss']:.6f} · "
        f"设备 {report.get('selected_device', 'unknown')}"
    )
    draw.text((60, 92), subtitle, font=_font(22), fill=MUTED)

    _plot_panel(
        draw, (50, 145, 1750, 535), "总对齐损失",
        epochs,
        [
            ("训练集", [float(item["train_loss"]) for item in records], TRAIN),
            ("验证集", [float(item["val_loss"]) for item in records], VALIDATION),
        ],
        best_epoch=best_epoch,
    )
    _plot_panel(
        draw, (50, 565, 1220, 1140), "分尺度验证损失",
        epochs,
        [
            (scale, [float(item[f"val_loss_{scale}"]) for item in records], color)
            for scale, color in SCALE_COLORS.items()
        ],
        value_format=lambda value: f"{value:.4f}",
        best_epoch=best_epoch,
    )
    _plot_panel(
        draw, (1250, 565, 1750, 1140), "学习率",
        epochs,
        [("LR", [float(item["lr"]) for item in records], "#8A63D2")],
        value_format=lambda value: f"{value:.1e}",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return output_path


def main() -> int:
    """解析命令行参数并生成训练曲线。"""
    parser = argparse.ArgumentParser(description="绘制 OffsetPredictor 训练损失曲线")
    parser.add_argument("--report", type=Path, default=Path("artifacts/step3/training/training_report.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/step3/training/alignment_loss_curves.png"))
    args = parser.parse_args()
    output = render_report(args.report, args.output)
    print(f"训练曲线已生成：{output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
