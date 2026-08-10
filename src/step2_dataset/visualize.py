from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import CLASS_NAMES
from .dataset import PairedQueryTemplateDataset
from .index import build_dataset_index


def render_visualizations(config: dict[str, Any], split: str | None = None, limit: int | None = None) -> dict[str, Any]:
    index = build_dataset_index(config)
    output_root = Path(config["_output_root"]) / "visualizations"
    selected_splits = (split,) if split else ("train", "val", "test")
    count = limit or int(config["visualization"]["samples_per_split"])
    outputs: dict[str, Any] = {}
    for split_name in selected_splits:
        dataset = PairedQueryTemplateDataset(index, split_name, config, training=split_name == "train", epoch=0)
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        rendered: list[Path] = []
        for position in range(min(count, len(dataset))):
            item = dataset[position]
            path = split_dir / f"{position:03d}_{item['meta']['sample_id']}.png"
            render_pair(item, config, path)
            rendered.append(path)
        contact_path = output_root / f"contact_sheet_{split_name}.png"
        render_contact_sheet(rendered, contact_path)
        outputs[split_name] = {
            "pairs": [str(path) for path in rendered],
            "contact_sheet": str(contact_path),
            "sha256": _sha256(contact_path),
        }
    return outputs


def render_pair(item: dict[str, Any], config: dict[str, Any], output_path: Path) -> None:
    panel_width = int(config["visualization"]["panel_width"])
    query = _tensor_to_pil(item["query"]).resize((panel_width, panel_width), Image.Resampling.BILINEAR)
    template = _tensor_to_pil(item["template"]).resize((panel_width, panel_width), Image.Resampling.BILINEAR)
    header, footer, gap = 92, 96, 24
    canvas = Image.new("RGB", (panel_width * 2 + gap + 48, panel_width + header + footer), config["visualization"]["background"])
    draw = ImageDraw.Draw(canvas)
    left_x, right_x, top = 16, panel_width + gap + 16, header
    canvas.paste(query, (left_x, top))
    canvas.paste(template, (right_x, top))
    draw.text((left_x, 16), "QUERY", fill="#0b0b0b", font=_font(24))
    draw.text((right_x, 16), "TEMPLATE", fill="#0b0b0b", font=_font(24))
    meta = item["meta"]
    draw.text((left_x, 50), f"{meta['sample_id']}  source={meta['source_id']}  variant={meta['variant']}  split={meta['split']}", fill="#52514e", font=_font(16))
    _draw_boxes(draw, item["query_target"], left_x, top, panel_width, config)
    _draw_boxes(draw, item["template_target"], right_x, top, panel_width, config)
    legend_y = top + panel_width + 12
    for index, class_name in enumerate(CLASS_NAMES):
        x = left_x + index * 150
        color = config["visualization"]["class_colors"][class_name]
        draw.rectangle((x, legend_y, x + 18, legend_y + 18), fill=color, outline="#0b0b0b", width=1)
        draw.text((x + 26, legend_y - 1), class_name, fill="#0b0b0b", font=_font(15))
    geometry = meta["transform_params"]["geometry"]
    q_photo = meta["transform_params"]["query_photometry"]
    t_photo = meta["transform_params"]["template_photometry"]
    draw.text((left_x, legend_y + 34), f"geometry: hflip={geometry['horizontal_flip']} vflip={geometry['vertical_flip']} rotate={geometry['rotation_k'] * 90}°", fill="#52514e", font=_font(14))
    draw.text((left_x, legend_y + 56), f"query photo: b={q_photo['brightness']:.2f} c={q_photo['contrast']:.2f} s={q_photo['saturation']:.2f}", fill="#52514e", font=_font(14))
    draw.text((right_x, legend_y + 56), f"template photo: b={t_photo['brightness']:.2f} c={t_photo['contrast']:.2f} s={t_photo['saturation']:.2f}", fill="#52514e", font=_font(14))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=False)


def render_contact_sheet(paths: list[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    if not images:
        raise ValueError("No pair visualizations to compose")
    thumb_width = 720
    thumbs = [image.resize((thumb_width, round(image.height * thumb_width / image.width)), Image.Resampling.LANCZOS) for image in images]
    gap = 16
    width = thumb_width * 2 + gap * 3
    rows = (len(thumbs) + 1) // 2
    height = gap + rows * (thumbs[0].height + gap)
    sheet = Image.new("RGB", (width, height), "#f9f9f7")
    for index, image in enumerate(thumbs):
        x = gap + (index % 2) * (thumb_width + gap)
        y = gap + (index // 2) * (image.height + gap)
        sheet.paste(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", optimize=False)
    for image in images:
        image.close()


def _draw_boxes(draw: ImageDraw.ImageDraw, target: dict[str, Any], offset_x: int, offset_y: int, panel_width: int, config: dict[str, Any]) -> None:
    scale = panel_width / float(target["size"][1])
    for object_index, (box, label) in enumerate(zip(target["boxes"].tolist(), target["labels"].tolist())):
        class_name = CLASS_NAMES[label]
        color = config["visualization"]["class_colors"][class_name]
        x1, y1, x2, y2 = [round(value * scale) for value in box]
        coordinates = (offset_x + x1, offset_y + y1, offset_x + x2, offset_y + y2)
        draw.rectangle(coordinates, outline="#0b0b0b", width=5)
        draw.rectangle(coordinates, outline=color, width=3)
        text = f"{class_name} #{object_index}"
        text_box = draw.textbbox((0, 0), text, font=_font(14))
        tw, th = text_box[2] - text_box[0], text_box[3] - text_box[1]
        tx, ty = offset_x + x1, max(offset_y, offset_y + y1 - th - 8)
        draw.rectangle((tx, ty, tx + tw + 8, ty + th + 6), fill="#fcfcfb", outline="#0b0b0b", width=1)
        draw.text((tx + 4, ty + 2), text, fill="#0b0b0b", font=_font(14))


def _tensor_to_pil(tensor) -> Image.Image:
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(np.clip(array, 0, 1) * 255).astype(np.uint8), mode="RGB")


def _font(size: int):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
