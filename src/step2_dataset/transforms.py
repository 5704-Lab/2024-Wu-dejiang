from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from .randomness import derive_seed


@dataclass(frozen=True)
class GeometricParams:
    horizontal_flip: bool
    vertical_flip: bool
    rotation_k: int
    output_width: int
    output_height: int


@dataclass(frozen=True)
class PhotometricParams:
    brightness: float
    contrast: float
    saturation: float
    hue_shift: float
    blur_radius: float
    noise_std: float
    noise_seed: int


class PairTransform:
    def __init__(self, config: dict[str, Any], training: bool) -> None:
        self.training = training
        self.config = config["augmentation"]["train"]
        self.output_width = int(config["image"]["output_width"])
        self.output_height = int(config["image"]["output_height"])

    def __call__(
        self,
        query: Image.Image,
        template: Image.Image,
        boxes: torch.Tensor,
        *,
        global_seed: int,
        split: str,
        sample_id: str,
        epoch: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        geometry_rng = random.Random(derive_seed(global_seed, split, sample_id, epoch, "geometry"))
        geometry = self.sample_geometry(geometry_rng)
        query, transformed_boxes = self.apply_geometry(query, boxes, geometry)
        template, template_boxes = self.apply_geometry(template, boxes, geometry)
        if not torch.equal(transformed_boxes, template_boxes):
            raise RuntimeError("Pair geometry produced inconsistent boxes")

        query_photo = self.sample_photometry(random.Random(derive_seed(global_seed, split, sample_id, epoch, "query-photo")), global_seed, split, sample_id, epoch, "query-noise")
        template_photo = self.sample_photometry(random.Random(derive_seed(global_seed, split, sample_id, epoch, "template-photo")), global_seed, split, sample_id, epoch, "template-noise")
        query = self.apply_photometry(query, query_photo)
        template = self.apply_photometry(template, template_photo)
        return (
            pil_to_tensor(query),
            pil_to_tensor(template),
            transformed_boxes,
            {"geometry": asdict(geometry), "query_photometry": asdict(query_photo), "template_photometry": asdict(template_photo)},
        )

    def sample_geometry(self, rng: random.Random) -> GeometricParams:
        if not self.training:
            return GeometricParams(False, False, 0, self.output_width, self.output_height)
        rotate = rng.randrange(1, 4) if rng.random() < float(self.config["rotate90_p"]) else 0
        return GeometricParams(
            horizontal_flip=rng.random() < float(self.config["horizontal_flip_p"]),
            vertical_flip=rng.random() < float(self.config["vertical_flip_p"]),
            rotation_k=rotate,
            output_width=self.output_width,
            output_height=self.output_height,
        )

    def sample_photometry(self, rng: random.Random, global_seed: int, split: str, sample_id: str, epoch: int, noise_view: str) -> PhotometricParams:
        if not self.training:
            return PhotometricParams(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, derive_seed(global_seed, split, sample_id, epoch, noise_view))
        factor = lambda amount: 1.0 + rng.uniform(-float(amount), float(amount))
        blur = rng.uniform(0.1, float(self.config["blur_radius_max"])) if rng.random() < float(self.config["blur_p"]) else 0.0
        noise = rng.uniform(0.001, float(self.config["noise_std_max"])) if rng.random() < float(self.config["noise_p"]) else 0.0
        return PhotometricParams(
            brightness=factor(self.config["brightness"]), contrast=factor(self.config["contrast"]),
            saturation=factor(self.config["saturation"]), hue_shift=rng.uniform(-float(self.config["hue"]), float(self.config["hue"])),
            blur_radius=blur, noise_std=noise, noise_seed=derive_seed(global_seed, split, sample_id, epoch, noise_view),
        )

    def apply_geometry(self, image: Image.Image, boxes: torch.Tensor, params: GeometricParams) -> tuple[Image.Image, torch.Tensor]:
        image = image.convert("RGB")
        result = boxes.clone().to(torch.float32)
        width, height = image.size
        if params.horizontal_flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            result[:, [0, 2]] = width - result[:, [2, 0]]
        if params.vertical_flip:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            result[:, [1, 3]] = height - result[:, [3, 1]]
        for _ in range(params.rotation_k):
            image = image.transpose(Image.Transpose.ROTATE_90)
            result = rotate_boxes_ccw(result, width)
            width, height = height, width
        if (width, height) != (params.output_width, params.output_height):
            scale_x, scale_y = params.output_width / width, params.output_height / height
            image = image.resize((params.output_width, params.output_height), Image.Resampling.BILINEAR)
            result[:, [0, 2]] *= scale_x
            result[:, [1, 3]] *= scale_y
        result[:, [0, 2]].clamp_(0, params.output_width)
        result[:, [1, 3]].clamp_(0, params.output_height)
        if result.numel() and not torch.all((result[:, 2] > result[:, 0]) & (result[:, 3] > result[:, 1])):
            raise ValueError("Geometric transform produced a degenerate box")
        return image, result

    def apply_photometry(self, image: Image.Image, params: PhotometricParams) -> Image.Image:
        image = ImageEnhance.Brightness(image).enhance(params.brightness)
        image = ImageEnhance.Contrast(image).enhance(params.contrast)
        image = ImageEnhance.Color(image).enhance(params.saturation)
        if params.hue_shift:
            hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
            hsv[..., 0] = (hsv[..., 0].astype(np.int16) + round(params.hue_shift * 255)) % 256
            image = Image.fromarray(hsv, mode="HSV").convert("RGB")
        if params.blur_radius:
            image = image.filter(ImageFilter.GaussianBlur(params.blur_radius))
        if params.noise_std:
            rng = np.random.default_rng(params.noise_seed)
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = np.clip(array + rng.normal(0.0, params.noise_std, array.shape), 0.0, 1.0)
            image = Image.fromarray(np.round(array * 255).astype(np.uint8), mode="RGB")
        return image


def rotate_boxes_ccw(boxes: torch.Tensor, width: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.clone()
    xmin, ymin, xmax, ymax = boxes.unbind(dim=1)
    return torch.stack((ymin, width - xmax, ymax, width - xmin), dim=1)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()
