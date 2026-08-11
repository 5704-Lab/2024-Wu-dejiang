"""Single-scale shared Siamese ResNet-50 backbone for Step 3.3-A."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class SharedSiameseBackbone(nn.Module):
    architecture = "resnet50"
    out_channels = 2048
    output_stride = 32

    def __init__(self, weights: str | None = "IMAGENET1K_V2") -> None:
        super().__init__()
        resolved_weights = _resolve_weights(weights)
        network = resnet50(weights=resolved_weights)
        self.encoder = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
            network.layer1,
            network.layer2,
            network.layer3,
            network.layer4,
        )
        self.weights_name = weights or "none"
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _validate_images(self, images: torch.Tensor, name: str) -> None:
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"{name} must have shape [B, 3, H, W]")
        if not images.is_floating_point():
            raise TypeError(f"{name} must use a floating dtype")
        if not torch.isfinite(images).all():
            raise ValueError(f"{name} contains NaN or Infinity")
        if torch.any(images < -1e-6) or torch.any(images > 1.0 + 1e-6):
            raise ValueError(f"{name} values must be in [0, 1]")

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        self._validate_images(images, "images")
        return (images - self.mean.to(dtype=images.dtype)) / self.std.to(dtype=images.dtype)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.normalize(images))

    def forward(
        self, query: torch.Tensor, template: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_images(query, "query")
        self._validate_images(template, "template")
        if query.shape != template.shape:
            raise ValueError(f"Query/template shapes differ: {query.shape}/{template.shape}")
        if query.device != template.device or query.dtype != template.dtype:
            raise ValueError("Query/template device and dtype must match")
        return self.encode(query), self.encode(template)


def _resolve_weights(name: str | None) -> ResNet50_Weights | None:
    if name is None or str(name).lower() == "none":
        return None
    if name == "IMAGENET1K_V2":
        return ResNet50_Weights.IMAGENET1K_V2
    raise ValueError(f"Unsupported ResNet-50 weights: {name}")
