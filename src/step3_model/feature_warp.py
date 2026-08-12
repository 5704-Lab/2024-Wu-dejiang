"""Per-scale DCN feature warp for Step 3.4-B (template alignment via deformable sampling)."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.ops import deform_conv2d

from .offset_predictor import OFFSET_SPEC

WARP_SPEC: tuple[tuple[str, int, int], ...] = OFFSET_SPEC


class FeatureWarp(nn.Module):
    """Warps template multi-scale features via deformable convolution with identity kernel.

    Each scale uses a per-channel identity kernel registered as a buffer.
    With zero offsets the warp is an exact identity; with non-zero offsets
    the DCN resamples template features at offset-shifted positions.
    """

    warp_spec = WARP_SPEC
    key_order = [name for name, _, _ in WARP_SPEC]

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        mid = kernel_size // 2
        for name, channels, _ in WARP_SPEC:
            weight = torch.zeros(channels, channels, kernel_size, kernel_size)
            weight[range(channels), range(channels), mid, mid] = 1.0
            self.register_buffer(f"weight_{name}", weight)

    def forward(
        self,
        template_features: dict[str, torch.Tensor],
        offsets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        expected_keys = self.key_order
        for features, name in [(template_features, "template_features"), (offsets, "offsets")]:
            if not isinstance(features, dict):
                raise TypeError(f"{name} must be a dict")
            if list(features.keys()) != expected_keys:
                raise ValueError(f"{name} keys must be exactly {expected_keys}, got {list(features.keys())}")
        for key in expected_keys:
            tf, off = template_features[key], offsets[key]
            if off.shape[1] != 2 * self.kernel_size * self.kernel_size:
                raise ValueError(
                    f"Offset channels at {key} must be {2 * self.kernel_size ** 2}, got {off.shape[1]}"
                )
            if tf.shape[0] != off.shape[0] or tf.shape[2:] != off.shape[2:]:
                raise ValueError(
                    f"Template/offset spatial mismatch at {key}: {tf.shape} vs {off.shape}"
                )
            if not torch.isfinite(tf).all():
                raise ValueError(f"Non-finite template features at {key}")
            if not torch.isfinite(off).all():
                raise ValueError(f"Non-finite offsets at {key}")
            if tf.dtype != off.dtype or tf.device != off.device:
                raise ValueError(f"dtype/device mismatch between template and offsets at {key}")
        warped: dict[str, torch.Tensor] = {}
        for name, channels, _stride in WARP_SPEC:
            weight = getattr(self, f"weight_{name}").to(dtype=template_features[name].dtype)
            warped[name] = deform_conv2d(
                input=template_features[name],
                offset=offsets[name],
                weight=weight,
                stride=1,
                padding=self.kernel_size // 2,
            )
        return warped
