"""CBAM attention and multi-scale fusion for Step 3.5-B."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .offset_predictor import OFFSET_SPEC

_STAGE_NAMES = [name for name, _, _ in OFFSET_SPEC]
_STAGE_CHANNELS = {name: channels for name, channels, _ in OFFSET_SPEC}
_STAGE_STRIDES = {name: stride for name, _, stride in OFFSET_SPEC}


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Global average pooling → bottleneck FC → sigmoid → per-channel weights.
    Models global context to suppress background texture channels.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        bottleneck = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.fc(x).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * weight


class SpatialAttention(nn.Module):
    """Spatial attention with large-kernel convolution.

    Mean + max pooling across channels → 7x7 conv → sigmoid → spatial weights.
    Focuses on local regions with strong feature differences.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        pooled = torch.cat([avg, mx], dim=1)  # [B, 2, H, W]
        weight = torch.sigmoid(self.conv(pooled))  # [B, 1, H, W]
        return x * weight


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention → spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class MultiScaleFusion(nn.Module):
    """Per-scale CBAM attention → projection to scalar → learned weighted fusion.

    Input:  dict of per-scale difference tensors {C3: [B,512,H/8,W/8], ...}
    Output: single [B, 1, 800, 800] enhanced heatmap.
    """

    def __init__(self, hidden: int = 128, reduction: int = 16) -> None:
        super().__init__()
        self.hidden = hidden
        self.key_order = _STAGE_NAMES

        # Per-scale: channel projection → CBAM → scalar projection
        self.project = nn.ModuleDict()
        self.cbam = nn.ModuleDict()
        self.fuse_conv = nn.ModuleDict()
        for name in _STAGE_NAMES:
            channels = _STAGE_CHANNELS[name]
            self.project[name] = nn.Conv2d(channels, hidden, 1)
            self.cbam[name] = CBAM(hidden, reduction)
            self.fuse_conv[name] = nn.Conv2d(hidden, 1, 1)

        # Learned per-scale fusion weights (softmax-normalized)
        self.fusion_weights = nn.Parameter(torch.zeros(3))

    def forward(self, diff_features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse per-scale difference features into unified heatmap.

        Args:
            diff_features: {C3: [B,512,H/8,W/8], C4: [B,1024,H/16,W/16], C5: [B,2048,H/32,W/32]}
                           computed as warped - query_features.
        Returns:
            [B, 1, 800, 800] enhanced heatmap.
        """
        expected_keys = self.key_order
        for key in expected_keys:
            if key not in diff_features:
                raise KeyError(f"Missing key {key} in diff_features, expected {expected_keys}")
            if not torch.isfinite(diff_features[key]).all():
                raise ValueError(f"Non-finite diff features at {key}")

        heatmaps: list[torch.Tensor] = []
        for name in expected_keys:
            d = diff_features[name]  # [B, C, H, W]
            h = self.project[name](d)           # → hidden channels
            h = self.cbam[name](h)             # CBAM attention
            h = self.fuse_conv[name](h)        # → 1 channel
            h = F.interpolate(h, size=(800, 800), mode="bilinear", align_corners=False)
            heatmaps.append(h)  # [B, 1, 800, 800]

        weights = F.softmax(self.fusion_weights, dim=0)  # [3]
        fused = sum(w * hm for w, hm in zip(weights, heatmaps))  # [B, 1, 800, 800]
        return fused
