"""Multi-scale Offset Predictor for Step 3.4-A (pre-DCN alignment)."""

from __future__ import annotations

import torch
from torch import nn

from .backbone import FEATURE_STAGES

OFFSET_SPEC: tuple[tuple[str, int, int], ...] = tuple(
    (name, channels, stride) for name, _, channels, stride in FEATURE_STAGES
)


class _OffsetHead(nn.Module):
    """Per-scale conv head: 3x3 conv(2C->C/4) -> GELU -> 3x3 conv(C/4->2*kH*kW)."""

    def __init__(self, in_channels: int, kernel_size: int = 3, hidden_ratio: int = 4) -> None:
        super().__init__()
        hidden = max(in_channels // hidden_ratio, 8)
        self.conv1 = nn.Conv2d(in_channels * 2, hidden, kernel_size, padding=kernel_size // 2)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(hidden, 2 * kernel_size * kernel_size, kernel_size, padding=kernel_size // 2)

    def forward(self, query: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        x = torch.cat([query, template], dim=1)
        x = self.activation(self.conv1(x))
        return self.conv2(x)


class OffsetPredictor(nn.Module):
    """Predicts per-scale DCN sampling offsets from paired multi-scale features.

    Each scale uses an independent lightweight conv head.  The final conv of each
    head is zero-initialised so that initial offsets are exactly zero, making
    the subsequent DCN behave like an ordinary convolution (safe identity).
    """

    offset_spec = OFFSET_SPEC
    key_order = [name for name, _, _ in OFFSET_SPEC]

    def __init__(
        self,
        kernel_size: int = 3,
        hidden_ratio: int = 4,
        zero_init: bool = True,
        max_offset: float = 3.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.hidden_ratio = hidden_ratio
        # 偏移量上限（特征图像素单位）。原实现用 stride 作为缩放因子，
        # 导致 C4/C5 允许的偏移过大（16/32 像素），tanh 输出冲到饱和区后
        # warped 退化为常数张量、梯度消失，训练死锁。改为统一的小值，
        # 让 DCN 只做 ±max_offset 像素的精细微调。
        self.max_offset = float(max_offset)
        self.heads = nn.ModuleDict({
            name: _OffsetHead(channels, kernel_size, hidden_ratio)
            for name, channels, _ in OFFSET_SPEC
        })
        if zero_init:
            self.zero_init_heads()

    def zero_init_heads(self) -> None:
        for head in self.heads.values():
            nn.init.zeros_(head.conv2.weight)
            nn.init.zeros_(head.conv2.bias)

    def forward(
        self,
        query_features: dict[str, torch.Tensor],
        template_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        expected_keys = self.key_order
        for features, name in [(query_features, "query_features"), (template_features, "template_features")]:
            if not isinstance(features, dict):
                raise TypeError(f"{name} must be a dict")
            if list(features.keys()) != expected_keys:
                raise ValueError(f"{name} keys must be exactly {expected_keys}, got {list(features.keys())}")
        for key in expected_keys:
            q, t = query_features[key], template_features[key]
            if q.shape != t.shape:
                raise ValueError(f"Shape mismatch at {key}: {q.shape} vs {t.shape}")
            if not torch.isfinite(q).all() or not torch.isfinite(t).all():
                raise ValueError(f"Non-finite features at {key}")
            if q.dtype != t.dtype or q.device != t.device:
                raise ValueError(f"dtype/device mismatch at {key}")
        offsets = {}
        for name, _, _stride in OFFSET_SPEC:
            offset = self.heads[name](query_features[name], template_features[name])
            offset = torch.tanh(offset) * self.max_offset
            offsets[name] = offset
        return offsets