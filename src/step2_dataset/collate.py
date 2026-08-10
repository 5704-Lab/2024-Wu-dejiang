from __future__ import annotations

from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader


def pair_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    shapes = {tuple(item["query"].shape) for item in batch} | {tuple(item["template"].shape) for item in batch}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent pair tensor shapes: {shapes}")
    for item in batch:
        for key in ("query_target", "template_target"):
            target = item[key]
            if target["boxes"].ndim != 2 or target["boxes"].shape[1] != 4:
                raise ValueError("Target boxes must have shape [N, 4]")
            if target["labels"].dtype != torch.int64:
                raise ValueError("Target labels must be int64")
            if len(target["boxes"]) != len(target["labels"]):
                raise ValueError("Target boxes and labels differ in length")
    return {
        "query": torch.stack([item["query"] for item in batch]),
        "template": torch.stack([item["template"] for item in batch]),
        "query_targets": [item["query_target"] for item in batch],
        "template_targets": [item["template_target"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }


def build_dataloader(dataset, *, batch_size: int, num_workers: int = 0, shuffle: bool | None = None) -> DataLoader:
    if shuffle is None:
        shuffle = getattr(dataset, "split", None) == "train"
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=pair_collate_fn, pin_memory=torch.cuda.is_available(),
    )
