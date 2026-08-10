from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from .randomness import stable_image_id
from .schema import DatasetIndex, SampleRecord
from .transforms import PairTransform


class PairedQueryTemplateDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        index: DatasetIndex,
        split: str,
        config: dict[str, Any],
        *,
        training: bool | None = None,
        epoch: int = 0,
    ) -> None:
        self.index = index
        self.split = split
        self.records = index.records(split)
        self.config = config
        self.training = split == "train" if training is None else training
        self.epoch = epoch
        self.transform = PairTransform(config, self.training)
        self.seed = int(config["seed"])
        self.expected_size = (int(config["image"]["width"]), int(config["image"]["height"]))

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.query_path) as image:
            query = image.convert("RGB")
        with Image.open(record.template_path) as image:
            template = image.convert("RGB")
        if query.size != self.expected_size or template.size != self.expected_size:
            raise ValueError(f"Unexpected pair size for {record.sample_id}: {query.size}/{template.size}")
        boxes = torch.tensor(record.boxes_xyxy, dtype=torch.float32).reshape(-1, 4)
        query_tensor, template_tensor, boxes, params = self.transform(
            query, template, boxes, global_seed=self.seed, split=self.split,
            sample_id=record.sample_id, epoch=self.epoch,
        )
        height, width = query_tensor.shape[-2:]
        target = _make_target(record, boxes, height, width, self.expected_size)
        return {
            "query": query_tensor,
            "template": template_tensor,
            "query_target": _clone_target(target),
            "template_target": _clone_target(target),
            "meta": {
                "sample_id": record.sample_id, "source_id": record.source_id, "variant": record.variant,
                "split": self.split, "query_path": str(record.query_path), "template_path": str(record.template_path),
                "epoch": self.epoch, "transform_params": params,
            },
        }


def _make_target(record: SampleRecord, boxes: torch.Tensor, height: int, width: int, original_size: tuple[int, int]) -> dict[str, torch.Tensor]:
    labels = torch.tensor(record.labels, dtype=torch.int64)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return {
        "boxes": boxes,
        "labels": labels,
        "image_id": torch.tensor([stable_image_id(record.sample_id)], dtype=torch.int64),
        "area": area.to(torch.float32),
        "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        "orig_size": torch.tensor([original_size[1], original_size[0]], dtype=torch.int64),
        "size": torch.tensor([height, width], dtype=torch.int64),
    }


def _clone_target(target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in target.items()}
