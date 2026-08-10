from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    source_id: str
    variant: str
    query_path: Path
    template_path: Path
    boxes_xyxy: tuple[tuple[float, float, float, float], ...]
    class_names: tuple[str, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class DatasetIndex:
    dataset_root: Path
    records_by_split: dict[str, tuple[SampleRecord, ...]]

    def records(self, split: str) -> tuple[SampleRecord, ...]:
        try:
            return self.records_by_split[split]
        except KeyError as exc:
            raise ValueError(f"Unknown split: {split}") from exc
