from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def resolved(path: str | Path, base: Path | None = None) -> Path:
    value = Path(path)
    if base is not None and not value.is_absolute():
        value = base / value
    return value.resolve()


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_roots(dataset_root: Path, output_root: Path) -> None:
    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if output_root == dataset_root or is_within(output_root, dataset_root):
        raise ValueError("Output root must not be inside the read-only dataset root")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" for v in values))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
