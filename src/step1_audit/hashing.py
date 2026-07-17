from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def snapshot_tree(root: Path, include_hashes: bool = True) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        record: dict[str, object] = {
            "path": path.relative_to(root).as_posix(),
            "size": stat.st_size,
        }
        if include_hashes:
            record["sha256"] = sha256_file(path)
        records.append(record)
    return records


def snapshot_digest(records: list[dict[str, object]]) -> str:
    canonical = "\n".join(
        f"{r['path']}\0{r['size']}\0{r.get('sha256', '')}" for r in records
    )
    return sha256_text(canonical)
