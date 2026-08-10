from __future__ import annotations

import hashlib


def derive_seed(global_seed: int, split: str, sample_id: str, epoch: int, view: str) -> int:
    value = f"step2-v1\0{global_seed}\0{split}\0{sample_id}\0{epoch}\0{view}"
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def stable_image_id(sample_id: str) -> int:
    return int.from_bytes(hashlib.sha256(sample_id.encode("utf-8")).digest()[:7], "big")
