from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from .hashing import sha256_text

_NAME = re.compile(r"^(?P<source>\d{4})_(?P<variant>test|b_test|b1_test|c_test|f_test|h_test|rc_test|rc1_test|s_test|s1_test)$")


def parse_sample_id(sample_id: str, allowed_variants: set[str]) -> tuple[str, str]:
    match = _NAME.fullmatch(sample_id)
    if not match:
        raise ValueError(f"Unrecognized sample name: {sample_id}")
    variant = match.group("variant")
    if variant not in allowed_variants:
        raise ValueError(f"Unexpected variant {variant!r}: {sample_id}")
    return match.group("source"), variant


def annotation_fingerprint(objects: list[dict[str, Any]]) -> str:
    normalized = sorted(
        ({"class_name": obj["class_name"], "bbox_xyxy": obj["bbox_xyxy"]} for obj in objects),
        key=lambda item: (item["class_name"], item["bbox_xyxy"]),
    )
    return sha256_text(json.dumps(normalized, sort_keys=True, separators=(",", ":")))


def build_groups(samples: list[dict[str, Any]], variants: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    expected = set(variants)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    issues: list[dict[str, str]] = []
    for sample in samples:
        try:
            source_id, variant = parse_sample_id(sample["sample_id"], expected)
            sample["source_id"] = source_id
            sample["variant"] = variant
            buckets[source_id].append(sample)
        except ValueError as exc:
            issues.append({"severity": "fatal", "sample_id": sample["sample_id"], "code": "invalid_sample_name", "message": str(exc)})

    groups: list[dict[str, Any]] = []
    for source_id, members in sorted(buckets.items()):
        seen = [member["variant"] for member in members]
        missing = sorted(expected - set(seen))
        duplicates = sorted({v for v in seen if seen.count(v) > 1})
        fingerprints = sorted({annotation_fingerprint(member["objects"]) for member in members})
        if missing:
            issues.append({"severity": "fatal", "sample_id": source_id, "code": "missing_variants", "message": ",".join(missing)})
        if duplicates:
            issues.append({"severity": "fatal", "sample_id": source_id, "code": "duplicate_variants", "message": ",".join(duplicates)})
        groups.append({
            "source_id": source_id,
            "sample_ids": sorted(member["sample_id"] for member in members),
            "variants": sorted(seen),
            "variant_count": len(members),
            "annotation_fingerprints": fingerprints,
            "annotation_consistent": len(fingerprints) == 1,
            "class_counts": _class_counts(members),
            "object_count": sum(len(member["objects"]) for member in members),
        })
    return groups, issues


def _class_counts(members: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for member in members:
        for obj in member["objects"]:
            counts[obj["class_name"]] += 1
    return dict(sorted(counts.items()))
