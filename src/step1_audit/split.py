from __future__ import annotations

import random
from collections import Counter
from typing import Any


def make_group_split(
    groups: list[dict[str, Any]],
    ratios: dict[str, float],
    seed: int,
    iterations: int,
    class_names: list[str],
    linked_source_sets: list[set[str]] | None = None,
) -> dict[str, list[str]]:
    names = ("train", "val", "test")
    if abs(sum(ratios[name] for name in names) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1")
    total = len(groups)
    group_by_id = {group["source_id"]: group for group in groups}
    components = _build_components(sorted(group_by_id), linked_source_sets or [])
    target_counts = {
        "train": total * ratios["train"],
        "val": total * ratios["val"],
        "test": total * ratios["test"],
    }
    global_counts = _sum_classes(groups, class_names)

    best: tuple[float, dict[str, list[str]]] | None = None
    rng = random.Random(seed)
    for _ in range(max(1, iterations)):
        shuffled = components.copy()
        rng.shuffle(shuffled)
        candidate = {name: [] for name in names}
        for component in shuffled:
            destination = min(
                names,
                key=lambda name: (
                    abs((len(candidate[name]) + len(component)) - target_counts[name])
                    - abs(len(candidate[name]) - target_counts[name]),
                    len(candidate[name]) / max(ratios[name], 1e-9),
                    name,
                ),
            )
            candidate[destination].extend(component)
        candidate = {name: sorted(ids) for name, ids in candidate.items()}
        score = _score(candidate, group_by_id, global_counts, ratios, class_names)
        score += sum(abs(len(candidate[name]) - target_counts[name]) / max(1.0, target_counts[name]) for name in names) * 5.0
        signature = tuple(candidate["train"] + candidate["val"] + candidate["test"])
        if best is None or (score, signature) < (best[0], tuple(best[1]["train"] + best[1]["val"] + best[1]["test"])):
            best = (score, candidate)
    assert best is not None
    return best[1]


def validate_split(split: dict[str, list[str]], groups: list[dict[str, Any]]) -> None:
    sets = {name: set(ids) for name, ids in split.items()}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError("Split groups overlap")
    expected = {group["source_id"] for group in groups}
    if set().union(*sets.values()) != expected:
        raise ValueError("Split does not cover all groups exactly")


def summarize_split(split: dict[str, list[str]], groups: list[dict[str, Any]], class_names: list[str]) -> dict[str, Any]:
    group_by_id = {group["source_id"]: group for group in groups}
    result: dict[str, Any] = {}
    for name, ids in split.items():
        members = [group_by_id[group_id] for group_id in ids]
        result[name] = {
            "group_count": len(ids),
            "sample_count": sum(group["variant_count"] for group in members),
            "object_count": sum(group["object_count"] for group in members),
            "class_counts": _sum_classes(members, class_names),
        }
    return result


def _build_components(source_ids: list[str], linked_source_sets: list[set[str]]) -> list[list[str]]:
    parent = {source_id: source_id for source_id in source_ids}

    def root(source_id: str) -> str:
        while parent[source_id] != source_id:
            parent[source_id] = parent[parent[source_id]]
            source_id = parent[source_id]
        return source_id

    def union(left: str, right: str) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    known = set(source_ids)
    for linked in linked_source_sets:
        members = sorted(linked & known)
        for member in members[1:]:
            union(members[0], member)

    components: dict[str, list[str]] = {}
    for source_id in source_ids:
        components.setdefault(root(source_id), []).append(source_id)
    return [sorted(component) for component in sorted(components.values(), key=lambda values: values[0])]


def _sum_classes(groups: list[dict[str, Any]], class_names: list[str]) -> dict[str, int]:
    counts = Counter({name: 0 for name in class_names})
    for group in groups:
        counts.update(group["class_counts"])
    return dict(counts)


def _score(
    split: dict[str, list[str]],
    group_by_id: dict[str, dict[str, Any]],
    global_counts: dict[str, int],
    ratios: dict[str, float],
    class_names: list[str],
) -> float:
    score = 0.0
    for name, ids in split.items():
        subset = [group_by_id[group_id] for group_id in ids]
        actual = _sum_classes(subset, class_names)
        for class_name in class_names:
            target = global_counts[class_name] * ratios[name]
            score += abs(actual[class_name] - target) / max(1.0, target)
    return score
