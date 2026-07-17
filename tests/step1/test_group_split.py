import pytest

from step1_audit.grouping import build_groups, parse_sample_id
from step1_audit.split import make_group_split, validate_split

VARIANTS = ["test", "b_test", "b1_test", "c_test", "f_test", "h_test", "rc_test", "rc1_test", "s_test", "s1_test"]


def _samples(group_count=10):
    samples = []
    for number in range(group_count):
        source = f"{number:04d}"
        for variant in VARIANTS:
            samples.append({
                "sample_id": f"{source}_{variant}",
                "objects": [{"class_name": "broke" if number % 2 else "dirty", "bbox_xyxy": [1, 2, 10, 20]}],
            })
    return samples


def test_name_parser_is_strict():
    assert parse_sample_id("0001_rc1_test", set(VARIANTS)) == ("0001", "rc1_test")
    with pytest.raises(ValueError):
        parse_sample_id("1_rc1_test", set(VARIANTS))


def test_groups_have_ten_variants():
    groups, issues = build_groups(_samples(3), VARIANTS)
    assert not issues
    assert len(groups) == 3
    assert all(group["variant_count"] == 10 for group in groups)


def test_linked_template_sources_stay_together():
    groups, _ = build_groups(_samples(20), VARIANTS)
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    split = make_group_split(groups, ratios, 42, 100, ["broke", "dirty"], linked_source_sets=[{"0001", "0002"}])
    assignments = {group_id: name for name, ids in split.items() for group_id in ids}
    assert assignments["0001"] == assignments["0002"]
    validate_split(split, groups)


def test_split_is_deterministic_and_disjoint():
    groups, _ = build_groups(_samples(20), VARIANTS)
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    first = make_group_split(groups, ratios, 42, 100, ["broke", "dirty"])
    second = make_group_split(groups, ratios, 42, 100, ["broke", "dirty"])
    assert first == second
    validate_split(first, groups)
    assert {name: len(ids) for name, ids in first.items()} == {"train": 16, "val": 2, "test": 2}
