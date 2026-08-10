import torch

from step2_dataset.collate import pair_collate_fn


def _item(count):
    target = {
        "boxes": torch.ones((count, 4), dtype=torch.float32),
        "labels": torch.zeros((count,), dtype=torch.int64),
    }
    return {
        "query": torch.zeros((3, 8, 8)), "template": torch.ones((3, 8, 8)),
        "query_target": {k: v.clone() for k, v in target.items()},
        "template_target": {k: v.clone() for k, v in target.items()}, "meta": {"count": count},
    }


def test_collate_preserves_variable_targets():
    batch = pair_collate_fn([_item(1), _item(3)])
    assert batch["query"].shape == (2, 3, 8, 8)
    assert [len(target["boxes"]) for target in batch["query_targets"]] == [1, 3]
