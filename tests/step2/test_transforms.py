import torch

from step2_dataset.transforms import rotate_boxes_ccw


def test_rotate_box_counter_clockwise():
    boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    transformed = rotate_boxes_ccw(boxes, 100)
    assert torch.equal(transformed, torch.tensor([[20.0, 70.0, 40.0, 90.0]]))
