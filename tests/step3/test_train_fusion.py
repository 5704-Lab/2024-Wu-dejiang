import torch

from step3_model.train_fusion import FusionHeatmapLoss, build_defect_heatmaps


def test_build_defect_heatmaps_scales_boxes():
    targets = [{"boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]), "size": torch.tensor([100, 100])}]
    heatmap = build_defect_heatmaps(targets, height=50, width=50)
    assert heatmap.shape == (1, 1, 50, 50)
    assert heatmap[0, 0, 10:20, 5:15].all()
    assert heatmap.sum().item() == 100


def test_fusion_heatmap_loss_prefers_correct_logits():
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1
    correct = torch.where(target.bool(), torch.tensor(8.0), torch.tensor(-8.0))
    wrong = -correct
    criterion = FusionHeatmapLoss()
    correct_loss, _ = criterion(correct, target)
    wrong_loss, _ = criterion(wrong, target)
    assert correct_loss < wrong_loss
