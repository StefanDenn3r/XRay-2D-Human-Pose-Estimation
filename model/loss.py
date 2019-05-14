import torch.nn.functional as F
from torch import nn


def nll_loss(output, target):
    return F.nll_loss(output, target)


def jointmseloss(output, target):
    criterion = nn.MSELoss(reduction='mean')
    batch_size = output.size(1)
    num_joints = output.size(0)
    heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
    heatmaps_gt = target.reshape((batch_size, 1, -1))

    loss = 0
    for idx in range(num_joints):
        heatmap_pred = heatmaps_pred[idx].squeeze()
        heatmap_gt = heatmaps_gt
        loss += criterion(heatmap_pred, heatmap_gt)

    return loss / num_joints
