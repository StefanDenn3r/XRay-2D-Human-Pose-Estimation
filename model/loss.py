import torch.nn.functional as F
from torch import nn


def nll_loss(output, target):
    return F.nll_loss(output, target)


def mse_loss(output, target):
    return nn.MSELoss(reduction='mean')(output, target)
