from utils import util
import torch.nn.functional as F
from torch import nn


def nll_loss(output, target):
    return F.nll_loss(output, target)


def jointmseloss(output, target):
    return util.apply_loss(nn.MSELoss(reduction='mean'), output, target)
