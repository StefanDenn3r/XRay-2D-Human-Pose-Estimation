from utils import util
import torch.nn.functional as F
from torch import nn


def nll_loss(output, target):
    return F.nll_loss(output, target)


def smooth_l1_loss(output, target):
    return nn.SmoothL1Loss(reduction='sum')(output, target)
