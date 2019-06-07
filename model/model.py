import torch.nn as nn

from base import BaseModel


class ConvolutionalPoseMachine(BaseModel):

    def __init__(self, num_classes=23):
        super(ConvolutionalPoseMachine, self).__init__()

        self.conv1 = nn.Conv2d(1, num_classes, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv1(x))
