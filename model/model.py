import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


class Bottleneck(BaseModel):
    def __init__(self, in_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x += identity

        return x


class Hourglass(BaseModel):
    def __init__(self, num_blocks=4, num_channels=32):
        super(Hourglass, self).__init__()

        self.channels = num_channels
        self.num_blocks = num_blocks

        self.pre_bottleneck_blocks, self.intermediate_bottleneck_blocks, self.post_bottleneck_blocks = [], [], []
        for _ in range(self.num_blocks):
            self.pre_bottleneck_blocks.append(Bottleneck(self.channels))
            self.intermediate_bottleneck_blocks.append(Bottleneck(self.channels))
            self.post_bottleneck_blocks.append(Bottleneck(self.channels))

        self.bottlenecks = [Bottleneck(self.channels) for _ in range(3)]
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        identities = []

        for i in range(self.num_blocks):
            x = self.pre_bottleneck_blocks[i](x)
            identities.append(self.intermediate_bottleneck_blocks[i](x))
            x = self.max_pool(x)

        for i in range(len(self.bottlenecks)):
            x = self.bottlenecks[i](x)

        for i in range(self.num_blocks):
            x = F.interpolate(x, scale_factor=2)
            x = self.post_bottleneck_blocks[i](x)
            x += identities[self.num_blocks - i - 1]

        return x


class StackedHourglassNet(BaseModel):
    """Hourglass model from Newell et al ECCV 2016"""

    def __init__(self, num_stacks=3, num_blocks=1, init_channels=32, num_classes=23):
        super(StackedHourglassNet, self).__init__()

        assert (1 <= num_blocks <= 2, "invalid number of blocks [1, 2]")

        self.num_stacks = num_stacks
        self.init_channels = init_channels
        self.channels = init_channels
        self.conv = nn.Conv2d(1, self.channels, 7, 2, padding=3)

        self.relu = F.relu

        self.hgs, self.intermediate_conv1, self.intermediate_conv2, self.loss_conv, self.intermediate_conv3 = [], [], [], [], []
        for i in range(self.num_stacks):
            self.hgs.append(Hourglass(num_blocks, self.channels))

            self.intermediate_conv1.append(Bottleneck(self.channels))

            self.loss_conv.append(nn.Conv2d(self.channels, num_classes, 1))
            if i < self.num_stacks - 1:
                self.intermediate_conv2.append(Bottleneck(self.channels))
                self.intermediate_conv3.append(nn.Conv2d(num_classes, self.channels, 1))

    def forward(self, x):
        out = []
        x = self.conv(x)
        x = self.relu(x)

        for i in range(self.num_stacks):
            hourglass_identity = x
            x = self.hgs[i](x)
            x = self.intermediate_conv1[i](x)
            intermediate_conv_identity = x

            loss_conv = self.loss_conv[i](x)
            out.append(loss_conv)
            if i < self.num_stacks - 1:
                x = self.intermediate_conv3[i](loss_conv) \
                    + self.intermediate_conv2[i](intermediate_conv_identity) \
                    + hourglass_identity

        return torch.stack(out)
