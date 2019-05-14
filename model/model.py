import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


# todo: generalize in num_stacks and num_blocks
# done! todo: add relu
# done! todo: add loss function => resize ground truth

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
    def __init__(self, num_blocks=1, num_channels=32):
        super(Hourglass, self).__init__()

        self.channels = num_channels

        # todo generalize depending on num_blocks
        self.block2 = Bottleneck(self.channels)

        self.block1 = Bottleneck(self.channels)

        self.bottleneck = Bottleneck(self.channels)

        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        # todo generalize depending on num_blocks

        identity2 = x  # 32 channels
        x = self.block2(x)  # 64 channels
        x = self.max_pool(x)

        identity1 = x  # 64
        x = self.block1(x)  # 128 channels
        x = self.max_pool(x)

        identity0 = x  # 128 todo: this should not be required?
        x = self.bottleneck(x)  # 256 channels
        x += identity0

        x = F.interpolate(x, scale_factor=2)
        x += identity1

        x = F.interpolate(x, scale_factor=2)
        x += identity2

        return x


class StackedHourglassNet(BaseModel):
    '''Hourglass model from Newell et al ECCV 2016'''

    def __init__(self, num_stacks=3, num_blocks=1, init_channels=32, num_classes=23):
        super(StackedHourglassNet, self).__init__()

        assert (1 <= num_blocks <= 5, "invalid number of blocks [1, 5]")

        self.num_stacks = num_stacks
        self.init_channels = init_channels
        self.channels = init_channels
        # reduce image size by factor 22
        self.conv = nn.Conv2d(1, self.channels, 7, 2, padding=3)  # size afterwards is (480/2, 616/2)
        self.bn = nn.BatchNorm2d(self.channels)

        self.hg1 = Hourglass(num_blocks, self.channels)
        self.intermediate_conv1 = Bottleneck(self.channels)
        self.intermediate_conv2 = Bottleneck(self.channels)

        self.loss_conv = nn.Conv2d(self.channels, num_classes, 1)  # size should be: (480/2, 616/2)

        self.intermediate_conv3 = nn.Conv2d(num_classes, self.channels, 1)

        self.hg2 = Hourglass(num_blocks, self.channels)
        self.relu = F.relu

    def forward(self, x):
        out = []
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        hourglass_identity = x
        x = self.hg1(x)
        x = self.intermediate_conv1(x)
        intermediate_conv_identity = x

        loss_conv = self.loss_conv(x)
        out.append(loss_conv)

        x = self.intermediate_conv3(loss_conv) \
            + self.intermediate_conv2(intermediate_conv_identity) \
            + hourglass_identity

        x = self.hg2(x)
        x = self.intermediate_conv1(x)
        x = self.loss_conv(x)

        out.append(x)

        return torch.stack(out)
