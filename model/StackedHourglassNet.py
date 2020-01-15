import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from model.layers import DepthwiseSeparableConvolution, same_padding


class Bottleneck(BaseModel):
    def __init__(self, in_channels, kernel_size, dilation=1, depthwise_separable_convolution=True):
        super(Bottleneck, self).__init__()

        if depthwise_separable_convolution:
            self.conv1 = DepthwiseSeparableConvolution(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1)
            self.conv2 = DepthwiseSeparableConvolution(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=kernel_size, dilation=dilation)
            self.conv3 = DepthwiseSeparableConvolution(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 1)
            self.conv2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size, padding=same_padding(kernel_size, dilation), dilation=dilation)
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
    def __init__(self, num_blocks=4, num_channels=32, kernel_size=3, dilation=1, depthwise_separable_convolution=True):
        super(Hourglass, self).__init__()

        self.channels = num_channels
        self.num_blocks = num_blocks
        self.dilation = dilation

        pre_bottleneck_blocks, intermediate_bottleneck_blocks, post_bottleneck_blocks = [], [], []
        for i in range(self.num_blocks):
            pre_bottleneck_blocks.append(Bottleneck(self.channels, kernel_size, self.dilation, depthwise_separable_convolution))
            intermediate_bottleneck_blocks.append(Bottleneck(self.channels, kernel_size, depthwise_separable_convolution=depthwise_separable_convolution))
            post_bottleneck_blocks.append(Bottleneck(self.channels, kernel_size, depthwise_separable_convolution=depthwise_separable_convolution))

        self.pre_bottleneck_blocks = nn.ModuleList(pre_bottleneck_blocks)
        self.intermediate_bottleneck_blocks = nn.ModuleList(intermediate_bottleneck_blocks)
        self.post_bottleneck_blocks = nn.ModuleList(post_bottleneck_blocks)

        self.bottlenecks = nn.ModuleList(
            [
                Bottleneck(self.channels, kernel_size, depthwise_separable_convolution=depthwise_separable_convolution)
                for _ in range(3)
            ])

        if self.dilation == 1:
            self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        identities = []

        for i in range(self.num_blocks):
            x = self.pre_bottleneck_blocks[i](x)
            identities.append(self.intermediate_bottleneck_blocks[i](x))

            if self.dilation == 1:
                x = self.max_pool(x)

        for i in range(len(self.bottlenecks)):
            x = self.bottlenecks[i](x)

        for i in range(self.num_blocks):
            if self.dilation == 1:
                x = F.interpolate(x, scale_factor=2)
            x = self.post_bottleneck_blocks[i](x)
            x += identities[self.num_blocks - i - 1]

        return x


class StackedHourglassNet(BaseModel):
    """Hourglass model from Newell et al ECCV 2016"""

    def __init__(self, num_stacks=3, num_blocks=1, num_channels=32, num_classes=23, kernel_size=3, dilation=1,
                 depthwise_separable_convolution=True):
        super(StackedHourglassNet, self).__init__()

        assert (1 <= num_blocks <= 7, "invalid number of blocks [1, 7]")

        self.num_stacks = num_stacks
        self.init_channels = num_channels
        self.channels = num_channels
        self.conv1 = (
            DepthwiseSeparableConvolution(in_channels=1, out_channels=self.channels, kernel_size=kernel_size)
            if depthwise_separable_convolution
            else nn.Conv2d(in_channels=1, out_channels=self.channels, kernel_size=kernel_size, padding=same_padding(kernel_size))
        )
        self.relu = nn.ReLU()

        hgs, intermediate_conv1, intermediate_conv2, loss_conv, intermediate_conv3 = [], [], [], [], []
        for i in range(self.num_stacks):
            hgs.append(Hourglass(num_blocks, self.channels, kernel_size, dilation, depthwise_separable_convolution))

            intermediate_conv1.append(Bottleneck(self.channels, kernel_size, dilation, depthwise_separable_convolution))

            loss_conv.append(
                DepthwiseSeparableConvolution(in_channels=self.channels, out_channels=num_classes, kernel_size=1)
                if depthwise_separable_convolution
                else nn.Conv2d(self.channels, num_classes, 1)
            )

            if i < self.num_stacks - 1:
                intermediate_conv2.append(Bottleneck(self.channels, kernel_size, dilation, depthwise_separable_convolution))

                intermediate_conv3.append(
                    DepthwiseSeparableConvolution(in_channels=num_classes, out_channels=self.channels, kernel_size=1)
                    if depthwise_separable_convolution
                    else nn.Conv2d(num_classes, self.channels, 1)
                )

        self.hgs = nn.ModuleList(hgs)
        self.intermediate_conv1 = nn.ModuleList(intermediate_conv1)
        self.intermediate_conv2 = nn.ModuleList(intermediate_conv2)
        self.loss_conv = nn.ModuleList(loss_conv)
        self.intermediate_conv3 = nn.ModuleList(intermediate_conv3)

    def forward(self, x):
        out = []
        x = self.conv1(x)
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
