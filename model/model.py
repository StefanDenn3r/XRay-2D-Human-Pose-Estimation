import torch
import torch.nn as nn

from base import BaseModel


def same_padding(kernel_size, dilation=1):
    return (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2


def calculate_kernel_size(receptive_field,  dilation):
    return (dilation - 1 + receptive_field) // dilation


class DepthwiseSeparableConvolution(BaseModel):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(DepthwiseSeparableConvolution, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.relu(x)

        x = self.pointwise(x)
        x = self.relu(x)

        return x


class GlobalAvgPool(BaseModel):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return x.view(*(x.shape[:-2]), -1).mean(-1)


class SqueezeExcitation(BaseModel):
    def __init__(self, channels, ratio=16):
        super(SqueezeExcitation, self).__init__()

        # not in original implementation. Throws error if not used, since original implementation uses greater reduction
        contract = max(2, channels // ratio)
        self.se = nn.Sequential(
            GlobalAvgPool(),
            nn.Linear(channels, contract),
            nn.ReLU(inplace=True),
            nn.Linear(contract, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x.mul(se_weight)


class X(BaseModel):
    def __init__(self, x_channels=128, depthwise_separable_convolution=True, squeeze_excitation=True, dilation=1):
        super(X, self).__init__()

        kernel_size = calculate_kernel_size(9, dilation)

        convs = [nn.Conv2d(in_channels=1, out_channels=x_channels, kernel_size=9, padding=same_padding(9))]

        if depthwise_separable_convolution:
            convs += [
                DepthwiseSeparableConvolution(in_channels=x_channels, out_channels=x_channels, kernel_size=9, padding=same_padding(9)),
                DepthwiseSeparableConvolution(in_channels=x_channels, out_channels=x_channels, kernel_size=9, padding=same_padding(9)),
                DepthwiseSeparableConvolution(in_channels=x_channels, out_channels=32, kernel_size=5, padding=same_padding(5))
            ]
        else:
            convs += [
                nn.Conv2d(in_channels=x_channels, out_channels=x_channels, kernel_size=kernel_size, padding=same_padding(kernel_size, dilation),
                          dilation=dilation),
                nn.Conv2d(in_channels=x_channels, out_channels=x_channels, kernel_size=kernel_size, padding=same_padding(kernel_size, dilation),
                          dilation=dilation),
                nn.Conv2d(in_channels=x_channels, out_channels=x_channels, kernel_size=kernel_size, padding=same_padding(kernel_size, dilation), dilation=dilation),
                nn.Conv2d(in_channels=x_channels, out_channels=x_channels, kernel_size=kernel_size, padding=same_padding(kernel_size, dilation), dilation=dilation),
                nn.Conv2d(in_channels=x_channels, out_channels=32, kernel_size=5, padding=same_padding(5))
            ]

        if squeeze_excitation:
            convs.insert(3, SqueezeExcitation(channels=x_channels, ratio=16))

        self.convs = nn.ModuleList(convs)

        self.max_pool = nn.MaxPool2d(3, 2, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = self.relu(x)
            if i < len(self.convs) - 1:
                x = self.max_pool(x)

        return x


class Stage1(BaseModel):
    def __init__(self, x_channels=128, stage_channels=512, num_classes=23, depthwise_separable_convolution=True, squeeze_excitation=True,
                 dilation=1):
        super(Stage1, self).__init__()
        self.X = X(x_channels, depthwise_separable_convolution, squeeze_excitation, dilation)

        if depthwise_separable_convolution:
            first_conv = DepthwiseSeparableConvolution(in_channels=32, out_channels=stage_channels, kernel_size=9, padding=same_padding(9))
        else:
            first_conv = nn.Conv2d(in_channels=32, out_channels=stage_channels, kernel_size=9, padding=same_padding(9))

        self.convs = nn.ModuleList([
            first_conv,
            nn.Conv2d(in_channels=stage_channels, out_channels=stage_channels, kernel_size=1),
            nn.Conv2d(in_channels=stage_channels, out_channels=num_classes, kernel_size=1)
        ])

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.X(x)
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < len(self.convs) - 1:
                x = self.relu(x)

        return x


class StageN(BaseModel):
    def __init__(self, x_channels=128, num_classes=23, depthwise_separable_convolution=True, squeeze_excitation=True):
        super(StageN, self).__init__()

        kernel_size = 11

        if depthwise_separable_convolution:
            first_convs = [
                DepthwiseSeparableConvolution(in_channels=32 + num_classes, out_channels=x_channels, kernel_size=kernel_size,
                                              padding=same_padding(kernel_size)),
                DepthwiseSeparableConvolution(in_channels=x_channels, out_channels=x_channels, kernel_size=kernel_size, padding=same_padding(kernel_size)),
                DepthwiseSeparableConvolution(in_channels=x_channels, out_channels=x_channels, kernel_size=kernel_size, padding=same_padding(kernel_size))
            ]
        else:
            first_convs = [
                nn.Conv2d(in_channels=32 + num_classes, out_channels=x_channels, kernel_size=11, padding=same_padding(kernel_size)),
                nn.Conv2d(in_channels=x_channels, out_channels=x_channels, kernel_size=11, padding=same_padding(kernel_size)),
                nn.Conv2d(in_channels=x_channels, out_channels=x_channels, kernel_size=11, padding=same_padding(kernel_size)),
            ]

        if squeeze_excitation:
            first_convs.insert(3, SqueezeExcitation(channels=x_channels, ratio=16))

        self.convs = nn.ModuleList([
            *first_convs,
            nn.Conv2d(in_channels=x_channels, out_channels=x_channels, kernel_size=1),
            nn.Conv2d(in_channels=x_channels, out_channels=num_classes, kernel_size=1)
        ])

        self.relu = nn.ReLU()

    def forward(self, x, image):
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < len(self.convs) - 1:
                x = self.relu(x)

        return x


class ConvolutionalPoseMachines(BaseModel):

    def __init__(self, x_channels=128, stage_channels=512, num_stages=3, num_classes=23,
                 depthwise_separable_convolution=True, squeeze_excitation=True, dilation=1):
        super(ConvolutionalPoseMachines, self).__init__()

        self.stage_1 = Stage1(x_channels, stage_channels, num_classes, depthwise_separable_convolution, squeeze_excitation, dilation)
        self.X = X(x_channels, depthwise_separable_convolution, squeeze_excitation, dilation)
        stages = []
        for _ in range(num_stages - 1):
            stages.append(StageN(x_channels, num_classes, depthwise_separable_convolution, squeeze_excitation))

        self.stages = nn.ModuleList(stages)

    def forward(self, image):
        out = []
        x = self.stage_1(image)
        out.append(x)
        for stage in self.stages:
            x_prime = self.X(image)
            x = torch.cat([x, x_prime], dim=1)
            x = stage(x, image)
            out.append(x)

        return torch.stack(out)
