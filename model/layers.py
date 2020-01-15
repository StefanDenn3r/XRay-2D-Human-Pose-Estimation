from torch import nn

from base import BaseModel


def same_padding(kernel_size, dilation=1):
    return (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2


def calculate_kernel_size(receptive_field, dilation):
    return (dilation - 1 + receptive_field) // dilation


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


class DepthwiseSeparableConvolution(BaseModel):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(DepthwiseSeparableConvolution, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=same_padding(kernel_size, dilation), dilation=dilation)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x
