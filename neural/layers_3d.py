import collections
import torch.nn as nn

from neural.layers import NormActivation


class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 3, 3), dilation=(1, 1, 1), bias=False):
        super().__init__()
        assert len(kernel_size) == 3, 'kernel_size must be a 3-tuple.'
        time_pad = (kernel_size[0] - 1) * dilation[0]
        height_pad = ((kernel_size[1] - 1) * dilation[1]) // 2
        width_pad = ((kernel_size[2] - 1) * dilation[2]) // 2

        # Pad temporally on the left
        self.pad = nn.ConstantPad3d(padding=(width_pad, width_pad, height_pad, height_pad, time_pad, 0), value=0)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, dilation=dilation, stride=1, padding=0, bias=bias)

    # pylint: disable=arguments-differ
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x


class Bottleneck3D(nn.Module):
    """
    Defines a 3D bottleneck module with a residual connection.
    """
    def __init__(self, in_channels, out_channels=None, kernel_size=(2, 3, 3), dilation=(1, 1, 1), low_rank=False, upsample=False,
                 downsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.half_channels = int(in_channels / 2)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.low_rank = low_rank
        self.upsample = upsample
        self.downsample = downsample
        self.out_channels = out_channels or self.in_channels

        # Define the main conv operation
        assert not (low_rank and upsample), 'Error, both upsample and low rank is not supported.'
        assert not (low_rank and downsample), 'Error, both downsample and low rank is not supported.'
        assert not (upsample and downsample), 'Error, both downsample and upsample is not supported.'

        if self.low_rank:
            raise NotImplementedError()
        elif self.upsample:
            raise NotImplementedError()
        elif self.downsample:
            raise NotImplementedError()
        else:
            bottleneck_conv = CausalConv3d(self.half_channels, self.half_channels, kernel_size=self.kernel_size,
                                           dilation=self.dilation, bias=False)

        self.layers = nn.Sequential(collections.OrderedDict([
            # First projection with 1x1 kernel
            ('conv_down_project', nn.Conv3d(self.in_channels, self.half_channels, kernel_size=1, bias=False)),
            ('abn_down_project', NormActivation(num_features=self.half_channels, dimension='3d', activation='leaky_relu')),
            # Second conv block
            ('conv', bottleneck_conv),
            ('abn', NormActivation(num_features=self.half_channels, dimension='3d', activation='leaky_relu')),
            # Final projection with 1x1 kernel
            ('conv_up_project', nn.Conv3d(self.half_channels, self.out_channels, kernel_size=1, bias=False)),
            ('abn_up_project', NormActivation(num_features=self.out_channels, dimension='3d', activation='leaky_relu')),
            # Regulariser
            ('dropout', nn.Dropout2d(p=0.2))
        ]))

        if self.out_channels != self.in_channels:
            raise NotImplementedError()
        else:
            self.projection = None

    # pylint: disable=arguments-differ
    def forward(self, *args):
        x, = args
        x_residual = self.layers(x)
        if self.downsample:
            x = nn.functional.max_pool3d(x, kernel_size=2, stride=2)
        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if self.out_channels != self.in_channels:
            x = self.projection(x)
        return x + x_residual