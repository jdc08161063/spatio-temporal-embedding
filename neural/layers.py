import torch

import torch.nn as nn

from functools import partial


class NormActivation(nn.Module):
    def __init__(self, num_features, dimension='2d', activation='none', momentum=0.05, slope=0.01):
        super().__init__()

        if dimension == '1d':
            self.norm = nn.BatchNorm1d(num_features=num_features, momentum=momentum)
        elif dimension =='2d':
            self.norm = nn.BatchNorm2d(num_features=num_features, momentum=momentum)
        elif dimension == '3d':
            self.norm = nn.BatchNorm3d(num_features=num_features, momentum=momentum)
        else:
            raise ValueError('Dimension={} not handled.'.format(dimension))

        if activation == "relu":
            self.activation_fn = lambda x: nn.functional.relu(x, inplace=True)
        elif activation == "leaky_relu":
            self.activation_fn = lambda x: nn.functional.leaky_relu(x, negative_slope=slope, inplace=True)
        elif activation == "elu":
            self.activation_fn = lambda x: nn.functional.elu(x, inplace=True)
        elif activation == "none":
            self.activation_fn = lambda x: x
        else:
            raise ValueError('Activation={} not handled.'.format(self.activation))

    def forward(self, x):
        x = self.norm(x)
        x = self.activation_fn(x)
        return x


class ConvBlock(nn.Module):
    """ Conv and optional (BN - ReLU)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm='none', activation='none', bias=False,
                 transpose=False):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d if not transpose else partial(nn.ConvTranspose2d, output_padding=1)

        if norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm =='bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            raise ValueError('Not recognised norm {}'.format(norm))

        if activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError('Not recognised activation {}'.format(activation))

        self.conv = self.conv(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

    def forward(self, x):
        x = self.conv(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class ResBlock(nn.Module):
    """ Conv - BN - ReLU - Conv - BN - ADD and then ReLU
        Same number of channels in and out.
    """
    def __init__(self, channels, norm='in', activation='lrelu', bias=False, last_block=False):
        super().__init__()
        if activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError('Not recognised activation {}'.format(activation))

        self.model = []

        self.model.append(ConvBlock(channels, channels, 3, 1, norm=norm, activation=activation, bias=bias))
        if last_block:
            norm = 'none'
            bias = True
            self.activation = None
        self.model.append(ConvBlock(channels, channels, 3, 1, norm=norm, activation='none', bias=bias))

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        identity = x
        x = self.model(x)
        x += identity
        if self.activation:
            x = self.activation(x)
        return x
