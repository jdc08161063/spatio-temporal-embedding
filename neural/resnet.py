import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

from collections import OrderedDict

from monodepth.layers import ConvBlock, Conv3x3, upsample

ENCODER_CHANNELS = np.array([64, 64, 128, 256, 512])
DECODER_CHANNELS = np.array([16, 32, 64, 128, 256])


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers=18, pretrained=True, use_skips=True):
        super().__init__()

        self.use_skips = use_skips
        self.num_ch_enc = ENCODER_CHANNELS
        self.num_ch_dec = DECODER_CHANNELS

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        # Upsample twice
        self.convs = OrderedDict()
        for i in range(4, 2, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        self.decoder = nn.ModuleList(list(self.convs.values()))

        self.output_features = [64, 64, 128]

    def forward(self, input_image):
        """
        Returns
        -------
            features0, features1 (from encoder) and output features after 2 steps of decoding
        """
        features = []
        x = input_image  # 3x96x320
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))  # 64x48x160
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))  # 64x28x80
        features.append(self.encoder.layer2(features[-1]))  # 128x12x40
        features.append(self.encoder.layer3(features[-1]))  # 256x6x20
        features.append(self.encoder.layer4(features[-1]))  # 512x3x10

        x = features[-1]
        for i in range(4, 2, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

        output_features = [features[0], features[1], x]  # x is shape 128x12x40

        return output_features


class ResnetDecoder(nn.Module):
    def __init__(self, num_output_channels=1, use_skips=True, depth=False, instance=False, segmentation=False,
                 mask=False, scales=range(3), n_classes=14):
        super().__init__()

        self.num_output_channels = num_output_channels  # is the instance number of output channels
        self.use_skips = use_skips
        self.depth = depth
        self.instance = instance
        self.segmentation = segmentation
        self.mask = mask
        self.scales = scales
        self.n_classes = n_classes
        self.upsample_mode = 'nearest'

        self.num_ch_enc = ENCODER_CHANNELS
        self.num_ch_dec = DECODER_CHANNELS

        # decoder
        self.convs = OrderedDict()
        for i in range(2, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[2] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        if self.depth:
            for s in self.scales:
                self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], 1)
            self.sigmoid = nn.Sigmoid()

        if self.instance:
            self.convs[("instance_conv")] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)

        if self.segmentation:
            self.convs[("segmentation_conv")] = Conv3x3(self.num_ch_dec[0], self.n_classes)

        if self.mask:
            self.convs[("mask_conv")] = Conv3x3(self.num_ch_dec[0], 2)

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        output = {}

        # decoder
        x = input_features[-1]
        for i in range(2, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales and self.depth:
                output[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        if self.instance:
            output['instance'] = self.convs[("instance_conv")](x)

        if self.segmentation:
            output['segmentation'] = self.convs[("segmentation_conv")](x)

        if self.mask:
            output['mask_logits'] = self.convs[('mask_conv')](x)

        return output
