import torch
import torchvision

import torch.nn as nn

from neural.resnet import ResnetEncoder, ResnetDecoder
from neural.dynamics.dynamics_factory import TemporalModel
from neural.utils import print_model_spec, require_grad
from monodepth.layers import ConvBlock, Conv3x3


class TemporalEncoder(nn.Module):
    """ Encoding + temporal model. Temporal model can be set to identity to have a static model."""
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        self.encoder = Encoder(
            encoder_name=self.config['encoder_name'],
            pretrained_encoder_path=self.config['pretrained_encoder_path']
        )
        self.temporal_model = TemporalModel(input_features=self.encoder.output_features[-1],
                                            model_name=config['dynamics_model_name'])
        self.receptive_field = self.temporal_model.receptive_field
        self.output_features = self.encoder.output_features  # works as long as the temporal model does not change
                                                             # the number of channels

    def forward(self, x):
        """
        Parameters
        ----------
            x: torch.tensor (B, T, 3, H, W)

        Returns
        -------
            z: torch.tensor (B, T, C, H, W)
                temporal embedding
        """
        b, seq_len, c, h, w = x.shape

        encoder_outputs = self.encoder(x.view(b * seq_len, c, h, w))
        encoder_outputs = [encoder_outputs[i].view(b, seq_len, *encoder_outputs[i].shape[1:])
                           for i in range(len(encoder_outputs))]
        z = self.temporal_model(encoder_outputs[-1])

        return encoder_outputs[:-1] + [z]


class Encoder(nn.Module):
    def __init__(self, encoder_name='', pretrained_encoder_path=''):
        super().__init__()
        self.output_features = None
        self.encoder_name = encoder_name

        if self.encoder_name == 'resnet':
            self.model = ResnetEncoder()
            self.output_features = self.model.output_features

            if pretrained_encoder_path:
                print('Loading encoder weights from {}'.format(pretrained_encoder_path))
                checkpoint = torch.load(pretrained_encoder_path)
                self.model.load_state_dict(checkpoint['encoder'])
        else:
            deeplab = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
            self.backbone = deeplab.backbone
            self.aspp = deeplab.classifier[0]
            require_grad(self, False)
            self.output_features = 256

        print_model_spec(self, 'Encoder')

    def forward(self, x):
        if self.encoder_name in ['resnet']:
            # Returns a list of elements
            x = self.model(x)
        else:
            # Returns one element
            x = self.backbone(x)['out']
            x = self.aspp(x)
        return x


class InstanceDecoder(nn.Module):
    def __init__(self, decoder_name='resnet', emb_dim=8, instance=False, mask=False, config=None):
        super().__init__()
        self.config = config
        if decoder_name == 'resnet':
            self.model = ResnetDecoder(num_output_channels=emb_dim, instance=instance, mask=mask)
            if self.config['pretrained_encoder_path']:
                print('Loading decoder weights from {}'.format(self.config['pretrained_encoder_path']))
                checkpoint = torch.load(self.config['pretrained_encoder_path'])
                # last layer do not have same nb of channels
                try:
                    self.model.load_state_dict(checkpoint['decoder'])
                except RuntimeError:
                    print('Not loading weights from the last layer of the decoder.')
                    checkpoint['decoder'].pop('decoder.6.conv.weight', None)
                    checkpoint['decoder'].pop('decoder.6.conv.bias', None)
                    self.model.load_state_dict(checkpoint['decoder'], strict=False)

        print_model_spec(self, 'Instance decoder')

    def forward(self, input_features):
        """
        Parameters
        ----------
            z: torch.tensor (B, T, C, H, W)
                temporal embedding
        """
        b, seq_len = input_features[-1].shape[:2]
        input_features = [input_features[i].view(b * seq_len, *input_features[i].shape[2:])
                          for i in range(len(input_features))]
        output = self.model(input_features)
        for key in output.keys():
            if output[key] is not None:
                output[key] = output[key].view(b, seq_len, *output[key].shape[1:])

        output['y'] = output['instance']
        output.pop('instance', None)

        return output


class DepthEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv1 = ConvBlock(in_channels=(self.config['emb_dim'] + 1),
                               out_channels=self.config['emb_dim'])
        self.conv2 = Conv3x3(in_channels=self.config['emb_dim'],
                             out_channels=self.config['emb_dim'])

    def forward(self, output):
        """
        Parameters
        ----------
            output: dict with keys:
                y: torch.tensor (batch_size, seq_len, emb_dim, H, W)
                depth: torch.tensor (batch_size, seq_len, 1, H, W)
        """
        b, seq_len, emb_dim, h, w = output['y'].shape

        y = output['y'].view(b*seq_len, emb_dim, h, w)
        depth = output['depth'].view(b*seq_len, 1, h, w)

        depth_y = torch.cat([y, depth], dim=1)

        # Convolution 1
        x = self.conv1(depth_y)
        # Convolution 2
        x = self.conv2(x)

        x = x.view(b, seq_len, emb_dim, h, w)

        return {'y': x}
