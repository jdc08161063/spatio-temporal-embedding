import torch.nn as nn

from neural.layers_3d import CausalConv3d, Bottleneck3D
from neural.layers import NormActivation
from neural.utils import print_model_spec


class TemporalModel(nn.Module):
    def __init__(self, input_features=128, model_name='baseline'):
        super().__init__()
        self.model_name = model_name
        self.model, receptive_field = DynamicsFactory(model_name=model_name, input_features=input_features).get_model()
        self.receptive_field = receptive_field
        self.output_features = input_features

        print_model_spec(self, 'Temporal')

    def forward(self, x):
        if self.model_name == 'gru':
            return self.model(x)

        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x


class DynamicsFactory:
    def __init__(self, model_name, input_features):
        self.model_name = model_name
        self.input_features = input_features
        self.receptive_field = 1

    def get_model(self):
        input_features = self.input_features
        if self.model_name == 'no_temporal':
            return nn.Sequential(), self.receptive_field
        elif self.model_name == 'baseline':
            self.receptive_field = 2
            return nn.Sequential(CausalConv3d(input_features, input_features, (2, 3, 3), dilation=(1, 1, 1)),
                                 NormActivation(input_features, dimension='3d', activation='leaky_relu'),
                                 CausalConv3d(input_features, input_features, (1, 3, 3), dilation=(1, 1, 1)),
                                 NormActivation(input_features, dimension='3d', activation='leaky_relu'),
                                 CausalConv3d(input_features, input_features, (1, 3, 3), dilation=(1, 1, 1)),
                                 NormActivation(input_features, dimension='3d', activation='leaky_relu'),
                                 ), self.receptive_field
        elif self.model_name == 'small':
            self.receptive_field = 3
            model = []
            for i in range(1):
                model.append(Bottleneck3D(input_features, kernel_size=(2, 3, 3), dilation=(1, 1, 1)))
            for i in range(10):
                model.append(Bottleneck3D(input_features, kernel_size=(1, 3, 3), dilation=(1, 1, 1)))
            model.append(Bottleneck3D(input_features, kernel_size=(2, 3, 3), dilation=(1, 1, 1)))

            return nn.Sequential(*model), self.receptive_field
        elif self.model_name == 'medium':
            self.receptive_field = 3
            model = []
            for i in range(1):
                model.append(Bottleneck3D(input_features, kernel_size=(2, 3, 3), dilation=(1, 1, 1)))
            for i in range(20):
                model.append(Bottleneck3D(input_features, kernel_size=(1, 3, 3), dilation=(1, 1, 1)))

            model.append(Bottleneck3D(input_features, kernel_size=(2, 3, 3), dilation=(1, 1, 1)))

            return nn.Sequential(*model), self.receptive_field
        elif self.model_name == 'large':
            self.receptive_field = 3
            model = []
            for i in range(1):
                model.append(Bottleneck3D(input_features, kernel_size=(2, 3, 3), dilation=(1, 1, 1)))
            for i in range(40):
                model.append(Bottleneck3D(input_features, kernel_size=(1, 3, 3), dilation=(1, 1, 1)))

            model.append(Bottleneck3D(input_features, kernel_size=(2, 3, 3), dilation=(1, 1, 1)))

            return nn.Sequential(*model), self.receptive_field
        else:
            raise ValueError('Dynamics model {} not implemented.'.format(self.model_name))

