from typing import Union, List, Dict, Any, cast, Tuple

import torch
import torch.nn as nn

__all__ = ['vgg_encoders']

from torch.nn import Sequential


class VGG(nn.Module):

    def __init__(
            self,
            out_channels: int,
            features: nn.Module,
            init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.out_channels = out_channels

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> Tuple[Sequential, int]:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    out_channels = in_channels
    return nn.Sequential(*layers), out_channels


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(cfg: str, batch_norm: bool, **kwargs: Any) -> VGG:
    feature, out_channels = make_layers(cfgs[cfg], batch_norm=batch_norm)
    model = VGG(out_channels, feature, **kwargs)

    return model


def vgg11() -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    """
    return _vgg('A', False)


def vgg11_bn() -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    """
    return _vgg('A', True)


def vgg13() -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    """
    return _vgg('B', False)


def vgg13_bn() -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    """
    return _vgg('B', True)


def vgg16() -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    """
    return _vgg('D', False)


def vgg16_bn() -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    """
    return _vgg('D', True)


def vgg19() -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    """
    return _vgg('E', False)


def vgg19_bn() -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    """
    return _vgg('E', True)


vgg_encoders = {
    "vgg11": {
        "encoder": vgg11,
        "params": {}
    },
    "vgg11_bn": {
        "encoder": vgg11_bn,
        "params": {}
    },
    "vgg13": {
        "encoder": vgg13,
        "params": {}
    },
    "vgg13_bn": {
        "encoder": vgg13_bn,
        "params": {}
    },
    "vgg16": {
        "encoder": vgg16,
        "params": {}
    },
    "vgg16_bn": {
        "encoder": vgg16_bn,
        "params": {}
    },
    "vgg19": {
        "encoder": vgg19,
        "params": {}
    },
    "vgg19_bn": {
        "encoder": vgg19_bn,
        "params": {}
    },
}
