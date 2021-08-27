
import torch.nn as nn
from pretrainedmodels.models.torchvision_models import pretrained_settings
from torchvision.models.vgg import VGG
from torchvision.models.vgg import make_layers

from ._base import EncoderMixin

# fmt: off
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGEncoder(VGG, EncoderMixin):

    def __init__(self, out_channels, config, batch_norm=False, depth=5, **kwargs):
        # 返回VGG模型, 在self参数里
        super().__init__(make_layers(config, batch_norm=batch_norm), **kwargs)
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3
        del self.classifier

    def make_dilated(self, stage_list, dilation_list):
        raise ValueError("'VGG' models do not support dilated mode due to Max Pooling"
                         " operations for down sampling!")

    def get_stages(self):
        stages = []
        stage_modules = []

        for module in self.features:
            if isinstance(module, nn.MaxPool2d):
                stages.append(nn.Sequential(*stage_modules))
                stage_modules = []
            stage_modules.append(module)
        stages.append(nn.Sequential(*stage_modules))
        return stages

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)
        # input: 1x3x224x224
        # 0-> 1x64x224x224
        # 1-> 1x128x112x112
        # 2-> 1x256x56x56
        # 3-> 1x512x28x28
        # 4-> 1x512x14x14
        # 5-> 1x512x7x7
        return features[2:]

    def load_state_dict(self, state_dict, **kwargs):
        keys = list(state_dict.keys())
        for k in keys:
            if k.startswith('classifier'):
                state_dict.pop(k)
        super(VGGEncoder, self).load_state_dict(state_dict, **kwargs)


vgg_encoders = {
    "vgg11": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg11"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["A"],
            "batch_norm": False,
        },
    },
    "vgg11_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg11_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["A"],
            "batch_norm": True,
        },
    },
    "vgg13": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg13"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["B"],
            "batch_norm": False,
        },
    },
    "vgg13_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg13_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["B"],
            "batch_norm": True,
        },
    },
    "vgg16": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg16"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["D"],
            "batch_norm": False,
        },
    },
    "vgg16_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg16_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["D"],
            "batch_norm": True,
        },
    },
    "vgg19": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg19"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["E"],
            "batch_norm": False,
        },
    },
    "vgg19_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg19_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["E"],
            "batch_norm": True,
        },
    },
}
