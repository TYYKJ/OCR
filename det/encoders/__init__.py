# @Time    : 2021/8/12 上午8:45
# @Author  : cattree
# @File    : __init__.py
# @Software: PyCharm
# @explain :
import torch.utils.model_zoo as model_zoo

from .resnet import resnet_encoders

encoders = {}
encoders.update(resnet_encoders)


def get_encoder(name, in_channels=3, depth=5, weights=None):

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(name, list(encoders.keys())))

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError("Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                weights, name, list(encoders[name]["pretrained_settings"].keys()),
            ))
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    return encoder


def get_encoder_names():
    return list(encoders.keys())
