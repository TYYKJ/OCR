# @Time    : 2021/9/11 上午10:59
# @Author  : 
# @File    : __init__.py
# @Software: PyCharm
# @explain :
import torch.utils.model_zoo as model_zoo

from .densenet import densenet_encoders
from .dpn import dpn_encoders
from .mobilenet import mobilenet_encoders
from .resnet import resnet_encoders
from .senet import senet_encoders

encoders = {}

encoders.update(resnet_encoders)
encoders.update(senet_encoders)
encoders.update(densenet_encoders)
encoders.update(mobilenet_encoders)
encoders.update(dpn_encoders)


def get_encoder(encoder_name: str, weights=None):
    """
    获取编码器

    :param encoder_name: 编码器名称
    :param weights:
    :return:
    """

    try:
        encoder_model = encoders[encoder_name]['encoder']
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(encoder_name, list(encoders.keys())))

    params = encoders[encoder_name]['params']
    encoder = encoder_model(**params)

    if weights is not None:
        try:
            settings = encoders[encoder_name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError("Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                weights, encoder_name, list(encoders[encoder_name]["pretrained_settings"].keys()),
            ))
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    return encoder
