# @Time    : 2021/7/30 下午5:12
# @Author  : cattree
# @File    : __init__.py
# @Software: PyCharm
# @explain :
from .RecResNetvd import resnet_encoders

encoders = {}

encoders.update(resnet_encoders)


def get_encoder(encoder_name: str):
    """
    获取编码器

    :param encoder_name: 编码器名称
    :return:
    """

    try:
        encoder_model = encoders[encoder_name]['encoder']
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(encoder_name, list(encoders.keys())))

    params = encoders[encoder_name]['params']
    encoder = encoder_model(**params)

    return encoder
