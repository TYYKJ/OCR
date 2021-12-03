# @Time    : 2021/9/28 下午2:37
# @Author  : 
# @File    : dbnet
# @Software: PyCharm
# @explain :
from __future__ import annotations

__all__ = ['DBDetModel']

from ..base import BaseModel
from ..criteria import DBLoss
from ..encoders import get_encoder
from ..fpn import DBFpn
from ..heads import DBHead
from ..metric import DetMetric
from ..postprocess import DBPostProcess


class DBDetModel(BaseModel):

    def __init__(
            self,
            encoder_name: str = 'resnet50',
            thresh: float = 0.3,
            unclip_ratio: float = 1.5,
            lr: float = 0.001,
            optimizer_name: str = 'sgd',
            weight_decay: float | None = None,
            momentum: float | None = None,
            weights: str = 'imagenet',
    ):
        super(DBDetModel, self).__init__()

        self.save_hyperparameters()

        self.encoder_name = encoder_name

        self.loss_func = DBLoss()

        self.lr = lr
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.encoder = get_encoder(encoder_name, weights)
        self.neck = DBFpn(self.encoder.out_channels)
        self.head = DBHead(in_channels=self.neck.out_channels)

        self.postprocess = DBPostProcess(thresh=thresh, unclip_ratio=unclip_ratio)
        self.metric = DetMetric()
