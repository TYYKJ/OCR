# @Time    : 2021/9/28 下午2:37
# @Author  : 
# @File    : dbnet
# @Software: PyCharm
# @explain :
__all__ = ['DBDetModel']

from ..base import DetModel
from ..criteria import DBLoss
from ..encoders import get_encoder
from ..fpn import DBFpn
from ..heads import DBHead
from ..metric import DetMetric
from ..postprocess import DBPostProcess


class DBDetModel(DetModel):

    def __init__(
            self,
            pretrained_model_path: str,
            encoder_name: str = 'resnet50',
            thresh: float = 0.3,
            unclip_ratio: float = 1.5,
            weight_decay: float = 1e-4,
            lr: float = 0.001,
            optimizer_name: str = 'sgd',
            train_loss_name: str = 'train_loss',
            val_loss_name: str = 'val_loss',
            weights: str = 'imagenet',
    ):
        super(DBDetModel, self).__init__()

        self.finetune = True if pretrained_model_path else False
        self.pretrained_model_path = pretrained_model_path

        self.save_hyperparameters(ignore='pretrained_model_path')

        self.encoder_name = encoder_name

        self.loss_func = DBLoss()

        self.lr = lr
        self.optimizer_name = optimizer_name
        self.train_loss_name = train_loss_name
        self.val_loss_name = val_loss_name
        self.weight_decay = weight_decay

        self.encoder = get_encoder(encoder_name, weights)
        self.neck = DBFpn(self.encoder.out_channels)
        self.head = DBHead(in_channels=self.neck.out_channels)

        self.postprocess = DBPostProcess(thresh=thresh, unclip_ratio=unclip_ratio)
        self.metric = DetMetric()
