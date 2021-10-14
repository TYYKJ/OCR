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
            encoder_name: str = 'resnet50vd',
            thresh: float = 0.3,
            unclip_ratio: float = 1.5,
            weight_decay: float = 1e-4,
            lr: float = 0.001,
            optimizer_name: str = 'adam',
            train_loss_name: str = 'train_loss',
            val_loss_name: str = 'val_loss',
    ):
        super(DBDetModel, self).__init__()

        self.save_hyperparameters()

        self.loss_func = DBLoss()

        self.lr = lr
        self.optimizer_name = optimizer_name
        self.train_loss_name = train_loss_name
        self.val_loss_name = val_loss_name
        self.weight_decay = weight_decay

        self.encoder = get_encoder(encoder_name)
        self.neck = DBFpn(self.encoder.out_channels)
        self.head = DBHead(in_channels=self.neck.out_channels)

        self.postprocess = DBPostProcess(thresh=thresh, unclip_ratio=unclip_ratio)
        self.metric = DetMetric()
