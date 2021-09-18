# @Time    : 2021/9/11 下午1:47
# @Author  : 
# @File    : db_model
# @Software: PyCharm
# @explain :
from ..base import DetModel, DBHead
from ..encoders import get_encoder
from ..fpn import DBFpn
from ..utils import DBLoss


class DBDetModel(DetModel):

    def __init__(
            self,
            encoder_name: str = 'resnet18vd',
            lr: float = 0.001,
            train_loss_name='train_loss'
    ):
        super(DBDetModel, self).__init__()

        self.loss_func = DBLoss()
        self.train_loss_name = train_loss_name
        self.lr = lr

        self.encoder = get_encoder(encoder_name)
        self.neck = DBFpn(self.encoder.out_channels)
        self.head = DBHead(in_channels=self.neck.out_channels)
