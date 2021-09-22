# @Time    : 2021/9/11 下午1:47
# @Author  : 
# @File    : db_model
# @Software: PyCharm
# @explain :
from ..base import DetModel, DBHead
from ..encoders import get_encoder
from ..fpn import DBFpn
from ..utils import DBLoss
from ..postprocess import DBPostProcess
from ..metric import DetMetric


class DBDetModel(DetModel):

    def __init__(
            self,
            encoder_name: str = 'resnet18vd',
            optimizer_name: str = 'adam',
            lr: float = 0.001,
            thresh: float = 0.3,
            unclip_ratio: float = 1.5,
            weight_decay: float = 1e-4,
            train_loss_name='train_loss'
    ):
        super(DBDetModel, self).__init__()

        self.loss_func = DBLoss()
        self.train_loss_name = train_loss_name
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.lr = lr

        self.encoder = get_encoder(encoder_name)
        self.neck = DBFpn(self.encoder.out_channels)
        self.head = DBHead(in_channels=self.neck.out_channels)

        self.postprocess = DBPostProcess(thresh=thresh, unclip_ratio=unclip_ratio)
        self.metric = DetMetric()
