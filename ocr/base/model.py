# @Time    : 2021/7/30 下午4:51
# @Author  : cattree
# @File    : model
# @Software: PyCharm
# @explain :
import pytorch_lightning as pl
from . import initialization as init
from ..utils import optim, CTCLabelConvert
from ..utils.metric import RecMetric


class OCRModel(pl.LightningModule):

    def __init__(self, character_path):
        super(OCRModel, self).__init__()
        self.converter = CTCLabelConvert.CTCLabelConverter(
            character=character_path)
        self.metric = RecMetric(self.converter)

    def initialize(self):
        # 以下变量都是放在self这里面的
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.neck)

    def forward(self, x):
        features = self.encoder(x)
        features = self.seq(features)
        features = self.neck(features)
        features = self.head(features)

        return features

    def training_step(self, batch, batch_idx):
        x, y = batch
        targets, targets_lengths = self.converter.encode(y)
        predict = self.forward(x)

        data = {
            'targets': targets,
            'targets_lengths': targets_lengths
        }

        loss = self.loss_func(predict, data)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        targets, targets_lengths = self.converter.encode(y)
        predict = self.forward(x)
        data = {
            'targets': targets,
            'targets_lengths': targets_lengths
        }
        loss = self.loss_func(predict, data)
        acc_dict = self.metric(predict, y)
        acc = acc_dict['n_correct'] / self.bs
        # self.log('str', acc_dict['show_str'])
        print(acc_dict['show_str'])
        self.log('acc', acc)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return optim.get_optimizer(self.parameters(), self.optimizer_name, self.lr)
