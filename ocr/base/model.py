# @Time    : 2021/7/30 下午4:51
# @Author  : cattree
# @File    : model
# @Software: PyCharm
# @explain :
import pytorch_lightning as pl
from . import initialization as init
from ..utils import optim


class OCRModel(pl.LightningModule):

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
        # TODO: CTC还需要两个参数
        x, y = batch
        predict = self.forward(x)
        loss = self.loss_func(predict, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predict = self.forward(x)
        loss = self.loss_func(predict, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return optim.get_optimizer(self.parameters(), self.optimizer_name, self.lr)
