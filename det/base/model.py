# @Time    : 2021/8/12 下午2:26
# @Author  : cattree
# @File    : model
# @Software: PyCharm
# @explain :
import pytorch_lightning as pl

from ocr.utils import optim


class OCRDetModel(pl.LightningModule):

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        head = self.head(decoder_output)

        return head

    def training_step(self, batch, batch_idx):
        x, y = batch
        predict = self.forward(x)
        loss = self.loss_func(predict, batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predict = self.forward(x)
        loss = self.loss_func(predict, batch)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return optim.get_optimizer(self.parameters(), self.optimizer_name, self.lr)
