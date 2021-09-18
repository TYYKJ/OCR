# @Time    : 2021/9/11 上午11:12
# @Author  : 
# @File    : model
# @Software: PyCharm
# @explain :
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl


class DetModel(pl.LightningModule):

    def forward(self, x):
        features = self.encoder(x)
        features = self.neck(features)
        features = self.head(features)

        return features

    def training_step(self, batch, batch_idx):
        data = batch
        output = self.forward(data['img'])
        loss_dict = self.loss_func(output, batch)

        self.log(name=self.train_loss_name, value=loss_dict['loss'])
        self.log(name='loss_shrink_maps', value=loss_dict['loss_shrink_maps'])
        self.log(name='loss_threshold_maps', value=loss_dict['loss_threshold_maps'])
        self.log(name='loss_binary_maps', value=loss_dict['loss_binary_maps'])

        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        data = batch
        output = self.forward(data['img'])
        loss_dict = self.loss_func(output, batch)

        self.log(name='val_loss', value=loss_dict['loss'])
        self.log(name='loss_shrink_maps', value=loss_dict['loss_shrink_maps'])
        self.log(name='loss_threshold_maps', value=loss_dict['loss_threshold_maps'])
        self.log(name='loss_binary_maps', value=loss_dict['loss_binary_maps'])

        return loss_dict['loss']

    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)

