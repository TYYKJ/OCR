# @Time    : 2021/9/26 下午2:34
# @Author  : 
# @File    : models
# @Software: PyCharm
# @explain :

import pytorch_lightning as pl
import torch
from ..optimizers import get_optimizer


class BaseModel(pl.LightningModule):

    # def initialize(self):
    #     init.initialize_backbone(self.encoder)
    #     init.initialize_neck(self.neck)
    #     init.initialize_head(self.head)

    def forward(self, x):
        features = self.encoder(x)
        # features = self.im_seq(features)
        features = self.neck(features)
        features = self.head(features)

        return features

    def training_step(self, batch, batch_idx):
        cur_batch_size = batch['img'].shape[0]
        targets, targets_lengths = self.convert.encode(batch['label'])

        batch['targets'] = targets
        batch['targets_lengths'] = targets_lengths

        predict = self.forward(batch['img'])

        loss = self.loss_func(predict, batch)

        acc_dict = self.metric(predict, batch['label'])
        acc = acc_dict['n_correct'] / cur_batch_size
        norm_edit_dis = 1 - acc_dict['norm_edit_dis'] / cur_batch_size

        self.log(name=self.train_loss_name, value=loss, on_step=False, on_epoch=True)
        self.log(name='train_acc', value=acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        cur_batch_size = batch['img'].shape[0]
        targets, targets_lengths = self.convert.encode(batch['label'])
        batch['targets'] = targets
        batch['targets_lengths'] = targets_lengths
        predict = self.forward(batch['img'])
        loss = self.loss_func(predict, batch)
        acc_dict = self.metric(predict, batch['label'])

        acc = acc_dict['n_correct'] / cur_batch_size
        self.log(name=self.val_loss_name, value=loss, on_step=False, on_epoch=True)
        self.log(name='eval_acc', value=acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), self.optimizer_name, self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, "monitor": self.val_loss_name}
