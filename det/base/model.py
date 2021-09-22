# @Time    : 2021/9/11 上午11:12
# @Author  : 
# @File    : model
# @Software: PyCharm
# @explain :
import torch
from torch.optim import Adam
import pytorch_lightning as pl
from ..utils import get_optimizer


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
        boxes, scores = self.postprocess(output.cpu().numpy(), batch['shape'])
        raw_metric = self.metric(batch, (boxes, scores))

        return raw_metric

    def validation_epoch_end(self, outputs):

        metric = self.metric.gather_measure(outputs)
        self.log('recall', value=metric['recall'].avg)
        self.log('precision', value=metric['precision'].avg)
        self.log('hmean', value=metric['fmeasure'].avg)
        return {'hmean': metric['fmeasure'].avg}

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), self.optimizer_name, self.lr, self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "hmean"}
