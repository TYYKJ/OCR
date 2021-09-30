# @Time    : 2021/9/28 上午9:44
# @Author  : 
# @File    : model
# @Software: PyCharm
# @explain :
import pytorch_lightning as pl
import torch

from ..optimizers import get_optimizer


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
        self.log(name='shrink_maps', value=loss_dict['loss_shrink_maps'])
        self.log(name='threshold_maps', value=loss_dict['loss_threshold_maps'])
        self.log(name='binary_maps', value=loss_dict['loss_binary_maps'])

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

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.optimizer_change_epoch:
            self.trainer.accelerator.setup_optimizers(self.trainer)

    def configure_optimizers(self):
        if self.optimizer_change:
            if self.current_epoch == self.optimizer_change_epoch:
                optimizer = get_optimizer(self.parameters(), 'sgd', self.lr)
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
                lr_scheduler.load_state_dict(self.trainer.lr_schedulers[0].state_dict())
                # only if you want to load the current state of the old learning rate.
            else:
                optimizer = get_optimizer(self.parameters(), 'adam', self.lr)
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, "monitor": self.val_loss_name}
        else:
            optimizer = get_optimizer(self.parameters(), self.optimizer_name, self.lr)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, "monitor": 'hmean'}
