import pytorch_lightning as pl
import torch

from ..base import initialization as init
from ..utils import optim


class OCRModel(pl.LightningModule):

    def initialize(self):
        init.initialize_backbone(self.encoder)
        init.initialize_neck(self.neck)
        init.initialize_head(self.head)

    def forward(self, x):
        features = self.encoder(x)
        features = self.im_seq(features)
        features = self.neck(features)
        features = self.head(features)

        return features

    def training_step(self, batch, batch_idx):
        targets, targets_lengths = self.convert.encode(batch['label'])
        batch['targets'] = targets
        batch['targets_lengths'] = targets_lengths

        predict = self.forward(batch['img'])

        loss = self.loss_func(predict, batch)

        self.log(name=self.train_loss_name, value=loss)

        return loss

    def validation_step(self, batch, batch_idx):
        targets, targets_lengths = self.convert.encode(batch['label'])
        batch['targets'] = targets
        batch['targets_lengths'] = targets_lengths

        predict = self.forward(batch['img'])

        loss = self.loss_func(predict, batch)

        self.log(name=self.val_loss_name, value=loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.get_optimizer(self.parameters(), self.optimizer_name, self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_loss"}
