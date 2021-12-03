import pytorch_lightning as pl
import torch

from ...utils import create_optimizer_v2


class BaseModel(pl.LightningModule):

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

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(self.parameters(), opt=self.optimizer_name, lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, "monitor": 'hmean'}
