
import pytorch_lightning as pl
import torch

from ..utils import optim


class OCRModel(pl.LightningModule):

    def forward(self, x):
        features = self.encoder(x)
        features = self.im_seq(features)
        features = self.neck(features)
        features = self.head(features)

        return features

    def training_step(self, batch, batch_idx):
        x, y, y_len = batch

        predict = self.forward(x)

        log_probs = predict.log_softmax(dim=2)
        batch_size = x.size(0)
        input_lengths = torch.LongTensor([predict.size(0)] * batch_size)
        target_lengths = torch.flatten(y_len)

        loss = self.loss_func(log_probs, y, input_lengths, target_lengths)

        self.log(name='train', value=loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, y_len = batch

        predict = self.forward(x)

        log_probs = predict.log_softmax(dim=2)
        batch_size = x.size(0)
        input_lengths = torch.LongTensor([predict.size(0)] * batch_size)
        target_lengths = torch.flatten(y_len)

        loss = self.loss_func(log_probs, y, input_lengths, target_lengths)

        self.log(name='val', value=loss)

        return loss

    def configure_optimizers(self):
        return optim.get_optimizer(self.parameters(), self.optimizer_name, self.lr)
