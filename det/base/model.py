
import pytorch_lightning as pl

from rec.utils import optim


class OCRDetModel(pl.LightningModule):

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        head = self.head(decoder_output)

        return head

    def training_step(self, batch, batch_idx):
        img, gt_score, gt_geo, ignored_map = batch
        pred_score, pred_geo = self.forward(img)

        loss = self.loss_func(gt_score, pred_score, gt_geo, pred_geo, ignored_map)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, gt_score, gt_geo, ignored_map = batch
        pred_score, pred_geo = self.forward(img)

        loss = self.loss_func(gt_score, pred_score, gt_geo, pred_geo, ignored_map)

        self.log('val_loss', loss)

    def configure_optimizers(self):
        return optim.get_optimizer(self.parameters(), self.optimizer_name, self.lr)
