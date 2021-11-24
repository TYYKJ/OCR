import pytorch_lightning as pl
import torch.optim

from ..encoders import get_encoder
from ..heads import CTC
from ..losses import CTCLoss
from ..metric import RecMetric
from ..necks import SequenceEncoder
from ..utils.label_convert import CTCLabelConverter


class CRNN(pl.LightningModule):

    def __init__(
            self,
            encoder_name: str,
            classes: int,
            alphabet_path: str,
            lr: float,
            train_loss_name: str = 'train_loss'
    ):
        super(CRNN, self).__init__()
        # model = CRNN.load_from_checkpoint(PATH, alphabet_path='xxx')
        self.save_hyperparameters(ignore=['alphabet_path'])
        self.encoder = get_encoder(encoder_name)
        self.neck = SequenceEncoder(in_channels=self.encoder.out_channels)
        self.head = CTC(self.neck.out_channels, classes)
        self.converter = CTCLabelConverter(alphabet_path)
        self.losses = CTCLoss(blank_idx=0)
        self.metric = RecMetric(self.converter)

        self.train_loss_name = train_loss_name
        self.val_loss_name = 'val_loss'
        self.lr = lr

    def forward(self, x):
        features = self.encoder(x)
        features = self.neck(features)
        features = self.head(features)
        return features

    def training_step(self, batch, batch_idx):
        cur_batch_size = batch['img'].shape[0]
        targets, targets_lengths = self.converter.encode(batch['label'])
        batch['targets'] = targets
        batch['targets_lengths'] = targets_lengths

        predict = self.forward(batch['img'])
        loss_dict = self.losses(predict, batch)

        acc_dict = self.metric(predict, batch['label'])
        acc = acc_dict['n_correct'] / cur_batch_size
        norm_edit_dis = 1 - acc_dict['norm_edit_dis'] / cur_batch_size

        self.log(name=self.train_loss_name, value=loss_dict.get('loss'), on_step=False, on_epoch=True)
        self.log(name='train_acc', value=acc, on_step=False, on_epoch=True)
        self.log(name='norm_edit_dis', value=norm_edit_dis, on_step=False, on_epoch=True)

        return loss_dict.get('loss')

    def validation_step(self, batch, batch_idx):
        cur_batch_size = batch['img'].shape[0]
        targets, targets_lengths = self.converter.encode(batch['label'])
        batch['targets'] = targets
        batch['targets_lengths'] = targets_lengths
        predict = self.forward(batch['img'])
        loss_dict = self.losses(predict, batch)
        acc_dict = self.metric(predict, batch['label'])

        acc = acc_dict['n_correct'] / cur_batch_size
        self.log(name=self.val_loss_name, value=loss_dict.get('loss'), on_step=False, on_epoch=True)
        self.log(name='eval_acc', value=acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, "monitor": self.val_loss_name}
