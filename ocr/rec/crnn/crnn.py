import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from ..encoders import get_encoder
from ..heads import CTC
from ..losses import CTCLoss
from ..metric import RecMetric
from ..necks import SequenceEncoder
from ..utils.label_convert import CTCLabelConverter
from ...utils import create_optimizer_v2


class CRNN(pl.LightningModule):

    def __init__(
            self,
            encoder_name: str,
            classes: int,
            alphabet_path: str,
            lr: float,
            momentum: float,
            weight_decay: float,
            optimizer_name: str,
            logger: WandbLogger,
    ):
        super(CRNN, self).__init__()
        self.save_hyperparameters(ignore=['alphabet_path'])
        self.optimizer_name = optimizer_name
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.encoder_name = encoder_name
        self.encoder = get_encoder(encoder_name)
        self.neck = SequenceEncoder(in_channels=self.encoder.out_channels)
        self.head = CTC(self.neck.out_channels, classes)
        self.converter = CTCLabelConverter(alphabet_path)
        self.losses = CTCLoss(blank_idx=0)
        self.metric = RecMetric(self.converter)

        self.lr = lr
        self.wandb_logger = logger

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

        self.log(name='train_loss', value=loss_dict.get('loss'), on_step=False, on_epoch=True)
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
        # df = acc_dict['show_str']
        # self.wandb_logger.log_text(key='Predict', dataframe=df)
        self.log(name='val_loss', value=loss_dict.get('loss'), on_step=False, on_epoch=True)
        self.log(name='val_acc', value=acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(self.parameters(), opt=self.optimizer_name, lr=self.lr,
                                        momentum=self.momentum, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, "monitor": 'val_acc'}
