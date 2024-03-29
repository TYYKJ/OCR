import pytorch_lightning as pl
import torch

from ..encoders import get_encoder
from ..heads import CTC
from ..losses import CTCLoss
from ..metric import RecMetric
from ..necks import SequenceEncoder
from ..utils.labelConvert import CTCLabelConverter
from ...utils import create_optimizer_v2


class CRNN(pl.LightningModule):

    def __init__(
            self,
            classes: int,
            alphabet_path: str,
            resume_model: bool,
            encoder_name: str = 'resnet18vd',
            optimizer_name: str = 'sgd',
            lr: float = 0.01,
            weight_decay: float = 0.,
            momentum: float = 0.9,
            hidden_size: int = 256,
            encoder_type: str = 'reshape',
    ):
        super(CRNN, self).__init__()
        self.save_hyperparameters(ignore=['alphabet_path'])
        self.optimizer_name = optimizer_name
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.encoder_name = encoder_name
        self.encoder = get_encoder(encoder_name)
        self.neck = SequenceEncoder(in_channels=self.encoder.out_channels,
                                    encoder_type=encoder_type, hidden_size=hidden_size)
        self.head = CTC(self.neck.out_channels, classes)
        self.converter = CTCLabelConverter(alphabet_path)
        self.losses = CTCLoss(blank_idx=0)
        self.metric = RecMetric(self.converter)

        self.lr = lr
        self.resume_model = resume_model

        self.all_train_acc = []
        self.all_acc = []

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

        self.all_train_acc.append(acc)

        self.log(name='train_loss', value=loss_dict.get('loss'))
        self.log(name='train_norm_edit_dis', value=norm_edit_dis)

        return loss_dict.get('loss')

    def training_epoch_end(self, outputs) -> None:
        avg_acc = sum(self.all_train_acc) / len(self.all_train_acc)
        self.log(name='train_acc', value=avg_acc)
        self.all_train_acc.clear()

    def validation_step(self, batch, batch_idx):
        cur_batch_size = batch['img'].shape[0]
        targets, targets_lengths = self.converter.encode(batch['label'])
        batch['targets'] = targets
        batch['targets_lengths'] = targets_lengths
        predict = self.forward(batch['img'])
        loss_dict = self.losses(predict, batch)
        acc_dict = self.metric(predict, batch['label'])

        acc = acc_dict['n_correct'] / cur_batch_size
        self.all_acc.append(acc)
        self.log(name='val_loss', value=loss_dict.get('loss'))
        # self.log(name='norm_edit_dis', value=loss_dict.get('norm_edit_dis'))

    def validation_epoch_end(self, outputs):
        avg_acc = sum(self.all_acc) / len(self.all_acc)
        self.log(name='val_acc', value=avg_acc)
        self.all_acc.clear()

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(self.parameters(), opt=self.optimizer_name, lr=self.lr,
                                        momentum=self.momentum, weight_decay=self.weight_decay)

        if not self.resume_model:
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                                 max_lr=self.lr,
                                                                 total_steps=self.trainer.max_steps,
                                                                 anneal_strategy='linear',
                                                                 cycle_momentum=False,
                                                                 pct_start=0.1),
                'interval': 'step',
                'frequency': 1
            }
        else:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, "monitor": 'val_acc'}
