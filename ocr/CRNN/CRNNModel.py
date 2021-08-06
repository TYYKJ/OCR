# @Time    : 2021/7/30 下午5:25
# @Author  : cattree
# @File    : stcmodel
# @Software: PyCharm
# @explain :

import torch.nn as nn

from ..base import CTCHead, OCRModel
from ..encoders import get_encoder
from ..necks import EncoderWithRNN, SequenceEncoder, Im2Seq
from ..utils import losses


class CRNN(OCRModel):

    def __init__(
            self,
            classes: int,
            character_path: str,
            encoder_name: str = 'resnet18vd',
            lstm_hidden_size: int = 256,
            necks_type: str = 'rnn',
            optimizer_name: str = 'adam',
            lr: int = 0.01,
            bs: int = 16,
    ):
        super(CRNN, self).__init__(character_path=character_path)
        self.save_hyperparameters()

        self.loss_func = losses.get_loss('ctc')

        self.optimizer_name = optimizer_name

        self.lr = lr

        self.bs = bs

        self.encoder = get_encoder(encoder_name)

        if necks_type == 'rnn':
            self.seq = Im2Seq(in_channels=self.encoder.out_channels)
        else:
            self.seq = nn.Identity()

        self.neck = EncoderWithRNN(in_channels=self.encoder.out_channels, hidden_size=lstm_hidden_size) \
            if necks_type == 'rnn' else SequenceEncoder(in_channels=self.encoder.out_channels)

        self.head = CTCHead(
            in_channels=lstm_hidden_size * 2 if necks_type == 'rnn' else self.neck.out_channels,
            n_class=classes
        )
