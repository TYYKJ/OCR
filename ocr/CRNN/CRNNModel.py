# @Time    : 2021/7/30 下午5:25
# @Author  : cattree
# @File    : stcmodel
# @Software: PyCharm
# @explain :

from ..base import CTCHead, OCRModel
from ..encoders import ResNet
from ..necks import EncoderWithRNN, SequenceEncoder, Im2Seq
import torch.nn as nn
from ..utils import losses


class CRNN(OCRModel):

    def __init__(
            self,
            classes: int,
            character_path: str,
            in_channels: int = 3,
            layers: int = 50,
            lstm_hidden_size: int = 48,
            necks_type: str = 'rnn',
            optimizer_name: str = 'adam',
            lr: int = 0.01
    ):
        super(CRNN, self).__init__(character_path=character_path)
        self.save_hyperparameters()

        self.loss_func = losses.get_loss('ctc')

        self.optimizer_name = optimizer_name

        self.lr = lr

        self.encoder = ResNet(in_channels=in_channels, layers=layers)

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
