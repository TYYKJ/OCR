
import torch.nn

from ..base import CTCHead, OCRModel, EncoderWithLSTM, Feature2Seq
from ..encoders import get_encoder


class CRNN(OCRModel):

    def __init__(
            self,
            classes: int,
            encoder_name: str = 'resnet18vd',
            lstm_hidden_size: int = 256,
            lr: float = 0.001,
            optimizer_name: str = 'adam',
            train_loss_name: str = 'train_loss',
            val_loss_name: str = 'val_loss',
    ):
        super(CRNN, self).__init__()
        self.save_hyperparameters()

        self.loss_func = torch.nn.CTCLoss()

        self.optimizer_name = optimizer_name

        self.lr = lr

        self.train_loss_name = train_loss_name
        self.val_loss_name = val_loss_name

        self.encoder = get_encoder(encoder_name)

        self.im_seq = Feature2Seq()

        self.neck = EncoderWithLSTM(in_channels=self.encoder.out_channels, hidden_size=lstm_hidden_size)

        self.head = CTCHead(
            in_channels=lstm_hidden_size * 2,
            n_class=classes
        )

        self.initialize()
