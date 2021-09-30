from ..base import BaseModel
from ..convert import CTCLabelConverter
from ..criteria import CTCLoss
from ..encoders import get_encoder
from ..heads import CTCHead
from ..necks import EncoderWithLSTM, Feature2Seq


class CRNN(BaseModel):
    r"""
    Build CRNN model.

    Args:
        classes: alphabet num.
            .. note::

                this num must +1.

        charset_path: dict path.
        encoder_name: backbone name.
        blank_idx: CTC loss blank idx, default idx is zero.
            lstm_hidden_size: neck lstm hidden size.
        lr: learning rate.
        optimizer_name: select a optimizer, if optimizer_change is False,
            model will use one optimizer, else will use Adam in the first ``'optimizer_change_epoch'`` epoch,
            and then use SGD optimizer.
        optimizer_change: change one optimizer to Adam+SGD.
        optimizer_change_epoch: change the optimizer to Adam+SGD epoch num.
        train_loss_name: custom train loss name.
        val_loss_name: custom val loss name

    Example::

        >>> from rec import CRNN
        >>> model = CRNN(classes=10+1, encoder_name='resnet50')
    """

    def __init__(
            self,
            classes: int,
            charset_path: str,
            encoder_name: str = 'resnet18vd',
            blank_idx: int = 0,
            lstm_hidden_size: int = 256,
            lr: float = 0.001,
            optimizer_name: str = 'adam',
            optimizer_change: bool = False,
            optimizer_change_epoch: int = 100,
            train_loss_name: str = 'train_loss',
            val_loss_name: str = 'val_loss',
    ):
        super(CRNN, self).__init__()

        self.save_hyperparameters()

        self.convert = CTCLabelConverter(character=charset_path)

        self.optimizer_name = optimizer_name
        self.lr = lr
        self.train_loss_name = train_loss_name
        self.val_loss_name = val_loss_name
        self.optimizer_change = optimizer_change
        self.optimizer_change_epoch = optimizer_change_epoch

        self.encoder = get_encoder(encoder_name)
        self.im_seq = Feature2Seq()
        self.neck = EncoderWithLSTM(in_channels=self.encoder.out_channels, hidden_size=lstm_hidden_size)
        self.head = CTCHead(
            in_channels=lstm_hidden_size * 2,
            n_class=classes
        )
        self.loss_func = CTCLoss(blank_idx=blank_idx)

        self.initialize()
