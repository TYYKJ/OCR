
from typing import Optional

from .decoder import FPNDecoder
from ..base import OCRDetModel, Head
from ..encoders import get_encoder
from ..utils.losses import EASTLoss


class FPN(OCRDetModel):

    def __init__(
            self,
            encoder_name: str = 'resnet18',
            encoder_weights: Optional[str] = 'imagenet',
            decoder_pyramid_channels: int = 256,
            decoder_ocr_channels: int = 128,
            decoder_merge_policy: str = "add",
            decoder_dropout: float = 0.2,
            lr: float = 0.01,
            optimizer_name: str = 'adam'
    ):
        super(FPN, self).__init__()
        self.save_hyperparameters()

        self.optimizer_name = optimizer_name
        self.lr = lr
        self.loss_func = EASTLoss()

        self.encoder = get_encoder(
            name=encoder_name,
            weights=encoder_weights,
        )
        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            pyramid_channels=decoder_pyramid_channels,
            ocr_channels=decoder_ocr_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )

        self.head = Head(
            in_channels=self.decoder.out_channels
        )
        # self.initialize()
