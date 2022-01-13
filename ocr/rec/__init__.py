from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from .crnn import CRNN
from .datamodule import RecDataModule

__all__ = ['CRNNTrainer', 'CRNN']


class CRNNTrainer:
    def __init__(
            self,
            image_path: str,
            train_label_path: str,
            val_label_path: str,
            checkpoint_save_path: str,
            classes: int,
            alphabet_path: str,
            input_h: int,
            use_augmentation: bool = False,
            encoder_name: str = 'resnet18vd',
            mean: float = 0.5,
            std: float = 0.5,
            batch_size: int = 16,
            num_workers: int = 16,
            optimizer_name: str = 'sgd',
            lr: float = 0.001,
            weight_decay: float = 0.,
            momentum: float = 0.9,
            weights: str = 'imagenet',
            resume_path: str | None = None,
    ):
        pl.seed_everything(1997)
        self.bs = batch_size
        self.nw = num_workers
        self.checkpoint_save_path = checkpoint_save_path
        self.image_path = image_path
        self.train_label_path = train_label_path
        self.val_label_path = val_label_path
        self.input_h = input_h

        self.use_augmentation = use_augmentation
        self.mean = mean
        self.std = std

        self.encoder_name = encoder_name

        self.lr = lr
        self.optimizer_name = optimizer_name
        self.weights = weights

        self.resume_path = resume_path
        self.classes = classes
        self.alphabet_path = alphabet_path
        self.momentum = momentum
        self.weight_decay = weight_decay

    def load_datamodule(self):
        return RecDataModule(
            alphabet_path=self.alphabet_path,
            image_path=self.image_path,
            train_label_path=self.train_label_path,
            val_label_path=self.val_label_path,
            input_h=self.input_h,
            mean=self.mean,
            std=self.std,
            batch_size=self.bs,
            num_workers=self.nw,
            use_augmentation=self.use_augmentation
        )

    def build_model(self):
        return CRNN(
            encoder_name=self.encoder_name,
            lr=self.lr,
            classes=self.classes,
            alphabet_path=self.alphabet_path,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            optimizer_name=self.optimizer_name,
        )

    def build_trainer(
            self,
            gpus: list,
            **kwargs
    ):
        if len(gpus) >= 2:
            strategy = "ddp"
            gpus = len(gpus)
        else:
            strategy = None
        logger = WandbLogger(name='CRNN')
        model = self.build_model()
        data = self.load_datamodule()

        early_stop = EarlyStopping(patience=20, monitor='val_acc', mode='max')
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            dirpath=self.checkpoint_save_path,
            filename='CRNN-' + model.encoder_name + '-{epoch:02d}-{val_acc:.2f}',
            save_last=True,
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        rp = RichProgressBar(leave=True)

        trainer = pl.Trainer(
            gpus=gpus,
            strategy=strategy,
            logger=logger,
            callbacks=[checkpoint_callback, lr_monitor, rp],
            **kwargs
        )

        trainer.fit(model, data, ckpt_path=self.resume_path)
