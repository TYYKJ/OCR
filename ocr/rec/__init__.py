from __future__ import annotations
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from ..utils import weight_init
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
            project_name: str = '',
            use_augmentation: bool = False,
            encoder_name: str = 'resnet18vd',
            mean: float = 0.5,
            std: float = 0.5,
            batch_size: int = 16,
            num_workers: int = 16,
            hidden_size: int = 256,
            optimizer_name: str = 'sgd',
            lr: float = 0.001,
            weight_decay: float = 0.,
            momentum: float = 0.9,
            weights: str = 'imagenet',
            resume_path: str | None = None,
            encoder_type: str = 'rnn',
    ):
        pl.seed_everything(42)
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
        self.project_name = project_name

        self.encoder_name = encoder_name

        self.lr = lr
        self.optimizer_name = optimizer_name
        self.weights = weights

        self.resume_path = resume_path
        self.classes = classes
        self.alphabet_path = alphabet_path
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.encoder_type = encoder_type
        self.hidden_size = hidden_size

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
        model = CRNN(
            encoder_name=self.encoder_name,
            lr=self.lr,
            classes=self.classes,
            alphabet_path=self.alphabet_path,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            optimizer_name=self.optimizer_name,
            encoder_type=self.encoder_type,
            hidden_size=self.hidden_size,
            resume_model=True if self.resume_path else False
        )

        # model.apply(weight_init)
        return model

    def build_trainer(
            self,
            gpus: list,
            epochs: int,
            gradient_accum: int = 1,
            **kwargs
    ):
        if len(gpus) >= 2:
            strategy = "ddp"
            gpus = len(gpus)
        else:
            strategy = None
        logger = WandbLogger(name=self.project_name, project='CRNN')

        model = self.build_model()
        # print(model)
        data = self.load_datamodule()

        # early_stop = EarlyStopping(patience=20, monitor='val_acc', mode='max')
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            dirpath=self.checkpoint_save_path,
            filename=f'{self.project_name}-CRNN-' + model.encoder_name + '-{epoch:02d}-{val_acc:.2f}',
            save_last=True,
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        rp = RichProgressBar(leave=True)

        if not self.resume_path:
            max_steps = epochs * int(np.ceil(len(data.train_dataloader()) / gradient_accum))
        else:
            max_steps = -1

        trainer = pl.Trainer(
            gpus=gpus,
            strategy=strategy,
            logger=logger,
            callbacks=[checkpoint_callback, lr_monitor, rp],
            max_steps=max_steps,
            max_epochs=epochs,
            **kwargs
        )

        trainer.fit(model, data, ckpt_path=self.resume_path)
