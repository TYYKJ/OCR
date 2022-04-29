from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from .DBNet import DBDetModel
from .datamodule import DetDataModule

__all__ = ['DBTrainer', 'DBDetModel']


class DBTrainer:
    def __init__(
            self,
            train_data_path: str,
            val_data_path: str,
            checkpoint_save_path: str,
            project_name: str = '',
            encoder_name: str = 'dpn68',
            lr: float = 0.001,
            batch_size: int = 16,
            num_workers: int = 16,
            optimizer_name: str = 'sgd',
            weight_decay: float = 0.,
            momentum: float = 0.9,
            weights: str = 'imagenet',
            resume_path: str | None = None,
    ):
        pl.seed_everything(42)
        self.bs = batch_size
        self.nw = num_workers
        self.project_name = project_name
        self.checkpoint_save_path = checkpoint_save_path
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.encoder_name = encoder_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.optimizer_name = optimizer_name
        self.weights = weights
        self.resume_path = resume_path

    def build_model(self):
        return DBDetModel(
            encoder_name=self.encoder_name,
            lr=self.lr,
            optimizer_name=self.optimizer_name,
            weights=self.weights,
            weight_decay=self.weight_decay,
            momentum=self.momentum
        )

    def load_datamodule(self):
        return DetDataModule(
            train_data_path=self.train_data_path,
            val_data_path=self.val_data_path,
            batch_size=self.bs,
            num_workers=self.nw,
        )

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

        model = self.build_model()
        data = self.load_datamodule()

        logger = WandbLogger(name=self.project_name, project='DBNet')

        early_stop = EarlyStopping(patience=100, monitor='hmean', mode='max')
        checkpoint_callback = ModelCheckpoint(
            monitor='hmean',
            mode='max',
            dirpath=self.checkpoint_save_path,
            filename=f'{self.project_name}-DB-' +
                     model.encoder_name + '-{epoch:02d}-{hmean:.2f}-{recall:.2f}-{precision:.2f}',
            save_last=True,
        )

        max_steps = epochs * int(np.ceil(len(data.train_dataloader()) / gradient_accum))

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        rp = RichProgressBar(leave=True)

        trainer = pl.Trainer(
            gpus=gpus,
            strategy=strategy,
            logger=logger,
            callbacks=[early_stop, checkpoint_callback, lr_monitor, rp],
            max_steps=max_steps,
            **kwargs
        )

        trainer.fit(model, data, ckpt_path=self.resume_path)
