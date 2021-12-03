from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from .classifyModel import ClassificationModel
from .datamodule import ClassificationDatamodule


class ClassifyTrainer:
    def __init__(
            self,
            batch_size: int,
            num_workers: int,
            train_root: str,
            val_root: str,
            model_name: str,
            classes_num: int,
            checkpoint_save_path: str,
            resume_path: str | None = None
    ):
        pl.seed_everything(1997)
        self.checkpoint_save_path = checkpoint_save_path
        self.resume_path = resume_path

        self.logger = WandbLogger(project='classify')
        self.datamodule = ClassificationDatamodule(
            batch_size=batch_size,
            num_workers=num_workers,
            train_root=train_root,
            val_root=val_root
        )

        self.model = ClassificationModel(
            classes_num=classes_num,
            model_name=model_name,
            logger=self.logger
        )

    def build_trainer(self, gpus: list, max_epochs: int, min_epochs: int = None):
        if len(gpus) >= 2:
            strategy = "ddp"
            gpus = len(gpus)
        else:
            strategy = None

        early_stop = EarlyStopping(patience=20, monitor='val_acc', mode='max')
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            dirpath=self.checkpoint_save_path,
            filename=self.model.model_name + '-{epoch:02d}-{val_acc:.2f}',
            save_last=True,
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        rp = RichProgressBar(leave=True)
        trainer = pl.Trainer(
            gpus=gpus,
            strategy=strategy,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            logger=self.logger,
            callbacks=[early_stop, checkpoint_callback, lr_monitor, rp]
        )

        trainer.fit(self.model, self.datamodule, ckpt_path=self.resume_path)
