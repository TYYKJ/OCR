import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from det import DBDetModel, DetDataModule


class TrainDB:
    def __init__(
            self,
            train_data_path: str,
            val_data_path: str,
            checkpoint_save_path: str,
            encoder_name: str = 'dpn68',
            lr: float = 0.001,
            batch_size: int = 16,
            num_workers: int = 16,
            optimizer_name: str = 'sgd',
            weights: str = 'imagenet',
            pretrained_path: str = '',
    ):
        pl.seed_everything(1997)
        self.bs = batch_size
        self.nw = num_workers
        self.checkpoint_save_path = checkpoint_save_path
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.encoder_name = encoder_name
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.weights = weights
        self.pretrained_path = pretrained_path

    def build_model(self):
        return DBDetModel(
            encoder_name=self.encoder_name,
            lr=self.lr,
            optimizer_name=self.optimizer_name,
            weights=self.weights,
            pretrained_model_path=self.pretrained_path,
        )

    def load_model(self):
        model = self.build_model()
        if model.pretrained_model_path:
            model.load_from_checkpoint(model.pretrained_model_path)
            return model
        else:
            return model

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
            max_epochs: int,
            min_epochs: int,
    ):
        if len(gpus) >= 2:
            strategy = "ddp"
            gpus = len(gpus)
        else:
            strategy = None

        model = self.load_model()
        data = self.load_datamodule()

        logger = WandbLogger()

        early_stop = EarlyStopping(patience=20, monitor='hmean', mode='max')
        checkpoint_callback = ModelCheckpoint(
            monitor='hmean',
            mode='max',
            dirpath=self.checkpoint_save_path,
            filename='DB-' + model.encoder_name + '-{epoch:02d}-{hmean:.2f}-{recall:.2f}-{precision:.2f}',
            save_last=True,
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        rp = RichProgressBar(leave=True)

        trainer = pl.Trainer(
            gpus=gpus,
            strategy=strategy,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            logger=logger,
            callbacks=[early_stop, checkpoint_callback, lr_monitor, rp]
        )

        trainer.fit(model, data)


if __name__ == '__main__':

    db = TrainDB(
        train_data_path='/home/cat/Documents/ZF/train.json',
        val_data_path='/home/cat/Documents/ZF/val.json',
        checkpoint_save_path='../weights',
        encoder_name='resnet50',
        weights='imagenet',
        lr=0.01,
        batch_size=16,
        num_workers=16,
        optimizer_name='sgd',
        pretrained_path=''
    )
    db.build_trainer(
        gpus=[1],
        max_epochs=150,
        min_epochs=80,
    )
