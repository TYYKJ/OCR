# @Time    : 2021/9/18 下午5:53
# @Author  : 
# @File    : train
# @Software: PyCharm
# @explain :
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import RichProgressBar
from det import DBDetModel, DetDataModule
import pytorch_lightning as pl

pl.seed_everything(1997)

model = DBDetModel(
    encoder_name='resnet18',
    lr=0.001,
    optimizer_name='sgd',
    weights='imagenet'
)

data = DetDataModule(
    train_data_path='/home/cat/Documents/ICDAR/ICDAR2019/train.json',
    val_data_path='/home/cat/Documents/icdar2015-ok/detection/train.json',
    batch_size=16,
    num_workers=16
)

logger = WandbLogger()

early_stop = EarlyStopping(patience=20, monitor='hmean', mode='max')
checkpoint_callback = ModelCheckpoint(
    monitor='hmean',
    mode='max',
    dirpath='../weights',
    filename='DB-' + model.encoder_name + '-{epoch:02d}-{hmean:.2f}-{recall:.2f}-{precision:.2f}',
    save_last=True,
)

lr_monitor = LearningRateMonitor(logging_interval='epoch')
rp = RichProgressBar(leave=True)

# DP 一机多卡
trainer = pl.Trainer(
    # open this, must drop last
    benchmark=True,
    # gpus=2, strategy="ddp",
    gpus=[1],
    max_epochs=100,
    min_epochs=80,
    logger=[logger],
    callbacks=[early_stop, checkpoint_callback, lr_monitor, rp],
)

trainer.fit(model, data)
