# @Time    : 2021/9/18 下午5:53
# @Author  : 
# @File    : train
# @Software: PyCharm
# @explain :
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from det import DBDetModel, DetDataModule
import pytorch_lightning as pl

pl.seed_everything(1997)

model = DBDetModel(
    encoder_name='resnet18vd',
    lr=0.01,
    optimizer_name='sgd',
)

data = DetDataModule(
    train_data_list='/home/data/OCRData/icdar2017_ocr/train.json',
    val_data_list='/home/data/OCRData/icdar2017_ocr/val.json',
    batch_size=16,
    num_workers=16
)

logger = WandbLogger()

early_stop = EarlyStopping(patience=20, monitor='hmean', mode='max')
checkpoint_callback = ModelCheckpoint(
    monitor='hmean',
    mode='max',
    dirpath='../weights',
    filename='DB-{epoch:02d}-{hmean:.2f}',
    save_last=True,
)

# DP 一机多卡
trainer = pl.Trainer(
    # open this, must drop last
    benchmark=True,
    checkpoint_callback=True,
    gpus=[0],
    max_epochs=1200,
    min_epochs=300,
    logger=[logger],
    callbacks=[early_stop, checkpoint_callback],
    # resume_from_checkpoint='../weights/DB-epoch=130-hmean=0.70.ckpt'
)

trainer.fit(model, data)
