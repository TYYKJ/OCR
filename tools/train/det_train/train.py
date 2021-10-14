# @Time    : 2021/9/18 下午5:53
# @Author  : 
# @File    : train
# @Software: PyCharm
# @explain :
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from det import DBDetModel, DetDataModule
import pytorch_lightning as pl

pl.seed_everything(1997)

model = DBDetModel(
    encoder_name='resnet50vd',
    lr=0.01,
    optimizer_name='sgd',
)

data = DetDataModule(
    train_data_path='/home/cat/文档/icdar2017/detection/train.json',
    val_data_path='/home/cat/文档/icdar2017/detection/val.json',
    batch_size=8,
    num_workers=8
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
    weights_summary='full',
    benchmark=True,
    checkpoint_callback=True,
    gpus=2,
    accelerator='ddp',
    max_epochs=1200,
    min_epochs=300,
    logger=[logger],
    callbacks=[early_stop, checkpoint_callback],
    plugins=DDPPlugin(find_unused_parameters=False),
    resume_from_checkpoint='../weights/last.ckpt'
)

trainer.fit(model, data)
