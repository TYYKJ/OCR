# @Time    : 2021/9/18 下午5:53
# @Author  : 
# @File    : train
# @Software: PyCharm
# @explain :
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor
from det import DBDetModel, DetDataModule
import pytorch_lightning as pl

pl.seed_everything(1997)

model = DBDetModel(
    encoder_name='resnet50',
    lr=0.001,
    optimizer_name='adam',
)

data = DetDataModule(
    train_data_path='/home/cat/Documents/PreTrainOCRData/train.json',
    val_data_path='/home/cat/Documents/PreTrainOCRData/val.json',
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

# DP 一机多卡
trainer = pl.Trainer(
    # open this, must drop last
    weights_summary='full',
    benchmark=True,
    # checkpoint_callback=True,
    gpus=[1],
    # accelerator='ddp',
    max_epochs=100,
    min_epochs=80,
    logger=[logger],
    callbacks=[early_stop, checkpoint_callback, lr_monitor],
    # plugins=DDPPlugin(find_unused_parameters=False),
    # resume_from_checkpoint='../weights/DB-resnet50-epoch=35-hmean=0.68-recall=0.61-precision=0.77.ckpt'
)

trainer.fit(model, data)
