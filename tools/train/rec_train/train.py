# @Time    : 2021/9/26 下午4:22
# @Author  : 
# @File    : train
# @Software: PyCharm
# @explain :
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from rec import CRNN, OCRDataModule

pl.seed_everything(1997)

model = CRNN(
    # 类别+1
    classes=3463 + 1,
    encoder_name='resnet50vd',
    lr=0.001,
    alphabet_path='./dict.txt',
)

data = OCRDataModule(
    alphabet_path='./dict.txt',
    image_path='/home/cat/文档/icdar2017/recognition/train',
    train_label_path='/home/cat/文档/icdar2017/recognition/train-no-space.txt',
    val_label_path='/home/cat/文档/icdar2017/recognition/val.txt',
    batch_size=8
)

logger = WandbLogger()
# logger.watch(model)
# logger.log_graph(model, input_array=torch.ones(1, 3, 32, 140))

early_stop = EarlyStopping(patience=20, monitor='eval_acc')
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='../weights',
    filename='CRNN-{epoch:02d}-{val_loss:.2f}--{eval_acc:.2f}',
    save_weights_only=False,
)

trainer = pl.Trainer(
    # fast_dev_run=True,
    # open this, must drop last
    weights_summary='full',
    # benchmark=True,
    checkpoint_callback=True,
    gpus=[1],
    # accelerator='ddp',
    max_epochs=200,
    min_epochs=100,
    logger=[logger],
    callbacks=[early_stop, checkpoint_callback],
    gradient_clip_algorithm='value',
    gradient_clip_val=5,
    # plugins=DDPPlugin(find_unused_parameters=False),
    # resume_from_checkpoint='../weights/last.ckpt'
)

trainer.fit(model, data)
