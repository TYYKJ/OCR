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
    classes=83 + 1,
    encoder_name='resnet18vd',
    lr=0.001,
    optimizer_name='sgd',
    lstm_hidden_size=512,
    charset_path='dict.txt',
)

data = OCRDataModule(
    charset_path='dict.txt',
    train_txt_path='/home/cat/文档/icdar2015/recognition/train-no-space.txt',
    val_txt_path='/home/cat/文档/icdar2015/recognition/test-no-space.txt',
    train_img_path='/home/cat/文档/icdar2015/recognition/train',
    val_img_path='/home/cat/文档/icdar2015/recognition/test',
    mean=0.5,
    std=0.5,
    input_h=32,
    batch_size=16,
    num_workers=16,
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
    gpus=[0],
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
