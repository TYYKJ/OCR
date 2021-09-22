# @Time    : 2021/9/18 上午9:24
# @Author  : 
# @File    : train
# @Software: PyCharm
# @explain :
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataloader import OCRDataModule
from rec import CRNN

pl.seed_everything(1997)

model = CRNN(
    # 类别+1
    classes=46 + 1,
    encoder_name='resnet18vd',
    lr=0.01,
    optimizer_name='sgd',
    lstm_hidden_size=128,
    charset_path='/home/cattree/PycharmProjects/torchOCR/data/dict.txt'
)


data = OCRDataModule(
    charset_path='/home/cattree/PycharmProjects/torchOCR/data/dict.txt',
    train_txt_path='/home/cattree/PycharmProjects/torchOCR/data/train.txt',
    val_txt_path='/home/cattree/PycharmProjects/torchOCR/data/val.txt',
    train_img_path='/home/cattree/PycharmProjects/torchOCR/data/trainimg',
    val_img_path='/home/cattree/PycharmProjects/torchOCR/data/valimg',
    mean=0.5,
    std=0.5,
    input_h=32,
    batch_size=8,
    num_workers=8
)

logger = WandbLogger()
logger.watch(model)
early_stop = EarlyStopping(patience=20, monitor='val_loss')
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='../weights',
    filename='{epoch:02d}-{val_loss:.2f}',
    save_weights_only=False,
)

trainer = pl.Trainer(
    checkpoint_callback=True,
    gpus=[0],
    max_epochs=200,
    auto_select_gpus=True,
    auto_scale_batch_size=True,
    logger=[logger],
    callbacks=[early_stop, checkpoint_callback],
    # accelerator='ddp'
    # resume_from_checkpoint='/home/cattree/PycharmProjects/torchOCR/BoatNumber/ocr_rec/weights/torchOCR-BoatNumber_ocr_rec_train/1dbgvqkj/checkpoints/epoch=18-step=8017.ckpt'
)

trainer.fit(model, data)
