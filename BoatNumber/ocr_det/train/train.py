# @Time    : 2021/8/13 下午4:46
# @Author  : cattree
# @File    : train
# @Software: PyCharm
# @explain :

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from BoatNumber.ocr_det.train.dataloader import DetDataModule
from det import FPN

model = FPN(encoder_name='resnet50')
logger = WandbLogger()
early_stop = EarlyStopping(patience=10, monitor='val_loss')

data = DetDataModule(
    train_gt_path='/home/data/OCRData/icdar2017rctw/detection/train/gt',
    train_img_path='/home/data/OCRData/icdar2017rctw/detection/train/image',
    val_gt_path='/home/data/OCRData/icdar2017rctw/detection/val/gt',
    val_image_path='/home/data/OCRData/icdar2017rctw/detection/val/image'
)

trainer = pl.Trainer(
    checkpoint_callback=True,
    gpus=1,
    max_epochs=200,
    # auto_lr_find=True,
    auto_select_gpus=True,
    auto_scale_batch_size=True,
    logger=[logger],
    weights_save_path='/home/cattree/PycharmProjects/torch-ocr/BoatNumber/ocr_det/train/weights',
    callbacks=[early_stop]
)

trainer.fit(model, data)
