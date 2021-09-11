# @Time    : 2021/8/13 下午4:46
# @Author  : cattree
# @File    : train
# @Software: PyCharm
# @explain :

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from BoatNumber.ocr_det.train.dataloader import DetDataModule
from det import East

if __name__ == '__main__':
    model = East(encoder_name='vgg16_bn', encoder_weights='imagenet')
    # data = torch.randn((1, 3, 640, 640))
    # model(data)

    logger = WandbLogger()
    early_stop = EarlyStopping(patience=10, monitor='val_loss')

    data = DetDataModule(
        train_gt_path='/home/data/OCRData/detection/train/gt',
        train_img_path='/home/data/OCRData/detection/train/image',
        val_gt_path='/home/data/OCRData/det0902/val/gt',
        val_image_path='/home/data/OCRData/det0902/val/img'
    )

    trainer = pl.Trainer(
        checkpoint_callback=True,
        gpus=[0],
        max_epochs=300,
        # auto_lr_find=True,
        auto_select_gpus=True,
        auto_scale_batch_size=True,
        logger=[logger],
        weights_save_path='/home/cattree/PycharmProjects/torchOCR/BoatNumber/ocr_det/train/weights',
        callbacks=[early_stop],
        # accelerator='dp'
        # resume_from_checkpoint='/home/cattree/PycharmProjects/torchOCR/BoatNumber/ocr_det/train/weights/torchOCR-BoatNumber_ocr_det_train/lkbwfq0s/checkpoints/epoch=131-step=265187.ckpt'
    )

    trainer.fit(model, data)
