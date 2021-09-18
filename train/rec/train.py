# @Time    : 2021/9/18 上午9:24
# @Author  : 
# @File    : train
# @Software: PyCharm
# @explain :
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from BoatNumber.ocr_rec.train.dataloader.dataloader import BoaNumberDataModule
from rec import CRNN

model = CRNN(
    # 类别+1
    classes=46 + 1,
    encoder_name='resnet18vd',
    lr=0.001,
    optimizer_name='adam',
    lstm_hidden_size=128,
    charset_path='/home/cattree/PycharmProjects/torchOCR/BoatNumber/ocr_rec/data/dict.txt'
)

data = BoaNumberDataModule(
    charset_path='/home/cattree/PycharmProjects/torchOCR/BoatNumber/ocr_rec/data/dict.txt',
    train_txt_path='/home/cattree/PycharmProjects/torchOCR/BoatNumber/ocr_rec/data/0916datarec/train.txt',
    val_txt_path='/home/cattree/PycharmProjects/torchOCR/BoatNumber/ocr_rec/data/0916datarec/val.txt',
    train_img_path='/home/cattree/PycharmProjects/torchOCR/BoatNumber/ocr_rec/data/0916datarec/trainimg',
    val_img_path='/home/cattree/PycharmProjects/torchOCR/BoatNumber/ocr_rec/data/0916datarec/valimg',
    mean=0.5,
    std=0.5,
    input_h=32,
    batch_size=8,
    num_workers=8
)

logger = WandbLogger()
early_stop = EarlyStopping(patience=20, monitor='val_loss')

trainer = pl.Trainer(
    checkpoint_callback=True,
    gpus=[0],
    max_epochs=200,
    # auto_lr_find=True,
    auto_select_gpus=True,
    auto_scale_batch_size=True,
    logger=[logger],
    weights_save_path='/home/cattree/PycharmProjects/torchOCR/BoatNumber/ocr_rec/weights',
    callbacks=[early_stop],
    # accelerator='ddp'
    # resume_from_checkpoint='/home/cattree/PycharmProjects/torchOCR/BoatNumber/ocr_rec/weights/torchOCR-BoatNumber_ocr_rec_train/1dbgvqkj/checkpoints/epoch=18-step=8017.ckpt'
)

trainer.fit(model, data)
