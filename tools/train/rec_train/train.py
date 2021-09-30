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
    classes=47 + 1,
    encoder_name='resnet18vd',
    lr=0.01,
    optimizer_name='sgd',
    lstm_hidden_size=256,
    charset_path='dict.txt',
    optimizer_change=False,
    optimizer_change_epoch=100
)

data = OCRDataModule(
    charset_path='dict.txt',
    train_txt_path='train.txt',
    val_txt_path='val.txt',
    train_img_path='/home/data/OCRData/BoatNum/trainimg',
    val_img_path='/home/data/OCRData/BoatNum/valimg',
    mean=0.5,
    std=0.5,
    input_h=32,
    batch_size=16,
    num_workers=16,
)

logger = WandbLogger()
# logger.watch(model)
# logger.log_graph(model, input_array=torch.ones(1, 3, 32, 140))

early_stop = EarlyStopping(patience=20, monitor='val_loss')
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='../weights',
    filename='CRNN-{epoch:02d}-{val_loss:.2f}',
    save_weights_only=False,
)

trainer = pl.Trainer(
    gpus=[0],
    max_epochs=500,
    min_epochs=200,
    logger=[logger],
    callbacks=[early_stop, checkpoint_callback],
)

trainer.fit(model, data)
