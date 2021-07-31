# @Time    : 2021/7/31 下午3:00
# @Author  : cattree
# @File    : train
# @Software: PyCharm
# @explain :
from ocr import CRNNModel
from config.config import Config
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
import torch
from dataloader.ocr_dataloader import OCRDataModule
from pytorch_lightning import seed_everything

seed_everything(42, workers=True)

conf = Config()

model = CRNNModel.CRNN(classes=conf.classes, character_path=conf.character_path)

data = OCRDataModule()

tb_logger = pl_loggers.TensorBoardLogger('logs/', log_graph=True)
# CTCLoss的输入为[probs, labels, probs_sizes, label_sizes]，即预测结果、标签、预测结果的数目和标签数目

trainer = pl.Trainer(
    checkpoint_callback=True,
    gpus=1,
    max_epochs=conf.EPOCH,
    auto_lr_find=True,
    auto_select_gpus=True,
    auto_scale_batch_size=True,
    logger=[tb_logger],
    weights_save_path='/home/cattree/PycharmProjects/limapOCR/BoatnumProject/trainModel/weights'
)

trainer.fit(model, data)
torch.save(model.state_dict(), '/home/cattree/PycharmProjects/limapOCR/BoatnumProject/trainModel/weights/final.pt')
trainer.test()
