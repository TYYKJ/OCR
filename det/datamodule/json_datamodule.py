# @Time    : 2021/9/28 下午2:59
# @Author  : 
# @File    : datamodule
# @Software: PyCharm
# @explain :
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .det_collate_fn import DetCollectFN
from .json_dataset import JsonDataset


class DetDataModule(pl.LightningDataModule):
    def __init__(self, train_data_path, val_data_path, batch_size, num_workers):
        super(DetDataModule, self).__init__()

        self.bs = batch_size
        self.nw = num_workers

        self.train = JsonDataset(data_list=train_data_path, is_train=True)
        self.val = JsonDataset(data_list=train_data_path, is_train=False)
        self.collate_fn = DetCollectFN()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.bs,
            num_workers=self.nw,
            shuffle=True,
            pin_memory=True,
            drop_last=True)

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True)
