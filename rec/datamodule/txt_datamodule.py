# @Time    : 2021/9/26 下午3:33
# @Author  : 
# @File    : txt_datamodule
# @Software: PyCharm
# @explain :
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .txt_collate_fn import RecCollateFn
from .txt_dataloader import RecDataLoader
from .txt_dataset import RecDataProcess, RecTextDataset


class OCRDataModule(pl.LightningDataModule):
    def __init__(
            self,
            charset_path,
            train_txt_path,
            val_txt_path,
            train_img_path,
            val_img_path,
            mean,
            std,
            input_h=32,
            batch_size=16,
            num_workers=8,
    ):
        super(OCRDataModule, self).__init__()

        self.num_workers = num_workers
        self.bs = batch_size
        self.input_h = input_h
        self.mean = mean
        self.std = std

        self.train = RecTextDataset(
            charset_path=charset_path,
            txt_file_path=train_txt_path,
            img_path=train_img_path,
            input_h=self.input_h,
            mean=self.mean,
            std=self.std,
            augmentation=True
        )

        self.val = RecTextDataset(
            charset_path=charset_path,
            txt_file_path=val_txt_path,
            img_path=val_img_path,
            input_h=self.input_h,
            mean=self.mean,
            std=self.std,
            augmentation=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=RecCollateFn(RecDataProcess(input_h=self.input_h, mean=self.mean, std=self.std))
        )

    def val_dataloader(self):
        return RecDataLoader(
            self.val,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.num_workers,
        )
