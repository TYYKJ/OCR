import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .rec_collate_fn import RecCollateFn
from .rec_dataset import RecTextLineDataset, RecDataLoader


class OCRDataModule(pl.LightningDataModule):
    def __init__(
            self,
            alphabet_path: str,
            image_path: str,
            train_label_path: str,
            val_label_path: str,
            input_h: int = 32,
            mean: float = 0.5,
            std: float = 0.5,
            batch_size: int = 16,
            num_workers: int = 8,
            use_augmentation: bool = False,
    ):
        super(OCRDataModule, self).__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.rec_collate_fn = RecCollateFn(
            input_h, mean=mean, std=std
        )

        self.train_dataset = RecTextLineDataset(
            alphabet_path,
            image_path,
            train_label_path,
            input_h,
            mean,
            std,
            use_augmentation
        )

        self.val_dataset = RecTextLineDataset(
            alphabet_path,
            image_path,
            val_label_path,
            input_h,
            mean,
            std,
            False
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.rec_collate_fn
        )

    def val_dataloader(self):
        return RecDataLoader(
            dataset=self.val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.rec_collate_fn
        )
