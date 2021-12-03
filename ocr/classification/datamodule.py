import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

__all__ = ['ClassificationDatamodule']


def get_dataset(root: str, is_train: bool = True):
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return datasets.ImageFolder(root=root, transform=transform[0])


class ClassificationDatamodule(pl.LightningDataModule):

    def __init__(self, train_root: str, val_root: str, batch_size: int, num_workers: int):
        super(ClassificationDatamodule, self).__init__()
        self.train_ds = get_dataset(train_root)
        self.val_ds = get_dataset(val_root)
        self.bs = batch_size
        self.nw = num_workers

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ds, batch_size=self.bs, num_workers=self.nw, shuffle=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_ds, batch_size=self.bs, num_workers=self.nw, shuffle=False
        )
