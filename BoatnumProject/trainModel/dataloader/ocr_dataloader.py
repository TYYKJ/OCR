# @Time    : 2021/7/31 上午10:39
# @Author  : cattree
# @File    : ocr_dataloader
# @Software: PyCharm
# @explain :
from typing import Optional

import PIL.Image
import pytorch_lightning as pl
import torchvision.transforms.functional
from torch.utils.data import DataLoader, random_split, Dataset

from BoatnumProject import Config
from ocr.utils.convert import load


class OCRDataset(Dataset):

    def __init__(self, img_dir, txt_path, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.data_list = load(txt_path)
        self.transform = transform
        self.target_transform = target_transform
        # self.height = 32
        # self.width = 100

    def __getitem__(self, item):
        line = self.data_list[item].split(' ')
        if self.img_dir.endswith('/'):
            img_path = self.img_dir + line[0]
        else:
            img_path = self.img_dir + f'/{line[0]}'
        im = torchvision.transforms.functional.to_tensor(PIL.Image.open(img_path))
        # im = cv2.imread(img_path)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # im = np.transpose(im, (2, 0, 1))
        # im = im / 255
        # im_size = im.shape
        #
        # if (im_size[1] / (im_size[0] * 1.0)) < 6.4:
        #     img_reshape = cv2.resize(im, (int(32.0 / im_size[0] * im_size[1]), self.height))
        #     mat_ori = np.zeros((self.height, self.width - int(32.0 / im_size[0] * im_size[1]), 3), dtype=np.uint8)
        #     out_img = np.concatenate([img_reshape, mat_ori], axis=1).transpose([1, 0, 2])
        # else:
        #     out_img = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        #     out_img = np.asarray(out_img).transpose([1, 0, 2])

        label = line[1]

        # if self.transform:
        #     im = self.transform(im)
        # if self.target_transform:
        #     label = self.target_transform(label)

        return im, label

    def __len__(self):
        return len(self.data_list)


class OCRDataModule(pl.LightningDataModule):

    def __init__(self):
        super(OCRDataModule, self).__init__()
        self.conf = Config()
        self.ocr_train, self.ocr_val, self.ocr_test = None, None, None

    def setup(self, stage: Optional[str] = None) -> None:
        ocr_train = OCRDataset(
            self.conf.img_path,
            self.conf.train_label_path,
        )
        self.ocr_test = OCRDataset(
            self.conf.img_path,
            self.conf.test_label_path
        )
        self.ocr_train, self.ocr_val = random_split(
            ocr_train,
            [
                int(len(ocr_train) * 0.8),
                len(ocr_train) - int(len(ocr_train) * 0.8)
            ]
        )

    def train_dataloader(self):
        return DataLoader(self.ocr_train, batch_size=self.conf.BATCH_SIZE, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.ocr_val, batch_size=self.conf.BATCH_SIZE, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.ocr_test, batch_size=self.conf.BATCH_SIZE, num_workers=8)
