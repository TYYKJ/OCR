# @Time    : 2021/9/18 上午9:23
# @Author  : 
# @File    : dataloader
# @Software: PyCharm
# @explain :
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from rec.utils.CreateRecAug import *


class RecDataProcess:
    def __init__(self, input_h, mean, std):
        """
        文本是被数据增广类

        :param config: 配置，主要用到的字段有 input_h, mean, std
        """

        self.input_h = input_h
        self.mean = mean
        self.std = std

        self.random_contrast = RandomContrast(probability=0.3)
        self.random_brightness = RandomBrightness(probability=0.3)
        self.random_sharpness = RandomSharpness(probability=0.3)
        self.compress = Compress(probability=0.3)
        self.rotate = Rotate(probability=0.5)
        self.blur = Blur(probability=0.3)
        self.motion_blur = MotionBlur(probability=0.3)
        self.salt = Salt(probability=0.3)
        self.adjust_resolution = AdjustResolution(probability=0.3)
        self.random_line = RandomLine(probability=0.3)
        self.random_contrast.setparam()
        self.random_brightness.setparam()
        self.random_sharpness.setparam()
        self.compress.setparam()
        self.rotate.setparam()
        self.blur.setparam()
        self.motion_blur.setparam()
        self.salt.setparam()
        self.adjust_resolution.setparam()

    def aug_img(self, img):
        img = self.random_contrast.process(img)
        img = self.random_brightness.process(img)
        img = self.random_sharpness.process(img)
        img = self.random_line.process(img)

        if img.size[1] >= 32:
            img = self.compress.process(img)
            img = self.adjust_resolution.process(img)
            img = self.motion_blur.process(img)
            img = self.blur.process(img)
        img = self.rotate.process(img)
        img = self.salt.process(img)
        return img

    def resize_with_specific_height(self, _img):
        """
        将图像resize到指定高度
        :param _img:    待resize的图像
        :return:    resize完成的图像
        """
        resize_ratio = self.input_h / _img.shape[0]
        return cv2.resize(_img, (0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)

    def normalize_img(self, _img):
        """
        根据配置的均值和标准差进行归一化
        :param _img:    待归一化的图像
        :return:    归一化后的图像
        """
        return (_img.astype(np.float32) / 255 - self.mean) / self.std

    def width_pad_img(self, _img, _target_width, _pad_value=0):
        """
        将图像进行高度不变，宽度的调整的pad
        :param _img:    待pad的图像
        :param _target_width:   目标宽度
        :param _pad_value:  pad的值
        :return:    pad完成后的图像
        """
        _height, _width, _channels = _img.shape
        to_return_img = np.ones([_height, _target_width, _channels], dtype=_img.dtype) * _pad_value
        to_return_img[:_height, :_width, :] = _img
        return to_return_img


class RecTextDataset(Dataset):

    def __init__(
            self,
            charset_path,
            txt_file_path,
            img_path,
            input_h=32,
            mean=0.5,
            std=0.5,
    ):
        self.img_path = img_path
        self.process = RecDataProcess(
            input_h=input_h,
            mean=mean,
            std=std
        )
        with open(charset_path, 'r', encoding='utf-8') as file:
            alphabet = ''.join([s.strip('\n') for s in file.readlines()])
        alphabet += ' '
        self.str2idx = {c: i for i, c in enumerate(alphabet)}
        self.labels = []
        with open(txt_file_path, 'r', encoding='utf-8') as f_reader:
            for m_line in f_reader.readlines():
                params = m_line.split('\t')
                if len(params) == 2:
                    m_image_name, m_gt_text = params
                    if m_gt_text.endswith('\n'):
                        m_gt_text = m_gt_text.strip('\n')
                    if True in [c not in self.str2idx for c in m_gt_text]:
                        continue
                    self.labels.append((m_image_name, m_gt_text))

    def _find_max_length(self):
        return max({len(_[1]) for _ in self.labels})

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # get img_path and trans
        img_name, trans = self.labels[index]
        # read img
        img_path = os.path.join(self.img_path, img_name)
        img = cv2.imread(img_path)
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            print(img_path)
            exit()

        return {'img': img, 'label': trans}


class RecCollateFn:
    def __init__(self, process_callback):
        self.process = process_callback
        self.t = transforms.ToTensor()

    def __call__(self, batch):
        resize_images = []

        all_same_height_images = [self.process.resize_with_specific_height(_['img']) for _ in batch]
        max_img_w = max({m_img.shape[1] for m_img in all_same_height_images})
        # make sure max_img_w is integral multiple of 8
        max_img_w = int(np.ceil(max_img_w / 8) * 8)
        labels = []
        for i in range(len(batch)):
            _label = batch[i]['label']
            labels.append(_label)
            img = self.process.normalize_img(all_same_height_images[i])
            img = self.process.width_pad_img(img, max_img_w)

            img = img.transpose([2, 0, 1])
            resize_images.append(torch.tensor(img, dtype=torch.float))
        resize_images = torch.stack(resize_images)
        return {'img': resize_images, 'label': labels}


class RecDataLoader:
    def __init__(self, dataset, batch_size, shuffle, num_workers, **kwargs):
        """
        自定义 DataLoader, 主要实现数据集的按长度划分，将长度相近的放在一个 batch

        :param dataset: 继承自 torch.utils.data.DataSet的类对象
        :param batch_size: 一个 batch 的图片数量
        :param shuffle: 是否打乱数据集
        :param num_workers: 后台进程数
        :param kwargs: **
        """
        self.dataset = dataset
        self.process = dataset.process
        self.len_thresh = self.dataset._find_max_length() // 2
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.iteration = 0
        self.dataiter = None
        self.queue_1 = list()
        self.queue_2 = list()

    def __len__(self):
        return len(self.dataset) // self.batch_size if len(self.dataset) % self.batch_size == 0 \
            else len(self.dataset) // self.batch_size + 1

    def __iter__(self):
        return self

    def pack(self, batch_data):
        batch = {'img': [], 'label': []}
        # img tensor current shape: B,H,W,C
        all_same_height_images = [self.process.resize_with_specific_height(_['img'][0].numpy()) for _ in batch_data]
        max_img_w = max({m_img.shape[1] for m_img in all_same_height_images})
        # make sure max_img_w is integral multiple of 8
        max_img_w = int(np.ceil(max_img_w / 8) * 8)
        for i in range(len(batch_data)):
            _label = batch_data[i]['label'][0]
            img = self.process.normalize_img(self.process.width_pad_img(all_same_height_images[i], max_img_w))
            img = img.transpose([2, 0, 1])
            batch['img'].append(torch.tensor(img, dtype=torch.float))
            batch['label'].append(_label)
        batch['img'] = torch.stack(batch['img'])
        return batch

    def build(self):
        self.dataiter = DataLoader(self.dataset, batch_size=1, shuffle=self.shuffle, num_workers=self.num_workers).__iter__()

    def __next__(self):
        if self.dataiter is None:
            self.build()
        if self.iteration == len(self.dataset) and len(self.queue_2):
            batch_data = self.queue_2
            self.queue_2 = list()
            return self.pack(batch_data)
        if not len(self.queue_2) and not len(self.queue_1) and self.iteration == len(self.dataset):
            self.iteration = 0
            self.dataiter = None
            raise StopIteration
        # start iteration
        try:
            while True:
                # get data from origin dataloader
                temp = self.dataiter.__next__()
                self.iteration += 1
                # to different queue
                if len(temp['label'][0]) <= self.len_thresh:
                    self.queue_1.append(temp)
                else:
                    self.queue_2.append(temp)

                # to store batch data
                batch_data = None
                # queue_1 full, push to batch_data
                if len(self.queue_1) == self.batch_size:
                    batch_data = self.queue_1
                    self.queue_1 = list()
                # or queue_2 full, push to batch_data
                elif len(self.queue_2) == self.batch_size:
                    batch_data = self.queue_2
                    self.queue_2 = list()

                # start to process batch
                if batch_data is not None:
                    return self.pack(batch_data)
        # deal with last batch
        except StopIteration:
            batch_data = self.queue_1
            self.queue_1 = list()
            return self.pack(batch_data)


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
        )

        self.val = RecTextDataset(
            charset_path=charset_path,
            txt_file_path=val_txt_path,
            img_path=val_img_path,
            input_h=self.input_h,
            mean=self.mean,
            std=self.std,
        )

    def train_dataloader(self):
        return RecDataLoader(
            self.train,
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=RecCollateFn(RecDataProcess(input_h=self.input_h, mean=self.mean, std=self.std)))

    def val_dataloader(self):
        return RecDataLoader(
            self.val,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=RecCollateFn(RecDataProcess(input_h=self.input_h, mean=self.mean, std=self.std)))


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     from ocr_rec import CRNN
#
#     model = CRNN(
#         # 类别+1
#         classes=47 + 1,
#         encoder_name='resnet34vd',
#         lr=0.001,
#         optimizer_name='adam',
#         lstm_hidden_size=512,
#         charset_path='/BoatNumber/ocr_rec/data/dict.txt'
#     )
#
#     ds = RecTextDataset(
#         charset_path='/BoatNumber/ocr_rec/data/dict.txt',
#         img_path='/home/cattree/PycharmProjects/torchOCR/BoatNumber/ocr_rec/data/0909recDatanoresize/recDatanoresize/trainimg',
#         txt_file_path='/home/cattree/PycharmProjects/torchOCR/BoatNumber/ocr_rec/data/0909recDatanoresize/recDatanoresize/train.txt')
#     dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=RecCollateFn(RecDataProcess(input_h=32, mean=0.5, std=0.5)))
#     for i, batch in enumerate(dl):
#         img = batch.get('img')
#         pre = model(img)
#         label = batch.get('label')
#         print(label)
#         img = img.squeeze()
#         img = img.transpose(2, 0)
#         img = img.transpose(1, 0)
#         print(img.size())
#         plt.imshow(img)
#         plt.show()
#
#         break
