# @Time    : 2021/9/26 下午3:30
# @Author  : 
# @File    : txt_dataloader
# @Software: PyCharm
# @explain :
import numpy as np
import torch
from torch.utils.data import DataLoader


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
        self.len_thresh = self.dataset.find_max_length() // 2
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
        self.dataiter = DataLoader(self.dataset, batch_size=1, shuffle=self.shuffle, num_workers=self.num_workers, drop_last=True).__iter__()

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
