# @Time    : 2021/9/26 下午3:25
# @Author  : 
# @File    : txt_dataset
# @Software: PyCharm
# @explain :
import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from ..transform import *
from ..transform.create_rec_aug import pil2cv, cv2pil


class RecDataProcess:
    def __init__(self, input_h, mean, std):
        """
        文本是被数据增广类

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

    @staticmethod
    def width_pad_img(_img, _target_width, _pad_value=0):
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
            augmentation=False
    ):
        self.img_path = img_path
        self.augmentation = augmentation
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

    def find_max_length(self):
        return max({len(_[1]) for _ in self.labels})

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # get img_path and trans
        img_name, trans = self.labels[index]
        # read img
        img_path = os.path.join(self.img_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augmentation:
            img = pil2cv(self.process.aug_img(cv2pil(img)))

        return {'img': img, 'label': trans}
