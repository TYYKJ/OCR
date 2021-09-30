# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 10:53
# @Author  : zhoujun

"""
此模块包含了检测算法的图片预处理组件，如随机裁剪，随机缩放，随机旋转，label制作等
"""
from .augment import *
from .iaa_augment import IaaAugment
from .make_border_map import MakeBorderMap
from .make_shrink_map import MakeShrinkMap
from .random_crop_data import EastRandomCropData, PSERandomCrop
