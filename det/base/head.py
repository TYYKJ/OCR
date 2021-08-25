# @Time    : 2021/8/12 下午2:54
# @Author  : cattree
# @File    : head
# @Software: PyCharm
# @explain :
import math

import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1,
                               bias=False)
        self.sigmoid1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(in_channels, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(in_channels, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.scope = 512

    def forward(self, x):
        score = self.sigmoid1(self.conv1(x))
        loc = self.sigmoid2(self.conv2(x)) * self.scope
        angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi
        geo = torch.cat((loc, angle), 1)
        return score, geo
