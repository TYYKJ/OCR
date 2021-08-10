# @Time    : 2021/8/9 下午5:13
# @Author  : cattree
# @File    : heads
# @Software: PyCharm
# @explain :
import torch.nn as nn


class CTCHead(nn.Module):
    def __init__(self, in_channels, n_class):
        super().__init__()
        self.fc = nn.Linear(in_channels, n_class)
        self.n_class = n_class

    def forward(self, x):
        return self.fc(x)
