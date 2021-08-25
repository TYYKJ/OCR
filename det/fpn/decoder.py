# @Time    : 2021/8/12 下午1:33
# @Author  : cattree
# @File    : decoder
# @Software: PyCharm
# @explain :
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3x3GNReLu(nn.Module):

    def __init__(self, in_channels, out_channels, up_sample=True):
        super(Conv3x3GNReLu, self).__init__()
        self.up_sample = up_sample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class FPNBlock(nn.Module):

    def __init__(self, pyramid_channels, skip_channels):
        super(FPNBlock, self).__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class OCRBlock(nn.Module):

    def __init__(self, in_channels, out_channels, n_up_sample=0):
        super(OCRBlock, self).__init__()

        blocks = [
            Conv3x3GNReLu(in_channels, out_channels, up_sample=bool(n_up_sample))
        ]

        if n_up_sample > 1:
            for _ in range(1, n_up_sample):
                blocks.append(
                    Conv3x3GNReLu(
                        out_channels, out_channels, up_sample=True
                    )
                )
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):

    def __init__(self, policy):
        super(MergeBlock, self).__init__()
        if policy not in ['add', 'cat']:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy)
            )


class FPNDecoder(nn.Module):

    def __init__(
            self,
            encoder_channels,
            pyramid_channels=256,
            ocr_channels=128,
            dropout=0.2,
            merge_policy='add'
    ):
        super(FPNDecoder, self).__init__()

        self.out_channels = ocr_channels if merge_policy == 'add' else ocr_channels * 4
        encoder_channels = encoder_channels[::-1]

        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.ocr_blocks = nn.ModuleList([
            OCRBlock(pyramid_channels, ocr_channels, n_up_sample=n_up_samples)
            for n_up_samples in [3, 2, 1, 0]
        ])

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, *x):
        c2, c3, c4, c5 = x

        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        feature_pyramid = [
            ocr_block(p) for ocr_block, p in zip(self.ocr_blocks, [p5, p4, p3, p2])
        ]

        x = self.merge(feature_pyramid)
        x = self.dropout(x)

        return x
