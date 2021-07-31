import torch.nn as nn

from . import base


__all__ = ['get_loss']


class CTCLoss(nn.CTCLoss, base.Loss):
    pass


def get_loss(name: str = 'ctc'):
    all_loss = ['ctc']
    loss_name = name.lower()
    if loss_name not in all_loss:
        raise ValueError(
            f'loss name must be {all_loss}, got `{name}`'
        )
    if loss_name == 'ctc':
        return CTCLoss(blank=0)
