from torch import nn


class Feature2Seq(nn.Module):

    def __init__(self):
        super(Feature2Seq, self).__init__()

    def forward(self, x):
        batch, channel, height, width = x.shape

        if height != 1:
            raise ValueError(
                f'feature height must be 1, but got `{height}`'
            )

        x = x.view(batch, channel * height, width)
        # width batch feature
        x = x.permute(2, 0, 1)
        return x


class EncoderWithLSTM(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(EncoderWithLSTM, self).__init__()
        hidden_size = kwargs.get('hidden_size')
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(in_channels, hidden_size, bidirectional=True, num_layers=2)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x
