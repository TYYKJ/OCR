
class EncoderMixin:
    @property
    def out_channels(self):
        return self._out_channels[-4:]
