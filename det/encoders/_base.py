# @Time    : 2021/8/12 下午2:32
# @Author  : cattree
# @File    : _base
# @Software: PyCharm
# @explain :
class EncoderMixin:
    @property
    def out_channels(self):
        return self._out_channels[-4:]
