from quagga.blocks import HorizontalStackBlock
from quagga.blocks import VerticalStackBlock


class MergeBlock(object):
    def __init__(self, *matrices, **kwargs):
        if kwargs['axis'] == 0:
            self.block = VerticalStackBlock(matrices, device_id=kwargs.get('device_id'))
        elif kwargs['axis'] == 1:
            self.block = HorizontalStackBlock(matrices, device_id=kwargs.get('device_id'))
        else:
            raise ValueError('MergeBlock can stack matrices only '
                             'horizontally or vertically.')

    def __getattr__(self, name):
        setattr(self, name, getattr(self.block, name))
        return getattr(self, name)