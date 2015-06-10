from quagga.blocks import HStackBlock
from quagga.blocks import VStackBlock


class MergeBlock(object):
    def __init__(self, f_matrix, s_matrix, axis, **kwargs):
        if axis == 0:
            self.block = VStackBlock(f_matrix, s_matrix, kwargs['max_ncols'])
        elif axis == 1:
            self.block = HStackBlock(f_matrix, s_matrix, kwargs['f_max_ncols'], kwargs['s_max_ncols'])
        else:
            raise ValueError('MergeBlock can stack matrices only '
                             'horizontally or vertically.')

    def __getattr__(self, name):
        setattr(self, name, getattr(self.block, name))
        return getattr(self, name)