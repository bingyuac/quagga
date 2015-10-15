from itertools import izip
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class LastSelectorBlock(object):
    def __init__(self, x):
        device_id = x[0].device_id
        learning = x[0].bpropagable
        self.context = Context(device_id)
        self.output = Matrix.empty_like(x[0])
        self.output = Connector(self.output, device_id if learning else None)
        if learning:
            self.x, self.dL_dx = izip(*x.register_usage(device_id, device_id))
        else:
            self.x = x.register_usage(device_id)
        self.last_idx = x.length - 1

    def fprop(self):
        self.output.assign(self.context, self.x[self.last_idx])
        self.output.fprop()

    def bprop(self):
        self.dL_dx[self.last_idx].add(self.context, self.output.backward_matrix)