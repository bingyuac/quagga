from quagga.matrix import Matrix
from quagga.context import Context
from quagga.matrix import MatrixList
from quagga.connector import Connector


class AddLastBlock(object):
    def __init__(self, x):
        self.x = x
        empty = Matrix.empty_like(x[0])
        self.context = Context(x[0].device_id)
        empty = Connector(empty, self.context, self.context)
        self.output = MatrixList(x[:] + [empty])

    def fprop(self):
        self.output.set_length(len(self.x) + 1)
        self.output[-1].fill(self.context, 0.0)

    def bprop(self):
        pass